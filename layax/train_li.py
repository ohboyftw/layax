"""Train the late-interaction option head.

The one non-obvious decision here is training-time option sampling. Encoding every
option for every row on every step means 77 (or 150) extra encoder passes per example,
and that is what would not fit on a T4. Training therefore sees the gold option plus a
sample of negatives, while evaluation always scores the complete label set. This is
standard practice for large label spaces, and it is also the mechanism that makes the
in-batch negative term worth having.

It runs as the training stage of ``python -m layax.cli run --config configs/banking77.json``.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import LIConfig
from .data import Example
from .li_head import (
    QTYPES,
    LateInteractionDecisionModel,
    build_option_sequence,
    build_state_sequence,
    pad_stack,
    render_options,
)
from .losses import combined_loss, in_batch_negative_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DecisionDataset(Dataset):
    """Examples -> tokenised tensors, with negatives sampled per epoch."""

    def __init__(self, rows: Sequence[Example], tok: Any, cfg: LIConfig,
                 train: bool = True, seed: int = 17):
        self.rows = list(rows)
        self.tok = tok
        self.cfg = cfg
        self.train = train
        self.rng = random.Random(seed)
        self.hard_negatives: Dict[int, List[int]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _pick_options(self, ex: Example, idx: int) -> Tuple[List[str], int]:
        """-> (option texts actually scored, index of the gold among them)."""
        texts = render_options(ex.qtype, ex.criteria)
        n = len(texts)
        k = self.cfg.train_options_per_row
        if not self.train or not k or k >= n:
            return texts, ex.label

        gold = ex.label
        pool = [i for i in range(n) if i != gold]
        if self.cfg.option_sampling == "hard" and idx in self.hard_negatives:
            hard = [i for i in self.hard_negatives[idx] if i != gold]
            # Half hard, half random: pure hard-negative sampling collapses onto a few
            # confusable labels and stops teaching the rest of the taxonomy.
            n_hard = min(len(hard), max(1, (k - 1) // 2))
            chosen = hard[:n_hard]
            rest = [i for i in pool if i not in set(chosen)]
            self.rng.shuffle(rest)
            chosen += rest[: (k - 1 - n_hard)]
        else:
            self.rng.shuffle(pool)
            chosen = pool[: k - 1]

        keep = [gold] + chosen
        self.rng.shuffle(keep)
        return [texts[i] for i in keep], keep.index(gold)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        opts, gold = self._pick_options(ex, idx)
        state_ids, truncated = build_state_sequence(
            self.tok, ex.state, ex.qtype, ex.instructions, self.cfg.state_max_len)
        opt_ids = [build_option_sequence(self.tok, t, self.cfg.option_max_len) for t in opts]
        return {"state_ids": state_ids, "opt_ids": opt_ids, "label": gold,
                "qtype": QTYPES[ex.qtype], "row": idx, "truncated": truncated,
                "n_options_total": len(render_options(ex.qtype, ex.criteria))}


def collate(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad to [B, L] states and [B, K, m] options, and build one-hot targets."""
    state_ids, state_mask = pad_stack([b["state_ids"] for b in batch], pad_id)
    K = max(len(b["opt_ids"]) for b in batch)
    flat, owner = [], []
    for i, b in enumerate(batch):
        for o in b["opt_ids"]:
            flat.append(o)
            owner.append(i)
    oid, oatt = pad_stack(flat, pad_id)
    m = oid.shape[1]
    B = len(batch)
    opt_ids = torch.full((B, K, m), pad_id, dtype=torch.long)
    opt_mask = torch.zeros((B, K, m), dtype=torch.long)
    option_mask = torch.zeros((B, K), dtype=torch.bool)
    pos = 0
    for i, b in enumerate(batch):
        k = len(b["opt_ids"])
        opt_ids[i, :k] = oid[pos: pos + k]
        opt_mask[i, :k] = oatt[pos: pos + k]
        option_mask[i, :k] = True
        pos += k

    label = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    target = torch.zeros((B, K), dtype=torch.float32)
    target[torch.arange(B), label] = 1.0
    return {
        "state_ids": state_ids, "state_mask": state_mask,
        "opt_ids": opt_ids, "opt_mask": opt_mask, "option_mask": option_mask,
        "label": label, "target": target,
        "qtype": torch.tensor([b["qtype"] for b in batch], dtype=torch.long),
        "row": torch.tensor([b["row"] for b in batch], dtype=torch.long),
    }


def build_optimizer(model: LateInteractionDecisionModel, cfg: LIConfig, total_steps: int):
    """Two learning rates: the pretrained encoder moves slowly, the new heads do not.

    The projections start from random init, so giving them the encoder's learning rate
    would spend the first epoch dragging a good encoder toward a bad head.
    """
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if n.startswith("encoder.") else head_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": cfg.lr_encoder},
         {"params": head_params, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay)
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


@torch.no_grad()
def refresh_hard_negatives(model, ds: DecisionDataset, tok, device, cfg: LIConfig,
                           max_rows: int = 2000, top_k: int = 16) -> int:
    """Record each row's highest-scoring wrong options, for the next epoch's sampling."""
    model.eval()
    n = min(len(ds), max_rows)
    updated = 0
    for idx in range(n):
        ex = ds.rows[idx]
        texts = render_options(ex.qtype, ex.criteria)
        if len(texts) <= cfg.train_options_per_row:
            continue
        sid, _ = build_state_sequence(tok, ex.state, ex.qtype, ex.instructions, cfg.state_max_len)
        ids, att = pad_stack([sid], tok.pad_token_id)
        oseqs = [build_option_sequence(tok, t, cfg.option_max_len) for t in texts]
        oid, oatt = pad_stack(oseqs, tok.pad_token_id)
        K, m = oid.shape
        logits, _, _ = model(ids.to(device), att.to(device),
                             oid.reshape(1, K, m).to(device), oatt.reshape(1, K, m).to(device),
                             torch.ones((1, K), dtype=torch.bool).to(device),
                             torch.tensor([QTYPES[ex.qtype]]).to(device))
        order = torch.argsort(logits[0], descending=True).cpu().tolist()
        ds.hard_negatives[idx] = [i for i in order if i != ex.label][:top_k]
        updated += 1
    model.train()
    return updated


def train(model: LateInteractionDecisionModel, tok: Any, train_rows: Sequence[Example],
          cfg: LIConfig, device: Optional[str] = None, eval_rows: Optional[Sequence[Example]] = None,
          log_every: int = 25, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Fit the head. Returns a run log; the caller decides what to persist."""
    set_seed(cfg.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).train()

    ds = DecisionDataset(train_rows, tok, cfg, train=True, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                    collate_fn=lambda b: collate(b, tok.pad_token_id), drop_last=False)

    steps_per_epoch = max(1, len(dl) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    opt, sched = build_optimizer(model, cfg, total_steps)

    use_amp = dev.type == "cuda" and cfg.amp_dtype != "fp32"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    # bf16 does not need loss scaling; fp16 on a T4 does.
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    history: List[Dict[str, Any]] = []
    step = 0
    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        if cfg.freeze_encoder_epochs and epoch < cfg.freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
        elif cfg.freeze_encoder_epochs and epoch == cfg.freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad_(True)

        if cfg.option_sampling == "hard" and epoch > 0:
            n = refresh_hard_negatives(model, ds, tok, dev, cfg)
            print("[layax] refreshed hard negatives for %d rows" % n, flush=True)

        # Reset per epoch while `step` keeps counting, so the first log of an epoch can
        # cover fewer than log_every steps: average over what was actually accumulated.
        running: Dict[str, float] = {}
        n_running = 0
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(dl):
            b = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logits, pooled, stats = model(
                    b["state_ids"], b["state_mask"], b["opt_ids"], b["opt_mask"],
                    b["option_mask"], b["qtype"])

                negatives = None
                if cfg.in_batch_negatives and cfg.interaction == "maxsim" and logits.size(0) > 1:
                    st_proj, _, _, st_mask = model.encode_state(
                        b["state_ids"], b["state_mask"], b["qtype"])
                    B, K, m = b["opt_ids"].shape
                    gold = b["label"]
                    g_ids = b["opt_ids"][torch.arange(B), gold]
                    g_att = b["opt_mask"][torch.arange(B), gold]
                    g_att = torch.where(g_att.sum(-1, keepdim=True) > 0, g_att,
                                        F.pad(torch.ones_like(g_att[:, :1]), (0, g_att.size(1) - 1)))
                    g_emb, g_mask = model.encode_options(g_ids, g_att)
                    negatives = in_batch_negative_loss(
                        st_proj, st_mask, g_emb, g_mask,
                        model.logit_scale[b["qtype"]])

                parts = combined_loss(logits.float(), b["target"], b["label"],
                                      b["option_mask"], b["qtype"], cfg, negatives)
                loss = parts["total"] / cfg.grad_accum

            scaler.scale(loss).backward()

            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1

                for k, v in parts.items():
                    running[k] = running.get(k, 0.0) + float(v.detach() if hasattr(v, "detach") else v)
                n_running += 1
                if step % log_every == 0:
                    rec = {"epoch": epoch, "step": step,
                           **{k: round(v / n_running, 4) for k, v in running.items()},
                           "lr_head": round(sched.get_last_lr()[-1], 8),
                           "elapsed_s": round(time.perf_counter() - t_start, 1)}
                    history.append(rec)
                    print("[layax] " + json.dumps(rec), flush=True)
                    running = {}
                    n_running = 0

        if eval_rows:
            from .evaluate import quick_accuracy
            acc = quick_accuracy(model, tok, eval_rows, cfg, dev)
            rec = {"epoch": epoch, "eval_accuracy": round(acc, 4)}
            history.append(rec)
            print("[layax] " + json.dumps(rec), flush=True)
            model.train()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    return {"history": history, "steps": step,
            "minutes": round((time.perf_counter() - t_start) / 60, 2)}
