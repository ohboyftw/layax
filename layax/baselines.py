"""Label-index classifier baselines on the same encoder and the same test rows.

Without these, layax is only ever compared to laya variants. For a fixed intent set the
standard answer is a classifier over label indices, and that is the arm the option
heads have to justify themselves against. Casanueva et al. (2020, arXiv 2003.04807)
report both shapes on Banking77 -- fine-tuned BERT-Large and frozen USE+ConveRT with an
MLP -- but on different encoders, so those are reference ranges, never these arms.

Both arms here:

* read the same state sequence as layax's state pass and use its CLS vector;
* start from the encoder weights they are handed, without mutating them;
* train on the decision-training split only;
* score every test row against the FULL label set.

A label index needs one shared label set. Rows whose criteria differ (per-row jsonl
criteria, distractor-augmented rows) have no fixed index, so ``label_space`` refuses them
rather than inventing one. The index comes from the training rows; a test label the
classifier never saw maps to -1, so it is scored wrong rather than skipped.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibrate import ece_before_after, fit_temperatures
from .competence import aurc, expected_calibration_error
from .config import LIConfig
from .data import Example
from .li_head import build_state_sequence, option_labels, pad_stack
from .runtime import TEMP_MAX, TEMP_MIN, temp_bucket


def label_space(rows: Sequence[Example]) -> List[str]:
    """The one label list every row shares, in index order. Raises if they differ."""
    first = option_labels(rows[0].qtype, rows[0].criteria)
    for ex in rows:
        if option_labels(ex.qtype, ex.criteria) != first:
            raise ValueError("classifier baselines need one shared label set; rows disagree "
                             "(%d vs %d labels)" % (len(first), len(option_labels(ex.qtype, ex.criteria))))
    return first


def test_gold(train_labels: Sequence[str], test_rows: Sequence[Example]) -> np.ndarray:
    """Test gold as indices into the training label list, -1 where the label is unseen."""
    index = {name: i for i, name in enumerate(train_labels)}
    return np.array([index.get(option_labels(ex.qtype, ex.criteria)[ex.label], -1)
                     for ex in test_rows])


def _state_batch(tok: Any, rows: Sequence[Example], cfg: LIConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = [build_state_sequence(tok, ex.state, ex.qtype, ex.instructions, cfg.state_max_len)[0]
            for ex in rows]
    return pad_stack(seqs, tok.pad_token_id)


def _cls(encoder: nn.Module, ids: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
    return encoder(input_ids=ids, attention_mask=att).last_hidden_state[:, 0]


@torch.no_grad()
def cls_features(encoder: nn.Module, tok: Any, rows: Sequence[Example], cfg: LIConfig,
                 device: Any, batch_size: int = 32) -> torch.Tensor:
    """CLS vectors of the state sequences, on CPU, in row order."""
    encoder.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        ids, att = _state_batch(tok, rows[i: i + batch_size], cfg)
        out.append(_cls(encoder, ids.to(device), att.to(device)).float().cpu())
    return torch.cat(out)


def _fit(forward: Callable[[torch.Tensor], torch.Tensor], labels: torch.Tensor,
         optimizer: torch.optim.Optimizer, epochs: int, batch_size: int,
         cfg: LIConfig, device: Any) -> List[float]:
    """Cross-entropy over label indices. ``forward`` maps a batch of row indices to logits."""
    dev_type = torch.device(device).type
    use_amp = dev_type == "cuda" and cfg.amp_dtype != "fp32"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    gen = torch.Generator().manual_seed(cfg.seed)
    history = []
    for _ in range(epochs):
        total = 0.0
        for idx in torch.randperm(len(labels), generator=gen).split(batch_size):
            with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                loss = F.cross_entropy(forward(idx).float(), labels[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(idx)
        history.append(round(total / len(labels), 5))
    return history


@torch.no_grad()
def _logits(head: nn.Module, x: torch.Tensor, device: Any) -> np.ndarray:
    head.eval()
    return head(x.to(device)).float().cpu().numpy()


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _as_features(logits: np.ndarray) -> List[Dict[str, Any]]:
    mask = np.ones(logits.shape[1], dtype=bool)
    return [{"logits": z, "option_mask": mask} for z in logits]


def fit_logit_temperature(cal_logits: np.ndarray, cal_gold: np.ndarray) -> Dict[str, Any]:
    """One temperature for a fixed-width classifier, fitted on calibration logits only.

    Reuses ``calibrate.fit_temperatures``: every row has the same option count, so every
    row lands in one bucket and the per-bucket fit is a single-temperature fit, with the
    same NLL grid, the same [0.5, 5.0] clamp and the same 50-row floor layax gets.
    """
    fit = fit_temperatures(_as_features(cal_logits), cal_gold)
    out = dict(fit["report"][temp_bucket(0, cal_logits.shape[1])])
    # The grid ends at the clamp, so a fit that wants T outside it lands on an endpoint
    # with clamped=False; ece_temp is then the clamped ECE, not the scaled optimum.
    out["at_bound"] = out["temperature"] in (TEMP_MIN, TEMP_MAX)
    return out


def temperature_scaled_ece(test_logits: np.ndarray, test_gold: np.ndarray,
                           t: float) -> float:
    """ECE on test after dividing the logits by ``t``; correctness as argmax == gold."""
    k = test_logits.shape[1]
    return ece_before_after(_as_features(test_logits), test_gold,
                            {temp_bucket(0, k): t})["ece_after"]


def group_report(correct: np.ndarray, conf: np.ndarray, test_rows: Sequence[Example],
                 key: str) -> Dict[str, Dict[str, Any]]:
    """Per-group n / accuracy / mean confidence, keyed by ``meta[key]`` as evaluate does."""
    groups = np.array([str(ex.meta.get(key, "unknown")) for ex in test_rows])
    return {g: {"n": int((groups == g).sum()), "accuracy": float(correct[groups == g].mean()),
                "mean_confidence": float(conf[groups == g].mean())}
            for g in sorted(set(groups))}


def logit_arm(name: str, test_logits: np.ndarray, cal_logits: np.ndarray,
              labels: Sequence[str], test_rows: Sequence[Example],
              cal_rows: Sequence[Example], t0: float,
              group_by: Optional[str] = None) -> Dict[str, Any]:
    """Same shape as every other arm in ``evaluate.compare``; MSP is the confidence.

    ``logits`` columns follow ``labels``. ``ece`` is raw (T = 1); ``ece_temp`` uses one
    temperature fitted on the calibration rows, never on test.
    """
    gold = test_gold(labels, test_rows)
    probs = _softmax(test_logits)
    correct = (probs.argmax(1) == gold).astype(np.float64)
    conf = probs.max(1)
    temp = fit_logit_temperature(cal_logits, test_gold(labels, cal_rows))
    out = {"arm": name, "n": len(gold), "accuracy": float(correct.mean()), "coverage": 1.0,
           "ece": expected_calibration_error(conf, correct), "aurc": aurc(conf, correct),
           "ece_temp": temperature_scaled_ece(test_logits, gold, temp["temperature"]),
           "temperature": temp, "errors": 0,
           "seconds": round(time.perf_counter() - t0, 2)}
    keys = [group_by] if group_by else []
    if any(ex.meta.get("label_seen") == "unseen" for ex in test_rows):
        keys.append("label_seen")
    for key in keys:
        out["by_" + key] = group_report(correct, conf, test_rows, key)
    return out


def frozen_mlp_baseline(encoder: nn.Module, tok: Any, train_rows: Sequence[Example],
                        cal_rows: Sequence[Example], test_rows: Sequence[Example],
                        cfg: LIConfig, device: Any, epochs: int = 30, hidden: int = 256,
                        lr: float = 1e-3, batch_size: int = 64,
                        group_by: Optional[str] = None) -> Dict[str, Any]:
    """Frozen encoder, CLS features extracted once, small MLP on top."""
    t0 = time.perf_counter()
    labels = label_space(train_rows)
    x_tr = cls_features(encoder, tok, train_rows, cfg, device)
    y_tr = torch.tensor([ex.label for ex in train_rows])
    torch.manual_seed(cfg.seed)
    mlp = nn.Sequential(nn.Linear(x_tr.shape[1], hidden), nn.GELU(), nn.Linear(hidden, len(labels))).to(device)
    mlp.train()
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    history = _fit(lambda idx: mlp(x_tr[idx].to(device)), y_tr, opt, epochs, batch_size, cfg, device)
    z_te, z_cal = (_logits(mlp, cls_features(encoder, tok, rows, cfg, device), device)
                   for rows in (test_rows, cal_rows))
    arm = logit_arm("classifier-frozen+mlp", z_te, z_cal, labels, test_rows, cal_rows, t0, group_by)
    arm["loss_history"] = history
    return arm


def train_finetuned_classifier(encoder: nn.Module, tok: Any, train_rows: Sequence[Example],
                               cfg: LIConfig, device: Any
                               ) -> Tuple[nn.Module, nn.Module, List[float]]:
    """Fine-tune a copy of ``encoder`` with a linear head on CLS. Returns (encoder, head, losses)."""
    torch.manual_seed(cfg.seed)
    enc = copy.deepcopy(encoder).to(device).train()
    head = nn.Linear(int(enc.config.hidden_size), len(label_space(train_rows))).to(device)

    def forward(idx: torch.Tensor) -> torch.Tensor:
        ids, att = _state_batch(tok, [train_rows[i] for i in idx.tolist()], cfg)
        return head(_cls(enc, ids.to(device), att.to(device)))

    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": cfg.lr_encoder},
                             {"params": head.parameters(), "lr": cfg.lr_head}],
                            weight_decay=cfg.weight_decay)
    labels = torch.tensor([ex.label for ex in train_rows])
    return enc, head, _fit(forward, labels, opt, cfg.epochs, cfg.batch_size, cfg, device)


def classifier_logits(enc: nn.Module, head: nn.Module, tok: Any, rows: Sequence[Example],
                      cfg: LIConfig, device: Any) -> np.ndarray:
    return _logits(head, cls_features(enc, tok, rows, cfg, device), device)


def finetuned_classifier_baseline(encoder: nn.Module, tok: Any, train_rows: Sequence[Example],
                                  cal_rows: Sequence[Example], test_rows: Sequence[Example],
                                  cfg: LIConfig, device: Any,
                                  group_by: Optional[str] = None) -> Dict[str, Any]:
    """Linear head on CLS, encoder unfrozen, cross-entropy. Trains a copy of ``encoder``.

    Uses layax's own learning rates, epochs and batch size, so the two differ in the
    head and the objective rather than in the training budget.
    """
    t0 = time.perf_counter()
    label_names = label_space(train_rows)
    enc, head, history = train_finetuned_classifier(encoder, tok, train_rows, cfg, device)
    z_te, z_cal = (classifier_logits(enc, head, tok, rows, cfg, device)
                   for rows in (test_rows, cal_rows))
    arm = logit_arm("classifier-finetuned", z_te, z_cal, label_names, test_rows, cal_rows, t0,
                    group_by)
    arm["loss_history"] = history
    return arm


def run_classifier_baselines(encoder: nn.Module, tok: Any, train_rows: Sequence[Example],
                             cal_rows: Sequence[Example], test_rows: Sequence[Example],
                             cfg: LIConfig, device: Any,
                             group_by: Optional[str] = None) -> List[Dict[str, Any]]:
    """Both arms, or one row saying why they could not run on this data.

    ``cal_rows`` only fit each arm's temperature; they are never trained on.
    """
    try:
        label_space(train_rows)
    except ValueError as e:
        return [{"arm": "classifier baselines (not run)", "n": 0, "errors": str(e)[:120]}]
    rows = (train_rows, cal_rows, test_rows, cfg, device)
    return [frozen_mlp_baseline(encoder, tok, *rows, group_by=group_by),
            finetuned_classifier_baseline(encoder, tok, *rows, group_by=group_by)]
