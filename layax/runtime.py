"""Inference runtime: a drop-in replacement for ``laya.Agent.predict``.

The returned dict keeps laya's shape (``answers``/``usage``, and per answer
``choice``/``score``/``noul`` plus ``probabilities`` and ``confidence``) so code written
for Laya does not have to change. Two things are added:

* ``answers[qid]["abstain"]`` when a gate threshold (``msp_threshold``) is set, and
  ``["competence"]`` too when the opt-in competence head is attached.
* ``routing``/``cache`` blocks describing what actually ran.

The production path is: warm the option cache once per schema, then every request is
one state forward pass plus a matmul.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .competence import CompetenceModel
from .config import LIConfig
from .li_head import (
    QTYPES,
    LateInteractionDecisionModel,
    OptionCache,
    build_option_sequence,
    build_state_sequence,
    download_laya,
    option_labels,
    pad_stack,
    render_options,
    serialize_state,
)

TEMP_MIN, TEMP_MAX = 0.5, 5.0


def clamp_temperature(t: Any, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """Same guard as upstream: a fitted temperature below 0.5 sharpens rather than softens.

    Upstream ships ``choice:11+`` at 0.1006, which multiplies logits about tenfold and
    publishes a 0.24 top probability as 0.99. Anything gating on confidence is then
    told a coin flip is a certainty.
    """
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % ({0: "choice", 1: "score", 2: "noul"}[int(qtype)], size)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


class LayaxAgent:
    """Late-interaction decision agent with an optional competence gate."""

    def __init__(self, model: LateInteractionDecisionModel, tokenizer: Any,
                 device: Optional[str] = None,
                 temperatures: Optional[Dict[str, float]] = None,
                 competence: Optional[CompetenceModel] = None,
                 revision: str = "layax-0.1.0",
                 option_chunk: int = 256,
                 msp_threshold: Optional[float] = None):
        self.model = model
        self.tok = tokenizer
        self.cfg: LIConfig = model.cfg
        self.competence = competence
        # Learn-then-Test threshold on the temperature-scaled max-softmax: the default gate
        # when no competence head is attached. None means no threshold met the target.
        self.msp_threshold = msp_threshold
        self.option_chunk = int(option_chunk)

        if device is not None:
            dev = torch.device(device)
            if dev.type == "cuda" and not torch.cuda.is_available():
                print("[layax] cuda requested but unavailable; running on cpu")
                dev = torch.device("cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.model.to(self.device).eval()

        self.dtype = torch.float32
        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)[0]
            self.dtype = torch.bfloat16 if (cap >= 8 and self.cfg.amp_dtype == "bf16") else torch.float16

        raw = temperatures or {}
        self.temperature_raw = dict(raw)
        self.temperature = {k: clamp_temperature(v) for k, v in raw.items()}
        self.cache = OptionCache(revision, self.cfg.interaction, self.cfg.proj_dim)
        self.model._option_cache = self.cache

    # -- construction --------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, directory: str, device: Optional[str] = None,
                        subfolder: str = "") -> "LayaxAgent":
        """Load a layax run directory, or ``subfolder`` of a Hugging Face Hub repo id."""
        from transformers import AutoTokenizer

        if not os.path.isdir(directory):
            from huggingface_hub import snapshot_download
            directory = snapshot_download(directory, allow_patterns=[subfolder + "/*" if subfolder else "*"])
        directory = os.path.join(directory, subfolder)
        with open(os.path.join(directory, "layax_config.json")) as f:
            blob = json.load(f)
        cfg = LIConfig.from_dict(blob["li"])
        enc_dir = os.path.join(directory, "encoder")
        tok_dir = os.path.join(directory, "tokenizer")
        tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else cfg.base_checkpoint)

        from transformers import AutoConfig, AutoModel
        if os.path.isdir(enc_dir):
            enc = AutoModel.from_config(AutoConfig.from_pretrained(enc_dir), attn_implementation="sdpa")
        else:
            raise FileNotFoundError("no encoder/ directory in %s" % directory)
        try:
            enc.config.reference_compile = False
        except Exception:
            pass

        model = LateInteractionDecisionModel(enc, cfg)
        st_path = os.path.join(directory, "model.safetensors")
        if os.path.exists(st_path):
            from safetensors.torch import load_model
            load_model(model, st_path, strict=True)
        else:  # training runs write a pickled state dict
            model.load_state_dict(torch.load(os.path.join(directory, "model.pt"), map_location="cpu"),
                                  strict=True)

        comp = None
        if os.path.exists(os.path.join(directory, "competence_meta.json")):
            comp = CompetenceModel.load(directory)

        return cls(model, tok, device=device,
                   temperatures=blob.get("temperatures", {}),
                   competence=comp, revision=blob.get("revision", "layax-0.1.0"),
                   msp_threshold=blob.get("msp_threshold"))

    # -- option cache ---------------------------------------------------------------

    @torch.no_grad()
    def warm_cache(self, questions: Dict[str, Dict[str, Any]], batch_size: int = 64) -> Dict[str, Any]:
        """Encode every option in a schema once.

        Call this at startup for each schema you serve. After it, per-request cost no
        longer scales with the option count -- which is the whole point of moving the
        options out of the state sequence.
        """
        texts: List[str] = []
        for q in questions.values():
            texts.extend(render_options(q["type"], q.get("criteria")))
        todo = [t for t in dict.fromkeys(texts) if self.cache.get(t) is None]
        t0 = time.perf_counter()
        for i in range(0, len(todo), batch_size):
            chunk = todo[i: i + batch_size]
            seqs = [build_option_sequence(self.tok, t, self.cfg.option_max_len) for t in chunk]
            ids, att = pad_stack(seqs, self.tok.pad_token_id)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                                enabled=self.device.type == "cuda"):
                emb, mask = self.model.encode_options(ids.to(self.device), att.to(self.device))
            for j, t in enumerate(chunk):
                self.cache.put(t, emb[j], mask[j])
        return {"encoded": len(todo), "total_options": len(set(texts)),
                "seconds": round(time.perf_counter() - t0, 4), **self.cache.stats()}

    def _option_tensors(self, option_texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cached-or-encode, returning [K, m, *] and [K, m] for one question."""
        missing = [t for t in option_texts if self.cache.get(t) is None]
        if missing:
            seqs = [build_option_sequence(self.tok, t, self.cfg.option_max_len) for t in missing]
            ids, att = pad_stack(seqs, self.tok.pad_token_id)
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype,
                                                 enabled=self.device.type == "cuda"):
                emb, mask = self.model.encode_options(ids.to(self.device), att.to(self.device))
            for j, t in enumerate(missing):
                self.cache.put(t, emb[j], mask[j])

        embs, masks = [], []
        for t in option_texts:
            e, m = self.cache.get(t)
            embs.append(e)
            masks.append(m)
        m_max = max(e.shape[0] for e in embs)
        d = embs[0].shape[1]
        out = torch.zeros((len(embs), m_max, d), dtype=torch.float32)
        out_mask = torch.zeros((len(embs), m_max), dtype=torch.bool)
        for i, (e, m) in enumerate(zip(embs, masks)):
            out[i, : e.shape[0]] = e.float()
            out_mask[i, : m.shape[0]] = m
        return out, out_mask

    # -- prediction -----------------------------------------------------------------

    @torch.no_grad()
    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]],
                return_features: bool = False) -> Dict[str, Any]:
        """Evaluate typed questions over one state. laya-compatible return shape."""
        qids = list(questions.keys())
        if not qids:
            raise ValueError("questions must not be empty")

        state_seqs, truncations, qtypes, opt_texts, labels = [], [], [], [], []
        for qid in qids:
            q = questions[qid]
            qt = q["type"]
            if qt not in QTYPES:
                raise ValueError("question %r has unknown type %r" % (qid, qt))
            ins = q["instructions"]
            if not isinstance(ins, str):
                ins = json.dumps(ins, ensure_ascii=False)
            seq, trunc = build_state_sequence(self.tok, state, qt, ins, self.cfg.state_max_len)
            state_seqs.append(seq)
            truncations.append(trunc)
            qtypes.append(QTYPES[qt])
            opt_texts.append(render_options(qt, q.get("criteria")))
            labels.append(option_labels(qt, q.get("criteria")))

        ids, att = pad_stack(state_seqs, self.tok.pad_token_id)
        K = max(len(o) for o in opt_texts)
        per_q = [self._option_tensors(o) for o in opt_texts]
        m_dim = max(t[0].shape[1] for t in per_q)
        d = per_q[0][0].shape[2]
        opts = torch.zeros((len(qids), K, m_dim, d), dtype=torch.float32)
        opt_tok_mask = torch.zeros((len(qids), K, m_dim), dtype=torch.bool)
        option_mask = torch.zeros((len(qids), K), dtype=torch.bool)
        for i, (e, mk) in enumerate(per_q):
            k, m = e.shape[0], e.shape[1]
            opts[i, :k, :m] = e
            opt_tok_mask[i, :k, :m] = mk
            option_mask[i, :k] = True

        qt_t = torch.tensor(qtypes, dtype=torch.long)
        t0 = time.perf_counter()
        with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                            enabled=self.device.type == "cuda"):
            logits, pooled, stats = self.model(
                ids.to(self.device), att.to(self.device),
                None, opt_tok_mask.to(self.device), option_mask.to(self.device),
                qt_t.to(self.device), option_embeddings=opts.to(self.device))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logits_np = logits.float().cpu().numpy()
        pooled_np = pooled.float().cpu().numpy()
        om_np = option_mask.numpy()
        state_text = serialize_state(state)

        feature_rows = []
        for r, qid in enumerate(qids):
            feature_rows.append({
                "pooled": pooled_np[r], "logits": logits_np[r], "option_mask": om_np[r],
                "max_sim": float(stats["max_sim"][r]), "mean_sim": float(stats["mean_sim"][r]),
                "std_sim": float(stats["std_sim"][r]),
                "n_state_tokens": int(att[r].sum()), "n_truncated": truncations[r],
                "n_options": int(om_np[r].sum()), "state_text": state_text,
            })

        comp_raw = comp_scores = None
        if self.competence is not None:
            comp_raw = self.competence.score_raw(feature_rows, device=str(self.device))
            comp_scores = self.competence.calibrate(comp_raw)

        answers: Dict[str, Any] = {}
        for r, qid in enumerate(qids):
            qt = questions[qid]["type"]
            k = int(om_np[r].sum())
            bucket = temp_bucket(QTYPES[qt], k)
            t_scale = self.temperature.get(bucket, 1.0)
            z = logits_np[r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            conf = round(confidence_from_probs(p, k), 4)
            lab = labels[r]

            if qt == "choice":
                ans: Dict[str, Any] = {
                    "type": "choice",
                    "choice": lab[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(lab, p)},
                    "confidence": conf,
                }
            elif qt == "score":
                ans = {
                    "type": "score",
                    "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(questions[qid].get("criteria", []))},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf,
                }
            else:
                ans = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                }

            ans["interaction"] = {"max_sim": round(float(stats["max_sim"][r]), 4),
                                  "mean_sim": round(float(stats["mean_sim"][r]), 4)}
            if truncations[r] > 0:
                ans["truncated_state_tokens"] = truncations[r]
            if comp_scores is not None:
                ans["competence"] = round(float(comp_scores[r]), 4)
                thr = self.competence.threshold
                ans["abstain"] = bool(comp_raw[r] < thr) if thr is not None else None
                if thr is None:
                    ans["abstain_note"] = ("no threshold met the risk target on calibration data; "
                                           "gate disabled, competence reported for inspection only")
            elif self.msp_threshold is not None:
                ans["abstain"] = bool(float(p.max()) < self.msp_threshold)
            answers[qid] = ans

        out = {
            "model": "layax-%s" % self.cfg.interaction,
            "answers": answers,
            "usage": {"input_tokens": int(att.sum()), "output_tokens": 0},
            "timing": {"forward_ms": round(elapsed_ms, 3)},
            "cache": self.cache.stats(),
        }
        if return_features:
            out["features"] = feature_rows
        return out

    # -- persistence ----------------------------------------------------------------

    def save(self, directory: str) -> None:
        from safetensors.torch import save_model

        os.makedirs(directory, exist_ok=True)
        save_model(self.model, os.path.join(directory, "model.safetensors"))
        self.model.encoder.config.save_pretrained(os.path.join(directory, "encoder"))
        self.tok.save_pretrained(os.path.join(directory, "tokenizer"))
        with open(os.path.join(directory, "layax_config.json"), "w") as f:
            json.dump({"li": self.cfg.to_dict(), "temperatures": self.temperature_raw,
                       "revision": self.cache.revision, "msp_threshold": self.msp_threshold},
                      f, indent=2)
        if self.competence is not None:
            self.competence.save(directory)


def load_base(cfg: LIConfig, token: Optional[str] = None,
              device: Optional[str] = None) -> Tuple[LateInteractionDecisionModel, Any]:
    """Download the Laya checkpoint and build an initialised late-interaction model."""
    from transformers import AutoTokenizer

    d = download_laya(cfg.base_checkpoint, cfg.base_subfolder, token=token)
    tok_dir = os.path.join(d, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else cfg.base_checkpoint)
    model = LateInteractionDecisionModel.from_laya(cfg, checkpoint_dir=d)
    return model, tok
