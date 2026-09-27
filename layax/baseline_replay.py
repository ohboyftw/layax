"""Replay dumps (see ``replay.py``) for the FastFit and fine-tuned classifier baselines."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .baselines import classifier_logits, train_finetuned_classifier
from .baselines_ext import PubCaps, _fastfit_logits, _fastfit_trainer, _label_texts
from .config import RunConfig
from .replay import replay_parts, write_replay


def _hidden_size(backbone: str) -> int:
    from transformers import AutoConfig
    return int(AutoConfig.from_pretrained(backbone).hidden_size)


def run_fastfit_replay(cfg: RunConfig, out_dir: str, device: str,
                       caps: Optional[PubCaps] = None) -> Dict[str, Any]:
    caps = caps or PubCaps()
    parts = replay_parts(cfg)
    train = dict(parts)["train"]
    texts = _label_texts(train)
    trainer = _fastfit_trainer(train, texts, cfg.li.seed, device, caps)
    trainer.train()
    # FastFit scores token embeddings with no projection head, so proj_dim is the backbone width.
    engine = {"interaction": "fastfit", "proj_dim": _hidden_size(caps.fastfit_backbone),
              "backbone": caps.fastfit_backbone,
              "caps": {"max_steps": caps.fastfit_max_steps, "batch_size": caps.fastfit_batch_size}}
    return write_replay(out_dir, parts, lambda rows: _fastfit_logits(trainer, rows, texts, caps),
                        trainer.model.state_dict(), engine)


def run_classifier_replay(cfg: RunConfig, out_dir: str, device: str,
                          hf_token: Optional[str] = None) -> Dict[str, Any]:
    """The ``classifier-finetuned`` baseline arm: same encoder, training and budget."""
    from .runtime import load_base
    parts = replay_parts(cfg)
    model, tok = load_base(cfg.li, token=hf_token)
    enc, head, _ = train_finetuned_classifier(model.encoder, tok, dict(parts)["train"],
                                              cfg.li, device)
    engine = {"interaction": "classifier", "proj_dim": int(head.in_features),
              "backbone": "%s/%s" % (cfg.li.base_checkpoint, cfg.li.base_subfolder)}
    state = {**{"encoder." + k: v for k, v in enc.state_dict().items()},
             **{"head." + k: v for k, v in head.state_dict().items()}}
    return write_replay(out_dir, parts, lambda rows: classifier_logits(enc, head, tok, rows, cfg.li, device),
                        state, engine)
