"""Replay dumps: an engine's raw logits on a run's own splits, for certifying a gate offline.

Every engine writes the same files from the same rows: competence, calibration and test
exactly as the layax run with this config and seed used, plus 400 MASSIVE rows in four
non-Latin scripts (gold null). Logits are raw, in the schema's option order; the consumer
applies ``engine.temperature``, which is fitted on calibration only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch

from .baselines import fit_logit_temperature, label_space, test_gold
from .config import RunConfig
from .data import Example

FOREIGN_LOCALES = ("km-KH", "am-ET", "te-IN", "ja-JP")


def replay_lines(split: str, rows: Sequence[Example], logits: np.ndarray) -> List[str]:
    gold = [None] * len(rows) if split == "foreign" else [int(ex.label) for ex in rows]
    return [json.dumps({"split": split, "state": ex.state, "gold": g,
                        "logits": [round(float(v), 5) for v in z],
                        "language": ex.meta.get("language", "")}, ensure_ascii=False)
            for ex, g, z in zip(rows, gold, logits)]


def replay_parts(cfg: RunConfig) -> List[tuple]:
    """(split name, rows) in dump order; the foreign rows carry the run's own schema."""
    from .pipeline import load_foreign_rows, load_splits
    splits = load_splits(cfg)
    ref = splits.test[0]
    foreign = [Example(e.state, ref.qtype, ref.instructions, ref.criteria, 0, dict(e.meta))
               for e in load_foreign_rows(FOREIGN_LOCALES, max_rows=100)[1]]
    return [("train", splits.train), ("competence", splits.competence),
            ("calibration", splits.calibration), ("test", splits.test), ("foreign", foreign)]


def write_replay(out_dir: str, parts: Sequence[tuple], score: Callable[[Sequence[Example]], np.ndarray],
                 state_dict: Dict[str, torch.Tensor], engine: Dict[str, Any]) -> Dict[str, Any]:
    """Score every split but train, write rows.jsonl + model.pt + manifest.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    named = dict(parts)
    dumped = [(n, r) for n, r in parts if n != "train"]
    logits = {n: score(r) for n, r in dumped}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for n, r in dumped:
            f.write("\n".join(replay_lines(n, r, logits[n])) + "\n")
    torch.save(state_dict, out / "model.pt")
    labels, ref = label_space(named["train"]), named["test"][0]
    temp = fit_logit_temperature(logits["calibration"], test_gold(labels, named["calibration"]))
    manifest = {"engine": {"run": out.name,
                           "model_sha256": hashlib.sha256((out / "model.pt").read_bytes()).hexdigest(),
                           "temperature": temp["temperature"], **engine},
                "schema": {"type": "choice", "instructions": ref.instructions,
                           "options": list(ref.criteria), "option_text": list(ref.criteria.values())},
                "counts": {n: len(r) for n, r in dumped}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
