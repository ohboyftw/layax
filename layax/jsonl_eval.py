"""Evaluate a trained run on its own JSONL test rows and on a separate shift file.

The pipeline's test and shift reports assume the intent datasets (foreign MASSIVE rows,
the ``language`` grouping). A JSONL task brings its own shift set and its own slices, and
often a downstream decision rule: act only when the model is confident in one outcome.
``proceed_stats`` reports that rule the way a controller would use it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .data import Example
from .evaluate import collect_predictions, softmax_rows


def probabilities(agent: Any, rows: Sequence[Example], temperatures: Dict[str, float],
                  device: str) -> np.ndarray:
    res = collect_predictions(agent.model, agent.tok, rows, agent.cfg, device, batch_size=32)
    return np.stack(softmax_rows(res["features"], temperatures))


def selective(probs: np.ndarray, gold: np.ndarray, threshold: float) -> Dict[str, float]:
    ok = probs.argmax(1) == gold
    keep = probs.max(1) >= threshold
    return {"accuracy": float(ok.mean()), "coverage": float(keep.mean()),
            "selective_risk": float(1 - ok[keep].mean()) if keep.any() else None}


def proceed_stats(probs: np.ndarray, gold: np.ndarray, labels: List[str], threshold: float,
                  go: str, worst: str) -> Dict[str, float]:
    """Proceed when the argmax is ``go`` and its probability clears the threshold."""
    g, w = labels.index(go), labels.index(worst)
    proceed = (probs.argmax(1) == g) & (probs[:, g] >= threshold)
    n = int(proceed.sum())
    return {"proceed_rate": float(proceed.mean()), "n_proceed": n,
            "not_%s" % go: float((gold[proceed] != g).mean()) if n else None,
            worst: float((gold[proceed] == w).mean()) if n else None}


def slice_report(probs: np.ndarray, rows: Sequence[Example], labels: List[str],
                 threshold: float, go: str, worst: str, key: str, value: Any) -> Dict[str, Any]:
    """The full report, then the same for rows with ``meta[key] == value``."""
    gold = np.array([ex.label for ex in rows])
    out = {"n": len(rows), **selective(probs, gold, threshold),
           **proceed_stats(probs, gold, labels, threshold, go, worst)}
    m = np.array([ex.meta.get(key) == value for ex in rows])
    if m.any():
        out["%s=%s" % (key, value)] = {"n": int(m.sum()), **selective(probs[m], gold[m], threshold),
                                       **proceed_stats(probs[m], gold[m], labels, threshold, go, worst)}
    return out
