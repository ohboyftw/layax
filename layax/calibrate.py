"""Temperature fitting for the decision logits, per (question type, option count) bucket.

Same bucketing as upstream (``choice:2``, ``choice:3-5``, ``choice:6-10``, ``choice:11+``,
and the same for score and noul) so fitted values are directly comparable to the ones a
Laya checkpoint ships.

Two guards, both learned from upstream's shipped values:

* A fitted temperature is clamped to [0.5, 5.0]. Below 0.5 it sharpens rather than
  softens: upstream ships ``choice:11+`` at 0.1006, which republishes a 0.24 top
  probability as 0.99.
* A bucket with too few rows is left at 1.0 rather than fitted to noise, and the row
  count is reported next to every value so a thinly-fitted bucket is visible.

Temperature fixes average miscalibration. It cannot fix confidently wrong: no monotone
rescaling separates right from wrong when the errors carry the highest scores. That is
what the competence head is for, and the two are complementary rather than alternatives.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .competence import expected_calibration_error
from .runtime import TEMP_MAX, TEMP_MIN, clamp_temperature, temp_bucket


def _nll(logits: np.ndarray, gold: int, k: int, t: float) -> float:
    z = logits[:k] / max(t, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return -float(np.log(max(p[gold], 1e-12))) if 0 <= gold < k else 0.0


def fit_temperatures(features: Sequence[Dict[str, Any]], gold: Sequence[int],
                     qtypes: Optional[Sequence[int]] = None,
                     min_rows: int = 50, grid: Optional[Sequence[float]] = None
                     ) -> Dict[str, Any]:
    """Fit one temperature per bucket by NLL on a grid.

    A grid rather than gradient descent: it is one dimension, the objective is cheap,
    and a grid cannot diverge or land outside the clamp.
    """
    grid = list(grid) if grid is not None else list(np.geomspace(TEMP_MIN, TEMP_MAX, 60))
    buckets: Dict[str, List[int]] = {}
    for i, r in enumerate(features):
        k = int(np.asarray(r["option_mask"], dtype=bool).sum())
        qt = int(qtypes[i]) if qtypes is not None else 0
        buckets.setdefault(temp_bucket(qt, k), []).append(i)

    fitted: Dict[str, float] = {}
    report: Dict[str, Any] = {}
    for b, idx in sorted(buckets.items()):
        if len(idx) < min_rows:
            fitted[b] = 1.0
            report[b] = {"temperature": 1.0, "n": len(idx),
                         "note": "under %d rows; left at 1.0 rather than fitted to noise" % min_rows}
            continue
        best_t, best_nll = 1.0, float("inf")
        for t in grid:
            total = 0.0
            for i in idx:
                k = int(np.asarray(features[i]["option_mask"], dtype=bool).sum())
                total += _nll(np.asarray(features[i]["logits"], dtype=np.float64),
                              int(gold[i]), k, t)
            if total < best_nll:
                best_nll, best_t = total, t
        t_clamped = clamp_temperature(best_t)
        fitted[b] = t_clamped
        report[b] = {"temperature": round(t_clamped, 4), "n": len(idx),
                     "raw": round(float(best_t), 4),
                     "clamped": bool(abs(t_clamped - best_t) > 1e-6)}

    return {"temperatures": fitted, "report": report}


def ece_before_after(features: Sequence[Dict[str, Any]], gold: Sequence[int],
                     temperatures: Dict[str, float],
                     qtypes: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """What the fit actually bought, on the split it is measured on."""
    def conf_correct(temps: Optional[Dict[str, float]]):
        confs, corr = [], []
        for i, r in enumerate(features):
            mask = np.asarray(r["option_mask"], dtype=bool)
            k = int(mask.sum())
            z = np.asarray(r["logits"], dtype=np.float64)[:k]
            if temps:
                qt = int(qtypes[i]) if qtypes is not None else 0
                z = z / max(temps.get(temp_bucket(qt, k), 1.0), 1e-6)
            p = np.exp(z - z.max())
            p = p / p.sum()
            confs.append(float(p.max()))
            corr.append(float(int(np.argmax(p)) == int(gold[i])))
        return np.array(confs), np.array(corr)

    c0, y0 = conf_correct(None)
    c1, y1 = conf_correct(temperatures)
    return {"ece_before": expected_calibration_error(c0, y0),
            "ece_after": expected_calibration_error(c1, y1),
            "mean_conf_before": float(c0.mean()) if len(c0) else float("nan"),
            "mean_conf_after": float(c1.mean()) if len(c1) else float("nan")}
