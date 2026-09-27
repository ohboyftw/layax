"""The default abstention gate: Learn-then-Test on max-softmax.

The learned competence head is off by default. At init_scale 20 it lost to max-softmax on
AURC in every Banking77 run, and max-softmax already separates foreign-script rows from
English ones (README, "What did not work"). This module is what replaces
it: the same Learn-then-Test procedure, on the temperature-scaled max-softmax.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .competence import aurc, fit_abstention_threshold, score_stats, walk_start
from .config import CompConfig
from .evaluate import softmax_rows


def msp(features: Sequence[Dict[str, Any]],
        temperatures: Optional[Dict[str, float]] = None) -> np.ndarray:
    return np.array([float(p.max()) for p in softmax_rows(features, temperatures)])


def fit_msp_gate(reference: Sequence[Dict[str, Any]], calibration: Sequence[Dict[str, Any]],
                 cal_correct: np.ndarray, temperatures: Dict[str, float],
                 cfg: CompConfig) -> Dict[str, Any]:
    """Walk start from ``reference`` (the competence split), threshold from calibration.

    Using a split other than calibration for the walk start keeps the tested threshold
    family independent of the rows the bound is computed on.
    """
    start = walk_start(msp(reference, temperatures), cfg.min_coverage)
    fit = fit_abstention_threshold(msp(calibration, temperatures), cal_correct, cfg.target_risk,
                                   cfg.delta, cfg.min_coverage, start_threshold=start,
                                   gate_score="max_softmax")
    fit.pop("curve", None)
    return fit


def msp_by_shift(features: Sequence[Dict[str, Any]], correct: np.ndarray,
                 shifts: Sequence[str],
                 temperatures: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
    """AURC of max-softmax and energy, plus mean max-softmax, per shift type and overall."""
    conf = msp(features, temperatures)
    energy = -np.array([score_stats(f["logits"], f["option_mask"])[5] for f in features])
    y = np.asarray(correct, dtype=np.float64)
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(shifts):
        groups.setdefault(s, []).append(i)
    groups["all"] = list(range(len(y)))
    out = {}
    for name, idx in sorted(groups.items()):
        ii = np.array(idx)
        out[name] = {"n": len(idx), "accuracy": float(y[ii].mean()),
                     "mean_msp": float(conf[ii].mean()),
                     "aurc_msp": aurc(conf[ii], y[ii]), "aurc_energy": aurc(energy[ii], y[ii])}
    return out
