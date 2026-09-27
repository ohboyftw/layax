"""Competence head: P(this answer is actually correct).

Not the same thing as ``act_head``
----------------------------------
Upstream Laya already has an action head: pooled CLS concatenated with four
score-distribution features (top1, top1-top2 margin, normalised entropy, k/255),
trained as part of the RLCD action cost and surfaced as ``act_probability``. It is
a useful signal and this head reuses its feature idea.

The difference is the training target and the training distribution. ``act_head``
learns an action cost on in-distribution data. This head is fitted directly against
*observed correctness* on a split the decision model never trained on, with
deliberately shifted rows mixed in: truncated states, languages the checkpoint
cannot read, injected distractor options, and a held-out domain.

That is aimed squarely at the failure temperature scaling cannot touch. Laya's README
reports 0.000 accuracy at 0.952 confidence on Khmer -- the probabilities are
confidently wrong, so no monotone rescaling of them separates right from wrong. The
upstream fix is the Router's pre-forward script heuristic, which works only for the
failure mode someone already enumerated. A learned head can cover the ones nobody did.

Output contract
---------------
``competence`` in [0, 1] after calibration (isotonic or Platt), reported for reading
only. ``abstain`` compares the RAW head sigmoid to a threshold chosen by Learn-then-Test
(Angelopoulos et al. 2021, arXiv 2110.01052): a fixed, data-independent threshold grid,
tested by fixed-sequence testing with a Clopper-Pearson bound. The guarantee is
P(selective risk <= target_risk) >= 1 - delta, and nothing more.

That is not conformal risk control (arXiv 2208.02814), which bounds *expected* risk. It
also holds only for data exchangeable with the calibration split. A threshold fitted on
English rows says nothing about an unseen script such as Khmer; weighted conformal
(Tibshirani et al. 2019) cannot repair that, because the likelihood ratio is unbounded
where the calibration data has no support. Realised risk on shifted test rows is an
empirical result, never something the bound covers.
"""
from __future__ import annotations

import json
import math
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import CompConfig

# --------------------------------------------------------------------------------------
# lightweight script / language signal (no hard dependency on laya.lang)
# --------------------------------------------------------------------------------------

SCRIPTS = ["latin", "cyrillic", "arabic", "devanagari", "han", "hangul", "hiragana",
           "katakana", "thai", "hebrew", "greek", "khmer", "tamil", "bengali", "other"]
_SCRIPT_INDEX = {s: i for i, s in enumerate(SCRIPTS)}


def script_profile(text: str, limit: int = 2000) -> np.ndarray:
    """Fraction of letters per script, as a fixed-width vector.

    Pure Python and sub-millisecond, same reasoning as the upstream Router: the signal
    has to be available *before* the forward pass to be worth anything.
    """
    counts = np.zeros(len(SCRIPTS), dtype=np.float32)
    n = 0
    for ch in text[:limit]:
        if not ch.isalpha():
            continue
        n += 1
        try:
            name = unicodedata.name(ch).split(" ")[0].lower()
        except ValueError:
            name = "other"
        counts[_SCRIPT_INDEX.get(name, _SCRIPT_INDEX["other"])] += 1
    if n:
        counts /= n
    return counts


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """Which blocks are present and how wide each is -- the header for the matrix."""
    names: List[str]
    widths: List[int]

    @property
    def dim(self) -> int:
        return int(sum(self.widths))

    def to_dict(self) -> Dict[str, Any]:
        return {"names": self.names, "widths": self.widths, "dim": self.dim}


def score_stats(logits: np.ndarray, option_mask: np.ndarray) -> np.ndarray:
    """Distribution shape of one row's option logits: [top1, margin, entropy, k, spread, energy]."""
    z = np.where(option_mask, logits, -1e4).astype(np.float64)
    k = int(option_mask.sum())
    zz = z[:k] if k else z[:1]
    m = zz.max()
    p = np.exp(zz - m)
    p = p / p.sum()
    srt = np.sort(p)[::-1]
    top1 = float(srt[0])
    top2 = float(srt[1]) if len(srt) > 1 else 0.0
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum() / math.log(max(k, 2)))
    spread = float(zz.max() - zz.min()) if k > 1 else 0.0
    # Energy: -logsumexp over raw logits. A standard free OOD signal -- it moves when the
    # whole logit vector is small, which softmax deliberately throws away.
    energy = float(-(m + np.log(np.exp(zz - m).sum())))
    return np.array([top1, top1 - top2, ent, min(k, 255) / 255.0, spread, energy], dtype=np.float32)


class FeatureBuilder:
    """Assembles the competence feature matrix and remembers how it was built.

    Mahalanobis statistics are fitted on the *clean* training rows only. Fitting them
    on the shifted rows would teach the detector that shift is normal, which is the
    one thing it must not learn.
    """

    def __init__(self, cfg: CompConfig, pooled_dim: int):
        self.cfg = cfg.validate()
        self.pooled_dim = int(pooled_dim)
        self.mu: Optional[np.ndarray] = None
        self.prec: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.spec = self._build_spec()

    def _build_spec(self) -> FeatureSpec:
        names, widths = [], []
        c = self.cfg
        if c.use_pooled:
            names.append("pooled")
            widths.append(self.pooled_dim)
        if c.use_score_stats:
            names.append("score_stats")
            widths.append(6)
        if c.use_interaction_stats:
            names.append("interaction_stats")
            widths.append(3)
        if c.use_length_stats:
            names.append("length_stats")
            widths.append(4)
        if c.use_lang_stats:
            names.append("lang_stats")
            widths.append(len(SCRIPTS) + 1)
        if c.use_energy:
            names.append("energy")
            widths.append(1)
        if c.use_mahalanobis:
            names.append("mahalanobis")
            widths.append(1)
        return FeatureSpec(names, widths)

    # -- fitting ------------------------------------------------------------------

    def fit_density(self, pooled_clean: np.ndarray, shrink: float = 0.1) -> None:
        """Gaussian fit of the clean pooled features, with shrinkage.

        Shrinkage is not optional: pooled dim is 768-1024 and the clean split here is a
        few thousand rows, so the empirical covariance is singular and its inverse is
        numerically meaningless.
        """
        x = np.asarray(pooled_clean, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("pooled_clean must be 2-D, got shape %s" % (x.shape,))
        self.mu = x.mean(0)
        xc = x - self.mu
        n, d = xc.shape
        cov = (xc.T @ xc) / max(1, n - 1)
        cov = (1.0 - shrink) * cov + shrink * np.eye(d) * (np.trace(cov) / d)
        self.prec = np.linalg.pinv(cov)

    def _mahalanobis(self, pooled: np.ndarray) -> np.ndarray:
        if self.mu is None or self.prec is None:
            return np.zeros((len(pooled), 1), dtype=np.float32)
        xc = np.asarray(pooled, dtype=np.float64) - self.mu
        d2 = np.einsum("nd,dk,nk->n", xc, self.prec, xc)
        # log1p keeps a far-out-of-distribution row from dominating the MLP's first layer.
        return np.log1p(np.clip(d2, 0, None)).astype(np.float32)[:, None]

    def fit_scaler(self, X: np.ndarray) -> None:
        # float64 and a 1e-4 floor: in float32 the std of a constant column (the option
        # count, when every fitting row shares one label set) is rounding noise just above
        # 1e-6, and a 60 -> 77 option change then z-scores to ~20000 and pins the gate.
        X = X.astype(np.float64)
        self.mean = X.mean(0).astype(np.float32)
        std = X.std(0)
        std[std < 1e-4] = 1.0
        self.std = std.astype(np.float32)

    def transform_scale(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return X
        return ((X - self.mean) / self.std).astype(np.float32)

    # -- building -----------------------------------------------------------------

    def build(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """rows carry: pooled, logits, option_mask, max_sim/mean_sim/std_sim,
        n_state_tokens, n_truncated, n_options, state_text."""
        c = self.cfg
        blocks: List[np.ndarray] = []
        pooled = np.stack([np.asarray(r["pooled"], dtype=np.float32) for r in rows])

        if c.use_pooled:
            blocks.append(pooled)
        if c.use_score_stats:
            blocks.append(np.stack([
                score_stats(np.asarray(r["logits"], dtype=np.float32),
                            np.asarray(r["option_mask"], dtype=bool)) for r in rows]))
        if c.use_interaction_stats:
            blocks.append(np.stack([
                np.array([r.get("max_sim", 0.0), r.get("mean_sim", 0.0), r.get("std_sim", 0.0)],
                         dtype=np.float32) for r in rows]))
        if c.use_length_stats:
            def _len(r):
                n = float(r.get("n_state_tokens", 0))
                trunc = float(r.get("n_truncated", 0))
                total = n + trunc
                return np.array([
                    math.log1p(n) / 10.0,
                    trunc / total if total > 0 else 0.0,     # what fraction was thrown away
                    math.log1p(float(r.get("n_options", 2))) / 7.0,
                    1.0 if trunc > 0 else 0.0,
                ], dtype=np.float32)
            blocks.append(np.stack([_len(r) for r in rows]))
        if c.use_lang_stats:
            def _lang(r):
                prof = script_profile(str(r.get("state_text", "")))
                return np.concatenate([prof, np.array([float(prof.max())], dtype=np.float32)])
            blocks.append(np.stack([_lang(r) for r in rows]))
        if c.use_energy:
            blocks.append(np.stack([
                score_stats(np.asarray(r["logits"], dtype=np.float32),
                            np.asarray(r["option_mask"], dtype=bool))[5:6] for r in rows]))
        if c.use_mahalanobis:
            blocks.append(self._mahalanobis(pooled))

        X = np.concatenate(blocks, axis=1).astype(np.float32)
        if X.shape[1] != self.spec.dim:
            raise RuntimeError("feature width %d != spec %d (%s)"
                               % (X.shape[1], self.spec.dim, self.spec.to_dict()))
        return X

    def save(self, path: str) -> None:
        np.savez(path, mu=self.mu if self.mu is not None else np.zeros(1),
                 prec=self.prec if self.prec is not None else np.zeros(1),
                 mean=self.mean if self.mean is not None else np.zeros(1),
                 std=self.std if self.std is not None else np.zeros(1),
                 has_density=np.array([self.mu is not None]),
                 has_scaler=np.array([self.mean is not None]),
                 pooled_dim=np.array([self.pooled_dim]),
                 spec=np.array([json.dumps(self.spec.to_dict())]))

    def load(self, path: str) -> "FeatureBuilder":
        z = np.load(path, allow_pickle=False)
        if bool(z["has_density"][0]):
            self.mu, self.prec = z["mu"], z["prec"]
        if bool(z["has_scaler"][0]):
            self.mean, self.std = z["mean"], z["std"]
        return self


# --------------------------------------------------------------------------------------
# head
# --------------------------------------------------------------------------------------

class CompetenceHead(nn.Module):
    """Small MLP over the feature matrix. One logit: P(argmax is correct)."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 64), dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.GELU(), nn.Dropout(dropout)]
            d = int(h)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.in_dim = int(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------

class IsotonicCalibrator:
    """Pool-adjacent-violators isotonic regression, no sklearn dependency."""

    def __init__(self):
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray, correct: np.ndarray) -> "IsotonicCalibrator":
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(correct, dtype=np.float64).ravel()
        if len(s) != len(y):
            raise ValueError("scores and correct must be the same length")
        order = np.argsort(s, kind="mergesort")
        s, y = s[order], y[order]
        vals = list(y)
        wts = [1.0] * len(y)
        i = 0
        while i < len(vals) - 1:
            if vals[i] <= vals[i + 1] + 1e-12:
                i += 1
                continue
            w = wts[i] + wts[i + 1]
            v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / w
            vals[i: i + 2] = [v]
            wts[i: i + 2] = [w]
            i = max(i - 1, 0)
        out, idx = np.empty(len(y)), 0
        for v, w in zip(vals, wts):
            n = int(round(w))
            out[idx: idx + n] = v
            idx += n
        self.x, self.y = s, np.clip(out, 0.0, 1.0)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.x is None:
            return np.asarray(scores, dtype=np.float64)
        return np.interp(np.asarray(scores, dtype=np.float64).ravel(), self.x, self.y)

    def save(self, path: str) -> None:
        np.savez(path, x=self.x if self.x is not None else np.zeros(1),
                 y=self.y if self.y is not None else np.zeros(1),
                 fitted=np.array([self.x is not None]))

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        c = cls()
        z = np.load(path, allow_pickle=False)
        if bool(z["fitted"][0]):
            c.x, c.y = z["x"], z["y"]
        return c


class PlattCalibrator:
    """One-dimensional logistic fit, by plain gradient descent."""

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, scores, correct, iters: int = 500, lr: float = 0.1) -> "PlattCalibrator":
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(correct, dtype=np.float64).ravel()
        a, b = 1.0, 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(a * s + b)))
            ga = float(((p - y) * s).mean())
            gb = float((p - y).mean())
            a -= lr * ga
            b -= lr * gb
        self.a, self.b = a, b
        return self

    def predict(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64).ravel()
        return 1.0 / (1.0 + np.exp(-(self.a * s + self.b)))


# --------------------------------------------------------------------------------------
# Learn-then-Test abstention threshold
# --------------------------------------------------------------------------------------

def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact binomial upper confidence bound on a rate, via bisection on the Beta CDF.

    Exact rather than Hoeffding because the error counts here are small (a handful of
    mistakes in a few hundred answered rows) and Hoeffding is very loose in that regime,
    which would cost real coverage for nothing.
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta  # noqa: F401
        from scipy.stats import beta as _b
        return float(_b.ppf(1.0 - delta, k + 1, n - k))
    except Exception:
        pass
    # scipy-free fallback: bisect on the regularised incomplete beta via the binomial tail.
    def tail(p: float) -> float:
        # P(Bin(n, p) <= k), by direct summation in log space.
        total = 0.0
        logp, log1p_ = math.log(max(p, 1e-300)), math.log(max(1.0 - p, 1e-300))
        for i in range(0, k + 1):
            lc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            total += math.exp(lc + i * logp + (n - i) * log1p_)
        return total

    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if tail(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def threshold_grid(grid: int = 100) -> np.ndarray:
    """The candidate thresholds, strictest first. Fixed before any calibration row is seen.

    Learn-then-Test needs the hypothesis family chosen independently of the calibration
    data; quantiles of the calibration scores would make the family itself a function of
    the data it is then tested on.
    """
    return np.linspace(1.0, 0.0, int(grid))


def _threshold_curve(scores: np.ndarray, correct: np.ndarray, grid: int) -> List[Dict[str, Any]]:
    rows = []
    for t in threshold_grid(grid):
        sel = scores >= t
        k = int(sel.sum())
        err = int((1 - correct[sel]).sum())
        rows.append({"threshold": float(t), "n_selected": k, "errors": err,
                     "coverage": k / len(scores), "risk": err / k if k else None})
    return rows


def walk_start(reference_scores: np.ndarray, min_coverage: float, grid: int = 100) -> float:
    """Strictest grid threshold that answers at least ``min_coverage`` of a reference set.

    The reference must be independent of the calibration split -- the caller passes clean
    competence-split scores -- so where the walk starts is fixed before any calibration
    row is seen. Choosing it from calibration coverage would make the tested family a
    function of the calibration data again.
    """
    ref = np.asarray(reference_scores, dtype=np.float64).ravel()
    for t in threshold_grid(grid):
        if len(ref) and float((ref >= t).mean()) >= min_coverage:
            return float(t)
    return 0.0


def fit_abstention_threshold(scores: np.ndarray, correct: np.ndarray,
                             target_risk: float = 0.05, delta: float = 0.1,
                             min_coverage: float = 0.30, grid: int = 100,
                             start_threshold: float = 1.0,
                             gate_score: str = "raw_sigmoid") -> Dict[str, Any]:
    """Learn-then-Test with fixed-sequence testing over a fixed grid on the raw head score.

    Walks the grid from strictest to loosest and tests H_t: selective risk(t) > target
    with a Clopper-Pearson bound at the full ``delta``, stopping at the first failure.
    Fixed-sequence controls the family-wise error without Bonferroni's delta/grid split,
    which buys a lot of power when risk is roughly monotone in the threshold.

    The walk begins at ``start_threshold`` (see ``walk_start``): from the very strict end,
    a handful of answered rows can never pass a bound and the walk would halt at once.
    ``min_coverage`` is only a ship check on the chosen threshold, applied after testing,
    so it never changes which threshold the procedure selects.

    ``threshold=None`` with ``feasible=False`` means no tested threshold met the target.
    The gate is then off; the closest miss is not shipped.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    if len(s) != len(y):
        raise ValueError("scores and correct must be the same length")
    base = {"procedure": "learn-then-test, fixed-sequence, Clopper-Pearson",
            "gate_score": gate_score, "target_risk": target_risk, "delta": delta,
            "min_coverage": min_coverage, "start_threshold": start_threshold,
            "n_calibration": len(s)}
    if len(s) == 0:
        return {**base, "threshold": None, "feasible": False,
                "reason": "empty calibration set", "curve": []}
    curve = _threshold_curve(s, y, grid)
    best = None
    for row in curve:
        if row["threshold"] > start_threshold:
            continue
        row["risk_upper"] = clopper_pearson_upper(row["errors"], row["n_selected"], delta)
        if row["risk_upper"] > target_risk:
            break
        best = row
    if best is None or best["coverage"] < min_coverage:
        return {**base, "threshold": None, "feasible": False, "curve": curve,
                "reason": "no threshold met risk<=%.3g at coverage>=%.3g"
                          % (target_risk, min_coverage)}
    return {**base, "threshold": best["threshold"], "feasible": True, "curve": curve,
            "coverage": best["coverage"], "risk": best["risk"], "risk_upper": best["risk_upper"]}


# --------------------------------------------------------------------------------------
# selective metrics
# --------------------------------------------------------------------------------------

def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray) -> Dict[str, np.ndarray]:
    """Risk as a function of coverage, answering the highest-scored rows first."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    n = len(y)
    cum_err = np.cumsum(1.0 - y)
    idx = np.arange(1, n + 1)
    return {"coverage": idx / n, "risk": cum_err / idx, "threshold": s[order]}


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better.

    The single number that says whether a confidence signal is useful for gating,
    which accuracy and ECE both fail to capture: a model can be well calibrated on
    average and still have no ability to tell its own errors apart.
    """
    c = risk_coverage_curve(scores, correct)
    return float(np.trapezoid(c["risk"], c["coverage"])) if hasattr(np, "trapezoid") \
        else float(np.trapz(c["risk"], c["coverage"]))


def selective_accuracy_at(scores: np.ndarray, correct: np.ndarray,
                          coverages: Sequence[float] = (0.5, 0.7, 0.9)) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    n = len(y)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        out["acc@cov%.2f" % c] = float(y[:k].mean())
    return out


def evaluate_selective(competence: np.ndarray, correct: np.ndarray,
                       threshold: float) -> Dict[str, Any]:
    """What the fitted gate actually does on a split it was not fitted on."""
    c = np.asarray(competence, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    sel = c >= threshold
    n_sel = int(sel.sum())
    return {
        "threshold": float(threshold),
        "coverage": float(n_sel / len(c)) if len(c) else 0.0,
        "selective_risk": float((1 - y[sel]).mean()) if n_sel else 0.0,
        "selective_accuracy": float(y[sel].mean()) if n_sel else 0.0,
        "abstained": int(len(c) - n_sel),
        "accuracy_on_abstained": float(y[~sel].mean()) if n_sel < len(c) else None,
        "full_accuracy": float(y.mean()) if len(y) else 0.0,
        "aurc": aurc(c, y),
        **selective_accuracy_at(c, y),
    }


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Same binning as laya.common.ece_score, so the numbers are comparable."""
    conf = np.asarray(conf, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


# --------------------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------------------

class CompetenceModel:
    """Feature builder + head + calibrator + threshold, saved and loaded as one unit.

    Two scores come out, on purpose. ``score_raw`` is the head sigmoid, and the only
    thing the abstention threshold is ever compared to. ``score`` is the calibrated
    probability, for reading. The calibrator is fitted on the same calibration split
    the threshold is chosen on, so letting it into threshold selection would use that
    split twice.
    """

    def __init__(self, cfg: CompConfig, builder: FeatureBuilder, head: CompetenceHead,
                 calibrator: Any = None, threshold_info: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.builder = builder
        self.head = head
        self.calibrator = calibrator
        self.threshold_info = threshold_info or {}

    @property
    def feasible(self) -> bool:
        return bool(self.threshold_info.get("feasible", False))

    @property
    def threshold(self) -> Optional[float]:
        """Raw-sigmoid threshold, or None when no threshold met the risk target."""
        t = self.threshold_info.get("threshold")
        return float(t) if self.feasible and t is not None else None

    @torch.no_grad()
    def score_raw(self, rows: Sequence[Dict[str, Any]], device: str = "cpu") -> np.ndarray:
        X = self.builder.transform_scale(self.builder.build(rows))
        # A head loaded from disk sits on cpu; the agent scores on its own device.
        self.head.to(device).eval()
        logits = self.head(torch.from_numpy(X).to(device)).float().cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return np.asarray(raw, dtype=np.float64)
        return np.asarray(self.calibrator.predict(raw), dtype=np.float64)

    def score(self, rows: Sequence[Dict[str, Any]], device: str = "cpu") -> np.ndarray:
        return self.calibrate(self.score_raw(rows, device))

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.head.state_dict(), os.path.join(directory, "competence_head.pt"))
        self.builder.save(os.path.join(directory, "competence_features.npz"))
        meta = {"cfg": self.cfg.to_dict(), "threshold_info":
                {k: v for k, v in self.threshold_info.items() if k != "curve"},
                "in_dim": self.head.in_dim, "pooled_dim": self.builder.pooled_dim,
                "calibrator": type(self.calibrator).__name__ if self.calibrator else None,
                "feature_spec": self.builder.spec.to_dict()}
        with open(os.path.join(directory, "competence_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if isinstance(self.calibrator, IsotonicCalibrator):
            self.calibrator.save(os.path.join(directory, "competence_calibrator.npz"))
        elif isinstance(self.calibrator, PlattCalibrator):
            with open(os.path.join(directory, "competence_calibrator.json"), "w") as f:
                json.dump({"a": self.calibrator.a, "b": self.calibrator.b}, f)

    @classmethod
    def load(cls, directory: str) -> "CompetenceModel":
        with open(os.path.join(directory, "competence_meta.json")) as f:
            meta = json.load(f)
        cfg = CompConfig.from_dict(meta["cfg"])
        builder = FeatureBuilder(cfg, meta["pooled_dim"])
        builder.load(os.path.join(directory, "competence_features.npz"))
        head = CompetenceHead(meta["in_dim"], cfg.hidden, cfg.dropout)
        head.load_state_dict(torch.load(os.path.join(directory, "competence_head.pt"),
                                        map_location="cpu"))
        cal = None
        iso = os.path.join(directory, "competence_calibrator.npz")
        pl = os.path.join(directory, "competence_calibrator.json")
        if meta.get("calibrator") == "IsotonicCalibrator" and os.path.exists(iso):
            cal = IsotonicCalibrator.load(iso)
        elif meta.get("calibrator") == "PlattCalibrator" and os.path.exists(pl):
            cal = PlattCalibrator()
            with open(pl) as f:
                d = json.load(f)
            cal.a, cal.b = d["a"], d["b"]
        return cls(cfg, builder, head, cal, meta.get("threshold_info"))
