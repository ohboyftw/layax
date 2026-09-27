"""Fit the competence head, its calibrator, and the Learn-then-Test abstention threshold.

Split discipline is the whole design, so it is stated once here and enforced in code:

* ``train``       -- the decision model's own training rows. The competence head must
                     never see these. A model is optimistic about rows it memorised, and
                     a head fitted on that optimism learns the wrong error rate.
* ``competence``  -- held out from decision training. Shift augmentation is applied
                     here, and the head is fitted on observed correctness.
* ``calibration`` -- held out from both. The threshold is chosen here, on the raw head
                     sigmoid. The calibrator is also fitted here, but it only shapes the
                     reported probability and never enters threshold selection, so the
                     split is not used twice for the gate. Reusing the competence split
                     would make the risk bound a statement about data already used to
                     choose the scores.
* ``test``        -- touched once, at the end.

Mahalanobis statistics are fitted on clean competence rows only: including the shifted
rows would teach the density model that shift is normal, which defeats the point.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .competence import (
    CompetenceHead,
    CompetenceModel,
    FeatureBuilder,
    IsotonicCalibrator,
    PlattCalibrator,
    aurc,
    evaluate_selective,
    fit_abstention_threshold,
    walk_start,
    score_stats,
)
from .config import SHIFT_FIELDS, CompConfig, LIConfig
from .data import Example, build_shift_set, shift_report
from .evaluate import collect_predictions


def training_shift_config(cfg: CompConfig) -> CompConfig:
    """The config competence training uses: ``cfg`` with the held-out shift switched off."""
    if cfg.heldout_shift is None:
        return cfg
    return dataclasses.replace(cfg, **{SHIFT_FIELDS[cfg.heldout_shift]: 0.0})


def _train_head(X: np.ndarray, y: np.ndarray, cfg: CompConfig,
                device: str = "cpu") -> CompetenceHead:
    """BCE with a positive-class weight, because errors are the rare class.

    Most answers are correct, so an unweighted head can reach high accuracy by
    predicting "correct" everywhere -- and be useless for the only thing it is for.
    """
    torch.manual_seed(cfg.seed)
    head = CompetenceHead(X.shape[1], cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    pos_weight = None
    if cfg.pos_weight_auto and n_pos > 0:
        # Weight is applied to the positive class (correct); errors are the minority, so
        # this downweights the majority rather than the other way round.
        pos_weight = torch.tensor([max(0.05, min(20.0, n_neg / n_pos))], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).float().to(device)
    n = len(Xt)
    history = []
    for epoch in range(cfg.epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i: i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = lossf(head(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        history.append(round(total / max(1, n), 5))
    head._history = history          # kept for the run log
    return head


def fit_competence(model, tok, cfg_li: LIConfig, cfg: CompConfig,
                   competence_rows: Sequence[Example],
                   calibration_rows: Sequence[Example],
                   device: Any = "cpu",
                   foreign_rows: Optional[Sequence[Example]] = None,
                   ood_rows: Optional[Sequence[Example]] = None,
                   batch_size: int = 16,
                   seed: int = 17) -> Dict[str, Any]:
    """Build the shift set, fit the head, calibrate, and fit the threshold.

    Returns ``{"competence": CompetenceModel, "report": {...}}``. The report is the
    thing to read: it says how many rows of each shift type were used, what the head
    bought over raw softmax (AURC), and whether the risk target was reachable at all.
    """
    dev = str(device)

    shifted = build_shift_set(competence_rows, training_shift_config(cfg),
                              foreign=foreign_rows, ood=ood_rows, seed=seed)
    report: Dict[str, Any] = {"shift_mix": shift_report(shifted),
                              "n_competence_rows": len(shifted),
                              "heldout_shift": cfg.heldout_shift}

    res = collect_predictions(model, tok, shifted, cfg_li, dev, batch_size=batch_size)
    y = res["correct"]
    report["competence_split_accuracy"] = float(y.mean()) if len(y) else float("nan")
    # Per-shift accuracy: if a shift type does not hurt accuracy, it is teaching the head
    # nothing and is costing training time.
    per_shift: Dict[str, Dict[str, float]] = {}
    for i, ex in enumerate(shifted):
        k = ex.meta.get("shift", "clean")
        e = per_shift.setdefault(k, {"n": 0, "correct": 0.0})
        e["n"] += 1
        e["correct"] += float(y[i])
    report["accuracy_by_shift"] = {k: round(v["correct"] / v["n"], 4) for k, v in per_shift.items()}

    pooled_dim = int(np.asarray(res["features"][0]["pooled"]).shape[0])
    builder = FeatureBuilder(cfg, pooled_dim)

    if cfg.use_mahalanobis:
        clean_idx = [i for i, ex in enumerate(shifted) if ex.meta.get("shift") is None
                     or ex.meta.get("shift") == "clean"]
        if len(clean_idx) >= 50:
            builder.fit_density(np.stack([res["features"][i]["pooled"] for i in clean_idx]))
            report["density_fitted_on"] = len(clean_idx)
        else:
            report["density_fitted_on"] = 0
            report["density_note"] = "fewer than 50 clean rows; mahalanobis left at zero"

    X = builder.build(res["features"])
    builder.fit_scaler(X)
    Xs = builder.transform_scale(X)
    head = _train_head(Xs, y, cfg, device=dev)
    report["head_loss_history"] = getattr(head, "_history", [])
    report["feature_spec"] = builder.spec.to_dict()

    # --- calibration split: threshold on the raw score, calibrator for reporting ------
    cal_res = collect_predictions(model, tok, calibration_rows, cfg_li, dev,
                                  batch_size=batch_size)
    cal_y = cal_res["correct"]
    comp = CompetenceModel(cfg, builder, head)
    raw = comp.score_raw(cal_res["features"], device=dev)
    if cfg.calibration == "isotonic":
        comp.calibrator = IsotonicCalibrator().fit(raw, cal_y)
    elif cfg.calibration == "platt":
        comp.calibrator = PlattCalibrator().fit(raw, cal_y)

    # The competence rows trained the head, so its scores there run optimistic and the
    # start lands a little strict. That costs coverage at worst, never validity.
    clean = [i for i, ex in enumerate(shifted) if ex.meta.get("shift") in (None, "clean")]
    ref = comp.score_raw([res["features"][i] for i in clean], device=dev) if clean else raw[:0]
    start = walk_start(ref, cfg.min_coverage)
    thr = fit_abstention_threshold(raw, cal_y, cfg.target_risk, cfg.delta, cfg.min_coverage,
                                   start_threshold=start)
    comp.threshold_info = thr
    report["threshold"] = {k: v for k, v in thr.items() if k != "curve"}
    report["calibration_split_accuracy"] = float(cal_y.mean()) if len(cal_y) else float("nan")
    report.update(_calibration_verdict(cal_res["features"], raw, cal_y, thr, cfg))
    return {"competence": comp, "report": report,
            "calibration_raw_scores": raw, "calibration_correct": cal_y}


def _calibration_verdict(features: Sequence[Dict[str, Any]], raw: np.ndarray,
                         correct: np.ndarray, thr: Dict[str, Any],
                         cfg: CompConfig) -> Dict[str, Any]:
    """Head vs max-softmax AURC on the calibration split, and what that means for the gate."""
    msp = np.array([score_stats(f["logits"], f["option_mask"])[0] for f in features])
    out = {"aurc_softmax_calibration": aurc(msp, correct),
           "aurc_competence_calibration": aurc(raw, correct)}
    out["aurc_improvement_calibration"] = (out["aurc_softmax_calibration"]
                                           - out["aurc_competence_calibration"])
    if out["aurc_improvement_calibration"] <= 0:
        out["verdict"] = ("competence head did NOT beat raw softmax on the calibration "
                          "split; do not ship it as a gate on this data")
    elif not thr.get("feasible"):
        out["verdict"] = ("head beats softmax but no threshold met risk<=%.3g at "
                          "coverage>=%.3g; report competence, keep the gate off"
                          % (cfg.target_risk, cfg.min_coverage))
    else:
        out["verdict"] = ("gate viable on exchangeable data: coverage %.3f at selective "
                          "risk %.3f (upper %.3f)"
                          % (thr["coverage"], thr["risk"], thr["risk_upper"]))
    return out


def evaluate_competence(comp: CompetenceModel, model, tok, cfg_li: LIConfig,
                        test_rows: Sequence[Example], device: Any = "cpu",
                        batch_size: int = 16, group_by: Optional[str] = None) -> Dict[str, Any]:
    """Run the fitted gate on the untouched test split.

    The Learn-then-Test bound holds only for data exchangeable with the calibration
    split. Test rows that are shifted are not, so a realised risk above target on them is
    the expected consequence of drift, not a bug -- and it is an empirical number, never
    one the bound promised anything about.

    With no feasible threshold the gate is off: no selective numbers are reported, only
    ``gate_enabled=False`` and the full-coverage metrics.
    """
    res = collect_predictions(model, tok, test_rows, cfg_li, device, batch_size=batch_size)
    raw = comp.score_raw(res["features"], device=str(device))
    out: Dict[str, Any] = {"gate_enabled": comp.threshold is not None}
    if comp.threshold is not None:
        out.update(evaluate_selective(raw, res["correct"], comp.threshold))
    else:
        out["threshold"] = None
        out["gate_note"] = ("no threshold met the risk target on calibration data; gate "
                            "off, every row answered")

    from .evaluate import evaluate_predictions
    out["metrics"] = evaluate_predictions(res, group_by=group_by,
                                          competence=comp.calibrate(raw), competence_rank=raw)
    return out


def aurc_by_shift(features: Sequence[Dict[str, Any]], correct: np.ndarray,
                  head_raw: np.ndarray, shifts: Sequence[str],
                  heldout: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """AURC of max-softmax, energy and the competence head, per shift type plus ``all``.

    Energy enters as ``logsumexp`` (negated energy) so that, like the other two, higher
    means more confident.
    """
    stats = np.stack([score_stats(f["logits"], f["option_mask"]) for f in features])
    signals = {"msp": stats[:, 0], "energy": -stats[:, 5], "competence": np.asarray(head_raw)}
    y = np.asarray(correct, dtype=np.float64)
    groups = {name: np.array([i for i, s in enumerate(shifts) if s == name])
              for name in sorted(set(shifts))}
    groups["all"] = np.arange(len(y))
    out: Dict[str, Dict[str, Any]] = {}
    for name, idx in groups.items():
        entry = {"n": int(len(idx)), "accuracy": float(y[idx].mean()), "heldout": name == heldout}
        entry.update({"aurc_" + k: aurc(v[idx], y[idx]) for k, v in signals.items()})
        out[name] = entry
    return out


def evaluate_shift_baselines(comp: CompetenceModel, model, tok, cfg_li: LIConfig,
                             shifted_test_rows: Sequence[Example], device: Any = "cpu",
                             batch_size: int = 16) -> Dict[str, Any]:
    """Head vs MSP vs energy, per shift type, on a shifted copy of the test split."""
    res = collect_predictions(model, tok, shifted_test_rows, cfg_li, device,
                              batch_size=batch_size)
    raw = comp.score_raw(res["features"], device=str(device))
    shifts = [ex.meta.get("shift", "clean") for ex in shifted_test_rows]
    heldout = comp.cfg.heldout_shift
    return {"shift_mix": shift_report(shifted_test_rows),
            # A held-out shift that could not be built (no foreign or ood rows supplied)
            # would otherwise just be a missing row in by_shift.
            "heldout_present": None if heldout is None else heldout in shifts,
            "by_shift": aurc_by_shift(res["features"], res["correct"], raw, shifts,
                                      heldout=heldout),
            "note": ("realised on shifted test rows; empirical only, not covered by the "
                     "Learn-then-Test bound")}


def save_run(directory: str, report: Dict[str, Any], name: str = "competence_report.json") -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=default)
    return path
