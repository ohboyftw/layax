"""End-to-end run: data -> splits -> train head -> temperatures -> competence -> eval.

One function so Kaggle, Modal and a local box all execute the same path. Anything that
only happens in the notebook is a difference between what you measured and what you
ship.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .baselines import run_classifier_baselines
from .calibrate import ece_before_after, fit_temperatures
from .config import RunConfig
from .data import (
    shift_report,
    Example,
    Splits,
    build_shift_set,
    hold_out_labels,
    load_dataset_by_name,
    load_jsonl,
    split_examples,
    splits_from_meta,
)
from .competence import evaluate_selective
from .evaluate import (
    LAYA_BASELINE_ARMS,
    collect_predictions,
    compare,
    evaluate_predictions,
    latency_benchmark,
)
from .li_head import QTYPES
from .msp_gate import fit_msp_gate, msp, msp_by_shift
from .runtime import LayaxAgent, clamp_temperature, load_base
from .train_competence import (
    evaluate_competence,
    evaluate_shift_baselines,
    fit_competence,
    save_run,
)
from .train_li import train


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_splits(cfg: RunConfig, jsonl_path: Optional[str] = None,
                jsonl_kwargs: Optional[Dict[str, Any]] = None) -> Splits:
    if cfg.dataset == "jsonl":
        if not jsonl_path:
            raise ValueError("dataset 'jsonl' needs --jsonl pointing at your rows")
        rows = load_jsonl(jsonl_path, max_rows=cfg.max_train_rows, **(jsonl_kwargs or {}))
        if cfg.split_key:
            return splits_from_meta(rows, cfg.split_key)
        return split_examples(rows, cfg.competence_frac, cfg.calibration_frac, seed=cfg.li.seed)
    train_rows, test_rows = load_dataset_by_name(cfg.dataset, max_rows=cfg.max_train_rows,
                                                 seed=cfg.li.seed)
    if cfg.max_eval_rows:
        test_rows = test_rows[: cfg.max_eval_rows]
    return split_examples(train_rows, cfg.competence_frac, cfg.calibration_frac,
                          test=test_rows, seed=cfg.li.seed)


def load_foreign_rows(locales: Sequence[str], max_rows: int = 400
                      ) -> Tuple[List[Example], List[Example]]:
    """MASSIVE rows in scripts the English checkpoint cannot read, as (train, test).

    The train rows are what make the competence head see confidently-wrong behaviour
    during training rather than only at eval. The test rows come from MASSIVE's own test
    split, so the shifted test set never reuses a row the head was fitted on. Khmer,
    Amharic, Telugu and Japanese are chosen to span four scripts, not four languages.
    """
    from .data import load_massive
    try:
        return load_massive(languages=locales, max_rows=max_rows)
    except Exception as e:                                  # dataset unavailable offline
        print("[layax] could not load foreign rows (%s); continuing without them" % e)
        return [], []


def competence_arm(test_eval: Dict[str, Any]) -> Dict[str, Any]:
    """The layax+competence row for the comparison table.

    ``accuracy`` stays the full-coverage accuracy, because the head never changes an
    answer. What the gate buys is shown as ``coverage`` and ``selective_accuracy``, and
    abstentions are not ``errors`` (that column counts rows an arm failed to run on).
    With the gate off there is no selective number to show, and none is shown.
    """
    m = test_eval["metrics"]
    row = {"arm": "layax+competence", "n": m["n"], "accuracy": m["accuracy"],
           "ece": m.get("ece_competence"), "aurc": m.get("aurc_competence"),
           "errors": 0, "seconds": ""}
    if not test_eval["gate_enabled"]:
        row.update(arm="layax+competence (gate off)", coverage=1.0)
        return row
    row.update(coverage=test_eval["coverage"],
               selective_accuracy=test_eval["selective_accuracy"])
    return row


def competence_stage(cfg: RunConfig, model: Any, tok: Any, splits: Splits, dev: str,
                     foreign_train: List[Example], foreign_test: List[Example],
                     report: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """The learned competence head (``competence_head: true``). Fills report["competence"],
    ["test"] and ["test_shift"]; returns the head and its comparison-table row."""
    fitted = fit_competence(model, tok, cfg.li, cfg.comp, splits.competence, splits.calibration,
                            device=dev, foreign_rows=foreign_train,
                            batch_size=cfg.eval_batch_size, seed=cfg.li.seed)
    report["competence"] = fitted["report"]
    print("[layax] competence: %s" % fitted["report"].get("verdict"), flush=True)
    comp = fitted["competence"]
    report["test"] = evaluate_competence(
        comp, model, tok, cfg.li, splits.test, device=dev, batch_size=cfg.eval_batch_size,
        group_by="label_seen" if cfg.heldout_labels else cfg.group_by)
    shifted = build_shift_set(splits.test, cfg.comp, foreign=foreign_test, seed=cfg.li.seed + 1)
    report["test_shift"] = evaluate_shift_baselines(comp, model, tok, cfg.li, shifted,
                                                    device=dev, batch_size=cfg.eval_batch_size)
    return comp, competence_arm(report["test"])


def msp_stage(cfg: RunConfig, model: Any, tok: Any, splits: Splits, dev: str,
              cal: Dict[str, Any], temperatures: Dict[str, float], foreign_test: List[Example],
              report: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Default gate: Learn-then-Test on the temperature-scaled max-softmax, with the same
    clamped temperatures the saved agent applies. Fills report["msp_gate"], ["test"] and
    ["test_shift"]; returns the threshold (None if infeasible) and the table row."""
    t = {k: clamp_temperature(v) for k, v in temperatures.items()}
    kw = {"device": dev, "batch_size": cfg.eval_batch_size}
    ref = collect_predictions(model, tok, splits.competence, cfg.li, **kw)
    gate = fit_msp_gate(ref["features"], cal["features"], cal["correct"], t, cfg.comp)
    report["msp_gate"] = gate
    res = collect_predictions(model, tok, splits.test, cfg.li, **kw)
    report["test"] = {"metrics": evaluate_predictions(
        res, group_by="label_seen" if cfg.heldout_labels else cfg.group_by)}
    row = {"arm": "layax+ltt(msp)", "n": len(res["correct"]), "errors": 0, "seconds": ""}
    if gate["feasible"]:
        sel = evaluate_selective(msp(res["features"], t), res["correct"], gate["threshold"])
        report["test"]["gate"] = sel
        row.update(accuracy=sel["full_accuracy"], coverage=sel["coverage"],
                   selective_accuracy=sel["selective_accuracy"], aurc=sel["aurc"])
    else:
        row["arm"] = "layax+ltt(msp) (no feasible threshold)"
    shifted = build_shift_set(splits.test, cfg.comp, foreign=foreign_test, seed=cfg.li.seed + 1)
    sres = collect_predictions(model, tok, shifted, cfg.li, **kw)
    report["test_shift"] = {"shift_mix": shift_report(shifted), "by_shift": msp_by_shift(
        sres["features"], sres["correct"], [e.meta.get("shift", "clean") for e in shifted], t)}
    return (gate["threshold"] if gate["feasible"] else None), row


def run(cfg: RunConfig, device: Optional[str] = None, hf_token: Optional[str] = None,
        jsonl_path: Optional[str] = None, jsonl_kwargs: Optional[Dict[str, Any]] = None,
        run_baselines: bool = True, foreign_locales: Sequence[str] = ("km-KH", "am-ET", "te-IN", "ja-JP"),
        skip_training: bool = False,
        on_checkpoint: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """The whole thing. Returns a report and writes it to ``cfg.output_dir``.

    The report is rewritten after every stage, with ``stage`` naming the last one that
    finished, so a run cancelled mid-baselines still leaves its training and test results.
    ``on_checkpoint`` runs after each write -- Modal passes its volume commit here.
    """
    t0 = time.perf_counter()
    cfg.validate()
    os.makedirs(cfg.output_dir, exist_ok=True)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    report: Dict[str, Any] = {"config": cfg.to_dict(), "device": dev,
                              "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def checkpoint(stage: str) -> None:
        report["stage"] = stage
        save_run(cfg.output_dir, report, "run_report.json")
        if on_checkpoint is not None:
            on_checkpoint()

    splits = load_splits(cfg, jsonl_path, jsonl_kwargs)
    if cfg.heldout_labels:
        splits, report["heldout_labels"] = hold_out_labels(
            splits, cfg.heldout_labels, seed=cfg.li.seed, keep_as_options=cfg.heldout_as_options)
        print("[layax] held out %d labels: %s" % (cfg.heldout_labels,
                                                  report["heldout_labels"]), flush=True)
    report["splits"] = splits.summary()
    print("[layax] splits: %s" % json.dumps(splits.summary()), flush=True)
    checkpoint("splits")

    model, tok = load_base(cfg.li, token=hf_token)
    model.to(dev)

    # Before layax training touches the encoder, so the classifiers start from the same
    # checkpoint weights layax does.
    classifier_arms: List[Dict[str, Any]] = []
    if cfg.classifier_baselines:
        classifier_arms = run_classifier_baselines(model.encoder, tok, splits.train,
                                                   splits.calibration, splits.test, cfg.li,
                                                   dev, group_by=cfg.group_by)
        report["classifier_arms"] = classifier_arms
        checkpoint("classifier_baselines")

    if not skip_training:
        log = train(model, tok, splits.train, cfg.li, device=dev,
                    eval_rows=splits.calibration[:512], output_dir=cfg.output_dir)
        report["training"] = log
        print("[layax] trained in %.1f min" % log["minutes"], flush=True)
        checkpoint("training")

    # --- temperatures, on the calibration split ------------------------------------
    cal = collect_predictions(model, tok, splits.calibration, cfg.li, dev,
                              batch_size=cfg.eval_batch_size)
    qts = [QTYPES[e.qtype] for e in splits.calibration]
    temps = fit_temperatures(cal["features"], [e.label for e in splits.calibration], qts)
    report["temperatures"] = temps["report"]
    report["temperature_effect"] = ece_before_after(
        cal["features"], [e.label for e in splits.calibration], temps["temperatures"], qts)
    print("[layax] temperatures: %s" % json.dumps(report["temperature_effect"]), flush=True)
    checkpoint("temperatures")

    # --- gate and test split (test touched once) ------------------------------------
    foreign_train, foreign_test = (load_foreign_rows(foreign_locales)
                                   if cfg.comp.shift_foreign > 0 else ([], []))
    if cfg.competence_head:
        comp, gate_arm = competence_stage(cfg, model, tok, splits, dev, foreign_train,
                                          foreign_test, report)
        msp_threshold = None
    else:
        comp = None
        msp_threshold, gate_arm = msp_stage(cfg, model, tok, splits, dev, cal,
                                            temps["temperatures"], foreign_test, report)
    print("[layax] test by shift: %s" % json.dumps(report["test_shift"]["by_shift"],
                                                   default=_jsonable), flush=True)
    m = report["test"]["metrics"]
    arms = [{"arm": "layax-" + cfg.li.interaction, "n": m["n"], "accuracy": m["accuracy"],
             "coverage": 1.0, "ece": m["ece_softmax"], "aurc": m["aurc_softmax"],
             "errors": 0, "seconds": m["seconds"]}, gate_arm] + classifier_arms
    report["arms"] = arms
    checkpoint("test")

    # --- baselines ---------------------------------------------------------------------
    if run_baselines:
        from .evaluate import evaluate_laya_baseline
        for kw in LAYA_BASELINE_ARMS:
            try:
                arms.append(evaluate_laya_baseline(
                    splits.test[: cfg.max_eval_rows or 1000],
                    checkpoint=cfg.li.base_checkpoint, subfolder=cfg.li.base_subfolder,
                    device=dev, **kw))
            except Exception as e:
                arms.append({"arm": "laya%s (failed)" % ("+" + ",".join(map(str, kw.values()))
                                                         if kw else ""),
                             "n": 0, "errors": str(e)[:120]})
            checkpoint("baseline %s" % (kw or "laya"))
    report["table"] = compare(arms)
    print("\n" + report["table"] + "\n", flush=True)

    # --- latency -------------------------------------------------------------------------
    agent = LayaxAgent(model, tok, device=dev, temperatures=temps["temperatures"],
                       competence=comp, msp_threshold=msp_threshold)
    sample = splits.test[0]
    try:
        report["latency"] = latency_benchmark(agent, {"q": sample.as_question()},
                                              sample.state, repeats=30)
        print("[layax] latency: %s" % json.dumps(report["latency"]), flush=True)
    except Exception as e:
        report["latency"] = {"error": str(e)}

    agent.save(cfg.output_dir)
    report["minutes_total"] = round((time.perf_counter() - t0) / 60, 2)
    checkpoint("done")
    print("[layax] wrote %s (%.1f min total)" % (os.path.join(cfg.output_dir, "run_report.json"),
                                                  report["minutes_total"]), flush=True)
    return report
