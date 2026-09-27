"""Publication baselines from other packages: FastFit and SetFit, plus the classifier arms.

FastFit (Yehudai & Bendel 2024, IBM ``fast-fit``) is the closest prior art: it scores a
query against every label name by token-level similarity, which is the late-interaction
idea layax applies to Laya's heads. SetFit (Tunstall et al. 2022) is the standard
few-shot sentence-transformer classifier. Both run on the rows the layax run for the
same config and seed used: ``pipeline.load_splits`` builds the splits, nothing re-splits.

* Train on ``splits.train``; the calibration split only fits a temperature
  (``baselines.logit_arm``); the test split is scored once, against the FULL label set.
* Each package uses the backbone its own README shows (FastFit: ``roberta-base``;
  SetFit: ``sentence-transformers/paraphrase-mpnet-base-v2``), not layax's Laya encoder.
  These arms compare methods with their recommended encoders, not heads on one encoder.
* Training budgets are capped (``PubCaps``) so one arm fits in a GPU batch job; the caps
  are written into every arm, so a number is never read without its budget.

A package that fails to import or run becomes an error arm with the reason. The batch
continues, and the failure is a result to report, not something to work around.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .baselines import label_space, logit_arm, run_classifier_baselines
from .config import RunConfig
from .data import Example, Splits
from .evaluate import compare

log = logging.getLogger(__name__)

BACKBONE_NOTE = ("FastFit and SetFit use the backbones their READMEs show, not layax's Laya "
                 "encoder; the classifier arms use the Laya encoder. Compare methods, not heads.")
LOG_EPS = 1e-12


@dataclass
class PubCaps:
    """Training budgets for the external arms. Written into every arm dict."""
    fastfit_backbone: str = "roberta-base"
    fastfit_max_steps: int = 1500
    fastfit_epochs: int = 40           # README value; max_steps is the binding cap
    fastfit_batch_size: int = 32
    fastfit_num_repeats: int = 4
    setfit_backbone: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    setfit_num_iterations: int = 5     # pairs per training row, instead of all pairs
    setfit_max_steps: int = 2000
    setfit_batch_size: int = 16
    max_text_length: int = 128
    infer_batch_size: int = 64


def _texts(rows: Sequence[Example]) -> List[str]:
    return [str(ex.state) for ex in rows]


def _label_texts(rows: Sequence[Example]) -> List[str]:
    """Readable label text in index order: the same strings layax's option tower embeds."""
    names = label_space(rows)
    crit = rows[0].criteria
    texts = [str(crit[n]) for n in names] if isinstance(crit, dict) else list(names)
    if len(set(texts)) != len(texts):
        raise ValueError("label texts are not unique; FastFit keys labels by text")
    return texts


def _to_label_order(scores: np.ndarray, trained: Sequence[Any], wanted: Sequence[Any],
                    fill: float) -> np.ndarray:
    """Reorder a package's columns into layax's label order; absent labels get ``fill``."""
    col = {key: j for j, key in enumerate(trained)}
    out = np.full((len(scores), len(wanted)), fill, dtype=np.float64)
    for i, key in enumerate(wanted):
        if key in col:
            out[:, i] = scores[:, col[key]]
    return out


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


# -- FastFit ------------------------------------------------------------------------------

def _import_fastfit() -> Any:
    """fast-fit 1.2.1 imports ``datasets.load_metric``, removed in datasets 3. It only uses
    it for accuracy during its own evaluate(), which is never called here."""
    import datasets
    if not hasattr(datasets, "load_metric"):
        acc = SimpleNamespace(compute=lambda predictions, references: {
            "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(references)))})
        datasets.load_metric = lambda *a, **kw: acc
    from fastfit import FastFitTrainer
    return FastFitTrainer


def _fastfit_trainer(train_rows: Sequence[Example], label_texts: List[str], seed: int,
                     device: str, caps: PubCaps) -> Any:
    import datasets
    trainer_cls = _import_fastfit()
    train = datasets.Dataset.from_dict({"text": _texts(train_rows),
                                        "label": [label_texts[ex.label] for ex in train_rows]})
    # FastFit's label set is the union of the train and test labels. A label-only "test"
    # set makes it the full label set without showing FastFit a single test row.
    labels_only = datasets.Dataset.from_dict({"text": label_texts, "label": label_texts})
    return trainer_cls(
        model_name_or_path=caps.fastfit_backbone, train_dataset=train, test_dataset=labels_only,
        label_column_name="label", text_column_name="text", num_train_epochs=caps.fastfit_epochs,
        max_steps=caps.fastfit_max_steps, per_device_train_batch_size=caps.fastfit_batch_size,
        max_text_length=caps.max_text_length, dataloader_drop_last=False,
        num_repeats=caps.fastfit_num_repeats, optim="adafactor", clf_loss_factor=0.1,
        fp16=device == "cuda", seed=seed, report_to="none")


@torch.no_grad()
def _fastfit_logits(trainer: Any, rows: Sequence[Example], label_texts: List[str],
                    caps: PubCaps) -> np.ndarray:
    """``inference_forward`` directly: ``export_model`` fails under transformers 4.57."""
    model = trainer.model.eval()
    dev = next(model.parameters()).device
    texts, out = _texts(rows), []
    for i in range(0, len(texts), caps.infer_batch_size):
        # FastFit rewrites the tokenizer's model_input_names, which drops the mask unless asked.
        enc = trainer.tokenizer(texts[i: i + caps.infer_batch_size], padding=True,
                                truncation=True, max_length=caps.max_text_length,
                                return_tensors="pt", return_attention_mask=True)
        out.append(model.inference_forward(enc["input_ids"].to(dev),
                                           enc["attention_mask"].to(dev)).float().cpu().numpy())
    return _to_label_order(np.concatenate(out), trainer.labels, label_texts, np.log(LOG_EPS))


def fastfit_baseline(splits: Splits, seed: int, device: str, group_by: Optional[str],
                     caps: PubCaps) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    label_texts = _label_texts(splits.train)
    trainer = _fastfit_trainer(splits.train, label_texts, seed, device, caps)
    if len(trainer.labels) != len(label_texts):
        raise ValueError("FastFit scores %d labels, expected %d"
                         % (len(trainer.labels), len(label_texts)))
    trainer.train()
    z_te, z_cal = (_fastfit_logits(trainer, rows, label_texts, caps)
                   for rows in (splits.test, splits.calibration))
    arm = logit_arm("fastfit", z_te, z_cal, label_space(splits.train), splits.test,
                    splits.calibration, t0, group_by)
    arm.update(package="fast-fit " + _package_version("fast-fit"),
               backbone=caps.fastfit_backbone, n_labels_scored=len(trainer.labels),
               caps={"max_steps": caps.fastfit_max_steps, "num_train_epochs": caps.fastfit_epochs,
                     "batch_size": caps.fastfit_batch_size, "num_repeats": caps.fastfit_num_repeats},
               confidence="softmax over FastFit similarity scores, as its pipeline reports",
               shims=["datasets.load_metric stub", "inference_forward instead of export_model"])
    return [arm]


# -- SetFit -------------------------------------------------------------------------------

def _setfit_model(train_rows: Sequence[Example], seed: int, caps: PubCaps) -> Any:
    import datasets
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(caps.setfit_backbone)
    args = TrainingArguments(batch_size=caps.setfit_batch_size, num_epochs=1,
                             num_iterations=caps.setfit_num_iterations,
                             max_steps=caps.setfit_max_steps, seed=seed, report_to="none",
                             save_strategy="no")
    train = datasets.Dataset.from_dict({"text": _texts(train_rows),
                                        "label": [ex.label for ex in train_rows]})
    Trainer(model=model, args=args, train_dataset=train).train()
    return model


def _setfit_logits(model: Any, rows: Sequence[Example], n_labels: int,
                   caps: PubCaps) -> np.ndarray:
    """Log of the head's probabilities; a label the head never saw gets probability 0."""
    p = model.predict_proba(_texts(rows), batch_size=caps.infer_batch_size, as_numpy=True)
    z = np.log(np.clip(np.asarray(p, dtype=np.float64), LOG_EPS, None))
    classes = [int(c) for c in model.model_head.classes_]
    return _to_label_order(z, classes, list(range(n_labels)), np.log(LOG_EPS))


def setfit_baseline(splits: Splits, seed: int, device: str, group_by: Optional[str],
                    caps: PubCaps) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    labels = label_space(splits.train)
    model = _setfit_model(splits.train, seed, caps)
    z_te, z_cal = (_setfit_logits(model, rows, len(labels), caps)
                   for rows in (splits.test, splits.calibration))
    arm = logit_arm("setfit", z_te, z_cal, labels, splits.test, splits.calibration, t0, group_by)
    arm.update(package="setfit " + _package_version("setfit"), backbone=caps.setfit_backbone,
               n_labels_scored=len(labels),
               n_labels_trained=len(model.model_head.classes_),
               head=type(model.model_head).__name__,
               caps={"num_iterations": caps.setfit_num_iterations,
                     "max_steps": caps.setfit_max_steps, "batch_size": caps.setfit_batch_size,
                     "num_epochs": 1})
    return [arm]


# -- the batch ----------------------------------------------------------------------------

def guarded(name: str, fn: Callable[[], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Run one arm; any failure becomes an error arm and the batch continues."""
    try:
        return fn()
    except Exception as e:  # third-party package boundary: every failure is a reportable result
        log.exception("baseline %s failed", name)
        return [{"arm": name + " (failed)", "n": 0,
                 "errors": "%s: %s" % (type(e).__name__, str(e)[:200])}]


def _classifier_arms(cfg: RunConfig, splits: Splits, device: str,
                     hf_token: Optional[str]) -> List[Dict[str, Any]]:
    from .runtime import load_base
    model, tok = load_base(cfg.li, token=hf_token)
    model.to(device)
    return run_classifier_baselines(model.encoder, tok, splits.train, splits.calibration,
                                    splits.test, cfg.li, device, group_by=cfg.group_by)


def run_pub_baselines(cfg: RunConfig, device: str, hf_token: Optional[str] = None,
                      caps: Optional[PubCaps] = None,
                      on_arm: Optional[Callable[[Dict[str, Any]], None]] = None
                      ) -> Dict[str, Any]:
    """Classifier, FastFit and SetFit arms on the run's own splits. No layax training.

    ``on_arm`` gets the report after every arm, so a cancelled job keeps what finished.
    """
    from .pipeline import load_splits
    if cfg.heldout_labels:
        raise ValueError("run_pub_baselines does not apply held-out labels")
    caps = caps or PubCaps()
    splits = load_splits(cfg)
    report: Dict[str, Any] = {"config": cfg.to_dict(), "device": device, "caps": asdict(caps),
                              "splits": splits.summary(), "note": BACKBONE_NOTE, "arms": []}
    steps = [("classifiers", lambda: _classifier_arms(cfg, splits, device, hf_token)),
             ("fastfit", lambda: fastfit_baseline(splits, cfg.li.seed, device, cfg.group_by, caps)),
             ("setfit", lambda: setfit_baseline(splits, cfg.li.seed, device, cfg.group_by, caps))]
    for name, fn in steps:
        report["arms"].extend(guarded(name, fn))
        report["table"] = compare(report["arms"])
        if on_arm is not None:
            on_arm(report)
    return report


def write_report(report: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
