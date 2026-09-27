"""Baseline arms: label-index classifiers, the laya shortlist k-sweep, and the table.

The `laya` package is not a test dependency, so the shortlist test installs a small
stand-in module that follows laya 0.3.5's public contract: ``predict_shortlist`` ranks
options by cosine on ``embed_fn`` output and returns a ``shortlist`` block with the kept
labels. The stub embed function is built so the expected recall is known exactly.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from layax.baselines import (
    finetuned_classifier_baseline,
    fit_logit_temperature,
    frozen_mlp_baseline,
    label_space,
    logit_arm,
    run_classifier_baselines,
)
from layax.config import LIConfig
from layax.data import Example
from layax.evaluate import COMPARE_KEYS, compare, evaluate_laya_baseline
from layax.pipeline import competence_arm

from test_integration import LABELS, make_rows


def _cfg(**kw) -> LIConfig:
    base = dict(base_checkpoint="stub", base_subfolder=None, state_max_len=48,
                option_max_len=8, epochs=3, batch_size=16, lr_encoder=3e-3, lr_head=3e-3,
                amp_dtype="fp32")
    base.update(kw)
    return LIConfig(**base)


def test_label_space_when_rows_disagree_then_raises():
    rows = make_rows(4)
    rows.append(Example("x", "choice", "q?", {"other": "o"}, 0, {}))
    with pytest.raises(ValueError):
        label_space(rows)


def test_run_classifier_baselines_when_no_shared_labels_then_reports_not_run(encoder, tok):
    rows = make_rows(8) + [Example("x", "choice", "q?", {"other": "o"}, 0, {})]
    arms = run_classifier_baselines(encoder, tok, rows, rows, rows, _cfg(), "cpu")
    assert len(arms) == 1 and arms[0]["n"] == 0 and "shared label set" in arms[0]["errors"]


def test_finetuned_classifier_when_task_learnable_then_beats_chance(encoder, tok):
    torch.manual_seed(0)
    test_rows = make_rows(120, seed=99)
    arm = finetuned_classifier_baseline(encoder, tok, make_rows(400, seed=1),
                                        make_rows(60, seed=3), test_rows, _cfg(), "cpu")
    assert arm["n"] == len(test_rows), "every test row is scored against every label"
    assert arm["accuracy"] > 1.5 / len(LABELS)
    assert set(COMPARE_KEYS) - {"selective_accuracy", "recall_at_k"} <= set(arm)


def test_finetuned_classifier_when_run_then_caller_encoder_unchanged(encoder, tok):
    """The pipeline hands over the encoder layax is about to train; it must come back as-is."""
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    finetuned_classifier_baseline(encoder, tok, make_rows(64), make_rows(16, seed=3),
                                  make_rows(16, seed=2), _cfg(epochs=1), "cpu")
    assert all(torch.equal(before[k], v) for k, v in encoder.state_dict().items())


def test_frozen_mlp_when_task_learnable_then_beats_chance(encoder, tok):
    torch.manual_seed(0)
    test_rows = make_rows(120, seed=99)
    arm = frozen_mlp_baseline(encoder, tok, make_rows(400, seed=1), make_rows(60, seed=3),
                              test_rows, _cfg(), "cpu")
    assert arm["n"] == len(test_rows)
    assert arm["accuracy"] > 1.5 / len(LABELS)


# -- temperature scaling of the classifier arms --------------------------------------------

def _calibrated_draw(n: int, k: int = 10, seed: int = 0):
    """Logits and gold sampled from softmax(logits): calibrated at T = 1 by construction."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, size=(n, k))
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    gold = np.array([rng.choice(k, p=row) for row in p])
    return z, gold


def test_fit_logit_temperature_when_logits_calibrated_then_temperature_near_one():
    z, gold = _calibrated_draw(2000)
    assert fit_logit_temperature(z, gold)["temperature"] == pytest.approx(1.0, abs=0.1)


def test_fit_logit_temperature_when_logits_tripled_then_temperature_near_three():
    """Multiplying calibrated logits by 3 is overconfidence the fit must undo."""
    z, gold = _calibrated_draw(2000)
    assert fit_logit_temperature(3.0 * z, gold)["temperature"] == pytest.approx(3.0, rel=0.1)


def test_fit_logit_temperature_when_optimum_beyond_clamp_then_flagged_at_bound():
    z, gold = _calibrated_draw(500)
    fit = fit_logit_temperature(20.0 * z, gold)
    assert fit["temperature"] == 5.0 and fit["at_bound"]


def _arm_rows(n: int, seed: int) -> tuple:
    rows = make_rows(n, seed=seed)
    z = np.random.default_rng(seed).normal(0.0, 3.0, size=(n, len(LABELS)))
    return rows, z


def test_logit_arm_when_only_test_changes_then_temperature_identical():
    """The temperature is a function of the calibration split alone."""
    cal_rows, z_cal = _arm_rows(200, seed=1)
    runs = [logit_arm("a", z_te, z_cal, LABELS, te_rows, cal_rows, 0.0)
            for te_rows, z_te in (_arm_rows(80, seed=2), _arm_rows(120, seed=3))]
    assert runs[0]["temperature"] == runs[1]["temperature"]
    assert runs[0]["ece"] != runs[1]["ece"], "test inputs really did differ"


def test_logit_arm_when_group_by_oos_then_every_test_row_in_one_group():
    rows, z = _arm_rows(90, seed=4)
    for i, ex in enumerate(rows):
        ex.meta["oos"] = i % 3 == 0
    arm = logit_arm("a", z, z, LABELS, rows, rows, 0.0, group_by="oos")
    assert set(arm["by_oos"]) == {"True", "False"}
    assert sum(g["n"] for g in arm["by_oos"].values()) == arm["n"] == 90


# -- laya shortlist k-sweep ---------------------------------------------------------------

def _fake_laya(embed) -> ModuleType:
    def predict(state, qs):
        labels = list(qs["q"]["criteria"])
        return {"answers": {"q": {"type": "choice", "choice": labels[0], "confidence": 0.5}}}

    def predict_shortlist(agent, state, qs, embed_fn, k):
        crit = qs["q"]["criteria"]
        texts = [state] + ["%s: %s" % (lab, d) for lab, d in crit.items()]
        m = embed_fn(texts)
        sims = m[1:] @ m[0] / (np.linalg.norm(m[1:], axis=1) * np.linalg.norm(m[0]) + 1e-12)
        kept = [list(crit)[i] for i in np.argsort(-sims, kind="mergesort")[:k]]
        res = predict(state, {"q": {**qs["q"], "criteria": {lab: crit[lab] for lab in kept}}})
        res["shortlist"] = {"q": {"labels": kept, "passthrough": False}}
        return res

    mod = ModuleType("laya")
    mod.load = lambda *a, **kw: SimpleNamespace(cfg={}, predict=predict)
    mod.embed_fn_from_agent = lambda agent: embed
    mod.predict_shortlist = predict_shortlist
    return mod


def test_laya_shortlist_when_gold_retrieved_for_half_then_recall_half(monkeypatch):
    """'hit' states embed onto their gold label; 'miss' states onto label 0.

    Golds are labels 10..29, and ties keep label order, so a miss row keeps labels 0..9
    and can never retrieve its gold.
    """
    labels = ["l%02d" % i for i in range(30)]
    crit = {lab: "d" for lab in labels}
    rows = [Example("%s %d" % ("hit" if i % 2 == 0 else "miss", 10 + i), "choice", "q?",
                    crit, 10 + i, {}) for i in range(20)]

    def embed(texts):
        kind, gold = texts[0].split()
        target = int(gold) if kind == "hit" else 0
        return np.array([np.eye(30)[target]] + [np.eye(30)[j] for j in range(30)])

    monkeypatch.setitem(sys.modules, "laya", _fake_laya(embed))
    arm = evaluate_laya_baseline(rows, shortlist_k=10)
    assert arm["arm"] == "laya+shortlist10"
    assert arm["recall_at_k"] == pytest.approx(0.5)
    assert arm["errors"] == 0 and arm["n"] == 20


# -- comparison table ---------------------------------------------------------------------

def _test_eval(gate_enabled: bool) -> dict:
    ev = {"gate_enabled": gate_enabled,
          "metrics": {"n": 100, "accuracy": 0.8, "ece_competence": 0.05, "aurc_competence": 0.1}}
    if gate_enabled:
        ev.update(coverage=0.6, selective_accuracy=0.97, abstained=40)
    return ev


def test_competence_arm_when_gate_off_then_no_selective_accuracy():
    row = competence_arm(_test_eval(False))
    assert row["accuracy"] == 0.8 and row["coverage"] == 1.0
    assert "selective_accuracy" not in row
    assert "gate off" in row["arm"]


def test_competence_arm_when_gate_on_then_accuracy_stays_full_coverage():
    """The fake-high headline: selective accuracy must never sit in the accuracy column."""
    row = competence_arm(_test_eval(True))
    assert row["accuracy"] == 0.8
    assert (row["coverage"], row["selective_accuracy"]) == (0.6, 0.97)
    assert row["errors"] == 0, "abstentions are not failures to run"


def test_compare_when_arm_lacks_a_column_then_cell_blank():
    table = compare([{"arm": "a", "n": 1, "accuracy": 0.5, "ece": None}])
    header, _, row = table.splitlines()
    assert "coverage" in header and "None" not in row
