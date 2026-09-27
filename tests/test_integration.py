"""End-to-end on a stub encoder: trainer, losses, calibration, competence, evaluation.

This covers everything except the Hugging Face download, which means a broken training
loop is caught in seconds locally instead of forty minutes into a Kaggle session.

The synthetic task has a real signal in it (the gold label's word appears in the state),
so a training loop that is wired up correctly can actually reduce the loss. A test that
only asserts "it ran" would pass with the gradients disconnected.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from layax.calibrate import ece_before_after, fit_temperatures
from layax.config import CompConfig, LIConfig, RunConfig
from layax.data import Example, build_shift_set, shift_report, split_examples
from layax.evaluate import collect_predictions, evaluate_predictions
from layax.li_head import LateInteractionDecisionModel
from layax.losses import in_batch_negative_loss, ordinal_cumlink_loss
from layax.train_competence import (
    aurc_by_shift,
    evaluate_competence,
    evaluate_shift_baselines,
    fit_competence,
    training_shift_config,
)
from layax.train_li import DecisionDataset, collate, train

from conftest import StubEncoder, StubTokenizer

LABELS = ["billing", "technical", "sales", "hardware", "access", "network"]
FILLER = ["the", "user", "reports", "an", "issue", "today", "please", "advise"]


def make_rows(n: int, seed: int = 0, language: str = "en") -> list[Example]:
    """State contains its gold label's keyword, plus noise. Learnable, but not trivially."""
    rng = random.Random(seed)
    criteria = {l: "requests about %s matters" % l for l in LABELS}
    rows = []
    for _ in range(n):
        gold = rng.randrange(len(LABELS))
        words = rng.choices(FILLER, k=8) + [LABELS[gold]] + rng.choices(FILLER, k=4)
        rng.shuffle(words)
        rows.append(Example(" ".join(words), "choice", "Which team should handle this?",
                            criteria, gold, {"language": language}))
    return rows


def _model(encoder, **kw) -> LateInteractionDecisionModel:
    base = dict(base_checkpoint="stub", base_subfolder=None, proj_dim=16,
                state_max_len=48, option_max_len=8, epochs=1, batch_size=8,
                grad_accum=1, amp_dtype="fp32", train_options_per_row=4, xattn_heads=2)
    base.update(kw)
    return LateInteractionDecisionModel(encoder, LIConfig(**base))


def test_collate_shapes_and_targets(tok):
    cfg = LIConfig(base_checkpoint="stub", base_subfolder=None, state_max_len=48,
                   option_max_len=8, train_options_per_row=4)
    ds = DecisionDataset(make_rows(8), tok, cfg, train=True)
    batch = collate([ds[i] for i in range(8)], tok.pad_token_id)
    B, K = batch["option_mask"].shape
    assert B == 8 and K == 4, "training should subsample to train_options_per_row"
    assert batch["opt_ids"].shape[:2] == (B, K)
    # Exactly one gold per row, and it sits inside the real options.
    assert torch.allclose(batch["target"].sum(-1), torch.ones(B))
    assert batch["option_mask"].gather(1, batch["label"][:, None]).all()


def test_eval_uses_full_option_set(tok):
    """Training subsamples; evaluation must not. This invariant is the headline number."""
    cfg = LIConfig(base_checkpoint="stub", base_subfolder=None, state_max_len=48,
                   option_max_len=8, train_options_per_row=4)
    rows = make_rows(4)
    train_ds = DecisionDataset(rows, tok, cfg, train=True)
    eval_ds = DecisionDataset(rows, tok, cfg, train=False)
    assert len(train_ds[0]["opt_ids"]) == 4
    assert len(eval_ds[0]["opt_ids"]) == len(LABELS)


def test_training_reduces_loss(encoder, tok):
    """Gradients reach the head and the loss moves. The wiring test."""
    torch.manual_seed(0)
    model = _model(encoder)
    log = train(model, tok, make_rows(160), model.cfg, device="cpu", log_every=1)
    losses = [h["total"] for h in log["history"] if "total" in h]
    assert len(losses) > 5
    assert np.mean(losses[-3:]) < np.mean(losses[:3]), "loss did not fall: check gradients"


def test_training_improves_accuracy(encoder, tok):
    """A stronger claim than loss: the task is learnable and the model learns it."""
    torch.manual_seed(0)
    model = _model(encoder, epochs=3, lr_head=3e-3, lr_encoder=3e-3)
    test_rows = make_rows(120, seed=99)
    before = collect_predictions(model, tok, test_rows, model.cfg, "cpu")["correct"].mean()
    train(model, tok, make_rows(400, seed=1), model.cfg, device="cpu")
    after = collect_predictions(model, tok, test_rows, model.cfg, "cpu")["correct"].mean()
    assert after > before, "accuracy %.3f -> %.3f" % (before, after)
    assert after > 1.5 / len(LABELS), "should clear random guessing"


def test_xattn_trains_too(encoder, tok):
    torch.manual_seed(0)
    model = _model(encoder, interaction="xattn")
    log = train(model, tok, make_rows(80), model.cfg, device="cpu", log_every=1)
    assert log["steps"] > 0
    assert all(np.isfinite(h.get("total", 0.0)) for h in log["history"])


def test_meanpool_when_trained_then_accuracy_improves(encoder, tok):
    """Gradients reach the meanpool path: the ablation arm is a real, trainable model."""
    torch.manual_seed(0)
    model = _model(encoder, interaction="meanpool", epochs=3, lr_head=3e-3, lr_encoder=3e-3)
    test_rows = make_rows(120, seed=99)
    before = collect_predictions(model, tok, test_rows, model.cfg, "cpu")["correct"].mean()
    train(model, tok, make_rows(400, seed=1), model.cfg, device="cpu")
    after = collect_predictions(model, tok, test_rows, model.cfg, "cpu")["correct"].mean()
    assert after > before and after > 1.5 / len(LABELS)


def test_in_batch_negatives_are_finite_and_directional():
    torch.manual_seed(0)
    B, L, m, p = 4, 6, 3, 8
    state = torch.nn.functional.normalize(torch.randn(B, L, p), dim=-1)
    gold = torch.nn.functional.normalize(torch.randn(B, m, p), dim=-1)
    mask = torch.ones(B, L, dtype=torch.bool)
    tm = torch.ones(B, m, dtype=torch.bool)
    scale = torch.full((B,), 4.0)
    loss = in_batch_negative_loss(state, mask, gold, tm, scale)
    assert loss.shape == (B,) and torch.isfinite(loss).all()
    # Aligning each state with its own gold must lower the loss.
    aligned = gold.mean(1, keepdim=True).expand(B, L, p)
    aligned = torch.nn.functional.normalize(aligned, dim=-1)
    better = in_batch_negative_loss(aligned, mask, gold, tm, scale)
    assert better.mean() < loss.mean()


def test_ordinal_loss_penalises_distance():
    """Predicting level 4 when the answer is 0 must cost more than predicting level 1."""
    probs_near = torch.tensor([[0.1, 0.7, 0.1, 0.05, 0.05]])
    probs_far = torch.tensor([[0.05, 0.05, 0.1, 0.1, 0.7]])
    label = torch.tensor([0])
    mask = torch.ones(1, 5, dtype=torch.bool)
    assert ordinal_cumlink_loss(probs_far, label, mask) > ordinal_cumlink_loss(probs_near, label, mask)


def test_temperature_fitting_improves_ece(encoder, tok):
    model = _model(encoder)
    rows = make_rows(300, seed=5)
    res = collect_predictions(model, tok, rows, model.cfg, "cpu")
    gold = [e.label for e in rows]
    fitted = fit_temperatures(res["features"], gold, min_rows=20)
    eff = ece_before_after(res["features"], gold, fitted["temperatures"])
    assert eff["ece_after"] <= eff["ece_before"] + 1e-6
    for b, v in fitted["temperatures"].items():
        assert 0.5 <= v <= 5.0, "temperature %s=%s escaped the clamp" % (b, v)


def test_shift_augmentation_mix(encoder, tok):
    """Clean rows must not be duplicated by a shift that could not be applied."""
    cfg = CompConfig(shift_foreign=0.0, shift_ood_domain=0.0)
    clean = make_rows(200, seed=3)
    shifted = build_shift_set(clean, cfg, seed=3)
    rep = shift_report(shifted)
    assert rep["clean"] == 200, "a no-op shift must be skipped, not appended as a clean copy"
    assert rep.get("truncate", 0) > 0
    # This dataset has one schema, so there are no labels outside a row's own option set
    # and the distractor shift correctly produces nothing.
    assert rep.get("distractors", 0) == 0


def test_distractors_apply_across_schemas(encoder, tok):
    """With a second label space available, distractors are injected and stay correct."""
    cfg = CompConfig(shift_foreign=0.0, shift_ood_domain=0.0, shift_distractors=1.0,
                     shift_truncate=0.0)
    other = {"shipping": "delivery status", "refund": "money back", "login": "account access"}
    clean = make_rows(100, seed=4)
    clean += [Example(e.state, e.qtype, e.instructions, other, 0, dict(e.meta))
              for e in make_rows(100, seed=5)]
    shifted = build_shift_set(clean, cfg, seed=4)
    rep = shift_report(shifted)
    assert rep.get("distractors", 0) > 0
    for ex in shifted:
        if ex.meta.get("shift") == "distractors":
            labels = list(ex.criteria.keys())
            assert 0 <= ex.label < len(labels), "gold index must survive the option growth"
            assert len(labels) > 3


def test_full_competence_cycle(encoder, tok):
    """splits -> fit -> calibrate -> threshold -> evaluate, with no network anywhere."""
    torch.manual_seed(0)
    model = _model(encoder)
    rows = make_rows(600, seed=7)
    splits = split_examples(rows, competence_frac=0.2, calibration_frac=0.15, seed=7)
    assert splits.summary()["train"] > 0

    train(model, tok, splits.train, model.cfg, device="cpu")

    cfg_c = CompConfig(epochs=4, hidden=[32, 16], shift_foreign=0.0, shift_ood_domain=0.0,
                       target_risk=0.2, min_coverage=0.2)
    out = fit_competence(model, tok, model.cfg, cfg_c, splits.competence,
                         splits.calibration, device="cpu", batch_size=8, seed=7)
    rep = out["report"]
    assert "verdict" in rep and "aurc_improvement_calibration" in rep
    assert set(rep["shift_mix"]) >= {"clean"}

    ev = evaluate_competence(out["competence"], model, tok, model.cfg, splits.test,
                             device="cpu", batch_size=8, group_by="language")
    assert ev["gate_enabled"] == rep["threshold"]["feasible"]
    if ev["gate_enabled"]:
        assert 0.0 <= ev["coverage"] <= 1.0
    else:
        assert ev["threshold"] is None and "selective_accuracy" not in ev
    assert "aurc_competence" in ev["metrics"]
    assert "by_language" in ev["metrics"]


def _fitted_with_heldout(encoder, tok, heldout):
    torch.manual_seed(0)
    model = _model(encoder)
    splits = split_examples(make_rows(400, seed=8), competence_frac=0.25,
                            calibration_frac=0.2, seed=8)
    cfg_c = CompConfig(epochs=2, hidden=[16], shift_foreign=0.0, shift_ood_domain=0.0,
                       shift_truncate=0.3, heldout_shift=heldout)
    out = fit_competence(model, tok, model.cfg, cfg_c, splits.competence,
                         splits.calibration, device="cpu", batch_size=16, seed=8)
    return model, splits, cfg_c, out


def test_fit_competence_when_shift_heldout_then_absent_from_training_mix(encoder, tok):
    _, _, _, out = _fitted_with_heldout(encoder, tok, "truncate")
    assert "truncate" not in out["report"]["shift_mix"]
    assert out["report"]["heldout_shift"] == "truncate"


def test_shift_baselines_when_shift_heldout_then_reported_on_test(encoder, tok):
    """The held-out shift is applied to the test set and flagged as never trained on."""
    model, splits, cfg_c, out = _fitted_with_heldout(encoder, tok, "truncate")
    shifted = build_shift_set(splits.test, cfg_c, seed=9)
    rep = evaluate_shift_baselines(out["competence"], model, tok, model.cfg, shifted,
                                   device="cpu", batch_size=16)
    assert rep["heldout_present"] is True
    assert rep["by_shift"]["truncate"]["heldout"] is True
    assert rep["by_shift"]["clean"]["heldout"] is False
    assert rep["by_shift"]["all"]["n"] == len(shifted)
    for k in ("aurc_msp", "aurc_energy", "aurc_competence"):
        assert 0.0 <= rep["by_shift"]["truncate"][k] <= 1.0


def test_aurc_by_shift_when_head_ranks_errors_then_beats_msp():
    rng = np.random.default_rng(3)
    n = 400
    correct = rng.binomial(1, 0.7, n).astype(float)
    feats = [{"logits": rng.normal(0, 2, 5).astype(np.float32),
              "option_mask": np.ones(5, dtype=bool)} for _ in range(n)]
    shifts = ["clean" if i % 2 else "truncate" for i in range(n)]
    out = aurc_by_shift(feats, correct, correct + rng.normal(0, 0.1, n), shifts)
    assert set(out) == {"clean", "truncate", "all"}
    assert out["clean"]["n"] + out["truncate"]["n"] == out["all"]["n"] == n
    assert out["all"]["aurc_competence"] < out["all"]["aurc_msp"]


def test_training_shift_config_when_heldout_then_only_that_fraction_zeroed():
    cfg = CompConfig(heldout_shift="foreign")
    t = training_shift_config(cfg)
    assert t.shift_foreign == 0.0
    assert (t.shift_truncate, t.shift_distractors) == (cfg.shift_truncate, cfg.shift_distractors)
    assert cfg.shift_foreign > 0, "the caller's config, used for the test set, is untouched"


def test_evaluate_predictions_flags_confidently_wrong(encoder, tok):
    """The Khmer signature must be surfaced by name, not left in a 51-row table."""
    model = _model(encoder)
    rows = make_rows(60, seed=11, language="en")
    res = collect_predictions(model, tok, rows, model.cfg, "cpu")
    # Force the signature: every row wrong, every row confident.
    res["correct"] = np.zeros(len(rows))
    for r in res["features"]:
        r["logits"] = np.array([20.0] + [0.0] * (len(LABELS) - 1), dtype=np.float32)
    for m in res["meta"]:
        m["language"] = "km"
    out = evaluate_predictions(res, group_by="language")
    assert "km" in out["confidently_wrong_groups"]


def test_run_config_round_trip(tmp_path):
    cfg = RunConfig(name="t", dataset="banking77", output_dir=str(tmp_path))
    p = tmp_path / "cfg.json"
    cfg.save(str(p))
    back = RunConfig.from_json(str(p))
    assert back.to_dict() == cfg.to_dict()


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        LIConfig(interaction="quantum").validate()
    with pytest.raises(ValueError):
        CompConfig(target_risk=1.5).validate()
    with pytest.raises(ValueError):
        CompConfig(use_pooled=False, use_score_stats=False, use_interaction_stats=False,
                   use_length_stats=False, use_lang_stats=False, use_energy=False,
                   use_mahalanobis=False).validate()
    with pytest.raises(ValueError):
        RunConfig(competence_frac=0.4, calibration_frac=0.3).validate()


def test_config_when_heldout_shift_unknown_then_raises():
    with pytest.raises(ValueError):
        CompConfig(heldout_shift="typos").validate()


def test_config_when_heldout_shift_has_zero_fraction_then_raises():
    with pytest.raises(ValueError):
        CompConfig(heldout_shift="ood", shift_ood_domain=0.0).validate()


def test_config_when_classifier_baselines_not_bool_then_raises():
    with pytest.raises(ValueError):
        RunConfig(classifier_baselines="yes").validate()
