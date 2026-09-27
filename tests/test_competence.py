"""Competence head, calibration and the Learn-then-Test abstention threshold.

The tests that matter here are the ones asserting the head does something the softmax
cannot: separate errors when the errors are the confident ones.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from layax.competence import (
    walk_start,
    CompetenceHead,
    FeatureBuilder,
    IsotonicCalibrator,
    PlattCalibrator,
    aurc,
    clopper_pearson_upper,
    evaluate_selective,
    expected_calibration_error,
    fit_abstention_threshold,
    risk_coverage_curve,
    score_stats,
    script_profile,
)
from layax.config import CompConfig


def test_script_profile_detects_non_latin():
    """The Khmer signal has to be available before the forward pass, as upstream's is."""
    latin = script_profile("duplicate charge on invoice 4411")
    khmer = script_profile("ខ្ញុំត្រូវបានគិតប្រាក់ពីរដង")
    deva = script_profile("मुझसे दो बार शुल्क लिया गया")
    assert latin.argmax() == 0
    assert khmer[11] > 0.9          # khmer index
    assert deva[3] > 0.9            # devanagari index


def test_score_stats_shape_and_energy():
    logits = np.array([4.0, 1.0, 0.5, -1.0])
    mask = np.array([True, True, True, True])
    s = score_stats(logits, mask)
    assert s.shape == (6,)
    top1, margin, ent, k, spread, energy = s
    assert 0 < top1 < 1 and margin > 0 and 0 <= ent <= 1
    # Energy must move with the absolute logit level, which softmax discards entirely.
    low = score_stats(logits - 10.0, mask)
    assert abs(low[0] - top1) < 1e-5, "softmax top1 is shift invariant"
    assert low[5] > energy, "energy is not shift invariant -- that is why it is a feature"


def test_score_stats_respects_mask():
    logits = np.array([2.0, 1.0, 99.0, 99.0])
    s_masked = score_stats(logits, np.array([True, True, False, False]))
    assert s_masked[3] == pytest.approx(2 / 255.0)
    assert s_masked[0] == pytest.approx(np.exp(2) / (np.exp(2) + np.exp(1)), abs=1e-6)


def test_aurc_rewards_a_signal_that_ranks_errors():
    rng = np.random.default_rng(0)
    correct = rng.binomial(1, 0.8, 500).astype(float)
    useless = rng.random(500)
    informative = correct + rng.normal(0, 0.2, 500)
    assert aurc(informative, correct) < aurc(useless, correct)


def test_aurc_is_blind_spot_of_ece():
    """A perfectly calibrated-on-average score can still be useless for gating.

    This is the reason the harness reports AURC next to ECE: upstream reports ECE, and
    ECE alone would call this signal excellent.
    """
    n = 1000
    correct = np.zeros(n)
    correct[: int(0.8 * n)] = 1.0
    # Shuffle, or the stable sort over tied scores would rank by construction order and
    # hand a signal-free score a perfect curve. Ties must be arbitrary to mean anything.
    np.random.default_rng(0).shuffle(correct)
    constant = np.full(n, 0.8)          # exactly the true accuracy, everywhere
    assert expected_calibration_error(constant, correct) < 0.01
    # ...and yet it ranks nothing: risk is flat at 0.2 across every coverage.
    c = risk_coverage_curve(constant, correct)
    assert abs(c["risk"][-1] - 0.2) < 1e-9
    assert aurc(constant, correct) > 0.15


def test_isotonic_is_monotone_and_calibrated():
    rng = np.random.default_rng(1)
    s = rng.random(2000)
    y = (rng.random(2000) < s ** 2).astype(float)      # miscalibrated on purpose
    iso = IsotonicCalibrator().fit(s, y)
    out = iso.predict(np.linspace(0, 1, 50))
    assert np.all(np.diff(out) >= -1e-9), "isotonic output must be non-decreasing"
    assert expected_calibration_error(iso.predict(s), y) < expected_calibration_error(s, y)


def test_platt_runs_and_bounds():
    rng = np.random.default_rng(2)
    s = rng.normal(0, 1, 500)
    y = (s > 0).astype(float)
    p = PlattCalibrator().fit(s, y).predict(s)
    assert np.all((p >= 0) & (p <= 1))


def test_clopper_pearson_is_above_the_point_estimate():
    assert clopper_pearson_upper(5, 100, 0.1) > 0.05
    assert clopper_pearson_upper(0, 100, 0.1) > 0.0
    assert clopper_pearson_upper(50, 50, 0.1) == 1.0
    # Tighter with more data at the same rate -- the reason exact beats Hoeffding here.
    assert clopper_pearson_upper(50, 1000, 0.1) < clopper_pearson_upper(5, 100, 0.1)


def test_threshold_meets_target_risk_on_calibration():
    rng = np.random.default_rng(3)
    n = 4000
    comp = rng.random(n)
    correct = (rng.random(n) < comp).astype(float)     # competence is genuinely predictive
    start = walk_start(rng.random(n), 0.1)              # independent reference, as in fit_competence
    info = fit_abstention_threshold(comp, correct, target_risk=0.1, delta=0.1, min_coverage=0.1,
                                    start_threshold=start)
    assert info["feasible"], info.get("reason")
    assert info["risk"] <= 0.1 + 1e-9
    assert info["coverage"] >= 0.1


def test_threshold_reports_infeasible_rather_than_pretending():
    """A gate that cannot hit the target must say so, not ship the closest threshold."""
    rng = np.random.default_rng(4)
    n = 2000
    comp = rng.random(n)
    correct = (rng.random(n) < 0.5).astype(float)      # score carries no information
    info = fit_abstention_threshold(comp, correct, target_risk=0.01, delta=0.1, min_coverage=0.3)
    assert info["feasible"] is False
    assert info["threshold"] is None, "an infeasible gate must not hand back a usable threshold"
    assert "reason" in info


def test_threshold_grid_when_calibration_scores_differ_then_grid_identical():
    """Learn-then-Test: the hypothesis family is fixed before the calibration data is seen."""
    rng = np.random.default_rng(20)
    a = fit_abstention_threshold(rng.random(500), rng.binomial(1, 0.9, 500), 0.1, 0.1, 0.1)
    b = fit_abstention_threshold(rng.beta(8, 2, 300), rng.binomial(1, 0.7, 300), 0.1, 0.1, 0.1)
    assert [r["threshold"] for r in a["curve"]] == [r["threshold"] for r in b["curve"]]


def test_walk_start_when_reference_uniform_then_start_answers_min_coverage():
    ref = np.random.default_rng(23).random(5000)
    start = walk_start(ref, 0.3)
    assert (ref >= start).mean() >= 0.3
    assert start == pytest.approx(0.7, abs=0.02), "strictest grid point that reaches 30%"


def test_fit_threshold_when_chosen_coverage_below_min_then_infeasible():
    """min_coverage is a ship check after the walk, not a filter on the tested family."""
    rng = np.random.default_rng(24)
    s = rng.random(4000)
    y = (rng.random(4000) >= 0.5 * (1 - s)).astype(float)       # only the top scores are safe
    info = fit_abstention_threshold(s, y, target_risk=0.05, delta=0.1, min_coverage=0.9,
                                    start_threshold=0.95)
    assert info["feasible"] is False and info["threshold"] is None


def _old_bonferroni_coverage(s, y, target, delta, min_cov, grid=100):
    """The procedure this replaced: quantile grid over the calibration scores, delta/grid."""
    best = 0.0
    for t in np.quantile(s, np.linspace(0.0, 0.99, grid)):
        sel = s >= t
        ub = clopper_pearson_upper(int((1 - y[sel]).sum()), int(sel.sum()), delta / grid)
        if ub <= target and sel.mean() >= min_cov:
            best = max(best, float(sel.mean()))
    return best


def test_fixed_sequence_when_risk_monotone_then_coverage_at_least_bonferroni():
    rng = np.random.default_rng(21)
    s = rng.random(4000)
    y = (rng.random(4000) >= 0.3 * (1 - s) ** 2).astype(float)   # error rate falls with s
    start = walk_start(rng.random(4000), 0.1)
    info = fit_abstention_threshold(s, y, target_risk=0.05, delta=0.1, min_coverage=0.1,
                                    start_threshold=start)
    old = _old_bonferroni_coverage(s, y, 0.05, 0.1, 0.1)
    assert info["feasible"] and old > 0, "setup check: both procedures should find a gate"
    assert info["coverage"] >= old


def test_learn_then_test_when_repeated_then_violation_rate_near_delta():
    """P(true selective risk at the chosen threshold > target) <= delta, by simulation.

    Scores are uniform and P(error | s) = 0.4 (1 - s), so the true selective risk at a
    threshold t is 0.2 (1 - t) in closed form and needs no test sample.
    """
    rng = np.random.default_rng(22)
    target, delta, repeats, violations, feasible = 0.1, 0.1, 200, 0, 0
    for _ in range(repeats):
        s = rng.random(400)
        y = (rng.random(400) >= 0.4 * (1 - s)).astype(float)
        start = walk_start(rng.random(400), 0.1)
        info = fit_abstention_threshold(s, y, target, delta, min_coverage=0.1,
                                        start_threshold=start)
        if info["feasible"]:
            feasible += 1
            violations += 0.2 * (1 - info["threshold"]) > target
    assert feasible > repeats // 2, "setup check: the gate should usually be feasible"
    assert violations / repeats <= delta + 0.05


def test_evaluate_selective_reports_both_sides():
    rng = np.random.default_rng(5)
    comp = rng.random(1000)
    correct = (rng.random(1000) < comp).astype(float)
    out = evaluate_selective(comp, correct, threshold=0.5)
    assert out["selective_accuracy"] > out["full_accuracy"], "gating should raise accuracy"
    assert out["accuracy_on_abstained"] < out["full_accuracy"]
    assert out["coverage"] + out["abstained"] / 1000 == pytest.approx(1.0)


def _rows(n, pooled_dim=8, k=4, seed=0):
    rng = np.random.default_rng(seed)
    return [{"pooled": rng.normal(0, 1, pooled_dim).astype(np.float32),
             "logits": rng.normal(0, 2, k).astype(np.float32),
             "option_mask": np.ones(k, dtype=bool),
             "max_sim": float(rng.random()), "mean_sim": float(rng.random()),
             "std_sim": float(rng.random()),
             "n_state_tokens": int(rng.integers(10, 500)), "n_truncated": 0,
             "n_options": k, "state_text": "some english text here"} for _ in range(n)]


def test_feature_builder_width_matches_spec():
    cfg = CompConfig()
    b = FeatureBuilder(cfg, pooled_dim=8)
    X = b.build(_rows(20))
    assert X.shape == (20, b.spec.dim)
    assert np.isfinite(X).all()


def test_feature_groups_can_be_ablated():
    rows = _rows(10)
    full = FeatureBuilder(CompConfig(), 8)
    lean = FeatureBuilder(CompConfig(use_pooled=False, use_mahalanobis=False), 8)
    assert lean.build(rows).shape[1] < full.build(rows).shape[1]


def test_mahalanobis_flags_shifted_rows():
    """The density feature must actually move for out-of-distribution pooled vectors."""
    cfg = CompConfig()
    b = FeatureBuilder(cfg, pooled_dim=8)
    rng = np.random.default_rng(7)
    clean = rng.normal(0, 1, (500, 8))
    b.fit_density(clean)
    near = b._mahalanobis(rng.normal(0, 1, (100, 8)))
    far = b._mahalanobis(rng.normal(6, 1, (100, 8)))
    assert far.mean() > near.mean() + 1.0


def test_competence_head_learns_confidently_wrong():
    """The load-bearing test.

    Construct data where high softmax confidence means WRONG. Softmax-based gating is
    then worse than useless, and a head with access to a shift feature should recover.
    If this ever fails, the head is not earning its place.
    """
    rng = np.random.default_rng(11)
    n = 3000
    rows, correct = [], []
    for i in range(n):
        foreign = i % 3 == 0
        # Foreign rows: very peaked logits (high confidence) but the answer is wrong.
        logits = rng.normal(0, 6 if foreign else 1.2, 5).astype(np.float32)
        ok = 0.0 if foreign else float(rng.random() < 0.85)
        rows.append({"pooled": rng.normal(4 if foreign else 0, 1, 8).astype(np.float32),
                     "logits": logits, "option_mask": np.ones(5, dtype=bool),
                     "max_sim": 0.1 if foreign else 0.8,
                     "mean_sim": 0.05 if foreign else 0.5, "std_sim": 0.1,
                     "n_state_tokens": 60, "n_truncated": 0, "n_options": 5,
                     "state_text": "ខ្ញុំត្រូវបាន" if foreign else "i lost my card"})
        correct.append(ok)
    y = np.array(correct)

    cfg = CompConfig(epochs=25, hidden=[32, 16])
    b = FeatureBuilder(cfg, 8)
    clean_idx = [i for i in range(n) if i % 3 != 0]
    b.fit_density(np.stack([rows[i]["pooled"] for i in clean_idx]))
    X = b.build(rows)
    b.fit_scaler(X)
    Xs = b.transform_scale(X)

    from layax.train_competence import _train_head
    head = _train_head(Xs, y, cfg, device="cpu")
    head.eval()
    with torch.no_grad():
        comp = torch.sigmoid(head(torch.from_numpy(Xs).float())).numpy()

    softmax_conf = np.array([score_stats(r["logits"], r["option_mask"])[0] for r in rows])
    assert aurc(softmax_conf, y) > 0.2, "setup check: softmax should be a bad gate here"
    assert aurc(comp, y) < aurc(softmax_conf, y) - 0.1, "competence head must beat softmax"
    assert comp[y == 0].mean() < comp[y == 1].mean()


def _agent_with_gate(encoder, tok, threshold_info):
    from layax.competence import CompetenceModel
    from layax.config import LIConfig
    from layax.li_head import LateInteractionDecisionModel
    from layax.runtime import LayaxAgent
    cfg = CompConfig()
    b = FeatureBuilder(cfg, pooled_dim=32)
    torch.manual_seed(0)
    # Calibrator that inverts the score: any gate reading it instead of the raw sigmoid
    # would disagree with the raw comparison on nearly every row.
    cal = IsotonicCalibrator()
    cal.x, cal.y = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    comp = CompetenceModel(cfg, b, CompetenceHead(b.spec.dim, [8]), cal, threshold_info)
    li = LIConfig(base_checkpoint="stub", base_subfolder=None, proj_dim=16,
                  state_max_len=64, option_max_len=8)
    return LayaxAgent(LateInteractionDecisionModel(encoder, li), tok, device="cpu", competence=comp)


def test_runtime_abstain_when_gate_infeasible_then_none(encoder, tok, choice_questions):
    agent = _agent_with_gate(encoder, tok, {"threshold": None, "feasible": False})
    out = agent.predict("we were billed twice", choice_questions)
    assert all(a["abstain"] is None for a in out["answers"].values())
    assert all("competence" in a for a in out["answers"].values())


def test_runtime_abstain_when_gate_feasible_then_compares_raw_score(encoder, tok, choice_questions):
    agent = _agent_with_gate(encoder, tok, {"threshold": 0.5, "feasible": True})
    out = agent.predict("we were billed twice", choice_questions, return_features=True)
    raw = agent.competence.score_raw(out["features"])
    for r, qid in enumerate(choice_questions):
        assert out["answers"][qid]["abstain"] == bool(raw[r] < 0.5)
        assert out["answers"][qid]["competence"] == pytest.approx(1.0 - raw[r], abs=1e-4)


def test_competence_model_when_infeasible_then_threshold_none():
    from layax.competence import CompetenceModel
    b = FeatureBuilder(CompConfig(), 8)
    m = CompetenceModel(CompConfig(), b, CompetenceHead(b.spec.dim, [8]), None,
                        {"threshold": 0.7, "feasible": False})
    assert m.threshold is None


def test_competence_head_round_trips(tmp_path):
    from layax.competence import CompetenceModel
    cfg = CompConfig(epochs=1)
    b = FeatureBuilder(cfg, 8)
    rows = _rows(60)
    X = b.build(rows)
    b.fit_scaler(X)
    head = CompetenceHead(b.spec.dim, cfg.hidden, cfg.dropout)
    cal = IsotonicCalibrator().fit(np.linspace(0, 1, 60), np.round(np.linspace(0, 1, 60)))
    m = CompetenceModel(cfg, b, head, cal, {"threshold": 0.4, "feasible": True})
    m.save(str(tmp_path))
    back = CompetenceModel.load(str(tmp_path))
    assert back.threshold == 0.4 and back.feasible
    assert np.allclose(m.score(rows), back.score(rows), atol=1e-5)
