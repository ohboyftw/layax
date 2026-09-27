import numpy as np

from layax.jsonl_eval import proceed_stats

LABELS = ["held", "slipped", "dropped"]


def test_proceed_stats_when_confident_held_then_only_those_rows_proceed():
    probs = np.array([[0.9, 0.05, 0.05], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8], [0.95, 0.03, 0.02]])
    gold = np.array([0, 0, 2, 2])
    s = proceed_stats(probs, gold, LABELS, 0.8, "held", "dropped")
    assert s["n_proceed"] == 2
    assert s["not_held"] == 0.5
    assert s["dropped"] == 0.5


def test_proceed_stats_when_nothing_clears_then_rates_are_none():
    s = proceed_stats(np.array([[0.5, 0.3, 0.2]]), np.array([0]), LABELS, 0.9, "held", "dropped")
    assert s["n_proceed"] == 0 and s["not_held"] is None
