"""Loader behaviour that decides what a capped run actually trains and tests on."""
from __future__ import annotations

from typing import Dict

import pytest

datasets = pytest.importorskip("datasets")
Dataset = datasets.Dataset

from layax import data  # noqa: E402


def _sorted_banking77(n_per_label: int) -> Dict[str, Dataset]:
    """Rows grouped by label, the way the Hub split is ordered."""
    def split(n):
        return Dataset.from_list([{"text": "msg %d %d" % (lab, i), "label": lab, "label_text": "intent_%02d" % lab}
                for lab in range(77) for i in range(n)])
    return {"train": split(n_per_label), "test": split(40)}


@pytest.fixture
def fake_hub(monkeypatch):
    monkeypatch.setattr(data, "_hf", lambda *a, **k: _sorted_banking77(130))


def test_load_banking77_when_train_capped_on_sorted_split_then_all_labels_present(fake_hub):
    train, _ = data.load_banking77(max_rows=2000)
    assert len(train) == 2000
    assert len({ex.label for ex in train}) == 77


def test_load_banking77_when_train_capped_then_test_not_capped(fake_hub):
    _, test = data.load_banking77(max_rows=500)
    assert len(test) == 77 * 40


def test_load_banking77_when_test_sliced_then_labels_spread(fake_hub):
    """The pipeline slices test with max_eval_rows, so test must arrive shuffled too."""
    _, test = data.load_banking77()
    assert len({ex.label for ex in test[:500]}) > 60


def test_load_banking77_when_same_seed_then_same_rows(fake_hub):
    a, _ = data.load_banking77(max_rows=300, seed=5)
    b, _ = data.load_banking77(max_rows=300, seed=5)
    assert [ex.state for ex in a] == [ex.state for ex in b]


def test_load_banking77_then_label_names_follow_label_ids(fake_hub):
    train, _ = data.load_banking77(max_rows=50)
    ex = train[0]
    assert list(ex.criteria)[ex.label] == "intent_%02d" % ex.label


def test_splits_from_meta_when_split_named_then_rows_follow_meta_and_others_unused():
    from layax.data import Example, splits_from_meta
    names = ["train", "train", "competence", "calibration", "test", "val"]
    rows = [Example("s%d" % i, "choice", "q", {"a": "a"}, 0, {"split": n}) for i, n in enumerate(names)]
    s = splits_from_meta(rows, "split")
    assert [len(s.train), len(s.competence), len(s.calibration), len(s.test)] == [2, 1, 1, 1]
    assert all(ex.meta["split"] != "val" for part in (s.train, s.test) for ex in part)
