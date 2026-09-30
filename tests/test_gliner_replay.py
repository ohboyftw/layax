"""GLiNER2 replay: label text must reach the model verbatim or not at all."""
from __future__ import annotations

import sys
import types

import pytest

from layax.data import Example
from layax.gliner_replay import _labels, _scorer


def _ref(criteria: dict) -> Example:
    return Example("text", "choice", "Which intent?", criteria, 0)


def test_labels_when_plain_text_then_option_text_in_schema_order():
    assert _labels(_ref({"card_arrival": "card arrival", "top_up": "top up"})) == \
        ["card arrival", "top up"]


def test_labels_when_option_text_has_marker_characters_then_rejected():
    """GLiNER2 would silently misalign logits to labels, so refuse rather than rewrite."""
    with pytest.raises(ValueError):
        _labels(_ref({"a": "fees (atm)", "b": "top up"}))


def test_labels_when_option_texts_repeat_then_rejected():
    """GLiNER2 keys logits by label text, so two options would share one score."""
    with pytest.raises(ValueError):
        _labels(_ref({"a": "top up", "b": "top up"}))


class _FakeScores:
    def __init__(self, by_label: dict):
        self.by_label = by_label

    def logit(self, task: str, label: str) -> float:
        return self.by_label[label]


class _FakeClassifier:
    """Stands in for gliner2's Classifier; logits are keyed by label text, as in gliner2."""

    by_label = {"zebra": 3.0, "apple": 1.0, "mango": 2.0}

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "_FakeClassifier":
        return cls()

    def to(self, device: str) -> "_FakeClassifier":
        return self

    def eval(self) -> "_FakeClassifier":
        return self

    def batch_score(self, texts: list, schema: object, config: object) -> list:
        return [_FakeScores(self.by_label) for _ in texts]


def _fake_gliner2(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = types.ModuleType("gliner2.classification.engine")
    engine.Classifier, engine.ClassificationConfig = _FakeClassifier, lambda **kw: kw
    schema = types.ModuleType("gliner2.classification.schema")
    schema.ClassificationSchema = lambda: types.SimpleNamespace(single=lambda *a, **kw: None)
    for name, mod in [("gliner2", types.ModuleType("gliner2")),
                      ("gliner2.classification", types.ModuleType("gliner2.classification")),
                      (engine.__name__, engine), (schema.__name__, schema)]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_scorer_when_labels_not_sorted_then_columns_follow_option_order(monkeypatch):
    _fake_gliner2(monkeypatch)
    ref = _ref({"z": "zebra", "a": "apple", "m": "mango"})
    score = _scorer(ref, "unused-dir", "cpu", batch_size=2)
    assert score([ref, ref]).tolist() == [[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]]
