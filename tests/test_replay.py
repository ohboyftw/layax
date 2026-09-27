import json

import numpy as np
import pytest

pytest.importorskip("torch")
from layax.data import Example  # noqa: E402
from layax.replay import replay_lines  # noqa: E402

ROWS = [Example("s%d" % i, "choice", "q", {"a": "a", "b": "b"}, i % 2, {"language": "en"})
        for i in range(3)]


def test_replay_lines_when_foreign_then_gold_is_null():
    lines = [json.loads(s) for s in replay_lines("foreign", ROWS, np.zeros((3, 2)))]
    assert all(r["gold"] is None for r in lines)


def test_replay_lines_when_labelled_then_gold_and_logits_match_rows():
    z = np.arange(6, dtype=float).reshape(3, 2)
    lines = [json.loads(s) for s in replay_lines("test", ROWS, z)]
    assert [r["gold"] for r in lines] == [0, 1, 0]
    assert [r["logits"] for r in lines] == z.tolist()
