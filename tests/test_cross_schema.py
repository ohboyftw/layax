from layax.cross_schema import drop_label, reword
from layax.data import Example

CRIT = {"a": "alpha", "oos": "oos", "b": "beta"}


def _rows():
    return [Example("s%d" % i, "choice", "q", CRIT, i % 3) for i in range(6)]


def test_drop_label_when_removed_then_gold_still_names_same_label():
    rows = _rows()
    out = drop_label(rows, "oos")
    kept = [ex for ex in rows if list(CRIT)[ex.label] != "oos"]
    assert [list(o.criteria)[o.label] for o in out] == [list(CRIT)[e.label] for e in kept]


def test_drop_label_when_removed_then_absent_from_schema():
    assert "oos" not in drop_label(_rows(), "oos")[0].criteria


def test_reword_when_applied_then_gold_index_and_keys_unchanged():
    rows = _rows()
    out = reword(rows, "about {text}")
    assert [o.label for o in out] == [r.label for r in rows]
    assert list(out[0].criteria) == list(CRIT)
    assert out[0].criteria["a"] == "about alpha"
