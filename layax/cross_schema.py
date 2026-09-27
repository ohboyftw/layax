"""Cross-schema scoring: a trained checkpoint against a label set it never saw.

Subsets, renames and merges of the trained labels are free for a plain classifier
(mask, relabel or sum its outputs), so they cannot show an advantage for option heads.
The one schema change a classifier cannot make is a label set with no trained labels in
it, where it scores exactly zero. This module builds those evaluations: another
dataset's test rows scored against that dataset's full label set, plus label-wording
variants of the training schema, which a classifier is immune to by construction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .data import Example


def drop_label(rows: Sequence[Example], name: str) -> List[Example]:
    """Remove one label from the schema and every row whose gold is that label."""
    names = list(rows[0].criteria)
    keep = {k: v for k, v in rows[0].criteria.items() if k != name}
    keys = list(keep)
    return [Example(ex.state, ex.qtype, ex.instructions, keep, keys.index(names[ex.label]),
                    dict(ex.meta)) for ex in rows if names[ex.label] != name]


def reword(rows: Sequence[Example], template: str) -> List[Example]:
    """Same rows, same gold index, option text rewritten as ``template.format(key, text)``."""
    crit = {k: template.format(key=k, text=v) for k, v in rows[0].criteria.items()}
    return [Example(ex.state, ex.qtype, ex.instructions, crit, ex.label, dict(ex.meta))
            for ex in rows]


def score(res: Dict[str, Any]) -> Dict[str, Any]:
    """Accuracy, chance and n from a ``collect_predictions`` result."""
    k = int(res["n_options"][0])
    return {"n": int(len(res["correct"])), "options": k,
            "accuracy": round(float(np.mean(res["correct"])), 4), "chance": round(1.0 / k, 4)}
