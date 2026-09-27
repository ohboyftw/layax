"""Offline test doubles.

Everything in the test suite runs with no network and no Laya checkpoint. That is
deliberate: shape and masking bugs in the interaction are the ones that survive a
casual eyeball and then quietly cost a training run, so they need to be catchable in
a second on a laptop rather than only on a GPU with a 2 GB download in front.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
import torch


from layax._stubs import StubEncoder, StubTokenizer  # noqa: E402,F401


@pytest.fixture
def tok() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture
def encoder() -> StubEncoder:
    torch.manual_seed(0)
    return StubEncoder()


@pytest.fixture
def choice_questions() -> Dict[str, Dict[str, Any]]:
    return {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {"billing": "invoices payments refunds",
                         "technical": "bugs outages system errors",
                         "sales": "pricing new contracts"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "critical blocking issue"],
        },
        "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
    }


def wide_choice(n: int) -> Dict[str, Dict[str, Any]]:
    """A question with ``n`` labels -- the case upstream truncates to noise."""
    return {"intent": {"type": "choice", "instructions": "Which banking intent is this?",
                       "criteria": {"label_%03d" % i: "description for intent number %d" % i
                                    for i in range(n)}}}
