"""Late-interaction head: shapes, masking, caching, and the option-count property."""
from __future__ import annotations

import pytest
import torch

from layax.config import LIConfig
from layax.li_head import (
    LateInteractionDecisionModel,
    OptionCache,
    build_option_sequence,
    build_state_sequence,
    pad_stack,
    render_options,
)
from layax.runtime import LayaxAgent

from conftest import wide_choice


def _model(encoder, **kw):
    cfg = LIConfig(base_checkpoint="stub", base_subfolder=None, proj_dim=16,
                   state_max_len=64, option_max_len=8, xattn_heads=2, **kw)
    return LateInteractionDecisionModel(encoder, cfg)


def test_state_sequence_excludes_options(tok, choice_questions):
    """The state sequence must not contain option text -- that is the whole change."""
    seq, trunc = build_state_sequence(tok, "duplicate charge on invoice 4411", "choice",
                                      "Which department?", 64)
    opt_ids = set(tok("billing: invoices payments refunds", add_special_tokens=False)["input_ids"])
    # The instruction is in the sequence, the option descriptions are not.
    assert len(seq) > 3
    assert not (opt_ids - {tok.cls_token_id, tok.sep_token_id}) <= set(seq)
    assert trunc == 0


def test_state_gets_whole_window(tok):
    """Long state fills the window; upstream would have spent ~200 tokens on options."""
    long_state = " ".join("word%d" % i for i in range(500))
    seq, trunc = build_state_sequence(tok, long_state, "choice", "Which?", 128)
    assert len(seq) == 128
    assert trunc > 0, "truncation count must be reported so the competence head can see it"


def test_forward_shapes_maxsim(encoder, tok, choice_questions):
    model = _model(encoder)
    agent = LayaxAgent(model, tok, device="cpu")
    out = agent.predict({"body": "we were billed twice please refund"}, choice_questions)
    assert set(out["answers"]) == set(choice_questions)
    assert out["answers"]["department"]["choice"] in ("billing", "technical", "sales")
    probs = out["answers"]["department"]["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert 0.0 <= out["answers"]["churn"]["noul"] <= 1.0
    assert out["answers"]["urgency"]["type"] == "score"


def test_forward_shapes_xattn(encoder, tok, choice_questions):
    model = _model(encoder, interaction="xattn")
    agent = LayaxAgent(model, tok, device="cpu")
    out = agent.predict("we were billed twice", choice_questions)
    probs = out["answers"]["department"]["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_ragged_option_counts_are_masked(encoder, tok):
    """Questions with different option counts share one padded tensor.

    If the padding slots leaked into the softmax, probabilities would not sum to 1 over
    the real labels -- so this is the test that the -1e4 fill is actually applied.
    """
    model = _model(encoder)
    agent = LayaxAgent(model, tok, device="cpu")
    qs = {
        "two": {"type": "choice", "instructions": "a or b?", "criteria": {"a": "first", "b": "second"}},
        "seven": {"type": "choice", "instructions": "which?",
                  "criteria": {"k%d" % i: "option %d" % i for i in range(7)}},
    }
    out = agent.predict("some state text", qs)
    assert len(out["answers"]["two"]["probabilities"]) == 2
    assert len(out["answers"]["seven"]["probabilities"]) == 7
    for qid in qs:
        assert abs(sum(out["answers"][qid]["probabilities"].values()) - 1.0) < 1e-3


MODES = ("maxsim", "xattn", "meanpool")


@pytest.mark.parametrize("mode", MODES)
def test_high_cardinality_runs_without_truncation(encoder, tok, mode):
    """77 labels: upstream raises or truncates to ~3 tokens each. Here every label is intact."""
    model = _model(encoder, interaction=mode)
    agent = LayaxAgent(model, tok, device="cpu")
    qs = wide_choice(77)
    out = agent.predict("i lost my card and need a replacement", qs)
    probs = out["answers"]["intent"]["probabilities"]
    assert len(probs) == 77
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    # Every option keeps its own tokens: distinct labels must not collapse to one value.
    assert len(set(round(v, 6) for v in probs.values())) > 1


@pytest.mark.parametrize("mode", MODES)
def test_state_length_is_independent_of_option_count(encoder, tok, mode):
    """The property that motivates the design: options no longer eat the state budget."""
    model = _model(encoder, interaction=mode)
    agent = LayaxAgent(model, tok, device="cpu")
    state = " ".join("token%d" % i for i in range(200))
    small = agent.predict(state, wide_choice(3))
    large = agent.predict(state, wide_choice(200))
    assert small["usage"]["input_tokens"] == large["usage"]["input_tokens"]


def test_option_cache_hits_on_second_call(encoder, tok, choice_questions):
    model = _model(encoder)
    agent = LayaxAgent(model, tok, device="cpu")
    warm = agent.warm_cache(choice_questions)
    assert warm["encoded"] > 0
    before = agent.cache.stats()["misses"]
    agent.predict("hello there", choice_questions)
    assert agent.cache.stats()["misses"] == before, "warm cache must not re-encode options"
    assert agent.cache.stats()["hits"] > 0


def test_cache_key_changes_with_interaction_mode():
    """A stale cache serving vectors from another mode would be invisible at eval time."""
    a = OptionCache("rev1", "maxsim", 128)
    b = OptionCache("rev1", "xattn", 128)
    c = OptionCache("rev2", "maxsim", 128)
    assert a.key("billing") != b.key("billing")
    assert a.key("billing") != OptionCache("rev1", "meanpool", 128).key("billing")
    assert a.key("billing") != c.key("billing")
    assert a.key("billing") == OptionCache("rev1", "maxsim", 128).key("billing")


@pytest.mark.parametrize("mode", MODES)
def test_cached_and_uncached_agree(encoder, tok, choice_questions, mode):
    """The cached path is the production path; it must not change the answer."""
    model = _model(encoder, interaction=mode)
    model.eval()
    agent = LayaxAgent(model, tok, device="cpu")
    cold = agent.predict("we were billed twice", choice_questions)
    agent.warm_cache(choice_questions)
    warm = agent.predict("we were billed twice", choice_questions)
    for qid in choice_questions:
        a, b = cold["answers"][qid], warm["answers"][qid]
        if a["type"] == "choice":
            assert a["choice"] == b["choice"]
            for k in a["probabilities"]:
                assert abs(a["probabilities"][k] - b["probabilities"][k]) < 1e-2


def test_maxsim_is_length_normalised(encoder, tok):
    """A verbose option must not win on word count alone."""
    model = _model(encoder)
    agent = LayaxAgent(model, tok, device="cpu")
    qs = {"q": {"type": "choice", "instructions": "which?",
                "criteria": {"short": "alpha",
                             "long": " ".join(["beta"] * 40)}}}
    out = agent.predict("gamma delta epsilon", qs)
    p = out["answers"]["q"]["probabilities"]
    # Untrained weights: the point is only that length does not produce a landslide.
    assert 0.02 < p["long"] < 0.98


def test_padded_option_slots_do_not_nan(encoder, tok):
    model = _model(encoder)
    logits, pooled, stats = model(
        *_dummy_batch(tok, model), option_embeddings=None)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(stats["max_sim"]).all()


def _dummy_batch(tok, model):
    """A batch where row 1 has fewer options than row 0, so slots are padded."""
    seqs = [build_state_sequence(tok, "state one text", "choice", "q?", 32)[0],
            build_state_sequence(tok, "state two text", "choice", "q?", 32)[0]]
    ids, att = pad_stack(seqs, tok.pad_token_id)
    opt_texts = [["alpha one", "beta two", "gamma three"], ["alpha one", "beta two"]]
    K = 3
    seqs_o, masks = [], torch.zeros((2, K), dtype=torch.bool)
    for i, opts in enumerate(opt_texts):
        for j in range(K):
            if j < len(opts):
                seqs_o.append(build_option_sequence(tok, opts[j], 8))
                masks[i, j] = True
            else:
                seqs_o.append([tok.pad_token_id])
    oid, oatt = pad_stack(seqs_o, tok.pad_token_id)
    m = oid.shape[1]
    oid = oid.reshape(2, K, m)
    oatt = oatt.reshape(2, K, m)
    oatt[~masks] = 0
    qt = torch.tensor([0, 0], dtype=torch.long)
    return ids, att, oid, oatt, masks, qt
