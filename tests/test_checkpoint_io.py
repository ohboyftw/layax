"""A saved agent reloads to identical predictions: the path every published checkpoint takes."""
import pytest

transformers = pytest.importorskip("transformers")
from layax.config import LIConfig  # noqa: E402
from layax.li_head import LateInteractionDecisionModel  # noqa: E402
from layax.runtime import LayaxAgent  # noqa: E402

WORDS = ["billing", "technical", "sales", "refund", "twice", "we", "were", "billed", "please"]


def _agent(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ":"] + WORDS))
    tok = transformers.BertTokenizerFast(vocab_file=str(vocab))
    enc = transformers.BertModel(transformers.BertConfig(
        vocab_size=len(WORDS) + 6, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32))
    cfg = LIConfig(base_checkpoint="stub", base_subfolder=None, proj_dim=8,
                   state_max_len=32, option_max_len=8)
    return LayaxAgent(LateInteractionDecisionModel(enc, cfg), tok, device="cpu")


def test_save_then_from_pretrained_when_reloaded_then_probabilities_identical(tmp_path):
    questions = {"q": {"type": "choice", "instructions": "which",
                       "criteria": {"billing": "billing", "technical": "technical", "sales": "sales"}}}
    agent = _agent(tmp_path)
    agent.save(str(tmp_path / "ckpt"))
    again = LayaxAgent.from_pretrained(str(tmp_path / "ckpt"), device="cpu")
    state = "we were billed twice please refund"
    before = agent.predict(state, questions)["answers"]["q"]["probabilities"]
    after = again.predict(state, questions)["answers"]["q"]["probabilities"]
    assert after == pytest.approx(before, abs=1e-6)
