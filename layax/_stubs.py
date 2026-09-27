"""Offline stand-ins for a tokenizer and encoder, for ``layax.cli smoke`` and the tests.

They ship with the package so the smoke check works from any install, not only a checkout.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Dict, List

import torch.nn as nn


class StubTokenizer:
    """Whitespace tokenizer with a stable hashed vocabulary."""

    vocab_size = 4096
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0
    mask_token_id = 3
    mask_token = "[MASK]"

    def __call__(self, text: str, add_special_tokens: bool = True) -> Dict[str, List[int]]:
        toks = str(text).split()
        ids = []
        for t in toks:
            h = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], 16)
            ids.append(4 + (h % (self.vocab_size - 4)))
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return {"input_ids": ids}

    def save_pretrained(self, path):  # pragma: no cover - persistence not under test
        return path


class StubEncoder(nn.Module):
    """Tiny bidirectional encoder with the transformers output surface layax uses."""

    def __init__(self, hidden_size: int = 32, vocab_size: int = 4096):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, 2, hidden_size * 2,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        h = self.emb(input_ids)
        pad = None if attention_mask is None else ~attention_mask.bool()
        h = self.enc(h, src_key_padding_mask=pad)
        # Padding positions come back as whatever the layer produced; zero them so a
        # downstream masking mistake shows up as a wrong number rather than hiding.
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)
        return SimpleNamespace(last_hidden_state=h)
