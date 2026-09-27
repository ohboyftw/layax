"""Late-interaction option heads for Laya decision models.

Why this exists
---------------
Upstream Laya builds one sequence per question::

    [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

and reads one logit off the hidden state at each ``[MASK]`` marker. Every option
therefore competes with every other option, and with the state, for a fixed
``head_max_len`` budget (192 tokens on the English checkpoint, 256 on the others).
At 77 labels that is ``(256 - 16) // 77`` which is about 3 tokens per label, and
measured accuracy falls to 0.453 on Banking77 (best upstream arm, same test rows).

``laya.shortlist`` works around this at the API level by retrieving a top-k subset
with a caller-supplied embedding function first. Its own docstring is explicit that
it "does not change ``DecisionModel.forward``". This module is the model-layer fix
that it deliberately leaves alone, and ``shortlist`` remains the baseline to beat.

How
---
Two passes, one shared encoder:

1. **State pass** -- ``[CLS] <type> question: instructions [SEP] state [SEP]``.
   No options inline, so the state gets the whole context window.
2. **Option pass** -- each option encoded on its own, once. The result is cacheable
   per schema, because option text changes far less often than input state does.

Scoring is then an interaction between the two token sets:

* ``maxsim``  -- ColBERT-style: every option token takes its best match over the
  state tokens, and those are averaged. Pure matmul, no extra encoder work, and the
  option side is entirely precomputable.
* ``xattn``   -- one cross-attention layer where option tokens attend to state
  tokens, then the scorer MLP. More expressive, still caches the option side.
* ``meanpool`` -- cosine between the mean-pooled state and option vectors: a plain
  bi-encoder with no token interaction. It is the floor the other two must beat.

Prior art, stated so nobody mistakes this for new: FastFit (Yehudai & Bendel, NAACL
2024, arXiv 2404.12365) already scores intents by ColBERT-style MaxSim over label-name
tokens on Banking77, CLINC150 and HWU64, and the ``xattn`` head sits in the
Poly-encoder family (Humeau et al., ICLR 2020, arXiv 1905.01969).

Consequences worth stating plainly:

* Option count stops consuming state tokens, so high-cardinality label sets are no
  longer truncated to noise.
* Per-request latency with a warm cache is one state pass plus a matmul, so it is
  roughly flat in the number of options rather than growing with them.
* Schemas become reusable artifacts (embed once, reuse across every request).

What this does NOT claim: late interaction is a known retrieval technique. The new
part is applying it to Laya's RLCD-trained typed decision heads. Numbers must be
measured before they are quoted -- nothing here is a benchmarked result.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LIConfig

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


# --------------------------------------------------------------------------------------
# sequence construction
# --------------------------------------------------------------------------------------

def serialize_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(qtype: str, criteria: Any) -> List[str]:
    """Option texts in label order. Mirrors laya.common.render_options.

    Kept as a local copy so layax can be imported and unit-tested without the laya
    package (and therefore without a network round trip) installed.
    """
    if qtype == "choice":
        if isinstance(criteria, (list, tuple)):
            return [str(c) for c in criteria]
        return [
            k if v is None or v == "" else "%s: %s" % (k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            for k, v in criteria.items()
        ]
    if qtype == "score":
        return ["level %d: %s" % (i, c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
                for i, c in enumerate(criteria)]
    crit = criteria or {}
    f, t = crit.get("false"), crit.get("true")
    return [
        "false: " + (f if f not in (None, "") else "no, the statement does not hold"),
        "true: " + (t if t not in (None, "") else "yes, the statement holds"),
    ]


def option_labels(qtype: str, criteria: Any) -> List[str]:
    if qtype == "choice":
        return [str(c) for c in (criteria if isinstance(criteria, (list, tuple)) else criteria.keys())]
    if qtype == "score":
        return [str(i) for i in range(len(criteria))]
    return ["false", "true"]


def build_state_sequence(tok, state: Any, qtype: str, instructions: str, max_len: int,
                         truncate_left: bool = False) -> List[int]:
    """``[CLS] <type> question: instructions [SEP] state [SEP]`` -- options are not here.

    The whole window minus the instruction belongs to the state, which is the point:
    upstream leaves ~320 usable state tokens on the English checkpoint after the option
    block, this leaves the full window.
    """
    mask_tok = tok.mask_token or ""
    ins = str(instructions).replace(mask_tok, " ") if mask_tok else str(instructions)
    head = tok("%s question: %s" % (qtype, ins), add_special_tokens=False)["input_ids"]
    # Never let a pathological instruction eat the whole window.
    head = head[: max(8, max_len // 3)]
    ids = [tok.cls_token_id] + head + [tok.sep_token_id]
    room = max(0, max_len - len(ids) - 1)
    text = serialize_state(state)
    if mask_tok:
        text = text.replace(mask_tok, " ")
    st = tok(text, add_special_tokens=False)["input_ids"]
    truncated = max(0, len(st) - room)
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], truncated


def build_option_sequence(tok, option_text: str, max_len: int) -> List[int]:
    """``[CLS] option text [SEP]`` -- one short sequence per option, encoded once."""
    mask_tok = tok.mask_token or ""
    txt = option_text.replace(mask_tok, " ") if mask_tok else option_text
    body = tok(txt, add_special_tokens=False)["input_ids"][: max(1, max_len - 2)]
    return [tok.cls_token_id] + body + [tok.sep_token_id]


def pad_stack(seqs: Sequence[Sequence[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(seqs)
    L = max((len(s) for s in seqs), default=1)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(list(s), dtype=torch.long)
        att[i, : len(s)] = 1
    return ids, att


# --------------------------------------------------------------------------------------
# option cache
# --------------------------------------------------------------------------------------

class OptionCache:
    """Per-schema cache of encoded option tokens.

    Keyed by the option text plus everything that would change its embedding: the
    checkpoint revision, the interaction mode and the projection width. A stale key
    silently serving old vectors is the one bug that would be invisible at eval time,
    so the key is explicit rather than clever.
    """

    def __init__(self, revision: str, interaction: str, proj_dim: int, max_entries: int = 200_000):
        self.revision = revision
        self.interaction = interaction
        self.proj_dim = proj_dim
        self.max_entries = max_entries
        self._store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0

    def key(self, option_text: str) -> str:
        h = hashlib.sha1()
        h.update(("%s|%s|%d|" % (self.revision, self.interaction, self.proj_dim)).encode("utf-8"))
        h.update(option_text.encode("utf-8"))
        return h.hexdigest()

    def get(self, option_text: str):
        k = self.key(option_text)
        v = self._store.get(k)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, option_text: str, emb: torch.Tensor, mask: torch.Tensor) -> None:
        if len(self._store) >= self.max_entries:
            self._store.pop(next(iter(self._store)))
        self._store[self.key(option_text)] = (emb.detach().to(torch.float16).cpu(),
                                              mask.detach().cpu())

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else None}

    def save(self, path: str) -> None:
        torch.save({"revision": self.revision, "interaction": self.interaction,
                    "proj_dim": self.proj_dim, "store": self._store}, path)

    @classmethod
    def load(cls, path: str) -> "OptionCache":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        c = cls(blob["revision"], blob["interaction"], blob["proj_dim"])
        c._store = blob["store"]
        return c


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

class LateInteractionDecisionModel(nn.Module):
    """Laya's encoder with the option block moved out of the sequence.

    Weight reuse from a Laya checkpoint: ``encoder.*`` and ``type_emb.*`` load directly,
    and ``scorer.*`` initialises the xattn scorer. The projections are new and start
    from scratch -- expect the first few hundred steps to be recovering what the marker
    head already knew.
    """

    def __init__(self, encoder: nn.Module, cfg: LIConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.encoder = encoder
        d = int(encoder.config.hidden_size)
        self.hidden_size = d
        p = cfg.proj_dim

        self.type_emb = nn.Embedding(3, d)
        self.proj_state = nn.Linear(d, p, bias=False)
        self.proj_opt = nn.Linear(d, p, bias=False)
        self.opt_norm = nn.LayerNorm(d)

        if cfg.interaction == "xattn":
            self.xattn = nn.MultiheadAttention(d, cfg.xattn_heads, dropout=cfg.xattn_dropout,
                                               batch_first=True)
            self.xattn_norm = nn.LayerNorm(d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        else:
            self.xattn = None
            self.scorer = None

        # Per-question-type affine on the interaction score. maxsim lives in [-1, 1];
        # without a scale the softmax is almost uniform and nothing trains.
        self.logit_scale = nn.Parameter(torch.full((3,), float(cfg.init_scale)),
                                        requires_grad=bool(cfg.learn_scale))
        self.logit_bias = nn.Parameter(torch.zeros(3), requires_grad=bool(cfg.learn_scale))

        self.register_buffer("temperature", torch.ones(3))
        self._option_cache: Optional[OptionCache] = None

    # -- encoding -----------------------------------------------------------------

    def encode_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     qtype: torch.Tensor):
        """-> (projected [B, L, p], hidden [B, L, d], pooled CLS [B, d], mask [B, L]).

        Both widths come back on purpose. ``maxsim`` and the interaction statistics want
        the normalised projection; the cross-attention layer needs the full hidden width,
        since its option queries live there too.
        """
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        pooled = h[:, 0]
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        z = F.normalize(self.proj_state(h), dim=-1) * mask
        return z, h * mask, pooled, attention_mask.bool()

    def encode_options(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """-> (option token embeddings [N, m, *], mask [N, m]).

        maxsim and meanpool return projected+normalised vectors; xattn returns hidden-size vectors,
        because the cross-attention layer needs the full width. Both are cacheable.
        """
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.opt_norm(h)
        mask = attention_mask.bool()
        if self.cfg.interaction == "xattn":
            return h * mask.unsqueeze(-1).to(h.dtype), mask
        z = F.normalize(self.proj_opt(h), dim=-1)
        return z * mask.unsqueeze(-1).to(z.dtype), mask

    # -- interaction --------------------------------------------------------------

    def _maxsim(self, state: torch.Tensor, state_mask: torch.Tensor,
                opts: torch.Tensor, opt_tok_mask: torch.Tensor,
                chunk: int = 256) -> torch.Tensor:
        """MaxSim score per option.

        state:        [B, L, p] (L2-normalised, padding zeroed)
        opts:         [B, K, m, p]
        returns:      [B, K] mean over option tokens of the best state match, in [-1, 1].

        Length-normalised rather than ColBERT's plain sum: option texts here vary from
        one word to a full rubric sentence, and a sum would hand long options a free
        advantage that has nothing to do with whether they are right.
        """
        B, K, m, p = opts.shape
        out = state.new_zeros((B, K))
        sm = (~state_mask).unsqueeze(1).unsqueeze(1)          # [B, 1, 1, L] True where padding
        for lo in range(0, K, chunk):
            hi = min(K, lo + chunk)
            o = opts[:, lo:hi]                                 # [B, C, m, p]
            sim = torch.einsum("bcmp,blp->bcml", o, state)     # [B, C, m, L]
            sim = sim.masked_fill(sm, -1e4)
            best = sim.max(dim=-1).values                      # [B, C, m]
            tm = opt_tok_mask[:, lo:hi].to(best.dtype)         # [B, C, m]
            denom = tm.sum(-1).clamp(min=1.0)
            out[:, lo:hi] = (best * tm).sum(-1) / denom
        return out

    def _meanpool_score(self, state: torch.Tensor, state_mask: torch.Tensor,
                        opts: torch.Tensor, opt_tok_mask: torch.Tensor) -> torch.Tensor:
        """Cosine of mean-pooled, re-normalised vectors: [B, L, p] x [B, K, m, p] -> [B, K].

        Pools the same projected token vectors maxsim uses, so the option cache is shared
        in shape and the ablation changes only the interaction.
        """
        sm = state_mask.unsqueeze(-1).to(state.dtype)
        s = F.normalize((state * sm).sum(1) / sm.sum(1).clamp(min=1.0), dim=-1)
        tm = opt_tok_mask.unsqueeze(-1).to(opts.dtype)
        o = F.normalize((opts * tm).sum(2) / tm.sum(2).clamp(min=1.0), dim=-1)
        return torch.einsum("bkp,bp->bk", o, s.to(o.dtype))

    def _xattn_score(self, state_h: torch.Tensor, state_mask: torch.Tensor,
                     opts: torch.Tensor, opt_tok_mask: torch.Tensor) -> torch.Tensor:
        """One cross-attention layer: option tokens query the state, then score.

        Flattened to [B*K, m, d] so every option attends independently; memory is the
        reason ``chunk_options`` exists on the caller side for very wide label sets.
        """
        B, K, m, d = opts.shape
        L = state_h.size(1)
        q = opts.reshape(B * K, m, d)
        kv = state_h.unsqueeze(1).expand(B, K, L, d).reshape(B * K, L, d)
        pad = (~state_mask).unsqueeze(1).expand(B, K, L).reshape(B * K, L)
        # A row whose state is entirely padding would make attention produce NaN.
        pad = pad & ~pad.all(dim=-1, keepdim=True)
        a, _ = self.xattn(q, kv, kv, key_padding_mask=pad, need_weights=False)
        h = self.xattn_norm(q + a)
        tm = opt_tok_mask.reshape(B * K, m, 1).to(h.dtype)
        pooled = (h * tm).sum(1) / tm.sum(1).clamp(min=1.0)
        return self.scorer(pooled).squeeze(-1).reshape(B, K)

    def score(self, state_proj: torch.Tensor, state_hidden: torch.Tensor,
              state_mask: torch.Tensor, opts: torch.Tensor, opt_tok_mask: torch.Tensor,
              option_mask: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
        """-> logits [B, K], padded option slots pushed to -1e4."""
        if self.cfg.interaction == "xattn":
            raw = self._xattn_score(state_hidden, state_mask, opts, opt_tok_mask)
        elif self.cfg.interaction == "meanpool":
            raw = self._meanpool_score(state_proj, state_mask, opts, opt_tok_mask)
        else:
            raw = self._maxsim(state_proj, state_mask, opts, opt_tok_mask)
        scale = self.logit_scale[qtype].unsqueeze(-1)
        bias = self.logit_bias[qtype].unsqueeze(-1)
        logits = raw.float() * scale.float() + bias.float()
        return logits.masked_fill(~option_mask, -1e4)

    def forward(self, state_ids, state_mask, opt_ids, opt_mask, option_mask, qtype,
                option_embeddings: Optional[torch.Tensor] = None,
                detach_encoder: bool = False):
        """Full pass.

        ``option_embeddings`` short-circuits the option tower with cached vectors
        ([B, K, m, *]); ``opt_ids``/``opt_mask`` are then ignored. That is the
        production path and the reason latency stays flat in the option count.

        Returns (logits [B, K], pooled CLS [B, d], interaction stats dict).
        """
        st_proj, st_hidden, pooled, st_mask = self.encode_state(state_ids, state_mask, qtype)
        if detach_encoder:
            st_proj, st_hidden, pooled = st_proj.detach(), st_hidden.detach(), pooled.detach()

        if option_embeddings is not None:
            opts = option_embeddings.to(st_proj.dtype)
            opt_tok_mask = opt_mask.bool()
        else:
            B, K, m = opt_ids.shape
            flat_ids = opt_ids.reshape(B * K, m)
            flat_att = opt_mask.reshape(B * K, m)
            # A fully padded slot is a real possibility (ragged option counts) and would
            # otherwise be a division by zero inside the encoder's attention.
            empty = flat_att.sum(-1) == 0
            if empty.any():
                flat_att = flat_att.clone()
                flat_att[empty, 0] = 1
            oh, om = self.encode_options(flat_ids, flat_att)
            opts = oh.reshape(B, K, m, -1)
            opt_tok_mask = om.reshape(B, K, m)
            if empty.any():
                opt_tok_mask = opt_tok_mask.clone().reshape(B * K, m)
                opt_tok_mask[empty] = False
                opt_tok_mask = opt_tok_mask.reshape(B, K, m)

        logits = self.score(st_proj, st_hidden, st_mask, opts, opt_tok_mask,
                            option_mask.bool(), qtype)
        stats = self._interaction_stats(st_proj, st_mask, opts, opt_tok_mask, option_mask.bool())
        return logits, pooled, stats

    @torch.no_grad()
    def _interaction_stats(self, state_tokens, state_mask, opts, opt_tok_mask, option_mask):
        """Cheap signals the competence head consumes.

        ``max_sim`` near zero means no option matched the state anywhere -- the shape of
        an out-of-domain or unreadable input, and exactly the case where the softmax
        alone still looks confident.
        """
        if self.cfg.interaction == "xattn":
            # Option vectors are hidden-size here; project so the statistic means the
            # same thing in both modes.
            o = F.normalize(self.proj_opt(opts), dim=-1)
        else:
            o = opts
        sim = torch.einsum("bkmp,blp->bkml", o, state_tokens)
        sim = sim.masked_fill((~state_mask).unsqueeze(1).unsqueeze(1), -1e4)
        best = sim.max(dim=-1).values                              # [B, K, m]
        tm = opt_tok_mask.to(best.dtype)
        per_opt = (best * tm).sum(-1) / tm.sum(-1).clamp(min=1.0)  # [B, K]
        om = option_mask.to(per_opt.dtype)
        n = om.sum(-1).clamp(min=1.0)
        mean = (per_opt * om).sum(-1) / n
        mx = per_opt.masked_fill(~option_mask, -1e4).max(-1).values
        var = ((per_opt - mean.unsqueeze(-1)) ** 2 * om).sum(-1) / n
        return {"max_sim": mx.float(), "mean_sim": mean.float(), "std_sim": var.clamp_min(0).sqrt().float()}

    # -- loading ------------------------------------------------------------------

    @classmethod
    def from_laya(cls, cfg: LIConfig, checkpoint_dir: Optional[str] = None,
                  encoder: Optional[nn.Module] = None, strict_report: bool = True
                  ) -> "LateInteractionDecisionModel":
        """Build from a downloaded Laya checkpoint directory.

        Pass ``encoder`` directly to build offline (tests do this with a stub), in which
        case no weights are copied.
        """
        if encoder is None:
            if checkpoint_dir is None:
                raise ValueError("pass checkpoint_dir or an encoder module")
            encoder = _load_encoder(checkpoint_dir)
        model = cls(encoder, cfg)
        if checkpoint_dir is not None:
            report = model.load_laya_weights(checkpoint_dir)
            if strict_report:
                print("[layax] initialised from Laya checkpoint: %s" % json.dumps(report))
        return model

    def load_laya_weights(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Copy ``encoder.*``, ``type_emb.*`` and (xattn only) ``scorer.*``.

        Returns a report rather than printing, so a training run can log exactly which
        tensors were reused and which started from scratch. Silent partial loads are
        how a run quietly trains a randomly initialised encoder for four hours.
        """
        from safetensors.torch import load_file

        path = os.path.join(checkpoint_dir, "model.safetensors")
        if not os.path.exists(path):
            raise FileNotFoundError("no model.safetensors in %s" % checkpoint_dir)
        src = load_file(path)
        own = self.state_dict()
        copied, skipped = [], []
        for k, v in src.items():
            if k in own and tuple(own[k].shape) == tuple(v.shape):
                own[k] = v
                copied.append(k)
            else:
                skipped.append(k)
        self.load_state_dict(own, strict=True)
        fresh = sorted({n.split(".")[0] for n in own if n not in set(copied)})
        return {"copied_tensors": len(copied), "skipped_tensors": len(skipped),
                "randomly_initialised_modules": fresh}


def _load_encoder(checkpoint_dir: str) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    enc_dir = os.path.join(checkpoint_dir, "encoder")
    if os.path.isdir(enc_dir):
        ecfg = AutoConfig.from_pretrained(enc_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        with open(os.path.join(checkpoint_dir, "rl_agent_config.json")) as f:
            cfg = json.load(f)
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    try:
        enc.config.reference_compile = False        # ModernBERT compiles by default; a loss here
    except Exception:
        pass
    return enc


def download_laya(repo: str = "convaiinnovations/laya", subfolder: Optional[str] = "typed-decisions",
                  token: Optional[str] = None) -> str:
    """Snapshot just the files a checkpoint needs, and return its directory."""
    from huggingface_hub import snapshot_download

    prefix = "%s/" % subfolder if subfolder else ""
    d = snapshot_download(repo, token=token or os.environ.get("HF_TOKEN"),
                          allow_patterns=[prefix + n for n in
                                          ("rl_agent_config.json", "model.safetensors",
                                           "tokenizer/*", "encoder/*")])
    return os.path.join(d, subfolder) if subfolder else d
