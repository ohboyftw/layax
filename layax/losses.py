"""Training objectives.

``proper_reward`` is a local copy of Laya's strictly-proper scoring rule (log score +
spherical score, plus RPS on ordinal questions), kept here so layax can be trained and
tested without importing the laya package. It is deliberately identical: mixing a
different reward in would make any comparison against the upstream checkpoint
meaningless.

Added on top:

* ``in_batch_negative_loss`` -- other rows' gold options as negatives. Cheap extra
  supervision, and the reason large label sets can be learned without every label
  appearing in every batch.
* ``ordinal_cumlink_loss`` -- a cumulative-link term for ``score`` questions.
  Upstream's reward already contains RPS, which is also ordinal-aware, so treat this
  as overlapping rather than novel: it is worth an ablation, not an assumption.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

QTYPE_SCORE = 1


def proper_reward(q: torch.Tensor, target: torch.Tensor, qtype: torch.Tensor,
                  mask: torch.Tensor, w_sph: float = 0.5, w_rps: float = 1.0,
                  log_floor: float = -9.21) -> torch.Tensor:
    """Strictly proper scoring rule: log score + spherical score, minus RPS on ordinals.

    Higher is better -- it is a reward, so the loss is its negative.
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPE_SCORE).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        rps = (((torch.cumsum(q, -1) - torch.cumsum(target, -1)) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Soft-target cross entropy over the real options only."""
    logits = logits.masked_fill(~mask, -1e4)
    logp = F.log_softmax(logits, dim=-1)
    return -(target * logp * mask).sum(-1)


def ordinal_cumlink_loss(probs: torch.Tensor, label: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Cumulative-link BCE for ordinal ``score`` questions.

    For K levels there are K-1 binary questions: is the true level above k? Predicting
    level 4 when the answer is 0 should cost more than predicting 1, which a flat
    softmax over levels does not know. SST-5 at 0.372 is the symptom this targets.
    """
    B, K = probs.shape
    if K < 2:
        return probs.new_zeros(B)
    cdf = torch.cumsum(probs, dim=-1).clamp(1e-6, 1 - 1e-6)
    surv = 1.0 - cdf[:, :-1]                                   # P(Y > k), k = 0..K-2
    idx = torch.arange(K - 1, device=probs.device).unsqueeze(0)
    tgt = (label.unsqueeze(1) > idx).float()
    valid = (idx < (mask.sum(-1, keepdim=True) - 1)).float()
    bce = -(tgt * torch.log(surv) + (1 - tgt) * torch.log(1 - surv))
    return (bce * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)


def in_batch_negative_loss(state_proj: torch.Tensor, state_mask: torch.Tensor,
                           gold_opts: torch.Tensor, gold_tok_mask: torch.Tensor,
                           scale: torch.Tensor) -> torch.Tensor:
    """Contrastive term over the batch's gold options.

    Row i's state should score its own gold option above every other row's. With 77 or
    150 labels most never appear as a negative inside one batch, and this is what
    supplies the missing contrast without widening the batch.

    state_proj:    [B, L, p] normalised
    gold_opts:     [B, m, p] normalised gold option tokens for each row
    returns:       [B]
    """
    B = state_proj.size(0)
    if B < 2:
        return state_proj.new_zeros(B)
    sim = torch.einsum("jmp,blp->bjml", gold_opts, state_proj)     # [B, B, m, L]
    sim = sim.masked_fill((~state_mask).unsqueeze(1).unsqueeze(1), -1e4)
    best = sim.max(dim=-1).values                                   # [B, B, m]
    tm = gold_tok_mask.unsqueeze(0).to(best.dtype)                  # [1, B, m]
    scores = (best * tm).sum(-1) / tm.sum(-1).clamp(min=1.0)        # [B, B]
    logits = scores * scale.view(-1, 1)
    labels = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits, labels, reduction="none")


def combined_loss(logits: torch.Tensor, target: torch.Tensor, label: torch.Tensor,
                  mask: torch.Tensor, qtype: torch.Tensor, cfg,
                  negatives: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Mix the terms and return every component, so a run log shows which one moved.

    A single scalar loss that goes down tells you nothing about whether the ordinal term
    is doing anything; the components are what make the ablation readable.
    """
    logits = logits.masked_fill(~mask, -1e4)
    probs = torch.softmax(logits, dim=-1) * mask
    probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)

    ce = masked_cross_entropy(logits, target, mask)
    rl = -proper_reward(probs, target, qtype, mask.float(), cfg.w_sph, cfg.w_rps)
    total = cfg.rlcd_weight * rl + (1.0 - cfg.rlcd_weight) * ce

    out = {"ce": ce.mean().detach(), "rlcd": rl.mean().detach()}

    if cfg.ordinal_score_loss:
        is_score = (qtype == QTYPE_SCORE).float()
        if is_score.any():
            ordl = ordinal_cumlink_loss(probs, label, mask) * is_score
            total = total + cfg.ordinal_weight * ordl
            out["ordinal"] = (ordl.sum() / is_score.sum().clamp(min=1)).detach()

    if negatives is not None:
        total = total + cfg.negative_weight * negatives
        out["negatives"] = negatives.mean().detach()

    out["total"] = total.mean()
    return out
