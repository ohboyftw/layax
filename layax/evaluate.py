"""Evaluation: full-label inference, metrics, per-group breakdowns, latency.

Every number reported here is measured in this process on the data you pass. Nothing
in this package ships a claimed benchmark result, and comparison figures published
elsewhere (including upstream's) are not side-by-side measurements -- so do not quote
them as if this harness produced them.

Reported per run:

* accuracy, macro-F1
* ECE and Brier (calibration, as upstream reports)
* AURC and selective accuracy at fixed coverage (whether confidence can gate)
* a breakdown by any ``meta`` key, so "which languages collapsed" is one call
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .competence import aurc, expected_calibration_error, selective_accuracy_at
from .config import LIConfig
from .data import Example
from .li_head import (
    QTYPES,
    build_option_sequence,
    build_state_sequence,
    option_labels,
    pad_stack,
    render_options,
    serialize_state,
)


@torch.no_grad()
def collect_predictions(model, tok, rows: Sequence[Example], cfg: LIConfig,
                        device: Any, batch_size: int = 16,
                        option_cache: Optional[Dict[str, Any]] = None,
                        progress_every: int = 0) -> Dict[str, Any]:
    """Score every row against its FULL option set.

    Training subsamples options; evaluation must not, or the headline number is a
    measurement of an easier task than the one being shipped.

    Returns predictions plus the feature rows the competence head consumes, so a single
    pass serves both.
    """
    model.eval()
    dev = torch.device(device)
    cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {} if option_cache is None else option_cache

    preds, correct, feature_rows, metas, golds = [], [], [], [], []
    n_options_all: List[int] = []
    t0 = time.perf_counter()

    for start in range(0, len(rows), batch_size):
        chunk = rows[start: start + batch_size]
        state_seqs, truncs, qts, opt_texts = [], [], [], []
        for ex in chunk:
            seq, trunc = build_state_sequence(tok, ex.state, ex.qtype, ex.instructions,
                                              cfg.state_max_len)
            state_seqs.append(seq)
            truncs.append(trunc)
            qts.append(QTYPES[ex.qtype])
            opt_texts.append(render_options(ex.qtype, ex.criteria))

        # Encode any option text not already cached. Option sets repeat heavily across
        # rows of one dataset, so this collapses to near zero after the first batch.
        todo = [t for t in dict.fromkeys(sum(opt_texts, [])) if t not in cache]
        for i in range(0, len(todo), 128):
            sub = todo[i: i + 128]
            oid, oatt = pad_stack([build_option_sequence(tok, t, cfg.option_max_len) for t in sub],
                                  tok.pad_token_id)
            emb, mask = model.encode_options(oid.to(dev), oatt.to(dev))
            for j, t in enumerate(sub):
                cache[t] = (emb[j].detach().cpu(), mask[j].detach().cpu())

        ids, att = pad_stack(state_seqs, tok.pad_token_id)
        K = max(len(o) for o in opt_texts)
        m = max(cache[t][0].shape[0] for t in dict.fromkeys(sum(opt_texts, [])))
        d = cache[opt_texts[0][0]][0].shape[1]
        opts = torch.zeros((len(chunk), K, m, d), dtype=torch.float32)
        otm = torch.zeros((len(chunk), K, m), dtype=torch.bool)
        om = torch.zeros((len(chunk), K), dtype=torch.bool)
        for i, texts in enumerate(opt_texts):
            for j, t in enumerate(texts):
                e, mk = cache[t]
                opts[i, j, : e.shape[0]] = e.float()
                otm[i, j, : mk.shape[0]] = mk
            om[i, : len(texts)] = True

        logits, pooled, stats = model(
            ids.to(dev), att.to(dev), None, otm.to(dev), om.to(dev),
            torch.tensor(qts, dtype=torch.long).to(dev),
            option_embeddings=opts.to(dev))

        lg = logits.float().cpu().numpy()
        pl = pooled.float().cpu().numpy()
        omn = om.numpy()

        for i, ex in enumerate(chunk):
            k = int(omn[i].sum())
            pred = int(np.argmax(lg[i, :k]))
            preds.append(pred)
            golds.append(ex.label)
            correct.append(int(pred == ex.label))
            metas.append(ex.meta)
            n_options_all.append(k)
            feature_rows.append({
                "pooled": pl[i], "logits": lg[i], "option_mask": omn[i],
                "max_sim": float(stats["max_sim"][i]), "mean_sim": float(stats["mean_sim"][i]),
                "std_sim": float(stats["std_sim"][i]),
                "n_state_tokens": int(att[i].sum()), "n_truncated": truncs[i],
                "n_options": k, "state_text": serialize_state(ex.state),
            })

        if progress_every and (start // batch_size) % progress_every == 0:
            print("[layax] scored %d/%d rows" % (min(start + batch_size, len(rows)), len(rows)),
                  flush=True)

    return {"pred": np.array(preds), "gold": np.array(golds),
            "correct": np.array(correct, dtype=np.float64),
            "features": feature_rows, "meta": metas,
            "n_options": np.array(n_options_all),
            "seconds": round(time.perf_counter() - t0, 3),
            "option_cache": cache}


def softmax_rows(features: Sequence[Dict[str, Any]],
                 temperatures: Optional[Dict[str, float]] = None) -> List[np.ndarray]:
    """Per-row probability vectors over real options, optionally temperature-scaled."""
    from .runtime import temp_bucket

    out = []
    for r in features:
        mask = np.asarray(r["option_mask"], dtype=bool)
        k = int(mask.sum())
        z = np.asarray(r["logits"], dtype=np.float64)[:k]
        if temperatures:
            t = temperatures.get(temp_bucket(0, k), 1.0)
            z = z / max(t, 1e-6)
        p = np.exp(z - z.max())
        out.append(p / p.sum())
    return out


def macro_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    labels = np.unique(gold)
    f1s = []
    for c in labels:
        tp = float(((pred == c) & (gold == c)).sum())
        fp = float(((pred == c) & (gold != c)).sum())
        fn = float(((pred != c) & (gold == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def brier_score(probs: Sequence[np.ndarray], gold: np.ndarray) -> float:
    """Multi-class Brier, averaged per row so ragged option counts stay comparable."""
    vals = []
    for p, g in zip(probs, gold):
        t = np.zeros_like(p)
        if 0 <= g < len(t):
            t[g] = 1.0
        vals.append(float(((p - t) ** 2).sum()))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_predictions(res: Dict[str, Any], temperatures: Optional[Dict[str, float]] = None,
                         group_by: Optional[str] = None,
                         competence: Optional[np.ndarray] = None,
                         competence_rank: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Headline metrics, plus an optional breakdown by a ``meta`` key.

    ``competence`` is the calibrated probability (ECE, per-group means).
    ``competence_rank`` is the raw head score the gate uses; when given it drives AURC
    and selective accuracy, because isotonic calibration merges scores into ties and
    would otherwise change the ranking being measured.
    """
    probs = softmax_rows(res["features"], temperatures)
    conf = np.array([float(p.max()) for p in probs])
    correct = res["correct"]

    out: Dict[str, Any] = {
        "n": int(len(correct)),
        "accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "macro_f1": macro_f1(res["pred"], res["gold"]),
        "ece_softmax": expected_calibration_error(conf, correct),
        "brier": brier_score(probs, res["gold"]),
        "mean_confidence": float(conf.mean()) if len(conf) else float("nan"),
        "aurc_softmax": aurc(conf, correct),
        **{k + "_softmax": v for k, v in selective_accuracy_at(conf, correct).items()},
        "mean_options": float(res["n_options"].mean()) if len(res["n_options"]) else 0.0,
        "seconds": res.get("seconds"),
    }

    # The gap between these two is the entire argument for the competence head: if the
    # softmax already ranked errors well, a learned head buys nothing.
    if competence is not None:
        rank = competence if competence_rank is None else competence_rank
        out["aurc_competence"] = aurc(rank, correct)
        out["ece_competence"] = expected_calibration_error(competence, correct)
        out.update({k + "_competence": v for k, v in selective_accuracy_at(rank, correct).items()})
        out["aurc_improvement"] = out["aurc_softmax"] - out["aurc_competence"]

    if group_by:
        groups: Dict[str, List[int]] = {}
        for i, m in enumerate(res["meta"]):
            groups.setdefault(str(m.get(group_by, "unknown")), []).append(i)
        per = {}
        for g, idx in sorted(groups.items()):
            ii = np.array(idx)
            entry = {"n": len(idx), "accuracy": float(correct[ii].mean()),
                     "mean_confidence": float(conf[ii].mean()),
                     "ece": expected_calibration_error(conf[ii], correct[ii])}
            if competence is not None:
                entry["mean_competence"] = float(np.asarray(competence)[ii].mean())
            per[g] = entry
        out["by_" + group_by] = per
        # The Khmer signature: near-zero accuracy at high confidence. Worth surfacing by
        # name rather than leaving for someone to spot in a table of 51 rows.
        out["confidently_wrong_groups"] = sorted(
            [g for g, e in per.items()
             if e["n"] >= 20 and e["accuracy"] < 0.15 and e["mean_confidence"] > 0.6])

    return out


@torch.no_grad()
def quick_accuracy(model, tok, rows: Sequence[Example], cfg: LIConfig, device: Any,
                   max_rows: int = 512) -> float:
    """Cheap in-training check. Not the reported number."""
    sub = list(rows)[:max_rows]
    res = collect_predictions(model, tok, sub, cfg, device, batch_size=16)
    return float(res["correct"].mean()) if len(res["correct"]) else 0.0


@torch.no_grad()
def latency_benchmark(agent, questions: Dict[str, Dict[str, Any]], state: Any,
                      repeats: int = 50, warmup: int = 5,
                      warm_cache: bool = True) -> Dict[str, Any]:
    """p50/p95 per request, cold and warm cache.

    Report both. The warm number is the production path; the cold number is what a
    process sees on its first request after a deploy, and it is the one that surprises
    people -- upstream measures 7-10 s on a language switch at default settings.
    """
    out: Dict[str, Any] = {"n_questions": len(questions),
                           "n_options": sum(len(render_options(q["type"], q.get("criteria")))
                                            for q in questions.values())}
    agent.cache.clear()
    t0 = time.perf_counter()
    agent.predict(state, questions)
    out["cold_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if warm_cache:
        agent.warm_cache(questions)
    for _ in range(warmup):
        agent.predict(state, questions)

    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        agent.predict(state, questions)
        times.append((time.perf_counter() - t) * 1000)
    arr = np.array(times)
    out.update({"p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "mean_ms": round(float(arr.mean()), 2),
                "ms_per_question": round(float(arr.mean() / max(1, len(questions))), 3),
                "repeats": repeats, "cache": agent.cache.stats()})
    return out


# --------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------

# Du et al. (AAAI 2023, arXiv 2212.00301) found an inline-options model peaked at k=25 on
# a 77-intent shortlist, so a single k=20 arm can understate the shortlist baseline.
SHORTLIST_KS = (10, 20, 25, 40)
LAYA_BASELINE_ARMS: Tuple[Dict[str, int], ...] = (
    ({},) + tuple({"shortlist_k": k} for k in SHORTLIST_KS) + ({"head_max_len": 512},))


def shortlist_hit(result: Dict[str, Any], qid: str, gold_label: str) -> bool:
    """Was the gold label among the options ``laya.predict_shortlist`` kept?

    Reads the ``shortlist`` block predict_shortlist returns, so it measures the
    retrieval that actually ran. When the shortlist cannot hold the gold, no downstream
    model can recover it, which is why recall@k is reported next to accuracy.
    """
    meta = result["shortlist"][qid]
    return bool(meta["passthrough"]) or gold_label in meta["labels"]


def evaluate_laya_baseline(rows: Sequence[Example], checkpoint: str = "convaiinnovations/laya",
                           subfolder: Optional[str] = "typed-decisions",
                           device: Optional[str] = None,
                           shortlist_k: Optional[int] = None,
                           head_max_len: Optional[int] = None) -> Dict[str, Any]:
    """Upstream Laya on the same rows -- the number layax has to beat.

    ``shortlist_k`` switches on ``laya.shortlist``, the API-level workaround, using the
    checkpoint's own encoder as the embedding function. That is the fair comparison:
    an honest result compares against upstream's best available answer for wide label
    sets, not against the configuration it is known to fail on.

    ``head_max_len`` raises the option budget at runtime, which the upstream README
    suggests for 50+ options. Worth including as a third arm.
    """
    import laya

    agent = laya.load(checkpoint, device=device, subfolder=subfolder)
    if head_max_len:
        agent.cfg["head_max_len"] = int(head_max_len)
        agent.cfg["max_len"] = max(agent.cfg.get("max_len", 512), int(head_max_len) * 2)

    embed_fn = None
    if shortlist_k:
        embed_fn = laya.embed_fn_from_agent(agent)

    correct, confs, hits, errors, t0 = [], [], [], 0, time.perf_counter()
    for ex in rows:
        qs = {"q": ex.as_question()}
        labels = option_labels(ex.qtype, ex.criteria)
        try:
            if shortlist_k:
                res = laya.predict_shortlist(agent, ex.state, qs, embed_fn, k=shortlist_k)
            else:
                res = agent.predict(ex.state, qs)
        except Exception:
            # Upstream raises when options overflow head_max_len. That is a real result
            # for this configuration, not something to hide behind a try/except -- it is
            # counted and reported.
            errors += 1
            correct.append(0.0)
            confs.append(0.0)
            continue
        if shortlist_k and ex.qtype == "choice":
            hits.append(float(shortlist_hit(res, "q", labels[ex.label])))
        a = res["answers"]["q"]
        if a["type"] == "choice":
            pred = labels.index(a["choice"]) if a["choice"] in labels else -1
        elif a["type"] == "score":
            pred = int(round(a["score"]))
        else:
            pred = int(a["noul"] >= 0.5)
        correct.append(float(pred == ex.label))
        confs.append(float(a.get("confidence", 0.0)))

    c = np.array(correct)
    conf = np.array(confs)
    out = {
        "arm": "laya" + (("+shortlist%d" % shortlist_k) if shortlist_k else "")
               + (("+head%d" % head_max_len) if head_max_len else ""),
        "n": len(rows), "accuracy": float(c.mean()) if len(c) else float("nan"),
        "coverage": 1.0, "ece": expected_calibration_error(conf, c), "aurc": aurc(conf, c),
        "errors": errors, "seconds": round(time.perf_counter() - t0, 2),
    }
    if shortlist_k:
        # Over rows where the shortlist ran; rows that raised are counted in ``errors``.
        out["recall_at_k"] = float(np.mean(hits)) if hits else float("nan")
    return out


COMPARE_KEYS = ["arm", "n", "accuracy", "coverage", "selective_accuracy", "recall_at_k",
                "ece", "aurc", "errors", "seconds"]


def compare(arms: Sequence[Dict[str, Any]]) -> str:
    """A small text table. The point is that arms sit next to each other in one run.

    ``accuracy`` is always over every row the arm was given. A gated arm's accuracy on
    the rows it answered goes in ``selective_accuracy``, next to its ``coverage``, and
    never in the accuracy column.
    """
    def cell(v: Any) -> str:
        if v is None:
            return ""
        return "%.4f" % v if isinstance(v, float) else "%s" % v

    width = {k: max(len(k), *(len(cell(a.get(k))) for a in arms)) for k in COMPARE_KEYS}
    lines = [" | ".join(k.ljust(width[k]) for k in COMPARE_KEYS),
             "-+-".join("-" * width[k] for k in COMPARE_KEYS)]
    for a in arms:
        lines.append(" | ".join(cell(a.get(k)).ljust(width[k]) for k in COMPARE_KEYS))
    return "\n".join(lines)
