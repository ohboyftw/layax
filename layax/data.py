"""Datasets, splits, and the shift augmentation the competence head trains against.

Registry
--------
``banking77``   77 intents -- the case upstream loses on (0.453 on the same test rows).
``massive``     60 intents across 51 languages -- the cross-lingual shift source.
``clinc150``    151 intents including out-of-scope -- a second high-cardinality check.
``jsonl``       your own rows, for example an export of support requests: one JSON object per line,
                no schema negotiation.

Every loader yields the same ``Example``, so a new dataset is one function.

Splitting
---------
Four splits, not two, and the reason matters. The competence head must be fitted on
rows the decision model never saw, or it learns the model's training-set optimism
rather than its real error rate. The Learn-then-Test threshold then needs a third split
it was not fitted on, or the risk bound is a statement about data already used twice.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple



@dataclass
class Example:
    """One typed decision.

    ``state`` is whatever the model reads (text or a JSON-able dict).
    ``criteria`` is the full option set for this question, in label order.
    ``label`` indexes into it. ``meta`` carries anything the eval wants to group by,
    for example ``{"language": "km", "shift": "foreign"}``.
    """
    state: Any
    qtype: str
    instructions: str
    criteria: Any
    label: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_question(self) -> Dict[str, Any]:
        return {"type": self.qtype, "instructions": self.instructions, "criteria": self.criteria}


@dataclass
class Splits:
    train: List[Example]
    competence: List[Example]
    calibration: List[Example]
    test: List[Example]

    def summary(self) -> Dict[str, int]:
        return {k: len(getattr(self, k)) for k in ("train", "competence", "calibration", "test")}


def split_examples(rows: Sequence[Example], competence_frac: float, calibration_frac: float,
                   test: Optional[Sequence[Example]] = None, seed: int = 17) -> Splits:
    """Carve train / competence / calibration out of the training rows.

    Shuffled with a fixed seed and nothing else: any grouping structure (the same
    request appearing twice, say) has to be handled by the caller, because only the
    caller knows what a duplicate means in its data.
    """
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_comp = int(n * competence_frac)
    n_cal = int(n * calibration_frac)
    comp = rows[:n_comp]
    cal = rows[n_comp: n_comp + n_cal]
    train = rows[n_comp + n_cal:]
    if test is None:
        # No dedicated test split: take one more slice off train rather than
        # silently evaluating on data the model trained on.
        n_test = max(1, int(len(train) * 0.15))
        test_rows, train = train[:n_test], train[n_test:]
    else:
        test_rows = list(test)
    return Splits(train, comp, cal, test_rows)


def splits_from_meta(rows: Sequence[Example], key: str) -> Splits:
    """Partition rows by ``meta[key]``; rows whose value is not a split name are unused."""
    by = {name: [ex for ex in rows if ex.meta.get(key) == name]
          for name in ("train", "competence", "calibration", "test")}
    empty = [name for name, part in by.items() if not part]
    if empty:
        raise ValueError("no rows with meta[%r] in %s" % (key, empty))
    return Splits(by["train"], by["competence"], by["calibration"], by["test"])


def hold_out_labels(splits: Splits, n: int, seed: int = 17,
                    keep_as_options: bool = False) -> Tuple[Splits, List[str]]:
    """Remove ``n`` labels from every split except test, which keeps the full label set.

    The question this answers is whether an option head can pick a label it never trained
    on. Train, competence and calibration lose the held-out rows entirely and their
    criteria are restricted to the seen labels, so the held-out names are never even
    negatives. Test is untouched apart from ``meta["label_seen"]`` ("seen" / "unseen"),
    which keeps its rows identical to a run without held-out labels.

    ``keep_as_options`` is the realistic variant: a new category that exists as text but
    has no examples yet. Held-out rows are still removed, but the labels stay in every
    row's criteria, so they appear as never-gold negatives during training.
    """
    crit = splits.train[0].criteria
    if not isinstance(crit, dict):
        raise ValueError("hold_out_labels needs dict criteria shared by every row")
    names = list(crit)
    if not 0 < n < len(names):
        raise ValueError("cannot hold out %d of %d labels" % (n, len(names)))
    heldout = set(random.Random(seed).sample(sorted(names), n))
    seen = dict(crit) if keep_as_options else {k: v for k, v in crit.items() if k not in heldout}
    seen_keys = list(seen)

    def restrict(rows: Sequence[Example]) -> List[Example]:
        out = []
        for ex in rows:
            if list(ex.criteria) != names:
                raise ValueError("hold_out_labels needs one criteria set shared by every row")
            gold = names[ex.label]
            if gold not in heldout:
                out.append(Example(ex.state, ex.qtype, ex.instructions, seen,
                                   seen_keys.index(gold), dict(ex.meta)))
        return out

    test = []
    for ex in splits.test:
        meta = dict(ex.meta)
        meta["label_seen"] = "unseen" if names[ex.label] in heldout else "seen"
        test.append(Example(ex.state, ex.qtype, ex.instructions, ex.criteria, ex.label, meta))
    return (Splits(restrict(splits.train), restrict(splits.competence),
                   restrict(splits.calibration), test), sorted(heldout))


# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------

def _hf(name: str, *args, **kw):
    from datasets import load_dataset
    return load_dataset(name, *args, **kw)


def shuffled_cap(rows: List[Example], max_rows: Optional[int], seed: int) -> List[Example]:
    """Seeded shuffle, then cap.

    Hub splits are often sorted by label (Banking77's first 2000 train rows hold 17 of
    its 77 intents), so capping before shuffling trains and tests on a label subset.
    """
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:max_rows] if max_rows else rows


def load_banking77(max_rows: Optional[int] = None, seed: int = 17
                   ) -> Tuple[List[Example], List[Example]]:
    """Banking77: 77 fine-grained banking intents.

    Read from the ``mteb/banking77`` parquet mirror: ``PolyAI/banking77`` is a dataset
    script, which ``datasets`` >= 4 refuses to run. The mirror has 9,993 / 3,076 rows
    against the original's 10,003 / 3,080, so its test split is not upstream's exactly.

    The label names are terse snake_case strings. They are expanded to readable phrases
    because the option tower embeds them as text -- ``card_arrival`` and
    ``card_delivery_estimate`` are nearly identical as tokens and quite different as
    questions. ``max_rows`` caps train only; the test split is shuffled, never capped.
    """
    ds = _hf("mteb/banking77")
    by_id = dict(zip(ds["train"]["label"], ds["train"]["label_text"]))
    names = [by_id[i] for i in range(len(by_id))]
    criteria = {n: n.replace("_", " ") for n in names}
    ins = "Which banking intent does this customer message express?"

    def conv(split):
        return [Example(r["text"], "choice", ins, criteria, int(r["label"]),
                        {"dataset": "banking77", "language": "en"}) for r in split]

    return shuffled_cap(conv(ds["train"]), max_rows, seed), shuffled_cap(conv(ds["test"]), None, seed)


def load_massive(languages: Sequence[str] = ("en-US",), max_rows: Optional[int] = None,
                 seed: int = 17) -> Tuple[List[Example], List[Example]]:
    """AmazonScience/massive: 60 intents, 51 locales.

    Used two ways: as a high-cardinality task, and as the source of the foreign-language
    rows that make the competence head see confidently-wrong behaviour during training.
    Read from the Hub's parquet conversion, since the original is a dataset script.
    ``max_rows`` caps train and test per locale, after a seeded shuffle.
    """
    train, test = [], []
    ins = "Which intent does this utterance express?"
    for loc in languages:
        ds = _hf("AmazonScience/massive", revision="refs/convert/parquet",
                 data_files={"train": "%s/train/*.parquet" % loc, "test": "%s/test/*.parquet" % loc})
        names = ds["train"].features["intent"].names
        criteria = {n: n.replace("_", " ").replace(":", " ") for n in names}
        meta = {"dataset": "massive", "language": loc.split("-")[0], "locale": loc}
        for split_name, bucket in (("train", train), ("test", test)):
            rows = [Example(r["utt"], "choice", ins, criteria, int(r["intent"]), dict(meta))
                    for r in ds[split_name]]
            bucket.extend(shuffled_cap(rows, max_rows, seed))
    return train, test


def load_clinc150(max_rows: Optional[int] = None, seed: int = 17
                  ) -> Tuple[List[Example], List[Example]]:
    """clinc/clinc_oos (plus config): 150 intents and an explicit out-of-scope class.

    ``max_rows`` caps train only; the test split is shuffled, never capped.
    """
    ds = _hf("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    criteria = {n: n.replace("_", " ") for n in names}
    ins = "Which intent does this request express?"

    def conv(split):
        return [Example(r["text"], "choice", ins, criteria, int(r["intent"]),
                        {"dataset": "clinc150", "language": "en",
                         "oos": names[int(r["intent"])] == "oos"}) for r in split]

    return shuffled_cap(conv(ds["train"]), max_rows, seed), shuffled_cap(conv(ds["test"]), None, seed)


def load_jsonl(path: str, instructions: Optional[str] = None,
               criteria: Optional[Any] = None, qtype: str = "choice",
               max_rows: Optional[int] = None) -> List[Example]:
    """Your own rows.

    Each line needs ``state`` (or ``text``) and ``label``. ``label`` may be the label
    string or its index. ``criteria``, ``instructions`` and ``type`` may be per row or
    passed once for the whole file.

    A wide, hierarchical label taxonomy is exactly the case the option head is for.
    """
    rows: List[Example] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_rows and len(rows) >= max_rows:
                break
            r = json.loads(line)
            crit = r.get("criteria", criteria)
            if crit is None:
                raise ValueError("row %d has no criteria and none was passed" % i)
            ins = r.get("instructions", instructions)
            if ins is None:
                raise ValueError("row %d has no instructions and none was passed" % i)
            qt = r.get("type", qtype)
            labels = list(crit.keys()) if isinstance(crit, dict) else [str(c) for c in crit]
            lab = r["label"]
            if isinstance(lab, str):
                if lab not in labels:
                    raise ValueError("row %d label %r is not in criteria" % (i, lab))
                lab = labels.index(lab)
            meta = dict(r.get("meta", {}))
            meta.setdefault("dataset", os.path.basename(path))
            rows.append(Example(r.get("state", r.get("text")), qt, ins, crit, int(lab), meta))
    return rows


REGISTRY: Dict[str, Callable[..., Any]] = {
    "banking77": load_banking77,
    "massive": load_massive,
    "clinc150": load_clinc150,
}


def load_dataset_by_name(name: str, **kw):
    if name not in REGISTRY:
        raise KeyError("unknown dataset %r; known: %s (or use load_jsonl)"
                       % (name, sorted(REGISTRY)))
    return REGISTRY[name](**kw)


# --------------------------------------------------------------------------------------
# shift augmentation
# --------------------------------------------------------------------------------------

def truncate_state(ex: Example, keep: float, rng: random.Random) -> Example:
    """Keep the first ``keep`` fraction of the state.

    Models an over-long input hitting the context limit. The answer may still be
    recoverable or may not -- which is the point: the competence head has to learn
    that a truncated state is a risk factor, not a guarantee of failure.
    """
    text = ex.state if isinstance(ex.state, str) else json.dumps(ex.state, ensure_ascii=False)
    words = text.split()
    n = max(1, int(len(words) * keep))
    meta = dict(ex.meta)
    meta["shift"] = "truncate"
    return Example(" ".join(words[:n]), ex.qtype, ex.instructions, ex.criteria, ex.label, meta)


def add_distractors(ex: Example, n_extra: int, pool: Sequence[str],
                    rng: random.Random) -> Optional[Example]:
    """Inject plausible but wrong extra options.

    Two things happen at once: the label set grows (the regime the late-interaction head
    is for), and near-miss options appear that a weak model will happily prefer.

    Returns ``None`` when nothing could be added -- a single-schema dataset has no
    distractors outside its own label set. Returning the row unchanged instead would
    append a duplicate of a clean row, quietly inflating the clean count and reweighting
    the competence training set toward exactly the rows it needs least.
    """
    if not isinstance(ex.criteria, dict):
        return None
    existing = set(ex.criteria)
    extra = [p for p in pool if p not in existing]
    rng.shuffle(extra)
    extra = extra[:n_extra]
    if not extra:
        return None
    gold_label = list(ex.criteria.keys())[ex.label]
    merged = dict(ex.criteria)
    for e in extra:
        merged[e] = e.replace("_", " ")
    keys = list(merged.keys())
    meta = dict(ex.meta)
    meta["shift"] = "distractors"
    meta["n_options"] = len(keys)
    return Example(ex.state, ex.qtype, ex.instructions, merged, keys.index(gold_label), meta)


def build_shift_set(clean: Sequence[Example], cfg, foreign: Optional[Sequence[Example]] = None,
                    ood: Optional[Sequence[Example]] = None, seed: int = 17) -> List[Example]:
    """Assemble the competence training set: clean rows plus deliberate shift.

    Proportions come from ``CompConfig``. The clean rows stay in, and stay the majority:
    a head trained only on broken inputs learns to distrust everything, which is a gate
    that abstains on all traffic and reports an excellent risk number for doing nothing.
    """
    rng = random.Random(seed)
    rows = list(clean)
    n = len(clean)
    pool: List[str] = []
    for ex in clean[:200]:
        if isinstance(ex.criteria, dict):
            pool.extend(ex.criteria.keys())
    pool = list(dict.fromkeys(pool))

    if cfg.shift_truncate > 0:
        for ex in rng.sample(list(clean), min(n, int(n * cfg.shift_truncate))):
            rows.append(truncate_state(ex, cfg.shift_truncate_keep, rng))

    if cfg.shift_distractors > 0 and pool:
        skipped = 0
        for ex in rng.sample(list(clean), min(n, int(n * cfg.shift_distractors))):
            aug = add_distractors(ex, cfg.shift_distractor_n, pool, rng)
            if aug is None:
                skipped += 1
                continue
            rows.append(aug)
        if skipped:
            # Single-schema datasets hit this for every row. Say so: a silently empty
            # shift type looks identical to one that was configured off.
            print("[layax] distractor shift skipped %d/%d rows: no labels outside the "
                  "row's own option set. Mix in a second dataset to enable it."
                  % (skipped, skipped + sum(1 for r in rows if r.meta.get("shift") == "distractors")))

    if cfg.shift_foreign > 0 and foreign:
        k = min(len(foreign), int(n * cfg.shift_foreign))
        for ex in rng.sample(list(foreign), k):
            meta = dict(ex.meta)
            meta["shift"] = "foreign"
            rows.append(Example(ex.state, ex.qtype, ex.instructions, ex.criteria, ex.label, meta))

    if cfg.shift_ood_domain > 0 and ood:
        k = min(len(ood), int(n * cfg.shift_ood_domain))
        for ex in rng.sample(list(ood), k):
            meta = dict(ex.meta)
            meta["shift"] = "ood"
            rows.append(Example(ex.state, ex.qtype, ex.instructions, ex.criteria, ex.label, meta))

    rng.shuffle(rows)
    return rows


def shift_report(rows: Sequence[Example]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = r.meta.get("shift", "clean")
        out[k] = out.get(k, 0) + 1
    return out
