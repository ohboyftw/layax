"""Configuration objects for layax.

Two independent pieces, two configs:

* ``LIConfig``     -- the late-interaction option head (removes the shared option
                      token budget that caps Laya at ~20 options).
* ``CompConfig``   -- the competence head (predicts P(this answer is correct),
                      trained on held-out errors under deliberate distribution shift).

Both are plain dataclasses so they serialise to JSON next to a checkpoint and can be
diffed in a run log.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


def _filter_known(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(
            "%s got unknown config keys %s; known keys are %s"
            % (cls.__name__, unknown, sorted(known))
        )
    return {k: v for k, v in d.items() if k in known}


INTERACTIONS = ("maxsim", "xattn", "meanpool")

# Shift type name (as written to ``Example.meta["shift"]``) -> the CompConfig fraction
# that controls it.
SHIFT_FIELDS = {"truncate": "shift_truncate", "foreign": "shift_foreign",
                "distractors": "shift_distractors", "ood": "shift_ood_domain"}


@dataclass
class LIConfig:
    """Late-interaction option scoring.

    The upstream model inlines every option into one sequence behind a shared
    ``head_max_len`` budget, so 77 labels get ~3 tokens each. Here the state and the
    options are encoded separately and scored by interaction, so the option count no
    longer competes with the state for tokens.
    """

    # Which base checkpoint the encoder and type embedding are initialised from.
    base_checkpoint: str = "convaiinnovations/laya"
    base_subfolder: Optional[str] = "typed-decisions"

    # "maxsim"  -- ColBERT-style token MaxSim, option embeddings fully cacheable.
    # "xattn"   -- one cross-attention layer over cached option tokens, then the scorer MLP.
    #              Strictly more expressive, still cacheable, ~1 extra layer of compute.
    # "meanpool" -- plain bi-encoder cosine of mean-pooled state and option vectors. No
    #              token interaction at all: the ablation floor that says whether maxsim
    #              or xattn earn their cost.
    interaction: str = "maxsim"

    # Projection dim for the late-interaction space. Smaller = smaller option cache.
    # 128 follows ColBERT; the cache for 1000 options x 24 tokens x 128 dims is ~6 MB in fp16.
    proj_dim: int = 128

    # Token budgets. Note the asymmetry with upstream: the state gets the WHOLE window
    # because options no longer live in the same sequence.
    state_max_len: int = 512
    option_max_len: int = 24

    # xattn only.
    xattn_heads: int = 8
    xattn_dropout: float = 0.1

    # Learned logit scale/bias on top of the interaction score, per question type.
    learn_scale: bool = True
    init_scale: float = 4.0

    # Score questions are ordinal; optionally add a CORN-style cumulative-link loss on top
    # of the usual proper-scoring-rule reward. Upstream's weakest primitive (SST-5 0.372).
    ordinal_score_loss: bool = True
    ordinal_weight: float = 0.3

    # Training.
    lr_encoder: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    # T4 has no bf16. Keep fp16 there, bf16 on Ampere+.
    amp_dtype: str = "fp16"
    freeze_encoder_epochs: int = 0
    seed: int = 17

    # Reward mixing: proper scoring rule (RLCD, as upstream) vs plain cross-entropy.
    # 1.0 = pure RLCD reward, 0.0 = pure CE. The middle is usually the fastest to fit.
    rlcd_weight: float = 0.7
    w_sph: float = 0.5
    w_rps: float = 1.0

    # In-batch negatives: score each state against other rows' gold options too.
    # This is what makes large label sets cheap to learn.
    in_batch_negatives: bool = True
    negative_weight: float = 0.5

    # Training-time option sampling. Encoding all 77 (or 150) options for every row in
    # every step is the thing that would not fit on a T4: the option tower is a real
    # encoder pass per option. Sample the gold plus this many negatives instead, and
    # evaluate on the full label set. Set to 0 to use every option (small label sets only).
    train_options_per_row: int = 16
    # "random" -- uniform negatives.
    # "hard"   -- negatives the model currently scores highest, refreshed each epoch.
    #             Faster to fit on confusable taxonomies, and more prone to collapse;
    #             keep "random" as the baseline you compare against.
    option_sampling: str = "random"

    def validate(self) -> "LIConfig":
        if self.interaction not in INTERACTIONS:
            raise ValueError("interaction must be one of %s, got %r"
                             % (INTERACTIONS, self.interaction))
        if self.proj_dim <= 0:
            raise ValueError("proj_dim must be positive")
        if self.option_max_len < 2:
            raise ValueError("option_max_len must be >= 2 (a marker plus at least one token)")
        if not 0.0 <= self.rlcd_weight <= 1.0:
            raise ValueError("rlcd_weight must be in [0, 1]")
        if self.option_sampling not in ("random", "hard"):
            raise ValueError("option_sampling must be 'random' or 'hard'")
        if self.train_options_per_row and self.train_options_per_row < 2:
            raise ValueError("train_options_per_row must be 0 (all options) or >= 2")
        if self.amp_dtype not in ("fp16", "bf16", "fp32"):
            raise ValueError("amp_dtype must be fp16, bf16 or fp32")
        return self

    to_dict = asdict

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LIConfig":
        return cls(**_filter_known(cls, d)).validate()


@dataclass
class CompConfig:
    """Competence head: P(the argmax answer is correct).

    Upstream already has ``act_head`` -- pooled CLS plus four score-distribution features,
    trained as part of the RLCD action cost. This is not that. This head is trained
    directly against observed correctness on a held-out split, with deliberately shifted
    inputs mixed in, so it learns the confidently-wrong regions (Laya's README reports 0.000 accuracy at
    0.952 confidence on Khmer) that temperature scaling cannot reach.
    """

    # Feature groups. Each is a knob so the ablation is one config edit.
    use_pooled: bool = True          # encoder CLS of the state pass
    use_score_stats: bool = True     # top1, margin, normalised entropy, k, logit spread
    use_interaction_stats: bool = True   # max/mean/std MaxSim, option-coverage
    use_length_stats: bool = True    # token count, truncation fraction, option-count bucket
    use_lang_stats: bool = True      # script id + language confidence from laya.lang
    use_energy: bool = True          # -logsumexp(logits): free OOD signal
    use_mahalanobis: bool = True     # distance of pooled CLS to the training feature mean

    hidden: List[int] = field(default_factory=lambda: [256, 64])
    dropout: float = 0.1
    detach_features: bool = True     # train the head without backprop into the encoder

    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 8
    batch_size: int = 128
    seed: int = 17

    # Class imbalance: most answers are correct, so errors are the rare class.
    pos_weight_auto: bool = True

    # --- shift augmentation -------------------------------------------------
    # Fractions are of the competence training set, applied on top of the clean rows.
    shift_truncate: float = 0.15     # cut the state to a fraction of its tokens
    shift_truncate_keep: float = 0.3
    shift_foreign: float = 0.25      # rows in languages the checkpoint cannot read
    shift_distractors: float = 0.2   # inject plausible-but-wrong extra options
    shift_distractor_n: int = 8
    shift_ood_domain: float = 0.15   # rows from a domain held out of decision training
    # One shift type (a key of SHIFT_FIELDS) kept out of competence training but still
    # applied to the shifted test set. It separates a head that generalises to shift from
    # one that learned the synthetic shortcuts of the shifts it was shown.
    heldout_shift: Optional[str] = None

    # --- calibration and abstention ----------------------------------------
    # Calibration only shapes the reported competence probability. The gate compares the
    # raw head sigmoid to its threshold.
    calibration: str = "isotonic"    # "isotonic" | "platt" | "none"
    # Learn-then-Test: pick the abstention threshold so that selective risk is
    # <= target_risk with probability >= 1 - delta, on data exchangeable with the
    # calibration split and nothing else.
    target_risk: float = 0.05
    delta: float = 0.1
    min_coverage: float = 0.30       # refuse to ship a gate that answers less than this

    def validate(self) -> "CompConfig":
        if self.calibration not in ("isotonic", "platt", "none"):
            raise ValueError("calibration must be isotonic, platt or none")
        for name in ("target_risk", "delta", "min_coverage"):
            v = getattr(self, name)
            if not 0.0 < v < 1.0:
                raise ValueError("%s must be in (0, 1), got %r" % (name, v))
        if not self.any_features():
            raise ValueError("competence head needs at least one feature group enabled")
        if self.heldout_shift is not None:
            if self.heldout_shift not in SHIFT_FIELDS:
                raise ValueError("heldout_shift must be one of %s or null, got %r"
                                 % (sorted(SHIFT_FIELDS), self.heldout_shift))
            if getattr(self, SHIFT_FIELDS[self.heldout_shift]) <= 0:
                raise ValueError("heldout_shift %r has fraction 0, so the test set would "
                                 "not contain it either" % self.heldout_shift)
        return self

    def any_features(self) -> bool:
        return any([
            self.use_pooled, self.use_score_stats, self.use_interaction_stats,
            self.use_length_stats, self.use_lang_stats, self.use_energy,
            self.use_mahalanobis,
        ])

    to_dict = asdict

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompConfig":
        return cls(**_filter_known(cls, d)).validate()


@dataclass
class RunConfig:
    """One experiment: a dataset, the two head configs, and where output goes."""

    name: str = "banking77"
    dataset: str = "banking77"       # see layax/data.py for the registry
    output_dir: str = "runs/banking77"
    li: LIConfig = field(default_factory=LIConfig)
    comp: CompConfig = field(default_factory=CompConfig)

    # Fractions of the training split reserved for the two calibration stages.
    # Competence must be fitted on rows the decision model did NOT train on, or it
    # learns the model's training-set optimism instead of its real error rate.
    competence_frac: float = 0.15
    calibration_frac: float = 0.10
    eval_batch_size: int = 32
    max_train_rows: Optional[int] = None
    max_eval_rows: Optional[int] = None
    # Label-index classifiers on the same encoder and test rows (layax/baselines.py):
    # fine-tuned linear head, and frozen encoder + MLP. Off by default because the
    # fine-tuned arm costs about as much as the layax training itself.
    classifier_baselines: bool = False
    # Number of labels removed from train, competence and calibration (test keeps all of
    # them, tagged seen/unseen). 0 = off. See data.hold_out_labels.
    heldout_labels: int = 0
    # Keep the held-out labels in train/competence/calibration criteria as never-gold
    # negatives, instead of removing them. Only meaningful with heldout_labels > 0.
    heldout_as_options: bool = False
    # Fit the learned competence head. Off by default: at init_scale 20 it lost to
    # max-softmax on AURC in every Banking77 run (README, "What did not work"). With it off, the gate is Learn-then-Test on max-softmax (msp_gate.py).
    competence_head: bool = False
    # Row meta key for the per-group test breakdown ("language", or "oos" for CLINC150,
    # whose loader tags out-of-scope rows). Held-out-label runs always group by label_seen.
    group_by: str = "language"
    # For dataset "jsonl": take the splits from this row meta key (values train,
    # competence, calibration, test; anything else is unused) instead of re-splitting at
    # random, so results line up with another system trained on the same split.
    split_key: Optional[str] = None

    def validate(self) -> "RunConfig":
        self.li.validate()
        self.comp.validate()
        if self.heldout_labels < 0:
            raise ValueError("heldout_labels must be >= 0")
        if self.competence_frac + self.calibration_frac >= 0.6:
            raise ValueError("competence_frac + calibration_frac leaves too little to train on")
        if not isinstance(self.classifier_baselines, bool):
            raise ValueError("classifier_baselines must be true or false")
        if not isinstance(self.competence_head, bool):
            raise ValueError("competence_head must be true or false")
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        d = dict(d)
        li = LIConfig.from_dict(d.pop("li", {}) or {})
        comp = CompConfig.from_dict(d.pop("comp", {}) or {})
        return cls(li=li, comp=comp, **_filter_known(cls, d)).validate()

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
