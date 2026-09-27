"""Modal entrypoint.

    pip install modal && modal setup
    modal run scripts/modal_app.py::main --config configs/banking77.json
    modal run scripts/modal_app.py::smoke                     # no GPU, no HF, ~20 s
    modal run scripts/modal_app.py::sweep --config configs/banking77.json
    modal run --detach scripts/modal_app.py::baselines_pub    # FastFit, SetFit, classifiers; 3 datasets x 3 seeds

Two volumes, because they have very different lifetimes: model weights and dataset
downloads are worth keeping between runs and are expensive to refetch, while run outputs
accumulate and are what you actually want to read afterwards.

Pick the GPU by what the step needs:

* ``A10G``  24 GB -- enough for ModernBERT-large with the option tower at batch 8.
* ``A100``  40/80 GB -- larger batches, roughly 3x faster wall clock.
* ``H100``  fastest, and the only one where bf16 is clearly the right dtype.
* ``T4``    16 GB, fp16 only. Matches Kaggle, so use it to reproduce a Kaggle run
            rather than to train quickly.
"""
from __future__ import annotations

import json
import os
import sys

import modal
import modal.exception

APP_NAME = "layax"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.48",
        "safetensors>=0.4",
        "huggingface_hub>=0.25",
        "datasets>=2.19",
        "numpy>=1.24",
        "scipy>=1.11",          # exact binomial bound; layax falls back without it
        "laya>=0.3.5",          # baseline arms only
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)

# FastFit and SetFit get their own image so the layax pipeline's environment is untouched.
# fast-fit 1.2.1 passes Trainer(tokenizer=...), which transformers 5 removed, hence the
# pin. These are the versions the CPU smoke ran with; the torch 2.5.1 + CUDA pairing has
# not been run before the batch. No `laya`: the baseline path never imports it.
baseline_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "safetensors>=0.4",
    "huggingface_hub>=0.25",
    "numpy>=1.24",
    "scipy>=1.11",
    "transformers==4.57.6",
    "datasets==4.5.0",
    "sentence-transformers==5.3.0",
    "accelerate==1.15.0",
    "evaluate==0.4.6",
    "scikit-learn>=1.3",
    "setfit==1.2.0",
    "fast-fit==1.2.1",
).env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})


def _with_source(img: modal.Image) -> modal.Image:
    # Ship the package itself rather than pip-installing it, so an edit is one
    # `modal run` away instead of a release. Local dirs must be the last layers.
    return (img.add_local_dir(os.path.join(HERE, "layax"), remote_path="/root/layax")
            .add_local_dir(os.path.join(HERE, "configs"), remote_path="/root/configs"))


image = _with_source(image)
baseline_image = _with_source(baseline_image)

app = modal.App(APP_NAME, image=image)
cache_vol = modal.Volume.from_name("layax-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("layax-runs", create_if_missing=True)

# A gated or rate-limited HF download needs a token. Create it once with
#   modal secret create huggingface HF_TOKEN=hf_...
# and it is picked up automatically; the run still works without it.


def _optional_hf_secret() -> list[modal.Secret]:
    # from_name is lazy, so a missing secret only fails at app start. Resolve it here on
    # the launching machine instead; inside the container the list is not consulted.
    if not modal.is_local():
        return []
    secret = modal.Secret.from_name("huggingface")
    try:
        secret.hydrate()
    except modal.exception.NotFoundError:
        return []
    return [secret]


HF_SECRET = _optional_hf_secret()

VOLUMES = {"/cache": cache_vol, "/runs": runs_vol}


@app.function(gpu="A10G", timeout=6 * 60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def run_pipeline(config: dict, jsonl_bytes: bytes | None = None,
                 jsonl_name: str = "rows.jsonl", run_baselines: bool = True) -> dict:
    """Full pipeline on GPU. Returns the report; artifacts land in the runs volume."""
    sys.path.insert(0, "/root")
    from layax.config import RunConfig
    from layax.pipeline import run

    cfg = RunConfig.from_dict(config)
    cfg.output_dir = os.path.join("/runs", cfg.name)
    os.makedirs(cfg.output_dir, exist_ok=True)

    jsonl_path = None
    if jsonl_bytes:
        jsonl_path = os.path.join("/runs", jsonl_name)
        with open(jsonl_path, "wb") as f:
            f.write(jsonl_bytes)

    report = run(cfg, device="cuda", hf_token=os.environ.get("HF_TOKEN"),
                 jsonl_path=jsonl_path, run_baselines=run_baselines,
                 on_checkpoint=runs_vol.commit)
    # Commit explicitly: without it the container can exit before the volume flushes and
    # a six-hour run leaves nothing behind.
    runs_vol.commit()
    cache_vol.commit()
    return {k: v for k, v in report.items() if k != "training"}


@app.function(gpu="A10G", timeout=4 * 60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def jsonl_arm(config: dict, rows: bytes, shift: bytes, go: str, worst: str,
              slice_key: str, slice_value) -> dict:
    """Train on JSONL rows, evaluate test and a shift file, write replay dumps with
    ``meta.episode`` as the state: <name>/replay and <name>/replay-shift."""
    sys.path.insert(0, "/root")
    import numpy as np

    from layax.config import RunConfig
    from layax.data import Example, load_jsonl
    from layax.evaluate import collect_predictions
    from layax.jsonl_eval import probabilities, slice_report
    from layax.pipeline import load_splits, run
    from layax.replay import write_replay
    from layax.runtime import LayaxAgent

    cfg = RunConfig.from_dict(config)
    cfg.output_dir = os.path.join("/runs", cfg.name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    paths = {}
    for name, blob in (("rows", rows), ("shift", shift)):
        paths[name] = os.path.join(cfg.output_dir, name + ".jsonl")
        with open(paths[name], "wb") as f:
            f.write(blob)
    report = run(cfg, device="cuda", hf_token=os.environ.get("HF_TOKEN"), jsonl_path=paths["rows"],
                 run_baselines=False, on_checkpoint=runs_vol.commit)
    agent = LayaxAgent.from_pretrained(cfg.output_dir, device="cuda")
    temps = {k: v["temperature"] for k, v in report["temperatures"].items()}
    gate = report["msp_gate"]
    thr = gate["threshold"] if gate.get("feasible") else float("inf")
    splits = load_splits(cfg, jsonl_path=paths["rows"])
    shift_rows = load_jsonl(paths["shift"])
    labels = list(splits.test[0].criteria)
    out = {"name": cfg.name, "msp_gate": gate, "temperatures": temps}
    for name, part in (("test", splits.test), ("shift", shift_rows)):
        p = probabilities(agent, part, temps, "cuda")
        out[name] = slice_report(p, part, labels, thr, go, worst, slice_key, slice_value)

    by_ep = {}

    def as_episode(part):
        eps = [Example(ex.meta["episode"], ex.qtype, ex.instructions, ex.criteria, ex.label, ex.meta)
               for ex in part]
        by_ep.update({e.state: ex for e, ex in zip(eps, part)})
        return eps

    def score(part):
        feats = collect_predictions(agent.model, agent.tok, [by_ep[e.state] for e in part], agent.cfg,
                                    "cuda", batch_size=32)["features"]
        return np.stack([f["logits"][: len(labels)] for f in feats])

    engine = {"interaction": cfg.li.interaction, "proj_dim": cfg.li.proj_dim}
    base = [("train", splits.train), ("competence", as_episode(splits.competence)),
            ("calibration", as_episode(splits.calibration))]
    sd = agent.model.state_dict()
    out["replay"] = write_replay(os.path.join(cfg.output_dir, "replay"),
                                 base + [("test", as_episode(splits.test))], score, sd, engine)["counts"]
    out["replay_shift"] = write_replay(os.path.join(cfg.output_dir, "replay-shift"),
                                       base + [("test", as_episode(shift_rows))], score, sd, engine)["counts"]
    with open(os.path.join(cfg.output_dir, "jsonl_eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    runs_vol.commit()
    return out


@app.local_entrypoint()
def jsonl_arms(config: str, rows: str, shift: str, go: str = "held", worst: str = "dropped",
               slice_key: str = "visible", seed_list: str = "17", epochs: int = 5,
               arms: str = "full,frozen"):
    """Two arms on the same rows: full fine-tune and a frozen encoder (freeze_encoder_epochs
    = epochs freezes every encoder parameter for the whole run)."""
    with open(config) as f:
        base = json.load(f)
    blobs = [open(p, "rb").read() for p in (rows, shift)]
    cfgs = []
    freeze = {"full": 0, "frozen": epochs}
    for seed in (int(s) for s in seed_list.split(",")):
        for arm in arms.split(","):
            c = json.loads(json.dumps(base))
            c["li"].update({"epochs": epochs, "seed": seed, "freeze_encoder_epochs": freeze[arm]})
            c["name"] = "%s-%s-s%d" % (base["name"], arm, seed)
            cfgs.append(c)
    kw = {"rows": blobs[0], "shift": blobs[1], "go": go, "worst": worst,
          "slice_key": slice_key, "slice_value": False}
    os.makedirs("runs/jsonl", exist_ok=True)
    for out in jsonl_arm.map(cfgs, kwargs=kw):
        with open(os.path.join("runs/jsonl", out["name"] + ".json"), "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(json.dumps({"name": out["name"], "test": out["test"], "shift": out["shift"]}, default=str))


@app.function(gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def benchmark_latency(model_dir: str, n_options: int = 77, repeats: int = 100) -> dict:
    """p50/p95 on GPU, cold and warm cache, at a given label-set width."""
    sys.path.insert(0, "/root")
    from layax.evaluate import latency_benchmark
    from layax.runtime import LayaxAgent

    agent = LayaxAgent.from_pretrained(model_dir, device="cuda")
    qs = {"intent": {"type": "choice", "instructions": "Which intent is this?",
                     "criteria": {"label_%03d" % i: "intent number %d" % i
                                  for i in range(n_options)}}}
    return latency_benchmark(agent, qs, "i lost my card and need a replacement", repeats=repeats)


@app.function(timeout=15 * 60)
def smoke() -> str:
    """Shapes only: no GPU, no checkpoint, no dataset. Run this first."""
    sys.path.insert(0, "/root")
    import torch.nn as nn
    from types import SimpleNamespace

    from layax.config import LIConfig
    from layax.li_head import LateInteractionDecisionModel
    from layax.runtime import LayaxAgent

    class Enc(nn.Module):
        def __init__(self, d=32, v=4096):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=d, vocab_size=v)
            self.emb = nn.Embedding(v, d)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            h = self.emb(input_ids)
            if attention_mask is not None:
                h = h * attention_mask.unsqueeze(-1).to(h.dtype)
            return SimpleNamespace(last_hidden_state=h)

    class Tok:
        cls_token_id, sep_token_id, pad_token_id, mask_token_id = 1, 2, 0, 3
        mask_token = "[MASK]"

        def __call__(self, text, add_special_tokens=True):
            ids = [4 + (hash(t) % 4000) for t in str(text).split()]
            return {"input_ids": ([1] + ids + [2]) if add_special_tokens else ids}

    cfg = LIConfig(base_checkpoint="stub", base_subfolder=None, proj_dim=16,
                   state_max_len=64, option_max_len=8)
    agent = LayaxAgent(LateInteractionDecisionModel(Enc(), cfg), Tok(), device="cpu")
    qs = {"q": {"type": "choice", "instructions": "which?",
                "criteria": {"l%02d" % i: "intent %d" % i for i in range(77)}}}
    out = agent.predict("i lost my card", qs)
    return "ok: %d options, forward %.1f ms" % (len(out["answers"]["q"]["probabilities"]),
                                                out["timing"]["forward_ms"])


@app.function(gpu="A10G", timeout=6 * 60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def sweep_one(config: dict, label: str) -> dict:
    """One arm of a sweep. Fanned out by ``sweep`` below."""
    sys.path.insert(0, "/root")
    from layax.config import RunConfig
    from layax.pipeline import run

    cfg = RunConfig.from_dict(config)
    cfg.name = "%s-%s" % (cfg.name, label)
    cfg.output_dir = os.path.join("/runs", cfg.name)
    report = run(cfg, device="cuda", hf_token=os.environ.get("HF_TOKEN"), run_baselines=False,
                 on_checkpoint=runs_vol.commit)
    runs_vol.commit()
    m = report["test"]["metrics"]
    return {"label": label, "accuracy": m["accuracy"], "aurc_softmax": m["aurc_softmax"],
            "aurc_competence": m.get("aurc_competence"), "ece": m["ece_softmax"],
            "verdict": report.get("competence", {}).get("verdict"),
            "msp_gate": report["test"].get("gate"),
            "test_by_shift": report["test_shift"]["by_shift"]}


def _block_max_z(comp, features) -> dict:
    """Largest |z-score| per competence feature block. A block far outside the range it
    was fitted on is the one driving the head when the option set changes."""
    import numpy as np
    z = np.abs(comp.builder.transform_scale(comp.builder.build(features)))
    out, start = {}, 0
    for name, width in zip(comp.builder.spec.names, comp.builder.spec.widths):
        out[name] = round(float(z[:, start: start + width].max()), 2)
        start += width
    return out


@app.function(gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def diagnose_heldout(run_name: str = "banking77-heldout") -> dict:
    """Re-score a finished held-out-labels run under three option sets, no training.

    * all77      -- every test row against all labels, as the run scored it.
    * seen_only  -- seen rows against the seen labels only: the option count the
                    competence head was fitted on. If the gate recovers here, the test-time
                    option count is what broke it.
    * unseen_only -- unseen rows against the held-out labels only. Accuracy above chance
                    (1/n_heldout) means the option tower matches labels it never trained on,
                    and the all77 miss is a prior toward trained labels.
    """
    sys.path.insert(0, "/root")
    import numpy as np

    from layax.config import RunConfig
    from layax.data import Example, hold_out_labels
    from layax.evaluate import collect_predictions
    from layax.pipeline import load_splits
    from layax.runtime import LayaxAgent

    run_dir = os.path.join("/runs", run_name)
    with open(os.path.join(run_dir, "run_report.json")) as f:
        report = json.load(f)
    cfg = RunConfig.from_dict(report["config"])
    splits, heldout = hold_out_labels(load_splits(cfg), cfg.heldout_labels, seed=cfg.li.seed)
    assert heldout == report["heldout_labels"], "held-out labels differ from the run's"
    agent = LayaxAgent.from_pretrained(run_dir, device="cuda")
    comp = agent.competence

    def restrict(ex: Example, keep: set) -> Example:
        crit = {k: v for k, v in ex.criteria.items() if k in keep}
        return Example(ex.state, ex.qtype, ex.instructions, crit,
                       list(crit).index(list(ex.criteria)[ex.label]), dict(ex.meta))

    names = list(splits.test[0].criteria)
    seen_names, held = set(names) - set(heldout), set(heldout)
    seen_rows = [e for e in splits.test if e.meta["label_seen"] == "seen"]
    unseen_rows = [e for e in splits.test if e.meta["label_seen"] == "unseen"]
    variants = {"all77": splits.test,
                "seen_only": [restrict(e, seen_names) for e in seen_rows],
                "unseen_only": [restrict(e, held) for e in unseen_rows]}

    out = {"threshold": comp.threshold, "calibration_gate": report["competence"]["threshold"]}
    for name, rows in variants.items():
        res = collect_predictions(agent.model, agent.tok, rows, agent.cfg, "cuda", batch_size=32)
        raw = comp.score_raw(res["features"], device="cuda")
        groups = np.array([e.meta["label_seen"] for e in rows])
        entry = {"max_abs_z_by_block": _block_max_z(comp, res["features"])}
        for g in sorted(set(groups)):
            ii = groups == g
            entry[g] = {"n": int(ii.sum()), "accuracy": float(res["correct"][ii].mean()),
                        "raw_min": float(raw[ii].min()), "raw_median": float(np.median(raw[ii])),
                        "raw_max": float(raw[ii].max()),
                        "coverage": (float((raw[ii] >= comp.threshold).mean())
                                     if comp.threshold is not None else None)}
        if name == "all77":
            pred_names = [names[p] for p in res["pred"]]
            unseen_ii = np.where(groups == "unseen")[0]
            entry["unseen"]["pred_in_heldout"] = float(np.mean([pred_names[i] in held
                                                                 for i in unseen_ii]))
            entry["seen"]["pred_in_heldout"] = float(np.mean([pred_names[i] in held
                                                               for i in np.where(groups == "seen")[0]]))
        out[name] = entry
        print("[layax] %s: %s" % (name, json.dumps(entry)), flush=True)
    out["chance_unseen_only"] = 1.0 / len(heldout)
    return out


@app.function(gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def label_prior(run_name: str = "banking77-heldout-s20") -> dict:
    """Does removing each label's average score recover unseen labels? No training.

    Reference rows are calibration-split states scored against the full label set, and
    only rows whose gold label was seen: their gold is never read, only their logits, so
    this is the unlabelled traffic a deployment would have. Test rows stay untouched.
    """
    sys.path.insert(0, "/root")
    import numpy as np

    from layax.config import RunConfig
    from layax.data import hold_out_labels
    from layax.evaluate import collect_predictions
    from layax.pipeline import load_splits
    from layax.runtime import LayaxAgent

    run_dir = os.path.join("/runs", run_name)
    with open(os.path.join(run_dir, "run_report.json")) as f:
        report = json.load(f)
    cfg = RunConfig.from_dict(report["config"])
    full = load_splits(cfg)
    held = set(report["heldout_labels"])
    names = list(full.test[0].criteria)
    ref_rows = [e for e in full.calibration if names[e.label] not in held]
    test = hold_out_labels(full, cfg.heldout_labels, seed=cfg.li.seed)[0].test
    agent = LayaxAgent.from_pretrained(run_dir, device="cuda")

    def logits(rows):
        res = collect_predictions(agent.model, agent.tok, rows, agent.cfg, "cuda", batch_size=32)
        return np.stack([f["logits"][: len(names)] for f in res["features"]])

    ref, lt = logits(ref_rows), logits(test)
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    gold = np.array([e.label for e in test])
    seen = np.array([e.meta["label_seen"] for e in test])
    held_idx = np.array([n in held for n in names])
    out = {"n_reference": len(ref_rows)}
    for name, scores in {"raw": lt, "centered": lt - mu, "zscore": (lt - mu) / sd}.items():
        pred = scores.argmax(1)
        out[name] = {g: {"accuracy": float((pred == gold)[seen == g].mean()),
                         "pred_in_heldout": float(held_idx[pred][seen == g].mean())}
                     for g in ("seen", "unseen")}
        out[name]["overall"] = float((pred == gold).mean())
        print("[layax] %s: %s" % (name, json.dumps(out[name])), flush=True)
    return out


@app.function(gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def msp_gate(run_name: str = "banking77-scale20") -> dict:
    """Learn-then-Test on max-softmax instead of the competence head. No training.

    Same procedure and splits as the pipeline's gate: walk start from competence-split
    scores, threshold fitted on calibration, applied once to test. Reported with the
    run's fitted temperature and with none, since temperature can reorder MSP across rows.
    """
    sys.path.insert(0, "/root")
    import numpy as np

    from layax.competence import evaluate_selective, fit_abstention_threshold, walk_start
    from layax.config import RunConfig
    from layax.evaluate import collect_predictions, softmax_rows
    from layax.pipeline import load_splits
    from layax.runtime import LayaxAgent

    run_dir = os.path.join("/runs", run_name)
    with open(os.path.join(run_dir, "run_report.json")) as f:
        report = json.load(f)
    cfg = RunConfig.from_dict(report["config"])
    splits = load_splits(cfg)
    agent = LayaxAgent.from_pretrained(run_dir, device="cuda")
    res = {name: collect_predictions(agent.model, agent.tok, rows, agent.cfg, "cuda", batch_size=32)
           for name, rows in (("competence", splits.competence),
                              ("calibration", splits.calibration), ("test", splits.test))}
    temps = {k: v["temperature"] for k, v in report["temperatures"].items()}
    out = {}
    for label, t in (("msp_temperature", temps), ("msp_raw", None)):
        msp = {k: np.array([p.max() for p in softmax_rows(r["features"], t)]) for k, r in res.items()}
        thr = fit_abstention_threshold(
            msp["calibration"], res["calibration"]["correct"], cfg.comp.target_risk,
            cfg.comp.delta, cfg.comp.min_coverage,
            start_threshold=walk_start(msp["competence"], cfg.comp.min_coverage),
            gate_score=label)
        thr.pop("curve")
        out[label] = {"calibration": thr, "test": (
            evaluate_selective(msp["test"], res["test"]["correct"], thr["threshold"])
            if thr["feasible"] else None)}
        print("[layax] %s: %s" % (label, json.dumps(out[label])), flush=True)
    return out


@app.function(gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def verify_msp_stage(run_name: str = "banking77-scale20-ep5") -> dict:
    """Run pipeline.msp_stage on a finished checkpoint, then save, reload and predict.

    Exercises the default gate end to end without training: the threshold fit, the test
    evaluation, the saved ``msp_threshold`` and ``abstain`` from ``LayaxAgent.predict``.
    The agent is saved to /tmp, not the volume, so the run directory is left untouched.
    """
    sys.path.insert(0, "/root")
    from layax.config import RunConfig
    from layax.evaluate import collect_predictions
    from layax.pipeline import load_foreign_rows, load_splits, msp_stage
    from layax.runtime import LayaxAgent

    run_dir = os.path.join("/runs", run_name)
    with open(os.path.join(run_dir, "run_report.json")) as f:
        cfg = RunConfig.from_dict(json.load(f)["config"])
    splits = load_splits(cfg)
    agent = LayaxAgent.from_pretrained(run_dir, device="cuda")
    cal = collect_predictions(agent.model, agent.tok, splits.calibration, agent.cfg, "cuda",
                              batch_size=cfg.eval_batch_size)
    foreign = load_foreign_rows(("km-KH", "am-ET", "te-IN", "ja-JP"))[1]
    report: dict = {}
    threshold, row = msp_stage(cfg, agent.model, agent.tok, splits, "cuda", cal,
                               agent.temperature_raw, foreign, report)
    saved = LayaxAgent(agent.model, agent.tok, device="cuda", temperatures=agent.temperature_raw,
                       msp_threshold=threshold)
    saved.save("/tmp/verify_agent")
    reloaded = LayaxAgent.from_pretrained("/tmp/verify_agent", device="cuda")
    ex = splits.test[0]
    answer = reloaded.predict(ex.state, {"q": ex.as_question()})["answers"]["q"]
    return {"threshold": threshold, "row": row, "gate": report["msp_gate"],
            "test_gate": report["test"].get("gate"),
            "test_accuracy": report["test"]["metrics"]["accuracy"],
            "by_shift": report["test_shift"]["by_shift"],
            "reloaded_threshold": reloaded.msp_threshold,
            "sample": {k: answer.get(k) for k in ("choice", "confidence", "abstain")},
            "sample_max_prob": max(answer["probabilities"].values())}


@app.local_entrypoint()
def verify_msp(run_name: str = "banking77-scale20-ep5"):
    print(json.dumps(verify_msp_stage.remote(run_name), indent=2, default=str))


@app.local_entrypoint()
def msp(run_name: str = "banking77-scale20"):
    print(json.dumps(msp_gate.remote(run_name), indent=2))


@app.local_entrypoint()
def prior(run_name: str = "banking77-heldout-s20"):
    print(json.dumps(label_prior.remote(run_name), indent=2))


@app.function(gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def dump_replay(run_name: str = "banking77-scale20-ep5") -> dict:
    """Cache real predictions as a replay dump for a downstream gate. No training.

    Competence, calibration and test splits exactly as the run used them, plus MASSIVE
    test rows in four non-Latin scripts scored against the same Banking77 schema (no
    gold). Raw logits are stored; the consumer applies the run's temperature itself.
    """
    sys.path.insert(0, "/root")
    import hashlib

    from layax.data import Example
    from layax.evaluate import collect_predictions
    from layax.pipeline import load_foreign_rows, load_splits
    from layax.config import RunConfig
    from layax.runtime import LayaxAgent

    run_dir = os.path.join("/runs", run_name)
    with open(os.path.join(run_dir, "run_report.json")) as f:
        report = json.load(f)
    cfg = RunConfig.from_dict(report["config"])
    splits = load_splits(cfg)
    ref = splits.test[0]
    foreign = [Example(e.state, ref.qtype, ref.instructions, ref.criteria, 0, dict(e.meta))
               for e in load_foreign_rows(("km-KH", "am-ET", "te-IN", "ja-JP"), max_rows=100)[1]]
    parts = [("competence", splits.competence), ("calibration", splits.calibration),
             ("test", splits.test), ("foreign", foreign)]
    agent = LayaxAgent.from_pretrained(run_dir, device="cuda")
    out_dir = os.path.join(run_dir, "replay")
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    n_opts = len(ref.criteria)  # never hardcode the label count
    with open(os.path.join(out_dir, "rows.jsonl"), "w", encoding="utf-8") as f:
        for split, rows in parts:
            res = collect_predictions(agent.model, agent.tok, rows, agent.cfg, "cuda", batch_size=32)
            for ex, feat in zip(rows, res["features"]):
                f.write(json.dumps({"split": split, "state": ex.state,
                                    "gold": None if split == "foreign" else ex.label,
                                    "logits": [round(float(v), 5) for v in feat["logits"][:n_opts]],
                                    "language": ex.meta.get("language", "")},
                                   ensure_ascii=False) + "\n")
            counts[split] = len(rows)
    with open(os.path.join(run_dir, "model.pt"), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {"engine": {"run": run_name, "model_sha256": sha,
                           "interaction": cfg.li.interaction, "proj_dim": cfg.li.proj_dim,
                           "temperature": report["temperatures"]["choice:11+"]["temperature"]},
                "schema": {"type": "choice", "instructions": ref.instructions,
                           "options": list(ref.criteria), "option_text": list(ref.criteria.values())},
                "counts": counts}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    runs_vol.commit()
    return manifest


@app.local_entrypoint()
def replay(run_name: str = "banking77-scale20-ep5"):
    print(json.dumps(dump_replay.remote(run_name)["counts"]))


REWORDINGS = {"own": "{text}", "raw_key": "{key}", "prefixed": "the customer wants help with {text}"}


@app.function(gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def cross_schema_one(run_name: str) -> dict:
    """One checkpoint scored against every dataset's full test label set, plus rewordings
    of its own schema. Inference only."""
    sys.path.insert(0, "/root")
    from layax.config import RunConfig
    from layax.cross_schema import drop_label, reword, score
    from layax.data import load_dataset_by_name
    from layax.evaluate import collect_predictions
    from layax.runtime import LayaxAgent

    run_dir = os.path.join("/runs", run_name)
    with open(os.path.join(run_dir, "run_report.json")) as f:
        own = RunConfig.from_dict(json.load(f)["config"]).dataset
    agent = LayaxAgent.from_pretrained(run_dir, device="cuda")
    tests = {name: load_dataset_by_name(name)[1] for name in ("banking77", "clinc150", "massive")}
    tests["clinc150-inscope"] = drop_label(tests["clinc150"], "oos")
    sets = {"target:" + k: v for k, v in tests.items()}
    sets.update({"reword:" + k: reword(tests[own], t) for k, t in REWORDINGS.items()})
    out = {"run": run_name, "own_dataset": own, "rewordings": REWORDINGS, "results": {}}
    for key, rows in sets.items():
        res = collect_predictions(agent.model, agent.tok, rows, agent.cfg, "cuda", batch_size=32)
        out["results"][key] = score(res)
        print("[layax] %s %s %s" % (run_name, key, out["results"][key]), flush=True)
    os.makedirs("/runs/cross_schema", exist_ok=True)
    with open(os.path.join("/runs", "cross_schema", run_name + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    runs_vol.commit()
    return out


@app.local_entrypoint()
def cross_schema(runs: str = ",".join("%s-pub-ep5-s%d" % (d, s) for d in
                                      ("banking77", "clinc150", "massive-en") for s in (17, 18, 19))):
    os.makedirs("runs/cross_schema", exist_ok=True)
    for out in cross_schema_one.map(runs.split(",")):
        with open(os.path.join("runs/cross_schema", out["run"] + ".json"), "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"run": out["run"], **{k: v["accuracy"] for k, v in out["results"].items()}}))


HF_RUNS = {"banking77": "banking77-pub-ep5-s17", "clinc150": "clinc150-pub-ep5-s17",
           "massive-en": "massive-en-pub-ep5-s17"}


@app.function(gpu="A10G", timeout=60 * 60, volumes=VOLUMES, secrets=HF_SECRET)
def hf_stage_one(sub: str) -> dict:
    """Re-save one run as safetensors under /runs/hf/<sub>, reload it, and check that the
    reloaded model gives the same logits and the same test accuracy as the run."""
    sys.path.insert(0, "/root")
    import numpy as np

    from layax.config import RunConfig
    from layax.evaluate import collect_predictions
    from layax.pipeline import load_splits
    from layax.runtime import LayaxAgent

    run_dir, out_dir = os.path.join("/runs", HF_RUNS[sub]), os.path.join("/runs", "hf", sub)
    with open(os.path.join(run_dir, "run_report.json")) as f:
        report = json.load(f)
    LayaxAgent.from_pretrained(run_dir, device="cuda").save(out_dir)
    rows = load_splits(RunConfig.from_dict(report["config"])).test
    res = {}
    for name, d in (("run", run_dir), ("staged", out_dir)):
        a = LayaxAgent.from_pretrained(d, device="cuda")
        res[name] = collect_predictions(a.model, a.tok, rows, a.cfg, "cuda", batch_size=32)
    z = [np.max(np.abs(np.asarray(x["logits"]) - np.asarray(y["logits"])))
         for x, y in zip(res["run"]["features"], res["staged"]["features"])]
    runs_vol.commit()
    return {"sub": sub, "run": HF_RUNS[sub], "n": len(rows), "max_abs_logit_diff": float(max(z)),
            "acc_run": float(res["run"]["correct"].mean()),
            "acc_staged": float(res["staged"]["correct"].mean()),
            "files": sorted(os.listdir(out_dir))}


# Read locally at launch, so a missing token never breaks the other entrypoints.
HF_WRITE = [modal.Secret.from_dict({"HF_WRITE_TOKEN": os.environ.get("HF_WRITE_TOKEN", "")})]


@app.function(timeout=60 * 60, volumes=VOLUMES, secrets=HF_WRITE)
def hf_upload_one(repo_id: str, docs: dict) -> str:
    """Uploads from the volume: `modal volume get` has corrupted large files before."""
    from huggingface_hub import HfApi
    for name, text in docs.items():
        with open(os.path.join("/runs", "hf", name), "w", encoding="utf-8") as f:
            f.write(text)
    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    api.create_repo(repo_id, exist_ok=True, private=True)
    api.upload_folder(repo_id=repo_id, folder_path="/runs/hf", commit_message="layax checkpoints")
    return "https://huggingface.co/" + repo_id


@app.local_entrypoint()
def hf_stage():
    for out in hf_stage_one.map(list(HF_RUNS)):
        print(json.dumps(out))


@app.local_entrypoint()
def hf_upload(repo_id: str):
    """Needs HF_WRITE_TOKEN (write scope) in the local environment. Creates the repo private."""
    docs = {}
    pairs = [("README.md", "hf/README.md"), ("LICENSE", "LICENSE"), ("NOTICE", "NOTICE")]
    pairs += [(sub + "/schema.json", "hf/%s/schema.json" % sub) for sub in HF_RUNS]
    for dst, src in pairs:
        with open(src, encoding="utf-8") as f:
            docs[dst] = f.read().replace("<HF_REPO_ID>", repo_id)
    print(hf_upload_one.remote(repo_id, docs))


@app.local_entrypoint()
def diagnose(run_name: str = "banking77-heldout"):
    print(json.dumps(diagnose_heldout.remote(run_name), indent=2))


@app.local_entrypoint()
def main(config: str = "configs/banking77.json", jsonl: str = "",
         no_baselines: bool = False, gpu_check: bool = False, wait: bool = False):
    """Spawn the pipeline and return; pass --wait to block and print the table.

    Without --wait, launch with ``modal run --detach``: a spawned call outlives this
    process only when the app is detached, and a blocking ``.remote()`` is cancelled the
    moment its caller dies, which is how a finished training run once lost its baselines.
    Results land in the ``layax-runs`` volume as ``<name>/run_report.json``, rewritten
    after every stage.
    """
    with open(config) as f:
        cfg = json.load(f)
    blob = None
    if jsonl:
        with open(jsonl, "rb") as f:
            blob = f.read()
    if gpu_check:
        print(smoke.remote())
        return
    args = (cfg, blob, os.path.basename(jsonl) if jsonl else "rows.jsonl", not no_baselines)
    if not wait:
        call = run_pipeline.spawn(*args)
        print("[layax] spawned %s; fetch with: modal volume get layax-runs %s/run_report.json"
              % (call.object_id, cfg["name"]))
        return
    report = run_pipeline.remote(*args)
    print(report.get("table", ""))
    print(json.dumps(report.get("test", {}).get("metrics", {}), indent=2)[:2000])


@app.local_entrypoint()
def quickwins(config: str = "configs/banking77_scale20.json"):
    """Three single-change arms off one config, spawned in parallel; launch with --detach.

    Each arm writes /runs/<name>-<label>/run_report.json like any sweep arm.
    """
    with open(config) as f:
        base = json.load(f)
    arms = {"ep5": {"epochs": 5},
            "alloptions": {"train_options_per_row": 0},
            "hardneg": {"option_sampling": "hard"}}
    for label, patch in arms.items():
        c = json.loads(json.dumps(base))
        c["li"].update(patch)
        call = sweep_one.spawn(c, label)
        print("[layax] spawned %s-%s: %s" % (base["name"], label, call.object_id))


@app.local_entrypoint()
def seeds(config: str = "configs/banking77_scale20.json", epochs: int = 5,
          seed_list: str = "18,19"):
    """Extra seeds of one arm, spawned in parallel; launch with --detach.

    The seed also drives the split shuffle, so each seed trains, calibrates and tests on
    its own splits: replicate runs of the whole pipeline, not of training alone.
    """
    with open(config) as f:
        base = json.load(f)
    for s in (int(x) for x in seed_list.split(",")):
        c = json.loads(json.dumps(base))
        c["li"].update({"epochs": epochs, "seed": s})
        label = "ep%d-s%d" % (epochs, s)
        call = sweep_one.spawn(c, label)
        print("[layax] spawned %s-%s: %s" % (base["name"], label, call.object_id))


@app.function(image=baseline_image, gpu="A10G", timeout=4 * 60 * 60, volumes=VOLUMES,
              secrets=HF_SECRET)
def pub_baselines_one(config: dict) -> dict:
    """Classifier, FastFit and SetFit arms for one config and seed. No layax training.

    Writes /runs/<name>-baselines.json after every arm, so a cancelled job keeps the
    arms that finished.
    """
    sys.path.insert(0, "/root")
    from layax.baselines_ext import run_pub_baselines, write_report
    from layax.config import RunConfig

    cfg = RunConfig.from_dict(config)
    path = os.path.join("/runs", cfg.name + "-baselines.json")

    def save(report: dict) -> None:
        write_report(report, path)
        runs_vol.commit()

    report = run_pub_baselines(cfg, device="cuda", hf_token=os.environ.get("HF_TOKEN"),
                               on_arm=save)
    return {"name": cfg.name, "table": report["table"]}


@app.local_entrypoint()
def baselines_pub(configs: str = "configs/banking77_pub.json,configs/clinc150_pub.json,"
                                 "configs/massive_en_pub.json",
                  seed_list: str = "17,18,19", epochs: int = 5):
    """Every config x seed as its own container, all spawned at once; launch with --detach.

    Names follow ``seeds``: <name>-ep<epochs>-s<seed>, so each job reads the same splits
    as the layax run it is compared with (the seed drives the split shuffle).
    """
    for path in configs.split(","):
        with open(path) as f:
            base = json.load(f)
        for s in (int(x) for x in seed_list.split(",")):
            c = json.loads(json.dumps(base))
            c["li"].update({"epochs": epochs, "seed": s})
            c["name"] = "%s-ep%d-s%d" % (base["name"], epochs, s)
            call = pub_baselines_one.spawn(c)
            print("[layax] spawned %s-baselines: %s" % (c["name"], call.object_id))


@app.function(image=baseline_image, gpu="A10G", timeout=2 * 60 * 60, volumes=VOLUMES,
              secrets=HF_SECRET)
def baseline_replay_one(config: dict, engine: str) -> dict:
    """Train one baseline engine ("fastfit" or "classifier") on one config and seed, then
    write replay files to /runs/<name>-<engine>. No layax training."""
    sys.path.insert(0, "/root")
    from layax.baseline_replay import run_classifier_replay, run_fastfit_replay
    from layax.config import RunConfig

    cfg = RunConfig.from_dict(config)
    out = os.path.join("/runs", cfg.name + "-" + engine)
    if engine == "fastfit":
        manifest = run_fastfit_replay(cfg, out, "cuda")
    else:
        manifest = run_classifier_replay(cfg, out, "cuda", hf_token=os.environ.get("HF_TOKEN"))
    runs_vol.commit()
    return manifest


@app.local_entrypoint()
def baseline_replay(engine: str = "fastfit",
                    configs: str = "configs/banking77_pub.json,configs/clinc150_pub.json,"
                                   "configs/massive_en_pub.json",
                    seed_list: str = "17,18,19", epochs: int = 5):
    """Names match ``baselines_pub``, so the splits are the layax run's own."""
    if engine not in ("fastfit", "classifier"):
        raise SystemExit("engine must be fastfit or classifier")
    cfgs = []
    for path in configs.split(","):
        with open(path) as f:
            base = json.load(f)
        for s in (int(x) for x in seed_list.split(",")):
            c = json.loads(json.dumps(base))
            c["li"].update({"epochs": epochs, "seed": s})
            c["name"] = "%s-ep%d-s%d" % (base["name"], epochs, s)
            cfgs.append(c)
    for m in baseline_replay_one.map(cfgs, kwargs={"engine": engine}):
        print(json.dumps({"run": m["engine"]["run"], "T": m["engine"]["temperature"],
                          "counts": m["counts"]}))


@app.local_entrypoint()
def sweep(config: str = "configs/banking77.json"):
    """The ablation that decides whether each piece earns its place.

    Runs in parallel: ``starmap`` fans out one container per arm, so seven arms cost one
    arm's wall clock rather than seven.
    """
    with open(config) as f:
        base = json.load(f)

    arms = []
    def arm(label, patch_li=None, patch_comp=None):
        c = json.loads(json.dumps(base))
        c.setdefault("li", {}).update(patch_li or {})
        c.setdefault("comp", {}).update(patch_comp or {})
        arms.append((c, label))

    arm("maxsim", {"interaction": "maxsim"})
    arm("xattn", {"interaction": "xattn"})
    # No token interaction at all: says whether maxsim/xattn earn their cost.
    arm("meanpool", {"interaction": "meanpool"})
    # Competence head without the pooled CLS block: Varshney et al. (2022) found
    # calibrators reading raw embeddings degraded selective prediction.
    arm("no-pooled", {}, {"use_pooled": False})
    arm("no-negatives", {"in_batch_negatives": False})
    arm("hard-negatives", {"option_sampling": "hard"})
    arm("no-shift", {}, {"shift_foreign": 0.0, "shift_truncate": 0.0,
                         "shift_distractors": 0.0, "shift_ood_domain": 0.0})

    for r in sweep_one.starmap(arms):
        print(json.dumps(r))
