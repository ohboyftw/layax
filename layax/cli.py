"""Command line.

    python -m layax.cli run       --config configs/banking77.json
    python -m layax.cli run       --config configs/custom_jsonl.json --jsonl data/rows.jsonl
    python -m layax.cli baseline  --config configs/banking77.json
    python -m layax.cli latency   --model-dir runs/banking77
    python -m layax.cli smoke                       # no network, no checkpoint
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

from .config import RunConfig


def _load_cfg(path: Optional[str], overrides: Optional[str]) -> RunConfig:
    cfg = RunConfig.from_json(path) if path else RunConfig()
    if overrides:
        patch = json.loads(overrides)
        base = cfg.to_dict()
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
        cfg = RunConfig.from_dict(base)
    return cfg


def cmd_run(args) -> int:
    from .pipeline import run
    cfg = _load_cfg(args.config, args.set)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run(cfg, device=args.device, hf_token=args.hf_token, jsonl_path=args.jsonl,
        run_baselines=not args.no_baselines, skip_training=args.skip_training)
    return 0


def cmd_baseline(args) -> int:
    from .evaluate import LAYA_BASELINE_ARMS, compare, evaluate_laya_baseline
    from .pipeline import load_splits
    cfg = _load_cfg(args.config, args.set)
    splits = load_splits(cfg, args.jsonl)
    rows = splits.test[: args.max_rows] if args.max_rows else splits.test
    arms = []
    for kw in LAYA_BASELINE_ARMS:
        arms.append(evaluate_laya_baseline(rows, cfg.li.base_checkpoint, cfg.li.base_subfolder,
                                           device=args.device, **kw))
    print(compare(arms))
    return 0


def cmd_latency(args) -> int:
    from .evaluate import latency_benchmark
    from .runtime import LayaxAgent
    agent = LayaxAgent.from_pretrained(args.model_dir, device=args.device)
    qs = {"intent": {"type": "choice", "instructions": "Which intent is this?",
                     "criteria": {"label_%03d" % i: "intent number %d" % i
                                  for i in range(args.n_options)}}}
    print(json.dumps(latency_benchmark(agent, qs, "i lost my card and need a replacement",
                                       repeats=args.repeats), indent=2))
    return 0


def cmd_smoke(args) -> int:
    """Shapes and masking on a random tiny encoder. No network, no checkpoint.

    Run this first on any new machine: it separates "the environment is wrong" from
    "the model is wrong", which is otherwise an hour of guessing.
    """
    from ._stubs import StubEncoder, StubTokenizer

    from .config import LIConfig
    from .li_head import LateInteractionDecisionModel
    from .runtime import LayaxAgent

    for mode in ("maxsim", "xattn", "meanpool"):
        cfg = LIConfig(base_checkpoint="stub", base_subfolder=None, interaction=mode,
                       proj_dim=16, state_max_len=64, option_max_len=8, xattn_heads=2)
        model = LateInteractionDecisionModel(StubEncoder(), cfg)
        agent = LayaxAgent(model, StubTokenizer(), device="cpu")
        qs = {"intent": {"type": "choice", "instructions": "which intent?",
                         "criteria": {"l%02d" % i: "intent number %d" % i for i in range(77)}}}
        out = agent.predict("i lost my card", qs)
        p = out["answers"]["intent"]["probabilities"]
        assert len(p) == 77 and abs(sum(p.values()) - 1.0) < 1e-3
        print("[layax] smoke %-8s ok: 77 options, sum(p)=%.4f, forward %.1f ms"
              % (mode, sum(p.values()), out["timing"]["forward_ms"]))
    print("[layax] smoke passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="layax")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config")
        p.add_argument("--set", help="JSON patch applied over the config file")
        p.add_argument("--device")
        p.add_argument("--jsonl", help="rows file when dataset is 'jsonl'")

    p = sub.add_parser("run", help="full pipeline")
    common(p)
    p.add_argument("--output-dir")
    p.add_argument("--hf-token")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--skip-training", action="store_true",
                   help="evaluate the freshly initialised head; useful only as a floor")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("baseline", help="upstream laya arms only")
    common(p)
    p.add_argument("--max-rows", type=int, default=1000)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("latency", help="p50/p95 with cold and warm cache")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--device")
    p.add_argument("--n-options", type=int, default=77)
    p.add_argument("--repeats", type=int, default=50)
    p.set_defaults(func=cmd_latency)

    p = sub.add_parser("smoke", help="offline shape check")
    p.set_defaults(func=cmd_smoke)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
