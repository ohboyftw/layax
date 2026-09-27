"""Linear probes on the kuka grasp-decision features, written as replay dumps.

Decision: [success, fail] for a commanded grasp, from the observation at the first
close command. Two arms: "fused" (frozen DINOv2 features + proprio) and "proprio"
(height_to_bottom + 7-d tool pose only), the control that says whether the image earns
its place.

Splits are by SHARD, never by episode: consecutive episodes in a shard share a robot,
bin and time. The shift set is grasps near one bin wall (tool y above the pooled 75th
percentile), which succeed less often; it is removed from every split and scored only
from test shards, so its rows share no shard with training.

    python scripts/kuka_head.py --out runs/kuka --replay runs/replay
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from layax.baselines import fit_logit_temperature

log = logging.getLogger("kuka")
OPTIONS = {"success": "the grasp lifts the object", "fail": "the grasp does not lift the object"}
INSTRUCTIONS = "Will this commanded grasp succeed?"
SPLIT_SHARDS = {"train": 18, "competence": 4, "calibration": 4, "test": 8}


def load(out: Path) -> Tuple[List[dict], np.ndarray]:
    rows = [json.loads(line) for p in sorted(out.glob("shard-*.jsonl"))
            for line in p.read_text(encoding="utf-8").splitlines()]
    z = np.load(out / "features.npz")
    assert [r["key"] for r in rows] == z["keys"].tolist(), "features out of row order"
    return rows, z["features"]


def assign(rows: List[dict], seed: int) -> Dict[str, np.ndarray]:
    shards = sorted({r["shard"] for r in rows})
    order = np.random.default_rng(seed).permutation(shards)
    cuts = np.cumsum([SPLIT_SHARDS[k] for k in SPLIT_SHARDS])
    groups = dict(zip(SPLIT_SHARDS, np.split(order, cuts[:-1])))
    y_cut = np.quantile([r["tool_pose"][1] for r in rows], 0.75)
    edge = np.array([r["tool_pose"][1] > y_cut for r in rows])
    shard = np.array([r["shard"] for r in rows])
    idx = {k: np.flatnonzero(np.isin(shard, g) & ~edge) for k, g in groups.items()}
    idx["shift"] = np.flatnonzero(np.isin(shard, groups["test"]) & edge)
    return idx


def features(rows: List[dict], image: np.ndarray, arm: str) -> np.ndarray:
    prop = np.array([[r["height_to_bottom"]] + r["tool_pose"] for r in rows], dtype=np.float64)
    return prop if arm == "proprio" else np.hstack([image.astype(np.float64), prop])


def fit(x: np.ndarray, y: np.ndarray, seed: int):
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed).fit(scaler.transform(x), y)
    return scaler, clf


def logits(scaler, clf, x: np.ndarray) -> np.ndarray:
    """Two columns in OPTIONS order: [success, fail]. Softmax of [z, 0] is the sigmoid."""
    z = clf.decision_function(scaler.transform(x))
    return np.stack([z, np.zeros_like(z)], 1)


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    r = np.empty(len(score))
    r[np.argsort(score)] = np.arange(1, len(score) + 1)
    p = y.sum()
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p)))


def write_dump(d: Path, parts: Dict[str, Tuple[List[dict], np.ndarray]], engine: dict) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "rows.jsonl").open("w", encoding="utf-8") as f:
        for split, (rs, z) in parts.items():
            for r, zz in zip(rs, z):
                f.write(json.dumps({"split": split, "state": "ep-" + r["key"], "gold": 0 if r["success"] else 1,
                                    "logits": [round(float(v), 5) for v in zz], "language": ""}) + "\n")
    manifest = {"engine": engine,
                "schema": {"type": "choice", "instructions": INSTRUCTIONS,
                           "options": list(OPTIONS), "option_text": list(OPTIONS.values())},
                "counts": {k: len(v[0]) for k, v in parts.items()}}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_arm(rows, image, idx, arm: str, seed: int, replay: Path) -> dict:
    x = features(rows, image, arm)
    y = np.array([0 if r["success"] else 1 for r in rows])  # gold index: 0 = success
    sel = lambda k: ([rows[i] for i in idx[k]], x[idx[k]], y[idx[k]])  # noqa: E731
    _, xtr, ytr = sel("train")
    scaler, clf = fit(xtr, (ytr == 0).astype(int), seed)
    z = {k: logits(scaler, clf, sel(k)[1]) for k in ("competence", "calibration", "test", "shift")}
    temp = fit_logit_temperature(z["calibration"], sel("calibration")[2])["temperature"]
    buf = io.BytesIO()
    np.savez(buf, mean=scaler.mean_, scale=scaler.scale_, coef=clf.coef_, intercept=clf.intercept_)
    run = "kuka-grasp-%s-s%d" % (arm, seed)
    engine = {"run": run, "model_sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
              "interaction": "linear-probe", "proj_dim": int(x.shape[1]), "temperature": temp,
              "backbone": "facebook/dinov2-small + proprio" if arm == "fused" else "proprio only"}
    base = {k: (sel(k)[0], z[k]) for k in ("competence", "calibration")}
    write_dump(replay / run, {**base, "test": (sel("test")[0], z["test"])}, engine)
    write_dump(replay / (run + "-shift"), {**base, "test": (sel("shift")[0], z["shift"])},
               {**engine, "run": run + "-shift"})
    out = {"run": run, "temperature": temp, "n": {k: len(idx[k]) for k in idx}}
    for k in ("test", "shift"):
        yy = sel(k)[2]
        out[k] = {"accuracy": float((z[k].argmax(1) == yy).mean()), "success_rate": float((yy == 0).mean()),
                  "auroc": auroc(z[k][:, 0], (yy == 0).astype(int))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/kuka")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--seeds", default="17,18,19")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rows, image = load(Path(args.out))
    results = []
    for seed in (int(s) for s in args.seeds.split(",")):
        idx = assign(rows, seed)
        for arm in ("fused", "proprio"):
            results.append(run_arm(rows, image, idx, arm, seed, Path(args.replay)))
            log.info(json.dumps(results[-1]))
    (Path(args.out) / "head_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
