"""Stream Open X-Embodiment `kuka` (QT-Opt) shards and keep only the grasp-decision moment.

For every episode that commands a grasp (gripper_closedness_action > 0.5 at some step),
keep the observation at the FIRST such step: the JPEG, height_to_bottom, the 7-d
base_pose_tool_reached and gripper_closed. The observation precedes the action, so
nothing after the grasp is kept. `steps/reward` equals `success` exactly and the final
height differs by outcome, so both are dropped. Episodes with no grasp command always
fail; they are counted per shard, not kept.

One shard is on disk at a time (downloaded, parsed, deleted), so peak disk use is about
one shard (0.8 GB) plus the kept frames. Re-running skips shards already done.

    python scripts/kuka_extract.py --out runs/kuka --n-shards 40
"""
from __future__ import annotations

import argparse
import json
import logging
import struct
import urllib.request
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

URL = "https://storage.googleapis.com/gresearch/robotics/kuka/0.1.0/kuka-train.tfrecord-%05d-of-01024"
ZLIB = ("base_pose_tool_reached", "gripper_closed", "height_to_bottom")
log = logging.getLogger("kuka")


def records(path: Path) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            data = f.read(struct.unpack("<Q", head[:8])[0])
            f.read(4)
            yield data


def _varint(b: bytes, i: int) -> Tuple[int, int]:
    out = shift = 0
    while True:
        c = b[i]
        i += 1
        out |= (c & 0x7F) << shift
        if c < 0x80:
            return out, i
        shift += 7


def _fields(b: bytes) -> Iterator[Tuple[int, object]]:
    """Protobuf wire format: (field number, bytes or int)."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        wt = key & 7
        if wt == 2:
            n, i = _varint(b, i)
            yield key >> 3, b[i:i + n]
            i += n
        elif wt == 0:
            v, i = _varint(b, i)
            yield key >> 3, v
        else:
            size = 4 if wt == 5 else 8
            yield key >> 3, b[i:i + size]
            i += size


def _feature(b: bytes):
    for num, sub in _fields(b):
        vals = [v for _, v in _fields(sub)]
        if num == 1:
            return vals
        if num == 2:
            return np.frombuffer(b"".join(vals), "<f4")
        if num == 3:
            raw, out, i = b"".join(vals), [], 0
            while i < len(raw):
                v, i = _varint(raw, i)
                out.append(v)
            return np.array(out, dtype=np.int64)
    return []


def parse(record: bytes, wanted: Tuple[str, ...]) -> Dict[str, object]:
    """tf.train.Example -> {name: value} for the wanted feature names only."""
    out = {}
    for _, feats in _fields(record):
        for _, entry in _fields(feats):
            kv = dict(_fields(entry))
            name = kv[1].decode()
            if name.endswith(wanted):
                v = _feature(kv[2])
                if name.endswith(ZLIB):  # TFDS zlib encoding: per-step float32 tensors
                    v = np.stack([np.frombuffer(zlib.decompress(x), "<f4") for x in v])
                out[name] = v
    return out


WANTED = ("success", "gripper_closedness_action", "image") + ZLIB


def extract_shard(path: Path, shard: int, img_dir: Path) -> Tuple[List[dict], dict]:
    rows, n, no_grasp, fails_no_grasp = [], 0, 0, 0
    for k, rec in enumerate(records(path)):
        ex = parse(rec, WANTED)
        n += 1
        success = bool(ex["success"][0])
        close = np.flatnonzero(ex["steps/action/gripper_closedness_action"] > 0.5)
        if not len(close):
            no_grasp += 1
            fails_no_grasp += int(not success)
            continue
        t = int(close[0])
        key = "%05d-%04d" % (shard, k)
        (img_dir / (key + ".jpg")).write_bytes(ex["steps/observation/image"][t])
        rows.append({"key": key, "shard": shard, "episode": k, "success": success, "close_step": t,
                     "height_to_bottom": float(ex["steps/observation/height_to_bottom"][t, 0]),
                     "tool_pose": ex["steps/observation/clip_function_input/base_pose_tool_reached"][t].round(5).tolist(),
                     "gripper_closed": float(ex["steps/observation/gripper_closed"][t, 0])})
    stats = {"shard": shard, "episodes": n, "no_grasp": no_grasp, "no_grasp_failed": fails_no_grasp,
             "kept": len(rows), "success_kept": round(float(np.mean([r["success"] for r in rows])), 4) if rows else None}
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/kuka")
    ap.add_argument("--n-shards", type=int, default=40)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # Spread across the index range, so any drift with shard number shows up.
    shards = [round(i * 1023 / (args.n_shards - 1)) for i in range(args.n_shards)]
    for shard in shards:
        done = out / ("shard-%05d.jsonl" % shard)
        if done.exists():
            continue
        tmp = out / ("shard-%05d.tfrecord.part" % shard)
        log.info("downloading shard %d", shard)
        urllib.request.urlretrieve(URL % shard, tmp)
        rows, stats = extract_shard(tmp, shard, img_dir)
        tmp.unlink()
        done.with_suffix(".tmp").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        done.with_suffix(".tmp").replace(done)
        with (out / "shard_stats.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        log.info("shard %d: %s", shard, stats)


if __name__ == "__main__":
    main()
