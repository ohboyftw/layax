"""Frozen DINOv2 features for the frames `kuka_extract.py` kept, in row order.

Features are the CLS token concatenated with the mean patch token (2 x hidden size).
Written as runs/kuka/features.npz with the row keys, so a head can be fitted without
touching the images again.

    python scripts/kuka_embed.py --out runs/kuka
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("kuka")


def load_rows(out: Path) -> list:
    rows = []
    for p in sorted(out.glob("shard-*.jsonl")):
        rows.extend(json.loads(line) for line in p.read_text(encoding="utf-8").splitlines())
    return rows


@torch.no_grad()
def embed(paths: list, model_name: str, batch_size: int) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(io.BytesIO(p.read_bytes())).convert("RGB") for p in paths[i:i + batch_size]]
        h = model(**proc(images=imgs, return_tensors="pt")).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy().astype(np.float32))
        if (i // batch_size) % 20 == 0:
            log.info("embedded %d/%d", i + len(imgs), len(paths))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/kuka")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(args.out)
    rows = load_rows(out)
    feats = embed([out / "images" / (r["key"] + ".jpg") for r in rows], args.model, args.batch_size)
    np.savez(out / "features.npz", keys=np.array([r["key"] for r in rows]), features=feats,
             model=args.model)
    log.info("wrote %s for %d rows", feats.shape, len(rows))


if __name__ == "__main__":
    main()
