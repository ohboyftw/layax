"""Replay dumps for GLiNER2.5-Decide (Fastino, Apache 2.0) scored zero-shot on a run's splits.

No training: the model sees each row's state with the schema's option text as label names
and the schema's instructions as the task instruction, and the raw per-label logits are
dumped in option order. That puts a zero-shot engine through the same Learn-then-Test
certificate as layax, the classifier and FastFit, on identical rows.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .config import RunConfig
from .data import Example
from .replay import replay_parts, write_replay

log = logging.getLogger(__name__)
GLINER_REPO = "fastino/GLiNER2.5-Decide"
TASK = "intent"
# GLiNER2 injects label strings into its prompt verbatim; these would break alignment.
RESERVED = ("[P]", "[L]", "[C]", "[E]", "[R]", "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]", "(", ")")


def _labels(ref: Example) -> List[str]:
    labels = list(ref.criteria.values())
    bad = [label for label in labels if any(t in label for t in RESERVED)]
    if bad:
        raise ValueError("labels GLiNER2 cannot take verbatim: %s" % bad[:5])
    # GLiNER2 keys logits by label text, so repeated texts would get one shared score.
    if len(set(labels)) != len(labels):
        raise ValueError("option texts repeat; GLiNER2 would score them identically")
    return labels


def _snapshot() -> Tuple[str, str]:
    """(commit sha, local dir) of the model repo at that exact commit.

    gliner2 2.0.0 applies ``revision=`` to config.json only; weights and tokenizer would
    still come from main. Loading from a local snapshot pins every file.
    """
    from huggingface_hub import HfApi, snapshot_download
    sha = HfApi().model_info(GLINER_REPO).sha
    return sha, snapshot_download(GLINER_REPO, revision=sha)


def _scorer(ref: Example, model_dir: str, device: str,
            batch_size: int) -> Callable[[Sequence[Example]], np.ndarray]:
    from gliner2.classification.engine import ClassificationConfig, Classifier
    from gliner2.classification.schema import ClassificationSchema

    labels = _labels(ref)
    # from_pretrained(device=...) records the device without moving the weights.
    clf = Classifier.from_pretrained(model_dir).to(device=device).eval()
    schema = ClassificationSchema().single(TASK, labels, instruction=ref.instructions)
    config = ClassificationConfig(batch_size=batch_size)

    def score(rows: Sequence[Example]) -> np.ndarray:
        out = clf.batch_score([r.state for r in rows], schema, config=config)
        log.info("gliner scored %d rows", len(rows))
        return np.array([[s.logit(TASK, label) for label in labels] for s in out], dtype=np.float64)

    return score


def run_gliner_replay(cfg: RunConfig, out_dir: str, device: str,
                      batch_size: int = 32) -> Dict[str, Any]:
    parts = replay_parts(cfg)
    ref = dict(parts)["test"][0]
    sha, model_dir = _snapshot()
    engine = {"interaction": "gliner2-zero-shot", "proj_dim": 0, "backbone": GLINER_REPO,
              "backbone_revision": sha,
              "label_text": "schema option_text", "instruction": ref.instructions}
    return write_replay(out_dir, parts, _scorer(ref, model_dir, device, batch_size), {}, engine)
