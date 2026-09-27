---
license: apache-2.0
base_model: convaiinnovations/laya
base_model_relation: finetune
tags:
  - text-classification
  - intent-classification
  - selective-prediction
datasets:
  - PolyAI/banking77
  - mteb/banking77
  - clinc/clinc_oos
  - AmazonScience/massive
language:
  - en
---

# layax intent checkpoints

> **Built on [Laya](https://github.com/NandhaKishorM/laya) by Nandakishor
> ([ConvAI Innovations](https://www.convaiinnovations.com)).** Laya's typed decision
> schema, its trained checkpoints
> ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)) and its
> scoring rules are the foundation of this work. layax only changes how a choice between
> many labels is scored, and adds an abstention gate. All credit for Laya itself belongs
> to its author. Laya is released under Apache 2.0, and so is layax.

Source code: [github.com/ohboyftw/layax](https://github.com/ohboyftw/layax)

These are three intent classification checkpoints trained with layax, an extension of
[Laya](https://github.com/NandhaKishorM/laya). layax encodes the
input and each label separately and scores them with token-level MaxSim, so a wide label
set does not share one token budget. The encoder is `convaiinnovations/laya`, which is
built on `answerdotai/ModernBERT-large`. Each checkpoint was fine-tuned on one dataset.

| folder | dataset | labels | test accuracy | abstain threshold |
|---|---|---|---|---|
| `banking77/` | Banking77 | 77 | 0.905 (3000 rows) | 0.808 |
| `clinc150/` | CLINC150, including out of scope | 151 | 0.902 (3000 rows) | 0.0, never abstains |
| `massive-en/` | MASSIVE, en-US | 60 | 0.875 (2974 rows) | 0.808 |

Test accuracy is for the file in that folder, re-measured after conversion with
`modal run scripts/modal_app.py::hf_stage`, and identical to the training run (seed 17
publication runs). The abstain threshold is `msp_threshold` in `layax_config.json`, a
Learn-then-Test threshold on the top probability for a selective risk of 0.05. On
CLINC150 the calibration rows met the target with every request answered, so the
threshold is 0.0 and it never abstains. Its test traffic missed the target; see
Limitations.

Each folder holds `model.safetensors` (all weights, including the fine-tuned encoder),
`layax_config.json` (head settings, temperatures and the gate threshold), `schema.json`
(the exact instruction and label text used in training), `tokenizer/` and `encoder/`
(the encoder config only).

## Intended use

Use these checkpoints for English intent classification on the label set each one was
trained with. They are published for Laya users and as a record of measured results. For
a new label set you need to train a new checkpoint. The checkpoints do not work on labels
they were not trained on.

## How to load

Install layax from source (https://github.com/ohboyftw/layax):

```bash
pip install "layax[inference] @ git+https://github.com/ohboyftw/layax"
```

```python
import json
from huggingface_hub import hf_hub_download
from layax.runtime import LayaxAgent

repo, sub = "<HF_REPO_ID>", "banking77"   # or "clinc150", "massive-en"
agent = LayaxAgent.from_pretrained(repo, subfolder=sub, device="cpu")
with open(hf_hub_download(repo, f"{sub}/schema.json")) as f:
    questions = {"intent": json.load(f)}

agent.warm_cache(questions)
out = agent.predict("I still have not received my new card", questions)
out["answers"]["intent"]["choice"]         # a label key such as "card_arrival"
out["answers"]["intent"]["probabilities"]
out["answers"]["intent"]["abstain"]        # True when the top probability is below the threshold
```

Use `schema.json` as it is. The label text and the instruction must match training
exactly. A different rendering of the labels lowered accuracy by up to 9 to 12 points on
the same test rows.

## Results

These results are over three training seeds (17, 18, 19), each with its own shuffled
splits. The published checkpoints are the seed 17 runs. All arms are scored on the same
test rows with the full label set. The layax value is the three seed mean. The other
columns give the range over the three seeds.

| dataset | layax | classifier on Laya encoder | FastFit | SetFit |
|---|---|---|---|---|
| Banking77 | 0.908 | 0.907 to 0.918 | 0.935 to 0.937 | 0.897 to 0.901 |
| CLINC150 | 0.894 | 0.883 to 0.904 | 0.893 to 0.904 | 0.865 to 0.883 |
| MASSIVE-en | 0.876 | 0.870 to 0.882 | 0.883 to 0.889 | 0.857 to 0.863 |

- Upstream Laya scored 0.453 on the same Banking77 test rows (best arm,
  `head_max_len=512`).
- The classifier is a fine-tuned linear head on the same encoder, with no separate tuning.
  FastFit uses `roberta-base` and SetFit uses `paraphrase-mpnet-base-v2`.
- Expected calibration error for layax is 0.017, 0.012 and 0.036. A temperature scaled
  classifier scores 0.015 to 0.024, 0.028 to 0.038 and 0.021 to 0.026. The two are level
  on Banking77, layax is better on CLINC150, and the classifier is better on MASSIVE-en.
- CPU latency with 77 cached options is p50 155.7 ms and p95 168.8 ms on a 16-thread
  AMD Zen 4 laptop CPU in fp32.

The runs used `modal run --detach scripts/modal_app.py::seeds` for layax and
`modal run --detach scripts/modal_app.py::baselines_pub` for the other arms.

## Limitations

- FastFit, a smaller model, is as accurate or more accurate on all three datasets. It ranks
  its own errors worse (AURC about twice layax's on MASSIVE-en and CLINC150), so it gives
  less safe coverage under an abstention gate. See the source README.
- Under a Learn-then-Test certificate (5% risk, 5 seeds, Banking77 and MASSIVE-en), layax
  passed 10 of 10 runs, a fine-tuned classifier 7 of 10 and FastFit 4 of 10. The
  classifier's failures were narrow and the seeds share most test rows, so read this as
  layax being the most reliable engine here, not as a large gap. Details are in the
  [source README](https://github.com/ohboyftw/layax#certification-across-engines).
- The `abstain` gate on the CLINC150 checkpoint does not meet its target. It was set for a
  selective risk of 0.05 but gave 0.098 to 0.119 on test. The calibration rows were 1.4%
  to 2.0% out of scope and the test rows were 17% to 19%. On Banking77 and MASSIVE-en the
  gate stayed under target on every seed.
- The gate's guarantee holds only on data like the calibration split. It says nothing
  about other languages or a different share of out-of-scope requests.
- Scored against another dataset's labels, accuracy was 0.02 to 0.34. That is above chance
  but not usable.
- Labels held out of training scored 0.065 to 0.078 on Banking77, and chance level when
  they were kept as options that are never correct.
- The competence head, a learned predictor of correctness, lost to plain max-softmax in
  every run. It is not included.
- Results are for English intent classification only.

## Training data and licenses

The weights are released under Apache 2.0. They are derived from
`convaiinnovations/laya` (Apache 2.0) and `answerdotai/ModernBERT-large` (Apache 2.0),
modified by fine-tuning.

- Banking77 by PolyAI, CC BY 4.0 (Casanueva et al. 2020). The data was loaded from the
  `mteb/banking77` mirror. That mirror carries an MIT tag, which should not be relied on.
- CLINC150, `clinc/clinc_oos` "plus" configuration, CC BY 3.0 (Larson et al. 2019).
- MASSIVE by Amazon, en-US locale, CC BY 4.0 (FitzGerald et al. 2022).

The data was modified. It was split into train, competence, calibration and test subsets,
and label names were rendered as option text with underscores and colons replaced by
spaces. No data is included in this repository.

## References

There is no paper for layax. Please cite the methods and datasets it uses.

- FastFit: Yehudai and Bendel, NAACL 2024. arXiv:2404.12365
- Learn then Test: Angelopoulos, Bates, Candès, Jordan and Lei, 2021. arXiv:2110.01052
- Banking77: Casanueva et al., 2020. arXiv:2003.04807
- CLINC150: Larson et al., EMNLP-IJCNLP 2019. ACL Anthology D19-1131
- MASSIVE: FitzGerald et al., 2022. arXiv:2204.08582
