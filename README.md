# layax

> **Built on [Laya](https://github.com/NandhaKishorM/laya) by Nandakishor
> ([ConvAI Innovations](https://www.convaiinnovations.com)).** Laya's typed decision
> schema, its trained checkpoints
> ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)) and its
> scoring rules are the foundation of this work. layax only changes how a choice between
> many labels is scored, and adds an abstention gate. All credit for Laya itself belongs
> to its author. Laya is released under Apache 2.0, and so is layax.

Checkpoints: [huggingface.co/ohboyFtw/layax](https://huggingface.co/ohboyFtw/layax)

layax changes how [Laya](https://github.com/NandhaKishorM/laya) (Apache 2.0) scores a
choice between many labels. Upstream Laya puts every option into one input sequence with a
shared token budget, so on a wide label set each option gets only a few tokens. layax
encodes the input and each option separately and scores them against each other with
token-level MaxSim, the method used by ColBERT and FastFit. Option embeddings are cached, so
the input keeps its full context window whatever the number of options. layax also adds an
optional abstention gate that answers only when the top probability clears a threshold
chosen with Learn-then-Test.

layax is published for Laya users and as a documented set of results, including the
negative ones. On the three intent datasets below it fixes Laya's wide label set problem
and is level with a fine-tuned classifier. FastFit, a smaller model, is as accurate or
more accurate on all three, but ranks its own errors worse, so it gives less safe coverage
under an abstention gate.

## Results

All numbers are measured with this repository on Modal A10G GPUs in September 2026. Each
dataset uses three seeds (17, 18, 19). The seed also shuffles the splits, so each seed is a
full rerun. Every arm is scored on the same test rows with the full label set. Test sets are
3000 rows, except MASSIVE-en, which uses all 2974.

### Accuracy

| dataset (labels) | layax | classifier on Laya encoder | FastFit | SetFit | upstream Laya |
|---|---|---|---|---|---|
| Banking77 (77) | 0.908 | 0.907 to 0.918 | 0.935 to 0.937 | 0.897 to 0.901 | 0.453 |
| CLINC150 (151) | 0.894 | 0.883 to 0.904 | 0.893 to 0.904 | 0.865 to 0.883 | not run |
| MASSIVE-en (60) | 0.876 | 0.870 to 0.882 | 0.883 to 0.889 | 0.857 to 0.863 | not run |

The layax column is the mean over three seeds. The standard deviation is 0.002 on
Banking77, 0.011 on CLINC150 and 0.002 on MASSIVE-en. The other columns give the range over
the three seeds. The CLINC150 label count includes the out-of-scope class.

- The classifier is a linear head on the Laya encoder, fine-tuned with cross-entropy. It
  was not tuned separately. It uses the layax learning rates, epochs and batch size.
- FastFit uses `roberta-base` with a 1500 step cap. SetFit uses
  `paraphrase-mpnet-base-v2` with a 2000 step cap. These are the backbones their own
  READMEs show, so they compare methods, not heads on one encoder.
- Upstream Laya is its best arm, `head_max_len=512`, on the seed 17 test rows.

Commands:

```bash
# layax, for each of banking77_pub, clinc150_pub, massive_en_pub
modal run --detach scripts/modal_app.py::seeds --config configs/<dataset>_pub.json --seed-list 17,18,19 --epochs 5
# classifier, FastFit and SetFit arms for all three datasets and seeds
modal run --detach scripts/modal_app.py::baselines_pub
# upstream Laya arms: the full pipeline, which trains layax and then scores the Laya arms
modal run --detach scripts/modal_app.py::main --config configs/banking77.json
```

### Calibration

ECE (expected calibration error) measures how far the predicted probabilities are from the
observed accuracy. Lower is better. The classifier ECE below is after one temperature was
fitted on the calibration split, which is a held-out part of the training data.

| dataset | layax ECE | classifier ECE, temperature scaled |
|---|---|---|
| Banking77 | 0.017 | 0.015 to 0.024 |
| CLINC150 | 0.012 | 0.028 to 0.038 |
| MASSIVE-en | 0.036 | 0.021 to 0.026 |

The two are level on Banking77. layax is better on CLINC150. The classifier is better on
MASSIVE-en. layax has no general calibration advantage over a temperature scaled
classifier.

### Error ranking

AURC (area under the risk and coverage curve) measures how well the top probability ranks
the model's own errors, which is what an abstention gate depends on. Lower is better.
Temperature scaling does not change it, because it does not change the order.

| dataset | layax | classifier on Laya encoder | FastFit | SetFit |
|---|---|---|---|---|
| Banking77 | 0.015 to 0.018 | 0.015 to 0.019 | 0.020 to 0.021 | 0.020 to 0.021 |
| CLINC150 | 0.015 to 0.017 | 0.014 to 0.026 | 0.034 to 0.042 | 0.026 to 0.033 |
| MASSIVE-en | 0.028 to 0.029 | 0.024 to 0.033 | 0.058 to 0.062 | 0.039 to 0.044 |

FastFit's accuracy lead does not carry over. On MASSIVE-en and CLINC150 it ranks its errors
about twice as badly as layax, so at the same risk target a gate on FastFit answers fewer
requests. layax and the classifier are level. The layax values come from each run's
`run_report.json`, the others from `baselines_pub`.

### Abstention gate

The gate uses the top probability after temperature scaling. Selective risk is the error
rate on the rows the model answers. Learn-then-Test tests thresholds on the calibration
split from strictest to loosest and stops at the first one that fails. The chosen
threshold comes with the guarantee P(selective risk ≤ 0.05) ≥ 0.9 on data like the
calibration split.

- On Banking77 the gate answered 82% to 88% of test rows at realised risk 0.033 to 0.039.
  It stayed under the target on every seed.
- On MASSIVE-en the gate answered 70% to 79% of test rows at realised risk 0.028 to 0.042.
  It stayed under the target on every seed.
- On CLINC150 the gate failed. Realised risk was 0.098 to 0.119, about double the target.
  The calibration rows come from the training split, where 1.4% to 2.0% of rows are
  out of scope. The test split is 17% to 19% out of scope. The model looked safe on
  calibration data, so the gate answered every test row. On in-scope test rows alone, risk
  was 0.040 to 0.049.

The CLINC150 result is the documented limit of the bound. It holds only on data that is
exchangeable with the calibration split. If your live traffic has more out-of-scope
requests than your calibration rows, calibrate on rows drawn from live traffic.

### Certification across engines

The same gate, compared across engines on identical rows. Each engine's per-row logits
were written as replay dumps (`modal run scripts/modal_app.py::dump_replay` for layax,
`::baseline_replay --engine fastfit|classifier` for the others) and certified in a
separate harness that is not part of this repository. Settings: target risk 0.05, delta
0.1, threshold fitted on the competence and calibration rows. A seed passes when the
realised test risk stays at or below the certified upper bound. Rows, splits and label
text are identical across engines for each seed.

| engine | Banking77 passed | Banking77 coverage | MASSIVE-en passed | MASSIVE-en coverage |
|---|---|---|---|---|
| layax | 5 of 5 | 0.897, 0.833, 0.883, 0.867, 0.870 | 5 of 5 | 0.773, 0.802, 0.738, 0.676, 0.781 |
| classifier on Laya encoder | 3 of 5 | 0.899, 0.872, 0.928, 0.860, 0.861 | 4 of 5 | 0.770, 0.843, 0.713, 0.699, 0.743 |
| FastFit | 4 of 5 | 0.978, 0.837, 0.885, 0.830, 0.935 | 0 of 5 | none, 0.685, none, 0.341, none |

- Seeds are 17 to 21, in that order. Coverage is listed whether or not the seed passed.
  "none" means no threshold met the target at coverage 0.3 or more.
- Over 10 runs, layax passed all 10, the classifier 7 and FastFit 4.
- The classifier's three failures are narrow (for example realised 0.0497 against a bound
  of 0.0488), and at seeds 20 and 21 its coverage was level with layax's. The seeds share
  most of their test rows, so they are not independent trials. Read this as layax being
  the most reliable engine here, not as a large gap to the classifier.
- FastFit's MASSIVE-en failures are not narrow, and match its worse error ranking above.
- The layax Banking77 rows for seeds 17 to 19 come from separate trainings with the same
  settings as the publication runs. Seeds 20 and 21 are publication runs. CLINC150 was
  not certified, because its gate fails on the traffic shift described above whatever
  the engine.

### CPU latency

With 77 cached options, one question takes p50 155.7 ms and p95 168.8 ms on a 16-thread
AMD Zen 4 laptop CPU with fp32 PyTorch. The benchmark uses synthetic option text and one
fixed input.

```bash
python -m layax.cli latency --model-dir <run> --device cpu --n-options 77
```

## What did not work

- Labels not seen in training. On a checkpoint trained on one dataset and scored on
  another dataset's full label set, accuracy was 0.02 to 0.34, against chance of 0.007 to
  0.017. The best case was CLINC150 to MASSIVE-en at 0.32 to 0.34. CLINC150 to Banking77
  scored 0.24 to 0.26, and Banking77 to CLINC150 scored 0.03 to 0.10. A classifier scores
  0 here by construction, but this is still not usable. Command:
  `modal run scripts/modal_app.py::cross_schema`.
- Held-out labels within one dataset. With 17 of the 77 Banking77 labels held out of
  training (`configs/banking77_heldout_s20.json`), accuracy on those labels was 0.065 to
  0.078. When the held-out labels stay in training as options that are never correct,
  accuracy on them falls to 0.0136, which is chance.
- Rewording the option text. On the same test rows, a different rendering of the labels
  lowers accuracy by up to 9 to 12 points. On Banking77, raw snake_case label keys score
  0.82 to 0.86 against 0.91. On CLINC150, a prefix template scores 0.78 to 0.82 against
  0.88 to 0.90. A classifier does not read label text, so it is not affected. Use the exact
  label rendering the checkpoint was trained with. The same `cross_schema` command runs
  this test.
- The competence head. This is a learned predictor of whether the model is right, trained
  on deliberately shifted rows. It lost to plain max-softmax on AURC in every run. AURC
  (area under the risk and coverage curve) measures how well a score ranks the model's own
  errors. The head is still in the code but is off by default (`competence_head: false`).

## Quickstart

layax installs from source.

```bash
pip install -e ".[train,dev]"
python -m layax.cli smoke    # shape check, no network, about 1 s
pytest -q                    # offline test suite
```

Train on Modal, or on Kaggle 2×T4 with `notebooks/layax_kaggle_2xT4.ipynb`:

```bash
modal run scripts/modal_app.py::smoke
modal run --detach scripts/modal_app.py::main --config configs/banking77_pub.json
```

To train on your own rows, write one JSON object per line in the form
`{"state": ..., "label": "...", "criteria": {...}, "meta": {...}}`. See
`data/rows.example.jsonl`. Then run:

```bash
python -m layax.cli run --config configs/custom_jsonl.json --jsonl data/rows.jsonl
```

## Inference

`LayaxAgent.from_pretrained` takes a local run directory, or a Hugging Face repo id with
`subfolder=`. Published checkpoints ship their training schema as `schema.json`.

```python
from layax.runtime import LayaxAgent

agent = LayaxAgent.from_pretrained("runs/banking77", device="cuda")
# or a Hugging Face repo id: LayaxAgent.from_pretrained(repo_id, subfolder="banking77")
agent.warm_cache(questions)   # encodes every option once per schema
out = agent.predict(state, questions)

out["answers"]["intent"]["choice"]         # the chosen criteria key
out["answers"]["intent"]["probabilities"]  # temperature scaled
out["answers"]["intent"]["abstain"]        # top probability below the gate threshold
```

`questions` maps a question id to `{"type": "choice", "instructions": ..., "criteria":
{label_key: description}}`. `abstain` is absent when no threshold met the risk target. The
return shape matches `laya.Agent.predict`, so code written for Laya does not need to
change.

## Limits

- The risk bound covers only data exchangeable with the calibration split. It says
  nothing about another language, another script or a different out-of-scope share.
  Per-shift test risk is an empirical result and is never covered by the bound.
- Training scores 16 sampled options per row so the option tower fits in GPU memory.
  Evaluation always uses the full label set.
- The trainer uses one GPU.
- The `xattn` interaction has no option chunking, so a very wide label set with a long
  input can run out of memory. `maxsim` chunks at 256 options.
- GPU training is not bit-reproducible. Two trainings of Banking77 seed 17 scored 0.9063
  (gate threshold 0.727) and 0.9053 (threshold 0.808). The published checkpoint is the
  second. Expect gate coverage to vary by
  a few points between trainings of the same seed.
- All results are English intent classification. Nothing here is measured on other tasks.

## Prior art

layax applies established methods to Laya's typed decision heads. Neither the late
interaction head nor the gate is new.

- FastFit (Yehudai and Bendel, NAACL 2024 demo,
  [arXiv 2404.12365](https://arxiv.org/abs/2404.12365)) scores intents with MaxSim over
  label tokens. It is direct prior art for the `maxsim` head.
- Poly-encoders (Humeau et al., ICLR 2020,
  [arXiv 1905.01969](https://arxiv.org/abs/1905.01969)) are the family of the `xattn` head.
- Learn-then-Test (Angelopoulos et al. 2021,
  [arXiv 2110.01052](https://arxiv.org/abs/2110.01052)) is the threshold procedure. It is
  not conformal risk control, which bounds expected risk.
- Kamath, Jia and Liang (ACL 2020,
  [2020.acl-main.503](https://aclanthology.org/2020.acl-main.503)) train a separate error
  predictor for selective prediction under shift. It is the nearest precedent for the
  competence head.
- Xin et al. (ACL 2021, [2021.acl-long.84](https://aclanthology.org/2021.acl-long.84))
  study selective prediction with error regularization.
- Varshney et al. 2022 ([arXiv 2203.00211](https://arxiv.org/abs/2203.00211)) found that no
  selective prediction method consistently beat max-softmax across 17 NLP datasets. The
  competence head result here agrees with them.
- Casanueva et al. 2020 ([arXiv 2003.04807](https://arxiv.org/abs/2003.04807)) introduced
  Banking77 and report 93.66% for fine-tuned BERT-Large. That is a published number on a
  different encoder, not a layax result.
- Du et al. (AAAI 2023, [arXiv 2212.00301](https://arxiv.org/abs/2212.00301)) could not fit
  77 intents into an inline options model and used a top-k shortlist.

## License

Apache 2.0, matching upstream Laya. See `NOTICE` for attribution of Laya, ModernBERT and the
training datasets.
