# layax — project brief for Claude Code

## What this is

Two model-layer extensions to Laya (https://github.com/NandhaKishorM/laya, Apache 2.0):
late-interaction option heads, and a Learn-then-Test abstention gate on max-softmax
(the learned competence head is an off-by-default ablation that lost to max-softmax).
Read `README.md` first, then `layax/li_head.py` and `layax/msp_gate.py` — the module
docstrings carry the reasoning, not just the API.

## Ground rules

**Do not invent benchmark numbers.** This package ships no measured accuracy claims.
Any number in a commit message, README edit or report must come from a run in that same
session, with the command that produced it. Comparison figures published upstream
are third-party, not measured — never restate them as this harness's results.

**Do not claim novelty without checking.** Late interaction (ColBERT; FastFit already
applies MaxSim to intent labels), learned error predictors under shift (Kamath et al.
2020), selective prediction, Learn-then-Test and OOD energy scores are all established.
The contribution is applying them to Laya's RLCD-trained typed heads. README "Prior art"
lists the sources. Search before writing "novel" anywhere.

**Call the threshold Learn-then-Test, not conformal.** It bounds P(selective risk ≤
target) ≥ 1 − delta, and only on data exchangeable with the calibration split. It says
nothing about an unseen script. Per-shift test risk is an empirical result, never covered.

**Report failures as results.** If the competence head does not beat softmax on AURC, or
no Learn-then-Test threshold meets the risk target, that is the finding. `fit_competence`
already writes a `verdict` field for exactly this — do not paper over it by loosening
`target_risk` until something passes.

**Evaluation uses the full label set.** Training subsamples options
(`train_options_per_row`); evaluation must not. If you touch `collect_predictions`, keep
that invariant or the headline number measures an easier task.

**Splits are load-bearing.** train → competence → calibration → test, in that order,
never overlapping. The competence head fitted on decision-training rows learns the
model's memorisation, and a threshold fitted on the competence split makes the risk bound
meaningless. See the docstring in `train_competence.py`.

## Working style

- Run `python -m layax.cli smoke` and `pytest -q` before and after any change to
  `li_head.py`, `competence.py` or `losses.py`. Both are offline and take seconds.
- New behaviour needs a test. The existing property tests are the model to follow:
  assert the thing that must be true, not that the code ran.
- State what you changed and what the result was. Not "improved the head" —
  "changed X; AURC went 0.14 → 0.11 on the calibration split; accuracy unchanged".
- Keep comments explaining *why*, and delete them when they stop being true.

## Pipeline

One entrypoint, `layax.pipeline.run`, used identically by Kaggle, Modal and local runs.
If something only happens in the notebook, it is a difference between what was measured
and what ships. Put it in the pipeline.

```
splits → train_li → fit_temperatures → fit_competence → evaluate → baselines → latency
```

## First tasks, in order

1. `python -m layax.cli smoke`, then `pytest -q`. Confirm all pass.
2. Small real run: `configs/banking77.json` with `max_train_rows: 2000`, one epoch, to
   confirm the checkpoint download, encoder init and training loop work end to end.
   `load_laya_weights` returns a report — check `randomly_initialised_modules` contains
   only the new projections, not `encoder`.
3. Full Banking77 run with baselines. The question being answered: does layax beat
   `laya`, `laya+shortlist(20)` and `laya+head_max_len=512` on the same test rows.
4. Read `verdict` and `aurc_improvement_calibration`. Decide whether the competence head
   is worth keeping on this data.
5. Ablate via `scripts/modal_app.py::sweep`. The `no-shift` arm is the one that says
   whether shift augmentation does anything.
6. Only then, own data: `configs/custom_jsonl.json` with your own rows.

## Things known to be unfinished

- `option_sampling="hard"` refreshes negatives once per epoch over at most 2000 rows.
  Fine for a first pass; it will need to scale for a large corpus.
- The trainer is single-GPU. Kaggle's second T4 runs baselines, not training. True
  multi-GPU means DDP via `torchrun`, not `DataParallel`.
- `evaluate_laya_baseline` counts upstream exceptions as errors and scores them 0. That
  is a real result for a configuration that cannot represent the label set, but the count
  must be reported next to the accuracy, never silently folded in.
- No hierarchical two-stage choice yet (category → subcategory). It is the obvious next
  feature for a large label taxonomy and the option cache makes the second stage cheap.
- The `xattn` path has no option-chunking, so a very wide label set with a long state can
  run out of memory. `maxsim` chunks at 256.
