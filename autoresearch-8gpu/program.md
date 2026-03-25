# Parameter Golf Autoresearch Program (8 GPU)

This file is the operating prompt for autonomous research on CUDA machines using `train_gpt.py` with 8 local NVIDIA GPUs.

The goal is to run disciplined, traceable experiments in `parameter-golf` without polluting the final challenge submission. Research artifacts are allowed in this folder. Final submission artifacts must later be reconstructed cleanly from a fresh base.

## Setup

To set up a new research session on 8 GPUs, do this first:

1. Read the in-scope files:
   - `autoresearch-8gpu/README.md`
   - `autoresearch-8gpu/program.md`
   - `train_gpt.py`
   - `data/README.md`
2. Verify the local scaffold, data, and visible GPUs:
   - Run `/Users/leonard/anaconda3/envs/env_ml/bin/python autoresearch-8gpu/verify_setup.py`
3. Confirm the default CUDA target:
   - `train_gpt.py`
4. Confirm the aggregate log exists:
   - `autoresearch-8gpu/results.tsv`
5. Confirm the per-run template exists:
   - `autoresearch-8gpu/runs/TEMPLATE_RUN/`
6. If the setup check fails because data is missing, use:
   - `HF_HOME="$(pwd)/.hf-cache" /Users/leonard/anaconda3/envs/env_ml/bin/python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`
7. Confirm this machine is intended to run 8-way local DDP:
   - use `torchrun --standalone --nproc_per_node=8 train_gpt.py`
   - if fewer than 8 GPUs are visible, treat setup as incomplete for this folder

Once setup passes, begin experimentation.

## What You Can Edit

Default research target:
- `train_gpt.py`

Do not edit unless the human explicitly approves:
- `train_gpt_mlx.py`
- anything under `data/`
- anything under `records/`
- `README.md`
- `requirements.txt`
- challenge docs or repo metadata

The point is to keep the solution surface narrow so improvements stay attributable and portable.

## Baseline First

The first meaningful run must establish a baseline before any optimization claims are made.

Baseline command:

```bash
RUN_ID=baseline_cuda_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For the first run:
1. Run the baseline command.
2. Save the stdout/stderr log.
3. Record the metrics.
4. Mark the run as `baseline`.
5. Only then begin iterative changes.

If there is no trustworthy baseline, do not claim a win.

## Objective

Primary objective:
- improve challenge-relevant `val_bpb`

Secondary objectives:
- stay within the artifact budget mindset
- keep code understandable
- preserve a clean path to final submission export
- make every experiment traceable and reproducible

## Metric Discipline

Do not confuse model improvement with evaluation improvement.

Track these separately whenever relevant:
- pre-quant quality
- post-quant quality
- artifact size
- any eval-procedure-specific gain
- any throughput gain from 8-GPU scaling

If a change mainly improves the score because of evaluation geometry, note that explicitly. Do not describe it as a pure training/model improvement.

If a change mainly helps because the 8-GPU setup changes throughput, note that explicitly too. Do not describe it as a pure architecture win.

## Required Traceability

Every run must have a stable `run_id`.

For every experiment, record:
- `run_id`
- base commit
- head commit
- target file changed
- GPU count used
- short hypothesis
- exact run command
- relevant env vars
- key metrics
- keep/discard/crash decision
- brief conclusion

Per-run folder:
- `autoresearch-8gpu/runs/<run_id>/`

Per-run files:
- `manifest.json`
- `stdout.log`
- `metrics.json`
- `decision.txt`
- `patch.diff`

Aggregate table:
- `autoresearch-8gpu/results.tsv`

TSV header:

```tsv
run_id	base_commit	head_commit	target_file	val_bpb	artifact_bytes	status	description
```

## Log Parsing

After each run, read the run log and extract the key output lines.

For `train_gpt.py`, the important lines include:
- `world_size:... grad_accum_steps:...`
- `train_loader:dataset:... train_shards:...`
- `step:... train_loss:...`
- `step:... val_loss:... val_bpb:...`
- `stopping_early: wallclock_cap ...`
- `serialized_model_int8_zlib:... bytes`
- `final_int8_zlib_roundtrip val_loss:... val_bpb:... eval_time:...`
- `final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...`
- `peak memory allocated: ... reserved: ...`

The main score to record is the final post-quant roundtrip `val_bpb`.

If the expected final lines are missing, treat the run as a crash unless the log clearly shows a completed alternative path.

## 8-GPU Comparison Rules

Keep the hardware setup fixed across comparisons.

That means:
- use `torchrun --standalone --nproc_per_node=8` for the baseline and all direct follow-up comparisons
- do not compare a 1-GPU or 4-GPU run against this folder's 8-GPU baseline as if they were the same setup
- if you temporarily debug on fewer GPUs, do not log that as a comparable result inside this 8-GPU loop
- remember that `train_gpt.py` derives `grad_accum_steps` from `WORLD_SIZE`, so the optimization geometry depends on the 8-GPU launch

## Experiment Design Rules

Each experiment should test one main idea.

Good experiments:
- one clear hypothesis
- one main change family
- easy to compare against the previous best

Bad experiments:
- many unrelated changes bundled together
- refactors with no measurable goal
- broad rewrites that make attribution impossible

When in doubt:
- choose the smaller, cleaner experiment

## Research Priorities

Prioritize search directions that already show signal in `records/`.

High-priority search axes:
- quantization-aware training and post-quant robustness
- artifact-size vs capacity tradeoffs
- model capacity reinvestment: depth, `MLP_MULT`, selected precision carve-outs
- optimizer and schedule tuning: Muon settings, weight decay, warmdown, momentum warmup, SWA
- cheap context features such as bigram-style input features or smear-style token mixing
- throughput tradeoffs under fixed wallclock with the 8-GPU setup

Lower-priority or deferred axes:
- tokenizer changes
- dataset pipeline changes
- large multi-file refactors
- complex test-time training
- LoRA-style adaptation
- eval-only tricks as the first line of attack

## The Experiment Loop

Once the baseline exists, loop forever:

1. Look at the current git state and identify the current best baseline or candidate.
2. Choose one narrow hypothesis from the allowed priority list.
3. Edit only `train_gpt.py` unless explicitly approved otherwise.
4. Save the exact diff for this run.
5. Run the experiment with a fresh `RUN_ID`.
6. Save the full stdout/stderr log to `autoresearch-8gpu/runs/<run_id>/stdout.log`.
7. Parse the final metrics and write `metrics.json`.
8. Write `decision.txt`.
9. Append one row to `autoresearch-8gpu/results.tsv`.
10. If the result is meaningfully better and reasonably clean, keep it and continue from there.
11. If the result is equal or worse, discard it and continue from the previous best state.
12. If the run crashed, record the crash and either retry once for a trivial bug or move on.

Be persistent, but not reckless.

## Candidate Promotion Rules

Mark a run as `candidate` only if it is meaningfully better and reasonably clean.

A candidate should usually satisfy most of:
- better challenge-relevant score
- no obvious artifact-budget regression
- no major complexity blow-up
- understandable implementation
- plausible portability to a clean submission version

Do not promote messy hacks just because they barely win.

## Crash Handling

If a run crashes:
- record it
- save the command and traceback
- classify whether the failure was implementation-related or idea-related

If the bug is trivial:
- fix it and retry once

If the idea is fundamentally unstable or causes repeated failure:
- mark it `crash` or `discard`
- move on

Do not get stuck on one bad direction.

## Simplicity Bias

All else equal, prefer:
- fewer moving parts
- smaller diffs
- ideas that can be cleanly ported into a final submission

A tiny gain from a complicated hack is often worse than a slightly smaller gain from a clean and portable idea.

## Research vs Submission

This folder is a research workspace, not the final submission state.

Rules:
- You may create research artifacts such as logs, notes, manifests, and summaries in this folder.
- You must not treat research scaffolding as part of the final solution.
- A promising result is only a `candidate`, not a final submission.
- Final submission must later be rebuilt from a clean checkout and must include only the actual solution files required by the challenge.

## Clean Submission Constraint

Always think ahead to the final clean export.

That means:
- do not rely on research-only helpers inside the final solution path
- do not spread important logic across research notes or ad hoc scripts
- keep the actual model/training idea localized to the target training file whenever possible

Assume that later a human will rebuild the final submission from a clean branch using only the winning idea.

## What Not To Do

Do not:
- change many files at once
- mutate the data pipeline without approval
- write directly into `records/` during research
- claim leaderboard-style progress without separating eval effects from model effects
- treat this research folder as the final submission

## Never Stop

Once the experiment loop has begun, do not pause to ask the human whether you should continue. Do not ask whether this is a good stopping point. Continue iterating until the human interrupts you.

If you run out of ideas:
- re-read `train_gpt.py`
- re-read `autoresearch-8gpu/program.md`
- re-read representative `records/...` submissions
- combine near-miss ideas
- prefer simple, high-signal changes before radical rewrites

## Current Intent

This folder is for traceable 8-GPU CUDA autoresearch.

Default target:
- `train_gpt.py`

Default long-term path:
- discover promising ideas on 8 local NVIDIA GPUs
- identify clean candidates
- later port or reconstruct the winning idea into a clean submission path
