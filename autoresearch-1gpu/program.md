# Parameter Golf Autoresearch Program (1 GPU)

This file is the operating prompt for autonomous research on CUDA machines using `train_gpt.py` with 1 local NVIDIA GPU.

The goal is to run disciplined, traceable experiments in `parameter-golf` without polluting the final challenge submission. Research artifacts are allowed in this folder. Final submission artifacts must later be reconstructed cleanly from a fresh base.

## Setup

To set up a new research session on 1 GPU, do this first:

1. Read the in-scope files:
   - `autoresearch-1gpu/README.md`
   - `autoresearch-1gpu/program.md`
   - `train_gpt.py`
   - `data/README.md`
2. Verify the local scaffold, data, and the visible GPU:
   - Run `/Users/leonard/anaconda3/envs/env_ml/bin/python autoresearch-1gpu/verify_setup.py`
3. Confirm the default CUDA target:
   - `train_gpt.py`
4. Confirm the aggregate log exists:
   - `autoresearch-1gpu/results.tsv`
5. Confirm the per-run template exists:
   - `autoresearch-1gpu/runs/TEMPLATE_RUN/`
6. If the setup check fails because data is missing, use:
   - `HF_HOME="$(pwd)/.hf-cache" /Users/leonard/anaconda3/envs/env_ml/bin/python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`
7. Confirm this machine is intended to run the single-GPU local loop:
   - use `torchrun --standalone --nproc_per_node=1 train_gpt.py`
   - if no NVIDIA GPUs are visible, treat setup as incomplete for this folder

Once setup passes, begin experimentation.

Before starting the first experiment loop:

1. Ensure you are on the shared research branch:
   - `git checkout research`
2. Confirm it tracks the remote branch:
   - `git branch --set-upstream-to=origin/research research`
   - if the upstream is already configured, leave it alone
3. Treat `research` as the branch that should continuously advance when experiments win.
4. Do not create throwaway local-only commits that never get pushed if they are being kept as the new best state.

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
RUN_ID=baseline_cuda_1gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
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
- any throughput gain or constraint caused by single-GPU execution

If a change mainly improves the score because of evaluation geometry, note that explicitly. Do not describe it as a pure training/model improvement.

If a change mainly helps because the 1-GPU setup changes throughput, note that explicitly too. Do not describe it as a pure architecture win.

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
- `autoresearch-1gpu/runs/<run_id>/`

Per-run files:
- `manifest.json`
- `stdout.log`
- `metrics.json`
- `decision.txt`
- `patch.diff`

Aggregate table:
- `autoresearch-1gpu/results.tsv`

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

## Runtime Budget

Each experiment should take about 10 minutes total, plus a small amount of startup and evaluation overhead.

Rules:
- the intended wallclock per run is approximately 10 minutes or less
- if a run exceeds 10 minutes, kill it and treat it as a failure
- after a timeout failure, discard the experiment and revert to the previous kept state
- do not let a single bad run stall the loop

## 1-GPU Comparison Rules

Keep the hardware setup fixed across comparisons.

That means:
- use `torchrun --standalone --nproc_per_node=1` for the baseline and all direct follow-up comparisons
- do not compare a multi-GPU run against this folder's 1-GPU baseline as if they were the same setup
- if you temporarily debug with a different device setup, do not log that as a comparable result inside this 1-GPU loop
- remember that `train_gpt.py` derives `grad_accum_steps` from `WORLD_SIZE`, so the optimization geometry depends on the 1-GPU launch

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
- throughput tradeoffs under fixed wallclock with the 1-GPU setup

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
2. Confirm you are on `research` before making the next experiment commit.
3. Choose one narrow hypothesis from the allowed priority list.
4. Edit only `train_gpt.py` unless explicitly approved otherwise.
5. Save the exact diff for this run.
6. Run the experiment with a fresh `RUN_ID`.
7. Save the full stdout/stderr log to `autoresearch-1gpu/runs/<run_id>/stdout.log`.
8. Parse the final metrics and write `metrics.json`.
9. Write `decision.txt`.
10. Append one row to `autoresearch-1gpu/results.tsv`.
11. If the result is meaningfully better and reasonably clean, keep it and continue from there.
12. Immediately push the kept commit to `origin research` so the remote branch stays current.
13. If the result is equal or worse, discard it and continue from the previous best state.
14. If the run crashed, record the crash and either retry once for a trivial bug or move on.

Be persistent, but not reckless.

## Git Push Rules

The `research` branch is the authoritative advancing branch for this loop.

Rules:
- keep all experimental code commits on `research`
- after every kept win, run `git push origin research`
- if a losing experiment is reset away, do not push that losing state
- do not push `autoresearch-1gpu/results.tsv`; leave it untracked locally
- if a run crashes and you decide to keep a trivial bugfix before retrying, only push after there is a kept winning result
- do not rewrite published history unless the human explicitly asks for it

The remote branch should reflect the current best known code state, not every failed attempt.

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

If the crash is something dumb and easy to fix:
- for example: a typo, missing import, obvious shape bug, or a simple OOM-inducing mistake
- fix it and re-run once using your judgment

If the idea is fundamentally unstable or causes repeated failure:
- mark it `crash` in `autoresearch-1gpu/results.tsv`
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

Do not stop to ask for permission of create/cat or change file, folder within the project once the loop started.

Assume the human may be asleep or away from the computer.

Rules:
- do not ask for permission to continue once the loop is running
- do not stop just because you need a new idea; think harder and keep going
- if you run out of ideas, re-read the in-scope files, inspect prior records, combine near-misses, and try more radical but still disciplined changes
- the loop runs indefinitely until it is manually interrupted

If you run out of ideas:
- re-read `train_gpt.py`
- re-read `autoresearch-1gpu/program.md`
- re-read representative `records/...` submissions
- combine near-miss ideas
- prefer simple, high-signal changes before radical rewrites

## Current Intent

This folder is for traceable 1-GPU CUDA autoresearch.

Default target:
- `train_gpt.py`

Default long-term path:
- discover promising ideas on 1 local NVIDIA GPU
- identify clean candidates
- later port or reconstruct the winning idea into a clean submission path
