# Autoresearch 1-GPU Feature Matrix

This file verifies that the essential workflow features from the original `autoresearch` setup have been copied into the separate `autoresearch-1gpu` scaffold, with CUDA-specific adaptations where needed.

## Core Setup Features

| Original autoresearch feature | Copied here? | Where it lives now |
|---|---|---|
| Dedicated research instructions | yes | `autoresearch-1gpu/program.md` |
| Explicit editable scope | yes | `autoresearch-1gpu/program.md` |
| Baseline-first policy | yes | `autoresearch-1gpu/program.md`, `autoresearch-1gpu/README.md` |
| Verify local data before first run | yes | `autoresearch-1gpu/verify_setup.py`, `autoresearch-1gpu/README.md` |
| Initialize aggregate results table | yes | `autoresearch-1gpu/results.tsv` |

## Experiment Loop Features

| Original autoresearch feature | Copied here? | Where it lives now |
|---|---|---|
| One main file as default edit surface | yes | `train_gpt.py`, documented in `autoresearch-1gpu/program.md` |
| One-experiment-at-a-time mindset | yes | `autoresearch-1gpu/program.md` |
| Keep / discard / crash decisions | yes | `autoresearch-1gpu/program.md`, `autoresearch-1gpu/runs/TEMPLATE_RUN/decision.txt` |
| Run-level logging | yes | `autoresearch-1gpu/runs/TEMPLATE_RUN/stdout.log` |
| Structured per-run metadata | yes | `autoresearch-1gpu/runs/TEMPLATE_RUN/manifest.json` |
| Structured metrics capture | yes | `autoresearch-1gpu/runs/TEMPLATE_RUN/metrics.json` |
| Patch capture for attribution | yes | `autoresearch-1gpu/runs/TEMPLATE_RUN/patch.diff` |
| Aggregate results row per run | yes | `autoresearch-1gpu/results.tsv` |
| Crash recording and retry policy | yes | `autoresearch-1gpu/program.md` |

## Submission Hygiene Features

| Original / desired feature | Copied here? | Where it lives now |
|---|---|---|
| Research branch may be messy | yes | `autoresearch-1gpu/program.md` |
| Final submission must be clean | yes | `autoresearch-1gpu/program.md`, `autoresearch-1gpu/submission-template/CHECKLIST.md` |
| Submission metadata template | yes | `autoresearch-1gpu/submission-template/submission.json` |
| Submission README template | yes | `autoresearch-1gpu/submission-template/README.md` |
| Clean export folder shape | yes | `autoresearch-1gpu/export-template/records/...` |
| Promotion from research to clean submission | yes | `autoresearch-1gpu/submission-template/CHECKLIST.md`, `autoresearch-1gpu/export-template/...` |

## Parameter Golf Adaptations

These are intentional differences from the tiny original repo:

- The default research target is `train_gpt.py`, not `train.py`.
- This scaffold is dedicated to `torchrun --standalone --nproc_per_node=1`.
- Data is fetched using `data/cached_challenge_fineweb.py`, not `prepare.py`.
- The local Python runtime is the conda environment `env_ml`, called directly as `/Users/leonard/anaconda3/envs/env_ml/bin/python`.
- Hugging Face cache is redirected to `.hf-cache/` inside the repo when needed to avoid sandbox and permission issues.
- Submission format is aligned to `parameter-golf/records/...`, not to the simpler TSV-only loop of the original repo.

## Verification Status

Current status after setup:

- scaffold copied: yes
- next session can start after 1-GPU verification: yes
