# Autoresearch Feature Matrix

This file verifies that the essential workflow features from the original `autoresearch` setup have been copied into the `parameter-golf` research scaffold, with local adaptations where needed.

## Core Setup Features

| Original autoresearch feature | Copied here? | Where it lives now |
|---|---|---|
| Dedicated research instructions | yes | `autoresearch/program.md` |
| Explicit editable scope | yes | `autoresearch/program.md` |
| Baseline-first policy | yes | `autoresearch/program.md`, `autoresearch/README.md` |
| Verify local data before first run | yes | `autoresearch/verify_setup.py`, `autoresearch/README.md` |
| Initialize aggregate results table | yes | `autoresearch/results.tsv` |

## Experiment Loop Features

| Original autoresearch feature | Copied here? | Where it lives now |
|---|---|---|
| One main file as default edit surface | yes | `train_gpt_mlx.py`, documented in `autoresearch/program.md` |
| One-experiment-at-a-time mindset | yes | `autoresearch/program.md` |
| Keep / discard / crash decisions | yes | `autoresearch/program.md`, `autoresearch/runs/TEMPLATE_RUN/decision.txt` |
| Run-level logging | yes | `autoresearch/runs/TEMPLATE_RUN/stdout.log` |
| Structured per-run metadata | yes | `autoresearch/runs/TEMPLATE_RUN/manifest.json` |
| Structured metrics capture | yes | `autoresearch/runs/TEMPLATE_RUN/metrics.json` |
| Patch capture for attribution | yes | `autoresearch/runs/TEMPLATE_RUN/patch.diff` |
| Aggregate results row per run | yes | `autoresearch/results.tsv` |
| Crash recording and retry policy | yes | `autoresearch/program.md` |

## Submission Hygiene Features

| Original / desired feature | Copied here? | Where it lives now |
|---|---|---|
| Research branch may be messy | yes | `autoresearch/program.md` |
| Final submission must be clean | yes | `autoresearch/program.md`, `autoresearch/submission-template/CHECKLIST.md` |
| Submission metadata template | yes | `autoresearch/submission-template/submission.json` |
| Submission README template | yes | `autoresearch/submission-template/README.md` |
| Clean export folder shape | yes | `autoresearch/export-template/records/...` |
| Promotion from research to clean submission | yes | `autoresearch/submission-template/CHECKLIST.md`, `autoresearch/export-template/...` |

## Parameter Golf Adaptations

These are intentional differences from the tiny original repo:

- The default local research target is `train_gpt_mlx.py`, not `train.py`.
- Data is fetched using `data/cached_challenge_fineweb.py`, not `prepare.py`.
- The local Python runtime is the conda environment `env_ml`, called directly as `/Users/leonard/anaconda3/envs/env_ml/bin/python`.
- Hugging Face cache is redirected to `.hf-cache/` inside the repo when needed to avoid sandbox and permission issues.
- Submission format is aligned to `parameter-golf/records/...`, not to the simpler TSV-only loop of the original repo.

## Verification Status

Current status after setup:

- scaffold copied: yes
- smoke dataset present: yes
- tokenizer present: yes
- next session can start from baseline run: yes
