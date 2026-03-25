# Submission Checklist

Use this checklist before copying a clean solution into `records/...`.

## Folder Placement

Choose the correct destination:
- `records/track_10min_16mb/<your-folder>/`
- `records/track_non_record_16mb/<your-folder>/`

The final pull request should only add one new folder under the appropriate `records/` track.

## Required Files

The submission folder must include:
- `README.md`
- `submission.json`
- `train_gpt.py`
- one or more train logs

If the solution needs extra files, they must be included and explained in `README.md`.

## Cleanliness Rules

Do not include research-only artifacts unless they are truly part of the submission:
- no `autoresearch/`
- no experiment manifests
- no scratch notes
- no temporary scripts
- no unrelated logs

The submission folder should contain only the files needed to explain and reproduce the result.

## `submission.json` Rules

Required in practice:
- `author`
- `github_id`
- `name`
- `blurb`
- `date`
- `val_bpb`
- `bytes_total`
- `bytes_code`

Commonly included and recommended:
- `val_loss`
- `val_loss_std`
- `val_bpb_std`
- `seeds`
- `seed_results`
- `pre_quant_val_loss`
- `pre_quant_val_bpb`
- `step_stop`
- `wallclock_seconds`
- `eval_time_seconds`

Size field note:
- if your artifact uses `int8+zlib`, a field like `bytes_model_int8_zlib` is fine
- if your artifact uses `int6+zstd`, rename accordingly, for example `bytes_model_int6_zstd`
- keep names honest and descriptive

## README Rules

Your `README.md` should clearly explain:
- what changed relative to a baseline
- why it helps
- the exact run command
- the reported metrics
- whether the result is single-seed or multi-seed
- whether the score is pre-quant or post-quant
- whether evaluation differs from baseline
- any dependency or reproduction caveat

## Reproducibility Rules

Before promoting a candidate to submission:
- make sure the script runs from inside the submission folder
- make sure logs are included
- make sure the recorded metrics match the logs
- make sure artifact bytes are accurate
- make sure the score belongs to the exact code snapshot being submitted

## Record Submission Rules

For new SOTA-style record submissions, also verify:
- the score beats the current SOTA by at least `0.005` nats
- there are enough logs to support the claimed significance
- the run reproducibly fits under 10 minutes on `8xH100`

## Non-Record Submission Rules

For non-record submissions:
- the idea should still be interesting, distinct, and justified
- the artifact must still satisfy the challenge's size spirit
- if compute exceeds the 10-minute record track, say so explicitly in `README.md`

## Final Promotion Rule

Do not submit directly from a messy research state.

Promote by:
1. starting from a clean base
2. copying only the actual winning solution
3. rebuilding the submission folder
4. verifying the included logs and metrics
