# TEMPLATE_RUN

This is a clean export template for a `parameter-golf` submission folder.

Use this folder as the final shape to reconstruct a real submission from research outputs. Do not submit this placeholder folder as-is.

## Track

`track_non_record_16mb`

Change this only if the final candidate belongs in `track_10min_16mb`.

## Summary

Replace this section with a concise explanation of:
- what changed
- why it helps
- what constraints it satisfies

Keep the writeup focused on the actual submission, not the full research history.

## Run Command

Replace with the exact command used for the reported run.

```bash
RUN_ID=replace_me \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- `val_bpb`: `REPLACE_ME`
- `val_loss`: `REPLACE_ME`
- `bytes_total`: `REPLACE_ME`
- `bytes_code`: `REPLACE_ME`
- `wallclock_seconds`: `REPLACE_ME`

## Reproducibility Notes

Document:
- whether the score is single-seed or multi-seed
- whether it is post-quant roundtrip
- whether evaluation differs from the baseline
- any extra dependency or helper file required by this folder

## Included Files

This folder should end up containing only the files needed for submission:
- `README.md`
- `submission.json`
- `train_gpt.py`
- one or more train logs
