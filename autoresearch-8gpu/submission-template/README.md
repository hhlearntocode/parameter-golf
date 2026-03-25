# Submission Title

Brief one-line summary of the submission and its score.

Example:
- `val_bpb: 1.1458` (mean of 3 seeds, post-quant roundtrip)

## Track

Choose one:
- `track_10min_16mb`
- `track_non_record_16mb`

If this is a non-record or unlimited-compute run, say so explicitly here.

## Summary

Describe the main idea in 1-3 short paragraphs:
- what changed
- why it should help
- what tradeoff it makes

Keep this focused on the actual solution, not on research history.

## Run Command

Provide the exact command used to reproduce the run.

```bash
# Setup if needed
# bash prepare.sh

# Exact training/eval command
RUN_ID=your_run_id \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If defaults are baked into `train_gpt.py`, say that explicitly.

## Key Metrics

- Mean `val_bpb`: `X.XXXX`
- Mean `val_loss`: `X.XXXX`
- Artifact size: `X` bytes
- Code size: `X` bytes
- Training wallclock: `X` seconds
- Eval time: `X` seconds

If you have multi-seed results, include standard deviation too.

## Seed Results

Use this table if the submission is based on multiple runs.

| Seed | val_loss | val_bpb | artifact_bytes | valid |
|------|----------|---------|----------------|-------|
| 42 | X | X | X | yes |
| 1337 | X | X | X | yes |
| 2024 | X | X | X | yes |
| Mean | X | X |  |  |
| Std | X | X |  |  |

## Technique Details

List the techniques that matter most. Keep it readable and concrete.

Examples:
- quantization scheme
- compression codec
- model depth / width / MLP ratio
- special input/context feature
- optimizer and schedule changes
- eval procedure if it differs from baseline

## Hyperparameters

Document the important hyperparameters used for the reported run.

Examples:
- `num_layers`
- `model_dim`
- `num_heads`
- `num_kv_heads`
- `mlp_mult`
- `train_seq_len`
- `train_batch_tokens`
- `warmdown_iters`
- `matrix_lr`
- `scalar_lr`
- `tied_embed_lr`
- `grad_clip_norm`
- quantization/compression settings

## Reproducibility Notes

State anything a reviewer would need to know:
- whether the score is mean over seeds
- whether this is pre-quant or post-quant
- whether evaluation uses a modified procedure
- whether the run fits the official 10-minute constraint
- any extra dependency or file needed beyond `train_gpt.py`

## Included Files

At minimum, the submission folder should contain:
- `README.md`
- `submission.json`
- `train_gpt.py`
- one or more train logs

If you need more files, explain exactly why they are required.
