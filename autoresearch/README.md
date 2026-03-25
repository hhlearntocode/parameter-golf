# Parameter Golf Autoresearch

This directory contains the minimal research scaffold for running traceable autoresearch on the `research` branch without polluting the final `records/...` submission.

## What Is Here

- `program.md`: the operating prompt for the research agent
- `results.tsv`: aggregate experiment table
- `runs/TEMPLATE_RUN/`: per-run folder shape
- `submission-template/`: clean submission metadata templates
- `export-template/`: example final folder shape for a clean export
- `verify_setup.py`: checks whether the scaffold and local data are ready
- `FEATURE_MATRIX.md`: verifies which core `autoresearch` features have been carried over

## Ready-To-Run Flow

Run these steps at the start of the next session:

1. Verify the scaffold and local data:

```bash
/Users/leonard/anaconda3/envs/env_ml/bin/python autoresearch/verify_setup.py
```

2. If data is missing, fetch it using the official repo path described in `data/README.md`.

Local smoke-sized example:

```bash
HF_HOME="$(pwd)/.hf-cache" /Users/leonard/anaconda3/envs/env_ml/bin/python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

Larger default challenge-style local setup:

```bash
HF_HOME="$(pwd)/.hf-cache" /Users/leonard/anaconda3/envs/env_ml/bin/python data/cached_challenge_fineweb.py --variant sp1024
```

3. Establish the baseline on the current research target:

```bash
RUN_ID=baseline_mlx \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
/Users/leonard/anaconda3/envs/env_ml/bin/python train_gpt_mlx.py
```

4. Create a real run folder by copying `runs/TEMPLATE_RUN/` to a new `run_id`.

5. Save the run log, metrics, decision, and patch into that folder, then append one row to `results.tsv`.

## Minimal Research Contract

The intended default local research surface is:
- `train_gpt_mlx.py`

The intended final clean submission surface is:
- a reconstructed folder under `records/...` containing only the exact code snapshot, logs, and metadata required by the challenge

## Notes

- The research scaffold is allowed to be noisy.
- The final submission must be clean.
- A good candidate run is not the same thing as a final submission.
