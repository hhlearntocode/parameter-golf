# autoresearch

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies the code, trains for the fixed challenge wallclock, checks whether the result improved, keeps or discards the change, and repeats. You wake up later to a log of experiments and, hopefully, a better model. This directory adapts that workflow to the CUDA path inside `parameter-golf`, centered on the challenge `train_gpt.py` script and an 8-GPU local `torchrun` loop. For the broader challenge rules, leaderboard, and submission format, see the main [`README.md`](../README.md).

## How it works

The setup is deliberately kept small and only really has three paths that matter:

- **`data/cached_challenge_fineweb.py`** - downloads the published FineWeb shards and tokenizer files used by the challenge.
- **`train_gpt.py`** - the single file the agent edits. It contains the full model, optimizer, distributed training loop, evaluation, and artifact-size checks. This is the default research surface for the CUDA path.
- **`autoresearch-8gpu/program.md`** - baseline instructions for one agent. Point your agent here and let it go.

By design, training runs for a **fixed 10-minute time budget** by default. The main metric is **`val_bpb`** (validation bits per byte). Lower is better, and it matches the challenge scoring target.

Research bookkeeping lives under `autoresearch-8gpu/runs/` and `autoresearch-8gpu/results.tsv`, but the spirit is the same as the original repo: the human edits the operating instructions, and the agent iterates on the one main training file.

## Quick start

**Requirements:** A machine with 8 visible NVIDIA GPUs, Python 3.10+, and a fresh virtual environment.

```bash
# 1. Create and activate an environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the challenge data and tokenizer files
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# 4. Verify the 8-GPU autoresearch scaffold
python3 autoresearch-8gpu/verify_setup.py

# 5. Manually run a baseline CUDA experiment
RUN_ID=baseline_cuda_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If the commands above work, your CUDA autoresearch path is ready.

For a larger training prefix, rerun the downloader without `--train-shards 1`. For dataset export and tokenizer rebuild details, see [`data/README.md`](../data/README.md).

## Running the agent

Spin up Claude, Codex, or your favorite coding agent in this repo and prompt it with something like:

```text
Hi, have a look at autoresearch-8gpu/program.md and let's kick off a new experiment. Let's do the setup first.
```

The `program.md` file is essentially a lightweight research skill, while `verify_setup.py`, `results.tsv`, and `runs/` provide the traceability layer around the original autoresearch loop.

## Project structure

```text
data/cached_challenge_fineweb.py   - challenge dataset downloader
train_gpt.py                       - CUDA model, optimizer, and training loop
autoresearch-8gpu/program.md       - agent instructions
autoresearch-8gpu/verify_setup.py  - scaffold + GPU + data checks
autoresearch-8gpu/results.tsv      - aggregate experiment log
autoresearch-8gpu/runs/            - per-run logs, metrics, and patches
requirements.txt                   - Python dependencies for the CUDA path
```

## Design choices

- **Single file to modify.** The agent should normally only touch `train_gpt.py`. This keeps diffs small and experiments attributable.
- **Fixed time budget.** Training defaults to the same 10-minute wallclock cap used by the challenge scripts. This makes iterations comparable under the real leaderboard constraint.
- **Hardware-locked comparisons.** This path assumes 8 visible GPUs and keeps the research loop aligned with `torchrun --standalone --nproc_per_node=8`, so results are compared under one consistent distributed setup.

## Platform support

This folder is the CUDA autoresearch path for `parameter-golf`. If you want the local Apple Silicon / MLX loop that targets `train_gpt_mlx.py`, use [`../autoresearch/`](../autoresearch/README.md) instead.

## License

MIT
