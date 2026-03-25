# autoresearch-macos

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies the code, trains for the fixed challenge wallclock, checks whether the result improved, keeps or discards the change, and repeats. You wake up later to a log of experiments and, hopefully, a better model. This directory adapts that workflow to the Apple Silicon / MLX path inside `parameter-golf`, using the local `train_gpt_mlx.py` baseline instead of the CUDA `train_gpt.py` path. For the overall challenge context, rules, and submission format, see the main [`README.md`](../README.md).

## How it works

The setup is deliberately kept small and only really has three paths that matter:

- **`data/cached_challenge_fineweb.py`** - downloads the published FineWeb shards and tokenizer files used by the challenge.
- **`train_gpt_mlx.py`** - the single file the agent edits. It contains the MLX model, optimizer, data loading, evaluation, and training loop. This is the default research surface for the local macOS path.
- **`autoresearch/program.md`** - baseline instructions for one agent. Point your agent here and let it go.

By design, training runs for a **fixed 10-minute time budget** by default. The main metric is **`val_bpb`** (validation bits per byte). Lower is better, and it stays comparable across tokenizer choices in the same way the challenge leaderboard does.

Research bookkeeping lives under `autoresearch/runs/` and `autoresearch/results.tsv`, but the idea remains the same as the original repo: the human edits the operating instructions in `program.md`, while the agent iterates on the training file.

## Quick start

**Requirements:** Apple Silicon Mac with MLX support, Python 3.10+, and a fresh virtual environment.

```bash
# 1. Create and activate an environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. Install the local MLX path dependencies
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# 3. Download the challenge data and tokenizer files
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# 4. Verify the autoresearch scaffold
python3 autoresearch/verify_setup.py

# 5. Manually run a baseline local experiment
RUN_ID=baseline_mlx \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python3 train_gpt_mlx.py
```

If the commands above work, your local MLX autoresearch path is ready.

For a larger local dataset prefix, rerun the downloader without `--train-shards 1`. For dataset export and tokenizer rebuild details, see [`data/README.md`](../data/README.md).

## Running the agent

Spin up Claude, Codex, or your favorite coding agent in this repo and prompt it with something like:

```text
Hi, have a look at autoresearch/program.md and let's kick off a new experiment. Let's do the setup first.
```

The `program.md` file is essentially a lightweight research skill, while `verify_setup.py`, `results.tsv`, and `runs/` provide the traceability layer around the original autoresearch loop.

## Project structure

```text
data/cached_challenge_fineweb.py  - challenge dataset downloader
train_gpt_mlx.py                  - MLX model, optimizer, and training loop
autoresearch/program.md           - agent instructions
autoresearch/verify_setup.py      - scaffold + data checks
autoresearch/results.tsv          - aggregate experiment log
autoresearch/runs/                - per-run logs, metrics, and patches
```

## Design choices

- **Single file to modify.** The agent should normally only touch `train_gpt_mlx.py`. This keeps diffs small and experiments attributable.
- **Fixed time budget.** Training defaults to the same 10-minute wallclock cap used by the challenge scripts. This makes iterations comparable and pushes the agent toward ideas that actually help under a hard runtime budget.
- **Platform-local workflow.** This path is for local Apple Silicon iteration. It is intentionally separate from the CUDA path so local experiments do not get mixed with the 8-GPU loop.

## Platform support

This folder is the Apple Silicon / MLX autoresearch path for `parameter-golf`. If you want the CUDA research loop that targets `train_gpt.py` on 8 visible NVIDIA GPUs, use [`../autoresearch-8gpu/`](../autoresearch-8gpu/README.md) instead.

## License

MIT
