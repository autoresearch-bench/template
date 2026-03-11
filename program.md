# autoresearch (from scratch)

This is an experiment to have LLM agents build and optimize a pretraining pipeline from scratch.

Unlike standard autoresearch, there is no starter `train.py`. You build everything yourself.

## Goal

**Minimize val_bpb** (bits per byte) on the validation set. This is the only metric that matters.

## What you have

- `prepare.py` — **read-only**. Contains:
  - Fixed constants: `MAX_SEQ_LEN = 2048`, `TIME_BUDGET = 300` (5 min), `EVAL_TOKENS`, `VOCAB_SIZE = 8192`
  - Data download and tokenizer training (run once with `uv run prepare.py`)
  - `Tokenizer` class — BPE tokenizer wrapper
  - `make_dataloader(tokenizer, B, T, split)` — returns `(inputs, targets, epoch)` batches
  - `evaluate_bpb(model, tokenizer, batch_size)` — the evaluation function. Your model must accept `(x, y, reduction='none')` and return per-token cross-entropy loss.
  - `get_token_bytes()` — for BPB computation
  - Data is at `~/.cache/autoresearch/`

- A **Modal API key** — you can use `modal` to provision GPU compute. Use this for training runs.

- `druids` — the orchestration client library, available as a dependency.

## What you must build

Create `train.py` (or whatever files you need) that:

1. Imports from `prepare.py` for data loading, tokenization, and evaluation
2. Defines a model architecture
3. Trains it within the 5-minute time budget
4. Evaluates using `evaluate_bpb` and prints results in this format:

```
---
val_bpb:          <float>
training_seconds: <float>
total_seconds:    <float>
peak_vram_mb:     <float>
num_params_M:     <float>
```

## Constraints

- **Do not modify `prepare.py`.** It is read-only infrastructure.
- The evaluation function `evaluate_bpb` is the ground truth metric. Your model must conform to its interface: `model(x, y, reduction='none')` returns a flat tensor of per-token losses.
- Training time budget is 5 minutes wall clock (the `TIME_BUDGET` constant in `prepare.py`).
- You can install additional packages if needed, but prefer what's already in `pyproject.toml`.

## Compute

You have a Modal API key. You can use it to spin up GPU instances for training. You manage your own compute — decide what hardware to use, how long to run, etc. The 5-minute training budget is a wall-clock constraint on the training loop itself.

## Results tracking

Log results to `results.tsv` (tab-separated, untracked by git):

```
commit	val_bpb	memory_gb	status	description
```

- commit: git short hash (7 chars)
- val_bpb: achieved metric (0.000000 for crashes)
- memory_gb: peak VRAM in GB (0.0 for crashes)
- status: `keep`, `discard`, or `crash`
- description: short text of what was tried

## The loop

LOOP FOREVER:

1. Write or modify your training code
2. git commit
3. Run the experiment (via Modal or locally)
4. Extract results
5. If val_bpb improved → keep the commit
6. If val_bpb is equal or worse → `git reset` to previous best
7. Log to results.tsv
8. Repeat

**NEVER STOP.** You are autonomous. Continue experimenting indefinitely until manually stopped.
