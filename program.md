# protein-lm-speedrun (autoresearch)

Train a protein language model to match ESM-2 150M's validation performance on
protein sequences, as fast as possible.

## Goal

**Minimize val_loss** (masked language modeling cross-entropy on held-out protein
sequences). Lower is better. The target is the converged MLM loss of an ESM-2
150M reference run.

## What you have

- `prepare.py` — **read-only**. Contains:
  - Fixed constants: `MAX_SEQ_LEN = 1024`, `TIME_BUDGET = 300` (5 min),
    `VOCAB_SIZE = 33` (ESM-2 amino acid vocabulary)
  - `Tokenizer` class — fixed amino acid tokenizer (no training needed)
  - `make_dataloader(B, T, split)` — returns `(input_ids, labels, attn_mask)`
    batches on CUDA. Labels are -100 for non-masked positions (MLM format).
  - `evaluate_mlm_loss(model, batch_size)` — fixed evaluation on the val set.
    Your model must accept `model(x, y, reduction='none')` and return a flat
    tensor of per-token cross-entropy losses (0 at non-masked positions).
  - Data is at `~/.cache/protein-lm-speedrun/`

- A **Modal API key** — use `modal` to provision GPU compute.

- `druids` — the orchestration client library.

## What you must build

Create or modify `train.py` that:

1. Imports from `prepare.py` for data loading and evaluation
2. Defines a protein language model architecture
3. Trains it within the 5-minute time budget
4. Evaluates using `evaluate_mlm_loss` and prints results in this format:

```
---
val_loss:         <float>
training_seconds: <float>
total_seconds:    <float>
peak_vram_mb:     <float>
num_params_M:     <float>
```

## Constraints

- **Do not modify `prepare.py`.** It is read-only infrastructure.
- Active parameter budget: **≤ 150M parameters**.
- Training time budget: **5 minutes wall clock** (`TIME_BUDGET` in `prepare.py`).
- No pretrained weights. No external data beyond what `prepare.py` provides.
- Model interface: `model(x, y, reduction='none')` returns flat per-token losses.

## Baseline architecture (train.py)

The reference baseline follows ESM-2 150M:

| Component          | Value                  |
|--------------------|------------------------|
| Parameters         | ~148M                  |
| Layers             | 30                     |
| Hidden dim         | 640                    |
| Attention heads    | 20                     |
| Intermediate dim   | 2560 (4× hidden)       |
| Positional enc.    | RoPE                   |
| Normalization      | Pre-LayerNorm          |
| Activation         | GELU                   |
| Vocabulary         | 33 tokens (ESM-2)      |
| Objective          | Masked LM (15% masked) |

## Compute

Use Modal to spin up GPU instances. The 5-minute budget is a wall-clock
constraint on the training loop itself (from first forward pass to the step
where budget is exceeded). Compilation and data loading count toward
`total_seconds` but not `training_seconds`.

## Results tracking

Log results to `results.tsv` (tab-separated, untracked by git):

```
commit	val_loss	memory_gb	status	description
```

- commit: git short hash (7 chars)
- val_loss: achieved metric (0.000000 for crashes)
- memory_gb: peak VRAM in GB (0.0 for crashes)
- status: `keep`, `discard`, or `crash`
- description: short text of what was tried

## The loop

LOOP FOREVER:

1. Look at current state: results.tsv, current train.py
2. Decide what to try. What might improve val_loss?
3. Modify train.py (architecture, optimizer, hyperparams, etc.)
4. Call `run_experiment` with a short description.
5. The tool commits, runs on Modal, parses metrics, keeps/discards.
6. Plan your next experiment.
7. Repeat.

## Research directions to explore

The interesting question is which NanoGPT speedrun techniques transfer to
protein sequences:

- **Optimizer**: Muon vs AdamW, Lion, schedule-free
- **Architecture**: QK-norm, RMSNorm vs LayerNorm, SwiGLU vs GELU
- **Attention**: RoPE variants, ALiBi, longer warmup schedules
- **MLM masking**: span masking, structure-aware masking, higher mask rate
- **Precision**: bf16, fp8 mixed, compile flags
- **Batch/LR**: gradient accumulation, warmup length, cosine vs linear decay
- **Data curriculum**: sequence length curriculum, amino acid frequency weighting

**NEVER STOP.** Continue experimenting until manually stopped or budget runs out.
