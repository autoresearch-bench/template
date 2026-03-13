# Scientific Observations

## Phase 1: Training Dynamics

### Baseline Analysis
- Model: GPT with 512 embd, 8 heads, 6 layers (~28M params)
- Default config: LR=3e-4, AdamW (default wd=0.01), batch_size=64, warmup=100
- Cosine decay to 0 (no min_lr floor), no gradient clipping, no torch.compile
- Training budget: 300s on H100 with bfloat16 autocast
- Context length: 2048, Vocab size: 8192
- LR schedule note: warmup is step-based but decay is time-based (uses total_training_time/TIME_BUDGET)

### Experiment Results
| # | Description | Key Changes | val_bpb | vs Best |
|---|-------------|-------------|---------|---------|
| *(awaiting results)* |||||

### Key Findings
*(will be updated as results arrive)*
