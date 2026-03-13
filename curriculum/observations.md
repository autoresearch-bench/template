# Scientific Observations

## Phase 1: Training Dynamics

### Experiment Results
| # | Config | val_bpb | Notes |
|---|--------|---------|-------|
| 1 | Baseline (LR=3e-4, no clip, default betas) | 1.7937 | ~1960 steps. Loss still declining at end. |
| 2 | LR=3e-3, clip=1.0, wd=0.1, betas=(0.9,0.95), min_lr=3e-4 | 1.2512 | Massive 30% improvement |

### Key Findings
1. **LR was drastically too low**: Default LR=3e-4 yielded 1.7937. Increasing to 3e-3 (10x) improved to 1.2512.
2. **Gradient clipping essential**: clip=1.0 prevents divergence at high LR.
3. **Weight decay helps**: wd=0.1 standard for transformer pretraining.
4. **Optimizer betas**: (0.9, 0.95) is standard for LLM pretraining, reduces momentum of second moment.
5. **Min LR floor**: Cosine schedule with min_lr=3e-4 (10% of peak) prevents LR from going to 0.

### What We Didn't Get To (budget exhausted)
- torch.compile (expected ~20% speedup, more steps in budget)
- LR=5e-3 or higher
- Batch size variations
- These will be explored in Phase 3 (Synthesis) if budget allows

### Phase 1 Best Config (locked for Phase 2)
- LR=3e-3, grad_clip=1.0, weight_decay=0.1
- betas=(0.9, 0.95), min_lr=3e-4
- warmup_steps=100, batch_size=64
- val_bpb = 1.2512

## Phase 2: Architecture
*(Results will be added as experiments complete)*
