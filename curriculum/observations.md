# Scientific Observations

## Phase 1: Training Dynamics

### Experiment Results
| # | Config | val_bpb | vs Baseline |
|---|--------|---------|-------------|
| 1 | Baseline (LR=3e-4, no clip, default betas) | 1.7998 | -- |
| 2 | LR=3e-3, clip=1.0, wd=0.1, betas=(0.9,0.95), min_lr=3e-4 | 1.2710 | -29.4% |

### Key Findings
1. **LR was drastically too low**: Default LR=3e-4 → 1.7998. LR=3e-3 (10x) → 1.2710.
2. **Gradient clipping essential**: clip=1.0 prevents divergence at high LR.
3. **Weight decay 0.1**: Standard for transformer pretraining, helps regularization.
4. **Optimizer betas (0.9, 0.95)**: Standard for LLM pretraining, less momentum on second moment.
5. **Min LR floor**: Cosine schedule with min_lr=3e-4 (10% of peak) prevents LR going to 0.

### Not Explored (budget exhausted)
- torch.compile (expected ~20% speedup → more training steps)
- LR=5e-3 or higher
- Batch size variations (32, 128)
- These should be explored in Phase 3 (Synthesis)

### Phase 1 Best Config (locked for Phase 2)
- LR=3e-3, grad_clip=1.0, weight_decay=0.1
- betas=(0.9, 0.95), min_lr=3e-4
- warmup_steps=100, batch_size=64
- val_bpb = 1.2710

## Phase 2: Architecture
*(Results will be added as experiments complete)*
