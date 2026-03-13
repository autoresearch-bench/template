# Research Plan

## Overall Goal
Minimize val_bpb on a ~28M-param GPT model trained for 300s on H100.

## Phase 1: Training Dynamics (Current)

### Strategy
The baseline LR=3e-4 is likely ~10-16x too low based on prior experience. We'll systematically optimize:

1. **Baseline** - Establish starting val_bpb with default config
2. **Learning Rate** - Test higher LRs: 1e-3, 3e-3, 5e-3, 8e-3
3. **Gradient Clipping** - Add grad_clip=1.0 (essential at high LR)
4. **Weight Decay** - Test wd=0.1 (standard for transformers)
5. **Optimizer Betas** - Try betas=(0.9, 0.95) (standard for pretraining)
6. **torch.compile** - Enable for ~20% speedup = more steps in budget
7. **Schedule** - Cosine with min_lr floor (10% of peak)
8. **Batch Size** - Test larger batches if memory allows

### Expected Progression
- Baseline: ~1.80 bpb
- Optimized LR alone: ~1.30-1.40 bpb
- Full optimization (LR + clip + wd + betas + compile + min_lr): ~1.20-1.25 bpb

## Phase 2: Architecture (Planned)
- SwiGLU activation, RMSNorm, RoPE, weight tying
- Depth/width trade-offs, FFN ratio adjustments

## Phase 3: Synthesis (Planned)
- Combine best Phase 1 training + best Phase 2 architecture
- Re-optimize LR for new architecture

## Phase 4: Advanced (Planned)
- Muon optimizer, QK-norm, exotic techniques
