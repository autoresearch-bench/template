# Research Plan

## Goal
Minimize val_bpb for a 28.3M param GPT model on climbmix data, 300s training budget on H100.

## Phase 1: Training Dynamics (COMPLETE)
**Best config**: LR=3e-3, grad_clip=1.0, wd=0.1, betas=(0.9, 0.95), min_lr=3e-4
**Result**: val_bpb = 1.2512 (30% improvement over baseline 1.7937)

## Phase 2: Architecture (CURRENT)
**Strategy**: Using Phase 1 training config, explore architecture improvements one at a time.

### Experiment Queue (Priority Order)
1. **Weight tying** (tok_emb weights = lm_head weights) - reduces params, often helps small models
2. **SwiGLU activation** - replace GELU MLP with SwiGLU (needs FFN ratio adjustment to keep params similar)
3. **torch.compile** - add model compilation for speedup (more steps in budget)
4. **Depth/width trade-off** - try n_layer=8,n_embd=448 or n_layer=10,n_embd=384
5. **FFN ratio** - try 8/3 ratio with SwiGLU (standard practice)

### Key Hypotheses
- Weight tying saves ~4M params that can be redistributed, and regularizes the model
- SwiGLU consistently outperforms GELU in modern transformers
- torch.compile gives free throughput improvement
- Prior experience: SwiGLU + weight tying together reached 1.167

## Phase 3: Synthesis (Planned)
- Combine best Phase 1 training + best Phase 2 architecture
- Re-tune LR for new architecture (may need different optimal LR)
- Add torch.compile if not already tested

## Phase 4: Advanced (Planned)
- Muon optimizer
- QK-layernorm
- Exotic techniques
