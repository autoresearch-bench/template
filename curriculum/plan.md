# Research Plan

## Objective
Minimize val_bpb on a 28.3M param GPT model trained for 300s on H100.

## Phase 1: Training Dynamics (COMPLETE)
**Best config**: LR=3e-3, grad_clip=1.0, wd=0.1, betas=(0.9, 0.95), min_lr=3e-4
**Result**: val_bpb = 1.2710 (29.4% improvement over baseline 1.7998)

## Phase 2: Architecture (NEXT)
Lock Phase 1 best training config, explore architecture improvements:

### Priority Order
1. **Weight tying** — share tok_emb and lm_head weights. Reduces params ~4M, acts as regularizer.
2. **SwiGLU activation** — replace GELU MLP with SwiGLU (8/3 hidden ratio to match param count).
3. **RMSNorm** — replace LayerNorm with RMSNorm (faster, works well in practice).
4. **RoPE** — replace learned positional embeddings with rotary position embeddings.
5. **torch.compile** — for throughput improvement (more steps in budget).
6. **Depth/width** — try deeper/narrower (n_layer=8, n_embd=448) or (n_layer=10, n_embd=384).
7. **Combined best** — all architecture improvements together.

### Expected Impact (from prior experience)
- SwiGLU + RMSNorm + RoPE + weight tying together reached ~1.167 previously.

## Phase 3: Synthesis (Planned)
- Combine best Phase 1 training + best Phase 2 architecture
- Re-optimize LR for new architecture
- Add torch.compile if not tested
- Try LR=5e-3, batch size variations

## Phase 4: Advanced (Planned)
- Muon optimizer, QK-layernorm, exotic techniques
