# Scientific Observations

## Phase 1: Training Dynamics

### Baseline Analysis
- Model: GPT with 512 embd, 8 heads, 6 layers (~28M params)
- Default config: LR=3e-4, AdamW (default wd=0.01), batch_size=64, warmup=100
- Cosine decay to 0 (no min_lr floor), no gradient clipping, no torch.compile
- Training budget: 300s on H100 with bfloat16 autocast
- Context length: 2048, Vocab size: 8192

### Experiment Results
| # | Description | Key Changes | val_bpb | vs Baseline |
|---|-------------|-------------|---------|-------------|
| 1 | Baseline | LR=3e-4, no clip, default betas/wd | 1.7876 | -- |
| 2 | High LR + stability | LR=3e-3, clip=1.0, wd=0.1, betas=(0.9,0.95), min_lr=3e-4 | 1.2632 | -29.3% |
| 3 | Higher LR | LR=5e-3, clip=1.0, wd=0.1, betas=(0.9,0.95), min_lr=5e-4 | 1.2411 | -30.6% |
| 4 | torch.compile | LR=3e-3 + torch.compile | 1.2828 | -28.2% |
| 5 | LR=8e-3 | LR=8e-3, clip=1.0, wd=0.1, min_lr=8e-4 | 1.2546 | -29.8% |
| 6 | Batch=128 | LR=5e-3, batch=128 | 1.3051 | -27.0% |
| 7 | **LR=6e-3** | **LR=6e-3, clip=1.0, wd=0.1, min_lr=6e-4** | **1.2231** | **-31.6%** |
| 8 | Batch=32 | LR=5e-3, batch=32, warmup=200 | 1.2278 | -31.3% |
| 9 | LR=7e-3 | LR=7e-3, diverged step 1450 | 1.5866 | -11.2% |
| 10 | Batch=32 + LR=6e-3 | LR too high for small batch | 1.3240 | -25.9% |
| 11 | Warmup=50 | Too fast warmup | 1.2780 | -28.5% |

### Key Findings
1. **LR is the dominant factor**: 20x increase (3e-4 → 6e-3) yielded 31.6% improvement.
2. **Optimal LR is 6e-3**: LR=7e-3 diverged, LR=8e-3 noisy. Peak at 6e-3.
3. **Stability measures essential**: grad_clip=1.0, wd=0.1, betas=(0.9,0.95) enable high LR.
4. **torch.compile hurts**: 15.9s overhead not amortized in 300s budget.
5. **Batch=64 optimal**: Larger batches = fewer steps, smaller = too noisy at high LR.
6. **LR-batch coupling**: Optimal LR scales with batch size.

### Phase 1 Best Config
- LR=6e-3, grad_clip=1.0, weight_decay=0.1, betas=(0.9, 0.95), min_lr=6e-4
- warmup_steps=100, batch_size=64 → **val_bpb = 1.2231**

## Phase 2: Architecture

### Experiment Results
| # | Description | Key Changes | val_bpb | vs P1 Best |
|---|-------------|-------------|---------|------------|
| 1 | Weight tying (naive) | Share tok_emb/lm_head | 1.6266 | -33.0% |
| 2 | SwiGLU (wrong config) | h=1344 + compile | 1.4007 | -14.5% |
| 3 | Weight tying + scaling | logits/sqrt(n_embd) | 1.4268 | -16.6% |
| 4 | SwiGLU v2 | h=1408, no compile | 1.2714 | -3.9% |
| 5 | RMSNorm (contaminated) | + accidental weight tying | 1.8181 | -48.6% |
| 6 | RMSNorm (clean) | RMSNorm only | 1.2728 | -4.1% |
| 7 | **RoPE** | Rotary position embeddings | **1.1830** | **+3.3%** |
| 8 | **RoPE + 8L/448** | RoPE + deeper/narrower | **1.1733** | **+4.1%** |
| 9 | RoPE + SwiGLU | RoPE + SwiGLU h=1088 | 1.1857 | +3.1% |
| 10 | **RoPE + 10L/384** | **RoPE + even deeper** | **1.1672** | **+4.6%** |

### Key Findings
1. **RoPE is the breakthrough**: Only arch change that improved val_bpb. Compute-neutral, removes 1M params.
2. **Throughput is king**: SwiGLU (-3.9%) and RMSNorm (-4.1%) both hurt due to slower per-step time.
3. **Depth + RoPE trend**: 6L→1.183, 8L→1.173, 10L→1.167. More depth consistently helps with RoPE.
4. **Fewer params + more depth = better**: 24M params at 10 layers beats 28M at 6 layers.
5. **Weight tying broken**: Init scale mismatch causes catastrophic failure. Needs training config adjustment.
6. **SwiGLU doesn't help**: Both alone and combined with RoPE, SwiGLU's throughput cost exceeds quality gain.

### Phase 2 Best Config
- RoPE, n_layer=10, n_embd=384, n_head=6 (24M params) → **val_bpb = 1.1672**

## Phase 3: Synthesis
*(Results will be added as experiments complete)*
