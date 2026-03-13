# Observations

## Baseline Analysis
- Model: GPT, 512 embed, 8 heads, 6 layers (~30M params)
- MLP: 4x expansion with GELU activation
- Training: AdamW LR=3e-4, batch_size=64, warmup=100 steps, cosine decay
- Context: 2048 tokens, vocab=8192 (BPE)
- Time budget: 300s on H100, bfloat16 autocast
- No gradient clipping, no explicit weight decay config
- LR schedule applied AFTER optimizer.step() (takes effect next step)
- No torch.compile, no gradient accumulation

## Potential Improvements Identified
1. **torch.compile** - Could significantly speed up training, allowing more steps in 5 min
2. **Gradient clipping** - Standard practice, may stabilize training
3. **Larger model** - With compile speedup, may fit more params in time budget
4. **Learning rate tuning** - 3e-4 is standard but may not be optimal
5. **Weight decay tuning** - Default 0.01, could try 0.1
6. **Architecture tweaks** - SwiGLU instead of GELU, RMSNorm instead of LayerNorm
7. **Batch size increase** - H100 has 80GB VRAM, likely underutilized
8. **Warmup length** - 100 steps may be too many/few
9. **Fix LR schedule** - Apply LR before optimizer step, not after
10. **Muon optimizer** - May converge faster than AdamW

## Experiment Results
(Pending baseline run)
