# Research Plan

## Phase 1: Establish Baseline
- Run train.py as-is to get baseline val_bpb
- Record VRAM usage and training throughput

## Phase 2: Low-hanging Fruit (parallel experiments)
1. **torch.compile** - Add `model = torch.compile(model)` for free speedup
2. **Gradient clipping** - Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. **Fix LR schedule** - Move LR update before optimizer.step()
4. **Weight decay 0.1** - Increase from default 0.01

## Phase 3: Architecture Improvements
5. **Larger model** - Increase n_embd=768, n_head=12, n_layer=8 (if compile gives enough speedup)
6. **SwiGLU activation** - Replace GELU MLP with SwiGLU for better quality
7. **RMSNorm** - Replace LayerNorm with RMSNorm (faster + modern)

## Phase 4: Training Improvements
8. **Higher learning rate** - Try 1e-3 or 6e-4
9. **Larger batch size** - Try 128 or 256 with gradient accumulation
10. **Muon optimizer** - Alternative to AdamW

## Phase 5: Combine Winners
- Stack all improvements that showed gains
- Fine-tune combined configuration

## Budget Strategy
- 4.0 GPU-hours = ~48 five-minute runs
- Use 2 assistants in parallel for throughput
- Prioritize changes most likely to help: compile, model size, architecture
