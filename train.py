"""
ESM-2 150M protein language model baseline.

Architecture: 30-layer bidirectional transformer with RoPE, Pre-LayerNorm,
GELU activation, and masked language modeling (15% masking).

Usage: python train.py
"""

import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, PAD_TOKEN,
    make_dataloader, evaluate_mlm_loss,
)

# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _build_rope_cache(head_dim: int, max_len: int, device) -> tuple:
    """Precompute (cos, sin) tables for RoPE. Returns (max_len, head_dim)."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t        = torch.arange(max_len, device=device).float()
    freqs    = torch.outer(t, inv_freq)           # (T, D/2)
    emb      = torch.cat([freqs, freqs], dim=-1)  # (T, D)
    return emb.cos(), emb.sin()


def _apply_rope(q, k, cos, sin):
    """Apply RoPE to query and key. q/k shape: (B, H, T, D)."""
    T   = q.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    q   = q * cos + _rotate_half(q) * sin
    k   = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.qkv  = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim,     bias=True)

    def forward(self, x, cos, sin, attn_bias=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k    = _apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class ESMBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim, bias=True),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim, bias=True),
        )

    def forward(self, x, cos, sin, attn_bias=None):
        x = x + self.attn(self.ln1(x), cos, sin, attn_bias)
        x = x + self.mlp(self.ln2(x))
        return x


class ESM2(nn.Module):
    """ESM-2 style masked language model (~148M parameters at default config)."""

    def __init__(
        self,
        vocab_size:       int = VOCAB_SIZE,
        hidden_dim:       int = 640,
        num_heads:        int = 20,
        num_layers:       int = 30,
        intermediate_dim: int = 2560,
        max_seq_len:      int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed  = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.blocks = nn.ModuleList([
            ESMBlock(hidden_dim, num_heads, intermediate_dim)
            for _ in range(num_layers)
        ])
        self.ln_f    = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        head_dim = hidden_dim // num_heads
        cos, sin = _build_rope_cache(head_dim, max_seq_len, device="cpu")
        self.register_buffer("rope_cos", cos)  # auto-moves with .to(device)
        self.register_buffer("rope_sin", sin)

    def forward(self, input_ids, labels=None, reduction="mean"):
        B, T = input_ids.shape

        # Float attention bias: 0 for real tokens, -inf for PAD keys
        pad_mask  = (input_ids == PAD_TOKEN)            # (B, T)
        attn_bias = input_ids.new_zeros(B, 1, 1, T, dtype=torch.float)
        attn_bias.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, attn_bias)
        x      = self.ln_f(x)
        logits = self.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            return loss
        return logits


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM       = 640
NUM_HEADS        = 20
NUM_LAYERS       = 30
INTERMEDIATE_DIM = 2560
BATCH_SIZE       = 32
LR               = 1e-4
WARMUP_STEPS     = 200

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda")

model = ESM2(
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_dim=INTERMEDIATE_DIM,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params / 1e6:.1f}M")

model = torch.compile(model)

optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
train_loader = make_dataloader(BATCH_SIZE, MAX_SEQ_LEN, "train")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

step                 = 0
total_training_time  = 0.0

while True:
    t0 = time.time()

    x, y, _ = next(train_loader)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # LR schedule: linear warmup → cosine decay over the training budget
    if step < WARMUP_STEPS:
        lr = LR * (step + 1) / WARMUP_STEPS
    else:
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lr       = LR * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 5:
        total_training_time += dt

    if step % 50 == 0:
        print(
            f"step {step:05d} | loss: {loss.item():.4f} | lr: {lr:.2e}"
            f" | dt: {dt*1000:.0f}ms | remaining: {max(0, TIME_BUDGET - total_training_time):.0f}s"
        )

    step += 1
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    val_loss = evaluate_mlm_loss(model, BATCH_SIZE)

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
