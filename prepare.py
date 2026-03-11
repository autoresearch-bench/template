"""
One-time data preparation for the Protein LM Speedrun.

Generates synthetic protein sequences (dry run) or downloads real data.

Usage:
    python prepare.py                         # 100k synthetic train + 5k val
    python prepare.py --num-train 200000      # larger synthetic set
    python prepare.py --seed 123              # different random seed

Data is stored in ~/.cache/protein-lm-speedrun/.
"""

import os
import sys
import time
import argparse

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN     = 1024   # context window (includes <cls> and <eos>)
TIME_BUDGET     = 300    # training time budget in seconds (5 min)
VOCAB_SIZE      = 33     # ESM-2 amino acid vocabulary size
MIN_PROTEIN_LEN = 30     # filter: discard sequences shorter than this
MAX_PROTEIN_LEN = 1022   # filter: discard sequences longer (leaves room for specials)
MLM_MASK_PROB   = 0.15   # fraction of tokens selected for masking
EVAL_BATCHES    = 100    # number of batches in the fixed validation evaluation

# ---------------------------------------------------------------------------
# Vocabulary (ESM-2 compatible, 33 tokens)
# ---------------------------------------------------------------------------

VOCAB = {
    "<cls>":    0,  "<pad>":    1,  "<eos>":    2,  "<unk>":    3,
    "L":  4,  "A":  5,  "G":  6,  "V":  7,  "S":  8,  "E":  9,  "R": 10,
    "T": 11,  "I": 12,  "D": 13,  "P": 14,  "K": 15,  "Q": 16,  "N": 17,
    "F": 18,  "Y": 19,  "M": 20,  "H": 21,  "W": 22,  "C": 23,  "X": 24,
    "B": 25,  "U": 26,  "Z": 27,  "O": 28,  ".": 29,  "-": 30,
    "<null_1>": 31, "<mask>": 32,
}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

CLS_TOKEN  = VOCAB["<cls>"]   # 0
PAD_TOKEN  = VOCAB["<pad>"]   # 1
EOS_TOKEN  = VOCAB["<eos>"]   # 2
UNK_TOKEN  = VOCAB["<unk>"]   # 3
MASK_TOKEN = VOCAB["<mask>"]  # 32

# Standard amino acid token IDs — used for random-token replacement in MLM
_AA_TOKEN_IDS = np.array([VOCAB[aa] for aa in "LAGVSERTIDPKQNFYMHWC"], dtype=np.uint8)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "protein-lm-speedrun")
DATA_DIR  = os.path.join(CACHE_DIR, "data")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal protein sequence tokenizer. Fixed ESM-2 compatible vocabulary."""

    def encode(self, seq: str) -> list:
        """Encode a protein sequence string to token IDs (includes <cls> and <eos>)."""
        tokens = [CLS_TOKEN]
        for aa in seq.upper():
            tokens.append(VOCAB.get(aa, UNK_TOKEN))
        tokens.append(EOS_TOKEN)
        return tokens

    def decode(self, ids) -> str:
        """Decode token IDs back to amino acid string (strips special tokens)."""
        skip = {CLS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN, 31}
        return "".join(ID_TO_TOKEN.get(i, "?") for i in ids if i not in skip)

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE


# ---------------------------------------------------------------------------
# Data generation (synthetic — for dry run and testing)
# ---------------------------------------------------------------------------

def _generate_synthetic_sequences(n: int, rng: np.random.Generator) -> list:
    """Generate n random protein sequences of valid lengths."""
    aa_chars = list("LAGVSERTIDPKQNFYMHWC")
    seqs = []
    for _ in range(n):
        length = int(rng.integers(MIN_PROTEIN_LEN, MAX_PROTEIN_LEN + 1))
        seq = "".join(rng.choice(aa_chars, size=length))
        seqs.append(seq)
    return seqs


def _save_split(seqs: list, split: str, tokenizer: Tokenizer):
    """Tokenize sequences and save as flat binary file + offset index."""
    flat = []
    offsets = []
    pos = 0
    for seq in seqs:
        tokens = tokenizer.encode(seq)
        flat.extend(tokens)
        offsets.append([pos, len(tokens)])
        pos += len(tokens)

    flat_arr    = np.array(flat,    dtype=np.uint8)
    offsets_arr = np.array(offsets, dtype=np.int64)

    flat_arr.tofile(os.path.join(DATA_DIR, f"{split}_data.bin"))
    np.save(os.path.join(DATA_DIR, f"{split}_offsets.npy"), offsets_arr)
    print(f"  {split}: {len(seqs):,} sequences, {len(flat_arr):,} tokens")


def prepare_data(n_train: int = 100_000, n_val: int = 5_000, seed: int = 42):
    """Generate synthetic protein sequences and save to disk (one-time setup)."""
    os.makedirs(DATA_DIR, exist_ok=True)

    needed = [f"{s}_{t}" for s in ("train", "val") for t in ("data.bin", "offsets.npy")]
    if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in needed):
        sizes = [
            os.path.getsize(os.path.join(DATA_DIR, f"{s}_data.bin"))
            for s in ("train", "val")
        ]
        print(f"Data already prepared at {DATA_DIR} "
              f"(train {sizes[0]/1e6:.1f}MB, val {sizes[1]/1e6:.1f}MB)")
        return

    rng = np.random.default_rng(seed)
    tok = Tokenizer()

    print(f"Generating {n_train:,} train + {n_val:,} val synthetic sequences...")
    t0 = time.time()
    _save_split(_generate_synthetic_sequences(n_train, rng), "train", tok)
    _save_split(_generate_synthetic_sequences(n_val,   rng), "val",   tok)
    print(f"Done in {time.time() - t0:.1f}s. Saved to {DATA_DIR}")


# ---------------------------------------------------------------------------
# MLM masking (applied on-the-fly during training/eval)
# ---------------------------------------------------------------------------

def _apply_mlm(input_ids: np.ndarray, rng: np.random.Generator):
    """Apply BERT-style MLM masking. Returns (masked_ids, labels).

    Labels: original token id at masked positions, -100 everywhere else.
    Of masked tokens: 80% → <mask>, 10% → random amino acid, 10% → unchanged.
    Special tokens (<cls>, <pad>, <eos>, <unk>) are never masked.
    """
    B, T = input_ids.shape
    labels = np.full((B, T), -100, dtype=np.int64)

    maskable = input_ids >= 4  # skip CLS(0), PAD(1), EOS(2), UNK(3)
    coin      = rng.random((B, T))
    is_masked = maskable & (coin < MLM_MASK_PROB)

    labels[is_masked] = input_ids[is_masked]

    r              = rng.random((B, T))
    replace_mask   = is_masked & (r < 0.80)
    replace_random = is_masked & (r >= 0.80) & (r < 0.90)
    # 10% keep-unchanged: is_masked & (r >= 0.90) — no action needed

    masked_ids = input_ids.copy()
    masked_ids[replace_mask]   = MASK_TOKEN
    rand_tokens                = rng.choice(_AA_TOKEN_IDS, size=(B, T))
    masked_ids[replace_random] = rand_tokens[replace_random]

    return masked_ids, labels


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def _load_split(split: str):
    data_path    = os.path.join(DATA_DIR, f"{split}_data.bin")
    offsets_path = os.path.join(DATA_DIR, f"{split}_offsets.npy")
    assert os.path.exists(data_path), (
        f"Data not found: {data_path}. Run `python prepare.py` first."
    )
    data    = np.frombuffer(open(data_path, "rb").read(), dtype=np.uint8)
    offsets = np.load(offsets_path)   # (N, 2) int64 array of [start, length]
    return data, offsets


def make_dataloader(B: int, T: int, split: str, seed: int = None):
    """Infinite MLM dataloader. Yields (input_ids, labels, attn_mask) on CUDA.

    input_ids  : (B, T) long  — token IDs with MLM masking applied
    labels     : (B, T) long  — original ids at masked positions, -100 elsewhere
    attn_mask  : (B, T) bool  — True for real tokens, False for padding

    split='train' → random sampling (non-deterministic unless seed is set)
    split='val'   → sequential order, wraps around (use a fixed seed for eval)
    """
    assert split in ("train", "val")
    data, offsets = _load_split(split)
    N   = len(offsets)
    rng = np.random.default_rng(seed if seed is not None else None)

    input_buf = np.full( (B, T), PAD_TOKEN, dtype=np.int64)
    attn_buf  = np.zeros((B, T),            dtype=bool)

    ptr = 0
    while True:
        if seed is not None:
            # Deterministic sequential order (for val evaluation)
            if ptr + B > N:
                ptr = 0
            batch_idx = np.arange(ptr, ptr + B) % N
            ptr += B
        else:
            batch_idx = rng.integers(0, N, size=B)

        input_buf[:] = PAD_TOKEN
        attn_buf[:]  = False

        for i, si in enumerate(batch_idx):
            start, length = offsets[si]
            seq_len = min(int(length), T)
            input_buf[i, :seq_len] = data[start : start + seq_len]
            attn_buf[i, :seq_len]  = True

        masked, labels = _apply_mlm(input_buf, rng)

        yield (
            torch.tensor(masked,    dtype=torch.long, device="cuda"),
            torch.tensor(labels,    dtype=torch.long, device="cuda"),
            torch.tensor(attn_buf,  dtype=torch.bool, device="cuda"),
        )


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mlm_loss(model, batch_size: int) -> float:
    """Mean MLM cross-entropy on the fixed validation set.

    Model interface: model(input_ids, labels, reduction='none')
    Returns a flat tensor of per-token losses; 0 at non-masked positions.

    Uses a fixed seed so results are deterministic across runs.
    """
    val_loader  = make_dataloader(batch_size, MAX_SEQ_LEN, "val", seed=42)
    total_nats  = 0.0
    total_masked = 0

    for _ in range(EVAL_BATCHES):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        mask      = y.view(-1) != -100
        total_nats   += loss_flat[mask].sum().item()
        total_masked += mask.sum().item()

    return total_nats / max(total_masked, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare protein sequence data")
    parser.add_argument("--num-train", type=int, default=100_000,
                        help="Number of synthetic training sequences (default: 100k)")
    parser.add_argument("--num-val",   type=int, default=5_000,
                        help="Number of synthetic validation sequences (default: 5k)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    prepare_data(n_train=args.num_train, n_val=args.num_val, seed=args.seed)
    print("Done! Ready to train.")
