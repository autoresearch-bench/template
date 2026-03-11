"""Autoresearch for the Protein LM Speedrun.

One agent, one repo, one loop. The agent modifies train.py, runs experiments
on Modal, keeps improvements, discards regressions.

Metric: val_loss (MLM cross-entropy on masked protein sequences). Lower = better.
"""

BUDGET_USD = 100.0
MODAL_ENV  = "ar-cc-starter"

SYSTEM_PROMPT = """\
You are an autonomous ML researcher running the Protein LM Speedrun. \
Your goal is to minimize val_loss (masked language modeling cross-entropy) \
on protein sequences by iterating on train.py.

## Setup

1. Read all files in the repo: prepare.py, train.py, program.md. \
Understand the full context before doing anything.
2. Create a fresh branch: `git checkout -b autoresearch/protein-run1`
3. Create results.tsv with just the header: \
`echo -e "commit\\tval_loss\\tmemory_gb\\tstatus\\tdescription" > results.tsv`
4. Run the baseline as-is to establish the starting val_loss.

## The experiment loop

LOOP FOREVER:

1. Look at the current state: results.tsv, current train.py
2. Decide what to try. Think about what might improve val_loss.
3. Modify train.py (or other non-prepare.py files you've created).
4. Call the `run_experiment` tool with a short description of what you're trying.
5. The tool handles everything: commits your changes, runs on Modal, parses \
metrics, appends to results.tsv, and auto keeps/discards based on val_loss.
6. Read the result summary and plan your next experiment.
7. Repeat from step 1.

## What you CAN modify

- train.py — architecture, optimizer, hyperparameters, training loop, everything.
- You can create new files if needed (custom CUDA kernels, etc.).
- You can install additional packages with `uv add`.

## What you CANNOT modify

- prepare.py — read-only. Contains the evaluation function, MLM dataloader,
  tokenizer, and fixed constants.

## Key differences from NanoGPT speedrun

This is MASKED language modeling (BERT-style), not autoregressive (GPT-style):
- The model sees the full sequence with 15% tokens masked
- Loss is computed only on masked positions
- Bidirectional attention (no causal mask)
- Tiny vocabulary: 33 amino acid tokens vs 50k BPE tokens
- Variable-length sequences, no documents

## Strategy tips

- Start by understanding the baseline: model size, optimizer, batch size, LR.
- Low-hanging fruit: torch.compile (already done), gradient clipping, bf16.
- Optimizer: Muon and Lion have shown gains on NanoGPT — try them here.
- Architecture: QK-norm, RMSNorm, SwiGLU, deeper vs wider tradeoffs.
- MLM-specific: span masking (mask contiguous blocks), higher mask rate, \
  warmup the mask rate.
- RoPE tuning: base frequency (10000 default), scaling for long sequences.
- Batch size: protein sequences are shorter than text on average, so you can \
  fit larger batches. Try larger batch + lower LR.
- Curriculum: start with short sequences, gradually increase length.
- Gradient accumulation: effective batch size matters more than step count.

## Budget

You have a fixed compute budget. Each experiment costs ~$0.55-2.00 depending \
on GPU type and duration. The orchestrator tells you remaining budget. \
Use experiments wisely — test hypotheses, don't just tune randomly.

## NEVER STOP

Do not pause to ask if you should continue. You are autonomous. Run experiments \
until told to stop or the budget runs out."""


import json


def _parse_metrics(output: str) -> dict:
    """Extract val_loss and other metrics from training output."""
    metrics = {}
    for line in output.split("\n"):
        line = line.strip()
        for key in ("val_loss", "peak_vram_mb", "training_seconds", "num_params_M"):
            if line.startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return metrics


async def _get_modal_spend(agent, env: str) -> float:
    """Query actual Modal spend for this environment."""
    result = await agent.exec(
        f"uv run modal billing report --for today --json 2>/dev/null || echo '[]'"
    )
    try:
        entries = json.loads(result.stdout or "[]")
        return sum(
            float(e["Cost"]) for e in entries if e.get("Environment") == env
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


async def program(ctx, repo_full_name="autoresearch-bench/template", **kwargs):
    working_dir   = "/home/agent/repo"
    spent         = 0.0
    best_loss     = float("inf")
    experiment_count = 0

    researcher = await ctx.agent(
        "researcher",
        system_prompt=SYSTEM_PROMPT,
        prompt="Begin. Read the repo, set up, establish baseline val_loss, then start experimenting.",
        git="write",
        working_directory=working_dir,
    )

    @researcher.on("run_experiment")
    async def run_experiment(description: str = ""):
        """Commit current changes, run train.py on a Modal H100, and
        auto keep/discard based on val_loss. Returns a summary with metrics
        and budget status."""
        nonlocal spent, best_loss, experiment_count

        # Check actual Modal spend
        spent = await _get_modal_spend(researcher, MODAL_ENV)
        if spent >= BUDGET_USD:
            ctx.done(
                f"Budget exhausted. Spent ${spent:.2f} across {experiment_count} experiments. "
                f"Best val_loss: {best_loss:.6f}"
            )
            return "BUDGET EXHAUSTED. No more experiments."

        experiment_count += 1

        # Auto-commit all changes
        await researcher.exec("cd /home/agent/repo && git add -A")
        commit_msg = f"experiment {experiment_count}: {description}"
        await researcher.exec(
            f"cd /home/agent/repo && git diff --cached --quiet"
            f" || git commit -m '{commit_msg}'"
        )

        # Run on Modal
        result = await researcher.exec(
            f"cd /home/agent/repo && uv run modal run --env {MODAL_ENV} run_modal.py 2>&1",
            timeout=1200,
        )

        output    = result.stdout or ""
        exit_code = result.exit_code
        metrics   = _parse_metrics(output)

        loss     = metrics.get("val_loss")
        vram_gb  = round(metrics.get("peak_vram_mb", 0) / 1024, 1)
        improved = loss is not None and loss < best_loss

        if improved:
            best_loss = loss

        hash_result  = await researcher.exec("git rev-parse --short HEAD")
        commit_hash  = (hash_result.stdout or "").strip()

        if exit_code != 0 or loss is None:
            status = "crash"
        elif improved:
            status = "keep"
        else:
            status = "discard"

        tsv_line = f"{commit_hash}\t{loss or 0.0:.6f}\t{vram_gb}\t{status}\t{description}"
        await researcher.exec(f"echo '{tsv_line}' >> /home/agent/repo/results.tsv")

        if status == "keep":
            await researcher.exec("cd /home/agent/repo && git push")
        elif status == "discard":
            await researcher.exec("cd /home/agent/repo && git reset --hard HEAD~1")

        spent           = await _get_modal_spend(researcher, MODAL_ENV)
        remaining_budget = BUDGET_USD - spent

        ctx.emit("experiment", {
            "number":           experiment_count,
            "description":      description,
            "status":           status,
            "val_loss":         loss,
            "best_loss":        best_loss,
            "vram_gb":          vram_gb,
            "spent":            spent,
            "remaining_budget": remaining_budget,
        })

        lines = [
            f"Experiment #{experiment_count}: {description}",
            f"Status: {status.upper()}",
        ]
        if loss is not None:
            lines.append(f"val_loss: {loss:.6f} (best: {best_loss:.6f})")
        if vram_gb:
            lines.append(f"peak_vram: {vram_gb} GB")
        if status == "keep":
            lines.append("Commit KEPT and pushed. This is your new baseline.")
        elif status == "discard":
            lines.append("Commit DISCARDED. Reverted to previous best.")
        elif status == "crash":
            lines.append("Run CRASHED. Last 50 lines of output:")
            lines.append("\n".join(output.strip().split("\n")[-50:]))
        lines.append(f"Budget: ${spent:.2f}/${BUDGET_USD:.2f} (${remaining_budget:.2f} remaining)")

        return "\n".join(lines)
