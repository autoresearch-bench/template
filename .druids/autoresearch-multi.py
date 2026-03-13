"""Autoresearch -- autonomous ML research with async human check-ins.

Scientist + lab assistant pattern for autonomous ML experimentation.
The scientist plans experiments, curates research state as committed
files in the repo, and maintains a dashboard. Lab assistants modify
code and run training on GPU VMs.

Usage:
  druids exec .druids/autoresearch.py --devbox owner/repo \
    spec="minimize val_bpb" gpu_hours=8 max_parallel=2
"""

from __future__ import annotations

import asyncio
import json
import time


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SCIENTIST_SYSTEM = """\
You are an autonomous ML research scientist. Your job is to design \
experiments that minimize val_bpb (validation bits per byte) on a language \
model training task.

You have lab assistants who run experiments for you. You never modify \
train.py or run training yourself. You think, plan, interpret, and curate.

## How experiments work

Each experiment modifies train.py and runs it for 5 minutes on a Modal \
H100 GPU. The output metric is val_bpb (lower is better). You specify \
what change to make in natural language, and a lab assistant implements \
it and runs it via `uv run modal run run_modal.py`.

Experiments run on git branches off of research/best. When a result \
improves, the assistant opens a PR into research/best. You decide \
what to merge. The research/best branch always represents the current \
best known configuration.

## Git workflow

You own the research/best branch. At the start, create it from main. \
Lab assistants branch off research/best for experiments. When they get \
an improvement, they open a PR. You review and merge with `gh pr merge`.

The PR history is a natural record of what worked. Your committed \
research files (dashboard, observations, plan) also live on \
research/best.

## Your tools

- read_state: Read the full research state (goal, budget, best result, \
branches, observations, journal, plan).
- run_experiments: Submit a batch of experiments as a JSON list. Each \
entry has {branch, change, hypothesis}. Returns immediately; results \
arrive as [Result] messages.
- update_observations: Rewrite your accumulated observations.
- update_plan: Rewrite your current research plan.
- log_entry: Write a timestamped journal entry.
- check_budget: Check remaining budget and assistant availability.

## State as committed files

You maintain research state as files in the repo under research/. \
These are committed artifacts that form the durable record of your \
work. Write and commit:

- research/dashboard.html -- a self-contained HTML page summarizing \
the current state of the research. This is the primary artifact the \
human reads when they check in. Make it clear, detailed, and honest.
- research/observations.md -- accumulated knowledge.
- research/plan.md -- what you intend to do next and why.

Commit these files regularly. Each commit is a temporal snapshot. The \
git history IS the journal of how the research evolved. Write good \
commit messages.

The structured state (budget, branches, runs) is available via \
read_state. Your committed files are your interpretation of that data.

## How you work

1. Read the state. Read train.py and prepare.py.
2. Write your initial dashboard and plan. Commit them.
3. Run the baseline (train.py as-is) before changing anything.
4. Plan experiments. Launch via run_experiments.
5. While experiments run: curate observations, update the dashboard, \
refine the plan. Commit regularly.
6. When results arrive as [Result] messages: interpret, update files, \
plan next batch.

## Branching strategy

Create experiment branches for distinct investigation threads. How \
many to explore depends on budget:

- Abundant budget: explore multiple directions in parallel.
- Limited budget: converge on the most promising branch.

## While waiting for results

You always have work to do:
- Update the dashboard with your current thinking.
- Curate observations. Are they still valid?
- Analyze trends. Is a 0.002 bpb difference signal or noise?
- Think about interactions between findings.
- Read train.py and prepare.py for new angles.

## Human interaction

The human checks in asynchronously -- possibly hours later. They will \
read your dashboard and committed files. They may send you messages \
with feedback, new directions, or additional budget. Incorporate their \
input when it arrives.

## Be a good scientist

- Simplicity criterion: prefer simpler code at equal performance.
- Control your variables. Change one thing at a time when possible.
- Track what you learn from failures, not just successes.
- The first run should always establish the baseline.
- When stuck, try something radical."""


ASSISTANT_SYSTEM = """\
You are a lab assistant running ML experiments. You receive experiment \
assignments, implement code changes, run training, and report results.

## Workflow for each experiment

1. Read the assignment message. It specifies: branch name, code change \
description, hypothesis.
2. Make sure your working tree is clean: `git stash` or `git checkout .`
3. Check out the branch. If it does not exist, create it from the \
current best branch (specified in the assignment).
4. Read train.py to understand the current state.
5. Implement the code change described in the assignment.
6. Commit the change: `git add -A && git commit -m "<description>"`.
7. Run on Modal: `uv run modal run --env ar-cc-starter run_modal.py 2>&1 | tee /tmp/run.log`
8. Extract results: `grep "^val_bpb:\\|^peak_vram_mb:" /tmp/run.log`
9. If grep output is empty, the run crashed. Run `tail -50 /tmp/run.log`.

## Reporting

Call submit_result with:
- branch: the branch name
- val_bpb: the val_bpb value (use "0" if crashed)
- peak_vram_mb: peak VRAM in MB (use "0" if crashed)
- config: one-line description of the change
- keep: "true" if val_bpb improved over the branch's previous best
- crashed: "true" if the run crashed
- notes: brief observations

## Git rules

- Always branch off research/best (or the base branch given in the \
assignment).
- If val_bpb improved (keep=true): push the commit and open a PR \
into research/best with `gh pr create --base research/best \
--title "<description>" --body "<hypothesis and result>"`.
- If val_bpb did not improve: `git reset --hard HEAD~1` to discard.
- If crashed: `git reset --hard HEAD~1` to discard.

After reporting, wait for the next assignment message."""


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

EXPERIMENT_HOURS = 5.0 / 60.0  # each 5-min run costs 1/12 of a GPU hour


async def program(ctx, spec="", gpu_hours="8", max_parallel="2", **kwargs):
    """Autonomous ML research with scientist + lab assistant pattern."""
    working_dir = "/home/agent/repo"
    gpu_hours_total = float(gpu_hours)
    n_parallel = int(max_parallel)
    next_journal_id = 1
    pending = 0

    state = {
        "goal": spec or "minimize val_bpb",
        "status": "running",
        "started_at": time.time(),
        "budget": {
            "gpu_hours_total": gpu_hours_total,
            "gpu_hours_used": 0.0,
            "max_parallel": n_parallel,
        },
        "best": None,
        "branches": {},
        "observations": "",
        "plan": "",
        "journal": [],
    }

    def add_journal(text):
        nonlocal next_journal_id
        entry = {"id": next_journal_id, "time": time.time(), "text": text}
        state["journal"].append(entry)
        next_journal_id += 1
        return entry

    def budget_remaining():
        return state["budget"]["gpu_hours_total"] - state["budget"]["gpu_hours_used"]

    def state_markdown():
        s = state
        b = s["budget"]
        rem = b["gpu_hours_total"] - b["gpu_hours_used"]
        lines = [
            f"# Research State\n",
            f"## Goal\n{s['goal']}\n",
            f"## Budget\n- Total: {b['gpu_hours_total']:.1f}h"
            f"\n- Used: {b['gpu_hours_used']:.1f}h"
            f"\n- Remaining: {rem:.1f}h"
            f"\n- Max parallel: {b['max_parallel']}\n",
        ]
        if s["best"]:
            lines.append(
                f"## Best Result\n- val_bpb: {s['best']['bpb']:.6f}"
                f"\n- branch: {s['best']['branch']}"
                f"\n- config: {s['best']['description']}\n"
            )
        else:
            lines.append("## Best Result\nNo results yet.\n")

        lines.append("## Branches")
        for name, br in s["branches"].items():
            lines.append(f"\n### {name} [{br['status']}]")
            lines.append(f"Hypothesis: {br['hypothesis']}")
            if br.get("best_bpb"):
                lines.append(f"Branch best: {br['best_bpb']:.6f}")
            for run in br["runs"]:
                tag = "CRASH" if run.get("crashed") else ("KEEP" if run["keep"] else "DISCARD")
                lines.append(f"  [{tag}] bpb={run['val_bpb']:.6f} | {run['config']}")

        lines.append(f"\n## Observations\n{s['observations'] or 'None yet.'}\n")
        lines.append(f"## Plan\n{s['plan'] or 'No plan yet.'}\n")

        recent = s["journal"][-10:]
        lines.append("## Recent Journal")
        for e in recent:
            t = time.strftime("%H:%M", time.localtime(e["time"]))
            lines.append(f"\n### Entry {e['id']} ({t})\n{e['text']}")

        return "\n".join(lines)

    # -- Agents --

    scientist = await ctx.agent(
        "scientist",
        system_prompt=SCIENTIST_SYSTEM,
        prompt=(
            f"Your research goal: {spec or 'minimize val_bpb'}\n\n"
            f"Budget: {gpu_hours_total} GPU-hours, {n_parallel} parallel "
            f"assistants.\n\n"
            "Start by calling read_state, then read train.py and prepare.py "
            "to understand the codebase. Create the research/best branch "
            "from main, set up research/ with your initial dashboard.html "
            "and plan.md, and commit them. Then run the baseline (train.py "
            "as-is) before experimenting."
        ),
        model="claude-opus-4-6",
        git="write",
        working_directory=working_dir,
    )

    add_journal(f"Research started. Goal: {state['goal']}. Budget: {gpu_hours_total}h.")

    idle_assistants: asyncio.Queue = asyncio.Queue()

    for i in range(n_parallel):
        asst = await ctx.agent(
            f"assistant-{i}",
            system_prompt=ASSISTANT_SYSTEM,
            prompt=f"You are lab assistant {i}. Wait for experiment assignments.",
            model="claude-opus-4-6",
            git="write",
            working_directory=working_dir,
        )

        @asst.on("submit_result")
        async def on_result(
            branch="", val_bpb="", peak_vram_mb="", config="",
            keep="", crashed="", notes="", caller=None,
        ):
            """Report experiment results back to the orchestrator."""
            nonlocal pending
            bpb = float(val_bpb) if val_bpb else 0.0
            vram = float(peak_vram_mb) if peak_vram_mb else 0.0
            is_keep = str(keep).lower() in ("true", "1", "yes")
            is_crash = str(crashed).lower() in ("true", "1", "yes")

            run = {
                "val_bpb": bpb,
                "peak_vram_mb": vram,
                "config": config,
                "keep": is_keep,
                "crashed": is_crash,
                "notes": notes,
                "time": time.time(),
            }

            if branch in state["branches"]:
                state["branches"][branch]["runs"].append(run)
                if is_keep and bpb > 0:
                    prev = state["branches"][branch].get("best_bpb")
                    if prev is None or bpb < prev:
                        state["branches"][branch]["best_bpb"] = bpb

            if is_keep and bpb > 0:
                if state["best"] is None or bpb < state["best"]["bpb"]:
                    state["best"] = {
                        "bpb": bpb, "branch": branch, "description": config,
                    }

            state["budget"]["gpu_hours_used"] += EXPERIMENT_HOURS
            tag = "CRASH" if is_crash else ("KEEP" if is_keep else "DISCARD")
            add_journal(f"{branch}: [{tag}] bpb={bpb:.6f} | {config}")
            pending -= 1

            await scientist.send(
                f"[Result] {branch}: val_bpb={bpb:.6f} [{tag}] | {config}"
                + (f"\nVRAM: {vram:.0f}MB" if vram else "")
                + (f"\nNotes: {notes}" if notes else "")
                + f"\nBudget remaining: {budget_remaining():.1f}h"
                + f" | Pending: {pending}"
            )

            if caller:
                await idle_assistants.put(caller)

            if budget_remaining() <= 0:
                state["status"] = "waiting"
                await scientist.send(
                    "[System] Budget exhausted. Curate your observations, "
                    "update the dashboard, and commit. Human feedback will "
                    "arrive when they check in."
                )

            return (
                f"Recorded. Best: "
                f"{state['best']['bpb']:.6f if state['best'] else '--'}. "
                f"Budget: {budget_remaining():.1f}h"
            )

        await idle_assistants.put(asst)

    # -- Scientist tools --

    @scientist.on("read_state")
    async def on_read_state():
        """Read the full research state as markdown."""
        return state_markdown()

    @scientist.on("run_experiments")
    async def on_run_experiments(experiments: str = ""):
        """Submit experiments. JSON list of {branch, change, hypothesis}.

        Returns immediately. Results arrive as [Result] messages."""
        nonlocal pending

        if budget_remaining() <= 0:
            return "Budget exhausted. Wait for human to add more."

        try:
            exps = json.loads(experiments)
        except json.JSONDecodeError:
            return "Invalid JSON. Expected: [{branch, change, hypothesis}, ...]"

        if not isinstance(exps, list):
            return "Expected a JSON list."

        launched = 0
        for exp in exps:
            branch = exp.get("branch", "")
            change = exp.get("change", "")
            hypothesis = exp.get("hypothesis", "")
            if not branch or not change:
                continue

            if branch not in state["branches"]:
                state["branches"][branch] = {
                    "hypothesis": hypothesis,
                    "status": "active",
                    "runs": [],
                    "best_bpb": None,
                }

            try:
                asst = idle_assistants.get_nowait()
            except asyncio.QueueEmpty:
                break

            pending += 1
            launched += 1

            branch_info = ""
            br = state["branches"][branch]
            if br.get("best_bpb"):
                branch_info = f"\nBranch current best: {br['best_bpb']:.6f}"

            base = "\nBase branch: research/best"

            await asst.send(
                f"[Experiment]\n"
                f"Branch: {branch}\n"
                f"Change: {change}\n"
                f"Hypothesis: {hypothesis}"
                f"{branch_info}"
                f"{base}\n\n"
                f"Implement the change, run the experiment, "
                f"call submit_result."
            )

        idle = idle_assistants.qsize()
        return (
            f"Launched {launched} experiment(s). "
            f"{idle} assistant(s) idle, {pending} pending. "
            f"Results will arrive as [Result] messages."
        )

    @scientist.on("update_observations")
    async def on_update_obs(observations: str = ""):
        """Rewrite the observations section of the research state."""
        state["observations"] = observations
        return "Observations updated."

    @scientist.on("update_plan")
    async def on_update_plan(plan: str = ""):
        """Rewrite the plan section of the research state."""
        state["plan"] = plan
        return "Plan updated."

    @scientist.on("log_entry")
    async def on_log_entry(text: str = ""):
        """Write a timestamped journal entry."""
        entry = add_journal(text)
        return f"Entry #{entry['id']} recorded."

    @scientist.on("check_budget")
    async def on_check_budget():
        """Check remaining budget and assistant status."""
        return (
            f"GPU hours total: {state['budget']['gpu_hours_total']:.1f}\n"
            f"GPU hours used: {state['budget']['gpu_hours_used']:.1f}\n"
            f"GPU hours remaining: {budget_remaining():.1f}\n"
            f"Pending experiments: {pending}\n"
            f"Idle assistants: {idle_assistants.qsize()}/{n_parallel}"
        )

    # -- Client events --

    @ctx.on_client_event("get_state")
    def on_get_state():
        """Return full experiment state."""
        return state

    @ctx.on_client_event("feedback")
    async def on_feedback(text="", extra_gpu_hours=""):
        """Send feedback to the scientist, optionally adding budget."""
        if extra_gpu_hours:
            extra = float(extra_gpu_hours)
            state["budget"]["gpu_hours_total"] += extra
            add_journal(f"Budget increased by {extra}h")
        msg = f"[Human Feedback] {text}"
        if extra_gpu_hours:
            msg += (
                f"\nBudget +{extra_gpu_hours}h. "
                f"Remaining: {budget_remaining():.1f}h"
            )
        state["status"] = "running"
        await scientist.send(msg)
        return {"ack": True}

    await ctx.wait()
