"""Compose shared instructions, a learning mode and public benchmark guidance."""

from pathlib import Path

from cam.config import BOTH_PROTOCOLS

ROOT = Path(__file__).parent

# Explicit user-authorized exceptions only; never generate these from a scorer.
TASK_CLARIFICATIONS = {
    ("mikasa_robo", "SeqOfColors3-VLA-v0"): "mikasa_robo/seq_of_colors3.md",
    ("mikasa_robo_v2", "SeqOfColors3-VLA-v0"): "mikasa_robo/seq_of_colors3.md",
}

# Static, task-independent public-interface guidance; no diagnostic input.
BENCHMARK_MODE_GUIDANCE = {
    # RMBench uses the shared demo-only prompt, including its final offline
    # code review. Its retired supplement is archived in legacy/rmbench/prompts/.
    ("mikasa_robo", "both", "demo_first"): "mikasa_robo/both.md",
    ("mikasa_robo_v2", "both", "v2"): "mikasa_robo/both.md",
    ("robomemarena", "both", "demo_first"): "robomemarena/both.md",
}


def benchmark_mode_guidance(config):
    relative = BENCHMARK_MODE_GUIDANCE.get(
        (config.benchmark, config.mode, config.both_protocol)
    )
    if relative is None:
        return None
    return {"source": f"cam/prompts/{relative}", "text": (ROOT / relative).read_text()}


def task_clarification(benchmark, task):
    relative = TASK_CLARIFICATIONS.get((benchmark, task))
    if relative is None:
        return None
    return {"source": f"cam/prompts/{relative}", "text": (ROOT / relative).read_text()}


def _shape(spec):
    return "x".join("N" if d is None else str(d) for d in spec["shape"]) or "scalar"


def describe_interface(interface):
    """Generated from the adapter's declared interface, so it cannot drift."""
    lines = [
        "## Declared observation and action interface",
        "",
        "Each `Frame` has `phase` ('demo' before execution, 'exec' after), `step`,",
        "`label`, `sensors` and these read-only NumPy arrays (N = any length):",
        "",
        "| Sensor | dtype | shape | Meaning |",
        "|---|---|---|---|",
    ]
    for name, spec in interface["sensors"].items():
        lines.append(f"| `{name}` | {spec['dtype']} | {_shape(spec)} | {spec['description']} |")
    action = interface["action"]
    bounds = ", ".join(
        f"{name} [{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"
        for name, lo, hi in zip(action["names"], action["low"], action["high"])
    )
    lines += [
        "",
        f"`robot.step({', '.join(action['names'])})`: {action['description']}",
        f"Bounds (values are clipped): {bounds}.",
    ]
    if action["ignored"]:
        lines.append(f"Ignored components (sent as zero): {', '.join(action['ignored'])}.")
    if action.get("motion_preview"):
        lines += ["", "## Robot motion dry-run", "", action["motion_preview"]["description"]]
    return "\n".join(lines) + "\n"


def render(config, task, episodes, training_episodes=None, adapter=None, interface=None, initial_policy=None):
    if adapter is not None and hasattr(adapter, "sensor_summary"):
        sensors = adapter.sensor_summary(task, config.depth)
    else:
        sensors = "RGB-D" if config.depth else "RGB (depth unavailable)"
    ids = [row["episode"] for row in episodes]
    training_ids = [
        row["episode"]
        for row in (training_episodes if training_episodes is not None else episodes)
    ]
    count = len(ids) if config.mode != "interaction" else 0
    mode_file = config.mode
    protocol = ""
    if config.mode == "both":
        mode_file = "both_v2" if config.both_protocol == "demo_first" else f"both_{config.both_protocol}"
        protocol = (
            f"Demo + interaction protocol: {config.both_protocol} "
            f"({BOTH_PROTOCOLS[config.both_protocol]}).\n"
        )
    schedule = (
        "Each attempt uses the next distinct training identity, disjoint from the assigned demonstrations; no repeats."
        if config.training_seed_policy == "disjoint_once"
        else "Attempts cycle automatically through these assigned training IDs."
    )
    if (config.training_seed_policy == "demo_cycle" and config.mode == "both"
            and len(training_ids) == count == config.budget
            and len(set(training_ids)) == count and set(training_ids) == set(ids)):
        schedule = (
            "Each attempt uses the next assigned demonstration identity and its recorded seed, "
            "exactly once within this budget. No new training seeds or retries. "
            "The host selects the identity; rollout accepts no seed or episode override."
        )
    physics = ""
    if config.benchmark in ("mikasa_robo", "mikasa_robo_v2"):
        from cam.interface.mikasa_robo.adapter import physics_backend_for_task

        physics = (
            f"Physics backend: {physics_backend_for_task(task, config.physics_backend)} "
            "for both training and held-out execution; "
            "host-selected and fixed for this run. Rendering uses the GPU independently.\n"
        )
    initialization = (
        "Continue the supplied existing Python policy for this same task in a user-requested repair. "
        "Review and fix code bugs, keeping its task strategy unless a code bug requires a change. "
        "Use only the assigned demos, public API and offline checks. Prior evaluation feedback is not supplied."
        if initial_policy else "Start from the supplied blank Python policy."
    )
    assignment = f"""Task: {task}
Method: Code-as-Memory. {initialization}
Benchmark: {config.benchmark}; learning mode: {config.mode}.
{protocol}Assigned expert demonstrations: {count}; demonstration IDs: {ids if count else []}.
Allowed simulator training episode IDs: {training_ids}.
Maximum simulator attempts: {config.budget} total. {schedule}
Sensors: {sensors}; task-provided reset video: {config.builtin_demo}.
{physics}Limits: {config.max_decisions} decisions per episode; {config.max_action_steps} step calls per action.
Use only the CaM tools: command, read_image, usage, snapshot, validate, rollout.
The writable directory is /workspace; save analysis files in /workspace/scratch.
Read the supplied api_reference before implementing the policy.
"""
    guide = getattr(adapter, "guide", "robomme/guide.md") if adapter is not None else "robomme/guide.md"
    benchmark = (ROOT.parent / "interface" / guide).read_text()
    if config.benchmark == "robomme_plus":
        benchmark += "\n" + (ROOT.parent / "interface/robomme/plus.md").read_text()
    if interface is not None and getattr(adapter, "describe_interface", False):
        benchmark += "\n" + describe_interface(interface)
    # Static public guidance and authorized corrections only. Never read
    # source-derived task guides, scorer contracts or diagnostic outcomes.
    clarification = task_clarification(config.benchmark, task)
    guidance = benchmark_mode_guidance(config)
    mode_text = (ROOT / "modes" / f"{mode_file}.md").read_text()
    if config.both_protocol == "demo_first":
        mode_text = mode_text.replace("Demonstrations + interaction v2:", "Demonstrations + interaction:", 1)
    return "\n\n".join(
        [
            assignment,
            *([clarification["text"]] if clarification else []),
            (ROOT / "core.md").read_text(),
            mode_text,
            *([guidance["text"]] if guidance else []),
            benchmark,
        ]
    )
