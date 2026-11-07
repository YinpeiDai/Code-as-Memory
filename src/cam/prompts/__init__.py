"""Compose one shared instruction, one learning mode and one benchmark guide."""

from pathlib import Path

ROOT = Path(__file__).parent


def render(config, task, episodes, training_episodes=None):
    sensors = "RGB-D" if config.depth else "RGB (depth unavailable)"
    ids = [row["episode"] for row in episodes]
    training_ids = [
        row["episode"]
        for row in (training_episodes if training_episodes is not None else episodes)
    ]
    count = len(ids) if config.mode != "interaction" else 0
    assignment = f"""Task: {task}
Method: Code-as-Memory. Start from the supplied blank Python policy.
Benchmark: {config.benchmark}; learning mode: {config.mode}.
Assigned expert demonstrations: {count}; demonstration IDs: {ids if count else []}.
Allowed simulator training episode IDs: {training_ids}.
Maximum simulator attempts: {config.budget} total, cycling only these training IDs.
Sensors: {sensors}; task-provided reset video: {config.builtin_demo}.
Limits: {config.max_decisions} decisions per episode; {config.max_action_steps} step calls per action.
Use only the CaM tools: command, read_image, usage, snapshot, validate, rollout.
The writable directory is /workspace; save analysis files in /workspace/scratch.
Read the supplied api_reference before implementing the policy.
"""
    benchmark = (ROOT.parent / "benchmarks/robomme.md").read_text()
    if config.benchmark == "robomme_plus":
        benchmark += "\n" + (ROOT.parent / "benchmarks/robomme_plus.md").read_text()
    return "\n\n".join(
        [
            assignment,
            (ROOT / "core.md").read_text(),
            (ROOT / "modes" / f"{config.mode}.md").read_text(),
            benchmark,
        ]
    )
