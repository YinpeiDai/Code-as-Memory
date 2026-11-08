"""Prepare a separate CPU evaluation of an existing frozen MIKASA v2 policy.

Uses normal CaM preparation, new snapshots and a new timetag. Copies only the
frozen policy payload; does not launch learning, copy test feedback, or alter
the source run. CPU scores must be reported separately from official GPU scores.
"""
import argparse
import json
from pathlib import Path

from cam.config import RunConfig
from cam.core.budget import program_hash
from cam.core.history import freeze, payload
from cam.harness.prepare import prepare_runs, validate_run
from cam.io import read, write


def prepare_cpu(source, base, timetag=None):
    source = Path(source).resolve()
    original = validate_run(source)
    if original["config"]["benchmark"] != "mikasa_robo_v2" or len(original["tasks"]) != 1:
        raise ValueError("Provide a single-task MIKASA-Robo v2 run")
    if original["config"]["mode"] != "demos":
        raise ValueError("This comparison imports demo-only policies (checkpoint R000)")
    task = original["tasks"][0]
    policy = source / "tasks" / task / "policy"
    frozen = read(policy / "freeze.json")
    if program_hash(policy) != frozen["program_hash"]:
        raise ValueError("Source frozen policy integrity mismatch")
    config = RunConfig.from_dict({**original["config"], "physics_backend": "cpu"})
    prepared = prepare_runs(
        base, Path(original["dataset_reference"]).parent, config, tasks=[task],
        benchmark_source=Path(original["benchmark_source_reference"]),
        timetag=timetag,
    )
    destination = Path(prepared["runs"][0]["root"])
    if read(destination / "run.json")["benchmark_hash"] != original["benchmark_hash"]:
        raise ValueError("New benchmark snapshot differs from the original; policy was not imported")
    task_root = destination / "tasks" / task
    files = payload(policy)
    for name in ("program", "workspace"):
        folder = task_root / name
        for path in (folder / "tools").glob("*.py"):
            path.unlink()  # newly prepared starter only
        for relative, data in files.items():
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    imported = freeze(read(task_root / "state/config.json"))
    if imported["program_hash"] != frozen["program_hash"]:
        raise ValueError("Imported policy differs from the original")
    provenance = {
        "schema": "cam-frozen-evaluation-import/1", "source_run": str(source),
        "source_program_hash": frozen["program_hash"], "source_frozen_at": frozen["frozen_at"],
        "physics_backend": "cpu", "rendering": "GPU", "new_learning_sessions": 0,
        "source_config": original["config"], "source_harness_hash": original["harness_hash"],
        "source_benchmark_hash": original["benchmark_hash"],
        "reporting": "CPU comparison only; preserve and separately report official GPU results",
    }
    write(destination / "evaluation_source.json", provenance)
    write(task_root / "logs/completion.json", {
        "status": "finished", "kind": "imported_frozen_policy", "source_run": str(source),
        "program_hash": frozen["program_hash"], "new_learning_sessions": 0,
    })
    readme = destination / "README.md"
    readme.write_text(
        "# CPU evaluation of a previously frozen policy\n\n"
        f"Imported unchanged from `{source}`. No new coding session or training rollout. "
        "`evaluation_source.json` records provenance. CPU physics and native CPU IK are "
        "the comparison condition; images still render on the GPU. Original GPU scores "
        "remain separate.\n\n" + readme.read_text()
    )
    validate_run(destination)
    return {"task": task, "run": str(destination), "source_run": str(source),
            "program_hash": frozen["program_hash"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--timetag")
    args = parser.parse_args()
    print(json.dumps(prepare_cpu(args.run, args.runs_dir, args.timetag), indent=2))
