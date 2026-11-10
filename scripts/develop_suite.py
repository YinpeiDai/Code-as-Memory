"""Develop comparison settings with a shared resource pool and export frozen solutions.

Each arm uses develop_batch.py, blank policies and its assigned demonstration
subset. Evaluation remains host-only and starts after each policy freezes.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def start(args):
    import cam
    from cam.config import COMPARISON_SUITES, RunConfig
    from cam.io import copy_tree, read, sha256, tree_hash, write

    spec = read(args.plan)
    suite = spec["suite"]
    declared = COMPARISON_SUITES[suite]
    if (spec.get("schema") != "cam-development-suite/1" or spec["benchmark"] != declared["benchmark"]
            or spec["tasks"] != list(declared["tasks"])):
        raise ValueError("Suite must declare the exact registered benchmark and tasks")
    if sum(arm["workers"] for arm in spec["arms"].values()) != 8:
        raise ValueError("The comparison uses eight total coding workers")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    campaign = args.output.resolve() / suite / stamp
    campaign.mkdir(parents=True)
    copy_tree(Path(cam.__file__).parent, campaign / "inputs/harness/cam", ("legacy",))
    shutil.copy2(__file__, campaign / "driver.py")
    inputs = {}
    for arm, item in spec["arms"].items():
        plan = read(ROOT / item["plan"])
        selected = plan["benchmarks"][spec["benchmark"]]
        config = read(ROOT / selected["config"])
        cfg = RunConfig.from_dict(config)
        if (cfg.suite != suite or cfg.benchmark != spec["benchmark"]
                or selected["tasks"] != spec["tasks"] or selected["demo_count"] != item["demo_count"]
                or cfg.budget != item["interaction_count"]):
            raise ValueError(f"Suite arm disagrees with its declared task/data/budget assignment: {arm}")
        config_path = campaign / "inputs/configs" / f"{arm}.json"
        write(config_path, config)
        selected["config"] = str(config_path)
        selected["datasets"] = [str(args.dataset.resolve())]
        plan_path = campaign / "inputs/plans" / f"{arm}.json"
        write(plan_path, plan)
        inputs[arm] = {**item, "plan": str(plan_path)}
    manifest = {**spec, "arms": inputs, "created": time.time(), "repository": str(ROOT),
                "python": sys.executable, "dataset": str(args.dataset.resolve()),
                "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
                "gpus": args.gpus, "workers": 8, "evaluation_workers_per_gpu": 5,
                "harness_hash": tree_hash(Path(cam.__file__).parent, ("legacy",)),
                "batch_driver_sha256": sha256(ROOT / "scripts/develop_batch.py")}
    write(campaign / "suite.json", manifest)
    write(campaign / "batches.json", {})
    refresh(campaign)
    with (campaign / "supervisor.log").open("ab") as log:
        proc = subprocess.Popen([sys.executable, str(campaign / "driver.py"), "_worker", str(campaign)],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    write(campaign / "process.json", {"pid": proc.pid, "started": time.time()})
    return {"campaign": str(campaign), "pid": proc.pid, "policies": spec["expected_policies"],
            "expected_evaluation_episodes": spec["expected_policies"] * spec["evaluation_episodes_per_task"]}


def refresh(campaign):
    from cam.io import read, write

    spec = read(campaign / "suite.json")
    launched = read(campaign / "batches.json")
    tasks, batches = {}, {}
    for arm in spec["arms"]:
        batch = Path(launched[arm]["batch"]) if arm in launched else None
        state = read(batch / "status.json") if batch and (batch / "status.json").exists() else {}
        batches[arm] = state.get("status", "starting" if batch else "pending")
        for task in spec["tasks"]:
            info = state.get("tasks", {}).get(f"{spec['benchmark']}/{task}", {})
            root = Path(info["run"]) if info.get("run") else None
            rows = []
            if root:
                budget = spec["arms"][arm]["interaction_count"]
                rows = [read(p) for p in (root / "tasks" / task / "rollouts/test" / f"R{budget:03d}").glob("ep*/result.json")]
            scored = [r for r in rows if r["scorable"]]
            tasks[f"{arm}/{task}"] = {"setting": arm, "task": task, "run": str(root) if root else None,
                "development_status": info.get("status", "pending"),
                "evaluation_status": info.get("evaluation_status", "pending"),
                "frozen": bool(root and (root / "tasks" / task / "policy/freeze.json").exists()),
                "expected": spec["evaluation_episodes_per_task"], "scored": len(scored),
                "successes": sum(r["success"] for r in scored),
                "infrastructure_errors": len(rows) - len(scored),
                "policy_errors": sum(r.get("error") is not None for r in scored),
                **{key: info[key] for key in ("error", "evaluation_error") if key in info}}
    all_done = all(s in ("finished", "needs_attention", "launch_failed") for s in batches.values())
    complete = all_done and all(s == "finished" for s in batches.values()) and all(
        row["scored"] == row["expected"] and not row["infrastructure_errors"] and not row["policy_errors"]
        for row in tasks.values())
    status = {"schema": "cam-development-suite-status/1", "suite": spec["suite"], "updated": time.time(),
              "status": "finished" if complete else "needs_attention" if all_done else "running",
              "expected_policies": spec["expected_policies"], "frozen": sum(r["frozen"] for r in tasks.values()),
              **{k: sum(r[k] for r in tasks.values()) for k in ("scored", "expected", "successes", "policy_errors", "infrastructure_errors")},
              "batches": batches, "tasks": tasks}
    write(campaign / "status.json", status)
    lines = [f"# {spec['suite']}", "", f"Status: {status['status']}; frozen policies {status['frozen']}/{status['expected_policies']}; "
             f"held-out coverage {status['scored']}/{status['expected']}.", "", spec["comparison_note"], "",
             "Fresh blank policies use Claude Code / Opus 5.5. Eight total coding workers (3/3/2 across settings). "
             "Training shares one simulator lease per GPU. One evaluation task runs at a time with five episode workers per GPU on GPUs "
             + ", ".join(map(str, spec["gpus"])) + ". Evaluation makes no model calls and never enters learner feedback.", "",
             "| Setting | Task | Development | Evaluation | Successes / scored / expected |",
             "|---|---|---|---|---|" ]
    for row in tasks.values():
        lines.append(f"| {row['setting']} | {row['task']} | {row['development_status']} | {row['evaluation_status']} | "
                     f"{row['successes']} / {row['scored']} / {row['expected']} |")
    lines += ["", "`arms/` contains batch logs, isolated assignments, development histories and full rollouts. "
              "`solutions/` and `configs/solutions/` receive byte-identical frozen code and provenance when each arm completes. "
              "`status.json` retains per-task errors and actual coverage. Historical full-90 results are a separate track.", ""]
    (campaign / "README.md").write_text("\n".join(lines))
    return status


def export_arm(campaign, arm):
    from dataclasses import replace
    from cam.config import RunConfig
    from cam.core.history import select_checkpoint
    from cam.harness.solutions import load_solutions, record_hash
    from cam.io import read, sha256, write

    spec = read(campaign / "suite.json")
    catalog = campaign / "configs/solutions" / spec["suite"] / f"{arm}.json"
    solution = campaign / "solutions" / spec["suite"] / arm
    if catalog.exists():
        load_solutions(solution, catalog=catalog)
        return
    batch = Path(read(campaign / "batches.json")[arm]["batch"])
    state = read(batch / "status.json")
    if state["status"] != "finished":
        return
    items, config = {}, None
    for task in spec["tasks"]:
        row = state["tasks"][f"{spec['benchmark']}/{task}"]
        root = Path(row["run"])
        plan = read(root / "run.json")
        checkpoint, frozen = select_checkpoint(root / "tasks" / task, spec["arms"][arm]["interaction_count"])
        if config is None:
            config = replace(RunConfig.from_dict(plan["config"]), mode="demos", budget=0,
                             both_protocol=None, training_seed_count=None).to_dict()
        target = solution / task
        target.mkdir(parents=True, exist_ok=True)
        code = {}
        for source in sorted((checkpoint / "tools").glob("*.py")):
            destination = target / source.name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError("Exported policy bytes changed")
            if not destination.exists():
                shutil.copyfile(source, destination)
            code[source.name] = sha256(destination)
        items[task] = {"task_spec": read(checkpoint / "task.json"), "code_sha256": code,
            "source_run": str(root), "source_run_tag": root.name, "source_checkpoint": frozen["checkpoint"],
            "source_program_hash": frozen["program_hash"], "source_harness_sha256": plan["harness_hash"],
            "source_prompt_sha256": sha256(root / "tasks" / task / "prompt.md"),
            "assigned_demos": plan["demonstrations_per_task"][task], "training_attempts": frozen["training"]["attempts"],
            "physics_backend": plan["physics_backend_by_task"][task],
            "interface_sha256": record_hash(plan["interfaces"][task]),
            "test_records_sha256": record_hash(read(root / "inputs/metadata/test" / f"record_dataset_{task}_metadata.json")["records"]),
            "archived_result": {key: row["evaluation"][key] for key in ("expected", "scored", "successes", "success_rate")}}
    native_hash = read(batch / "plan.json")["benchmarks"][spec["benchmark"]]["source_hash"]
    write(catalog, {"schema": "cam-solutions/1", "benchmark": spec["benchmark"], "suite": spec["suite"],
                    "experiment": arm, "benchmark_source_sha256": native_hash, "evaluation_config": config,
                    "description": "Fresh comparison-suite policies; exact frozen executable bytes and original scored outcomes.",
                    "tasks": items})
    load_solutions(solution, catalog=catalog)


def worker(campaign):
    from cam.io import read, sha256, tree_hash, write

    spec = read(campaign / "suite.json")
    repo = Path(spec["repository"])
    launched = read(campaign / "batches.json")
    for arm, item in spec["arms"].items():
        if arm in launched:
            continue
        if (tree_hash(repo / "src/cam", ("legacy",)) != spec["harness_hash"]
                or sha256(repo / "scripts/develop_batch.py") != spec["batch_driver_sha256"]):
            raise ValueError("Checkout changed before all suite arms were launched; prepare a fresh campaign")
        command = [spec["python"], str(repo / "scripts/develop_batch.py"), "start", item["plan"],
                   "--output", str(campaign / "arms" / arm), "--workers", str(item["workers"]),
                   "--gpus", *map(str, spec["gpus"]), "--resource-root", str(campaign / "resources"),
                   "--evaluate-on-freeze", "--evaluation-workers-per-gpu", "5"]
        write(campaign / "launches" / f"{arm}.command.json", command)
        with (campaign / "launches" / f"{arm}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=repo)
        if result.returncode:
            raise RuntimeError(f"Suite launch failed for {arm}; see launches/{arm}.log")
        launched[arm] = read(campaign / "launches" / f"{arm}.log")
        write(campaign / "batches.json", launched)
        refresh(campaign)
    previous = None
    while True:
        status = refresh(campaign)
        for arm in spec["arms"]:
            export_arm(campaign, arm)
        key = (status["status"], status["frozen"], status["scored"])
        if key != previous:
            print(json.dumps({k: status[k] for k in ("status", "frozen", "scored", "expected", "successes")}), flush=True)
            previous = key
        if status["status"] != "running":
            return
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start")
    p.add_argument("plan", type=Path)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    for command in ("status", "_worker"):
        sub.add_parser(command).add_argument("campaign", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.command == "_worker":
        sys.path.insert(0, str(args.campaign / "inputs/harness"))
        try:
            worker(args.campaign.resolve())
        except Exception as exc:
            from cam.io import read, write
            state = read(args.campaign / "status.json")
            write(args.campaign / "status.json", {**state, "status": "needs_attention", "supervisor_error": str(exc)})
            raise
    else:
        sys.path.insert(0, str(ROOT / "src"))
        if args.command == "start":
            print(json.dumps(start(args), indent=2))
        else:
            from cam.io import read
            print(json.dumps(read(args.campaign / "status.json"), indent=2))


if __name__ == "__main__":
    main()
