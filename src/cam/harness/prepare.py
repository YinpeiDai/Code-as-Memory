"""Prepare fresh CaM studies with pinned code, metadata and task-private evidence."""

import os
import json
from pathlib import Path
import shutil
import sys
import time
import importlib.metadata
import subprocess
from datetime import datetime, timezone
import re
from urllib.parse import quote

from cam import __file__ as package_file
from cam.benchmarks.registry import get_adapter
from cam.config import RunConfig
from cam.core.budget import Ledger
from cam.core.history import materialize, snapshot
from cam.data.extract import validate_dataset
from cam.io import copy_tree, read, sha256, tree_hash, write
from cam.prompts import render
from cam.harness.run_readme import write_run_readme, write_runs_readme

SETTINGS = {
    "demos": "demo_only",
    "interaction": "interaction_only",
    "both": "demo_interaction",
}

STARTER = '''def perceive_decide(obs, goal, memory, last):
    """Replace this observation-only template with a general task program."""
    return {"memory": {}, "act": {"tool": "act_observe", "args": {}}}

def act_observe(robot):
    """Observe one native step; this template does not solve the task."""
    robot.step(*robot.now().tcp, 1.0)
    return {"observed": True}
'''


def make_runtime(output, demonstrations=True):
    source = Path(package_file).parent
    package = output / "cam"
    for folder in ("", "core", "runtime"):
        (package / folder).mkdir(parents=True, exist_ok=True)
        (package / folder / "__init__.py").write_text(
            '"""Code-as-Memory isolated policy API."""\n'
        )
    names = {
        "core": ("workspace", "tools", "sandbox", "code_policy"),
        "runtime": ("observations", "wire", "robot", "policy_worker")
        + (("trajectory",) if demonstrations else ()),
    }
    for folder, modules in names.items():
        for name in modules:
            shutil.copy2(
                source / folder / (name + ".py"), package / folder / (name + ".py")
            )
    return output


def preparation_inputs(dataset, config, tasks=None, benchmark_source=None):
    dataset = Path(dataset).resolve()
    config = config if isinstance(config, RunConfig) else RunConfig.from_dict(config)
    adapter = get_adapter(config.benchmark, benchmark_source)
    if config.perceptual_variations:
        raise ValueError(
            "Perceptual variation protocols are not implemented in this RoboMME release"
        )
    selected = validate_dataset(dataset)
    if selected["benchmark"] != config.benchmark or selected[
        "benchmark_hash"
    ] != tree_hash(adapter.source):
        raise ValueError("Dataset and benchmark snapshot do not match")
    if selected["depth"] != config.depth:
        raise ValueError("Extraction and policy sensor settings must agree")
    tasks = tasks or list(selected["tasks"])
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) - set(selected["tasks"])
    ):
        raise ValueError("Choose distinct extracted tasks")
    return dataset, config, tasks, adapter, selected


def prepare(root, dataset, config, tasks=None, benchmark_source=None):
    """Prepare an exact run path; retained for embedded callers and old runs."""
    return _prepare(
        Path(root).resolve(),
        *preparation_inputs(dataset, config, tasks, benchmark_source),
    )


def coding_run_name(coding):
    """Short display folder; exact client/provider/model remain in the manifest."""
    agent = {"claude_code": "cc", "codex": "cx"}[coding.agent]
    model = re.sub(r"^(?:(?:us|eu|apac|global)\.)?anthropic\.", "", coding.model)
    model = re.sub(
        r"^claude-(opus|sonnet|haiku)-(\d+)-(\d{1,2})(?=$|[-.:])",
        r"\1\2.\3",
        model,
    )
    model = re.sub(r"^claude-(opus|sonnet|haiku)-(\d+)(?=$|[-.:])", r"\1\2", model)
    model = re.sub(r"^gpt-(?=\d)", "gpt", model)
    return agent + "-" + quote(model, safe="._-")


def prepare_runs(
    base, dataset, config, tasks=None, benchmark_source=None, timetag=None
):
    """Create one task per benchmark/task/client-model/setting/timetag leaf."""
    dataset, config, tasks, adapter, selected = preparation_inputs(
        dataset, config, tasks, benchmark_source
    )
    timetag = timetag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if timetag in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", timetag):
        raise ValueError("Timetag must be a single alphanumeric path component")
    base = Path(base).resolve()
    client = coding_run_name(config.coding)
    roots = [
        base / config.benchmark / task / client / SETTINGS[config.mode] / timetag
        for task in tasks
    ]
    for root in roots:
        if root.exists():
            raise FileExistsError(f"Run already exists; choose a new timetag: {root}")
    base.mkdir(parents=True, exist_ok=True)
    write_runs_readme(base)
    results = []
    for task, root in zip(tasks, roots):
        _prepare(root, dataset, config, [task], adapter, selected)
        results.append({"task": task, "root": str(root)})
    return {
        "schema": "cam-preparation/1",
        "benchmark": config.benchmark,
        "setting": SETTINGS[config.mode],
        "timetag": timetag,
        "runs": results,
    }


def _prepare(root, dataset, config, tasks, adapter, selected):
    if root.exists():
        raise FileExistsError(
            "Use a new run directory; existing logs and budgets cannot be reset"
        )
    root.mkdir(parents=True)
    benchmark = root / "inputs/benchmark" / adapter.package
    copy_tree(adapter.source, benchmark)
    package = root / "inputs/harness/cam"
    copy_tree(Path(package_file).parent, package)
    runtime = make_runtime(
        root / "runtime", demonstrations=config.mode != "interaction"
    )
    write(root / "inputs/dataset.json", selected)
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    }
    write(root / "inputs/environment.json", environment)
    metadata = root / "inputs/metadata"
    inputs = {
        name: sha256(root / name)
        for name in ("inputs/dataset.json", "inputs/environment.json")
    }
    for parent in Path(package_file).resolve().parents:
        lock = parent / "uv.lock"
        if lock.is_file():
            shutil.copy2(lock, root / "inputs/uv.lock")
            inputs["inputs/uv.lock"] = sha256(root / "inputs/uv.lock")
            break
    counts = {}
    for task in tasks:
        item = selected["tasks"][task]
        records = item["episodes"]
        training_records = adapter.select_training(records, config.training_seed_count)
        counts[task] = len(records) if config.mode != "interaction" else 0
        train = {int(row["episode"]): row for row in adapter.records(task, "train")}
        test = adapter.records(task, "test")
        for row in records:
            native = train[int(row["episode"])]
            if (
                int(native["seed"]) != int(row["seed"])
                or native["difficulty"] != row["difficulty"]
            ):
                raise ValueError("Selected training identity changed")
        if {int(row["seed"]) for row in test} & {int(row["seed"]) for row in records}:
            raise ValueError("Training/test seed overlap")
        for split, rows in [
            ("train", [train[int(row["episode"])] for row in training_records]),
            ("test", test),
        ]:
            path = metadata / split / f"record_dataset_{task}_metadata.json"
            write(path, {"env_id": task, "records": rows})
            inputs[path.relative_to(root).as_posix()] = sha256(path)
        task_root = root / "tasks" / task
        workspace = task_root / "workspace"
        files = {
            "task.json": (
                '{"task": '
                + json.dumps(task)
                + ', "runtime": "python", "memory": []}\n'
            ).encode(),
            "policy.md": b"Document the generated task policy and its evidence.\n",
            "tools/policy.py": STARTER.encode(),
        }
        materialize(workspace, files)
        materialize(task_root / "program", files)
        (workspace / "scratch").mkdir()
        (task_root / "feedback").mkdir()
        (task_root / "logs").mkdir()
        prompt = render(config, task, records, training_records)
        (workspace / "PROTOCOL.md").write_text(prompt)
        (task_root / "prompt.md").write_text(prompt)
        inputs[(task_root / "prompt.md").relative_to(root).as_posix()] = sha256(
            task_root / "prompt.md"
        )
        copy_tree(runtime / "cam", workspace / "api_reference")
        cfg = {
            "task": task,
            "mode": config.mode,
            "budget": config.budget,
            "task_root": str(task_root),
            "ledger": str(task_root / "state/ledger.sqlite"),
            "train_episodes": [int(row["episode"]) for row in training_records],
            "train_records": training_records,
            "depth": config.depth,
        }
        write(task_root / "state/config.json", cfg)
        inputs[(task_root / "state/config.json").relative_to(root).as_posix()] = sha256(
            task_root / "state/config.json"
        )
        Ledger(cfg["ledger"]).initialize(cfg)
        snapshot(cfg, "initialized")
        if config.mode != "interaction":
            target = task_root / "demonstrations" / Path(item["file"]).name
            target.parent.mkdir()
            try:
                os.link(dataset / item["file"], target)
            except OSError:
                shutil.copy2(dataset / item["file"], target)
            inputs[target.relative_to(root).as_posix()] = item["sha256"]
    (root / "runner.py").write_text("""from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'inputs/harness'))
from cam.cli import main
raise SystemExit(main())
""")
    revision = subprocess.run(
        ["git", "-C", str(adapter.source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    manifest = {
        "schema": "cam-run/1",
        "method": "Code-as-Memory",
        "created": time.time(),
        "config": config.to_dict(),
        "tasks": tasks,
        "demonstrations_per_task": counts,
        "dataset_reference": str(dataset / "manifest.json"),
        "raw_data_reference": selected["raw_data"],
        "benchmark_package": adapter.package,
        "benchmark_hash": tree_hash(benchmark),
        "benchmark_source_reference": str(adapter.source),
        "benchmark_git_revision": revision.stdout.strip()
        if revision.returncode == 0
        else None,
        "harness_hash": tree_hash(package),
        "runtime_hash": tree_hash(runtime),
        "inputs": inputs,
        "test_episodes": {
            task: [int(row["episode"]) for row in adapter.records(task, "test")]
            for task in tasks
        },
        "evaluation_api_calls": 0,
        "test_feedback_to_coding": False,
        "fresh_policy": True,
        "checkpoint_rule": "R_N is saved before attempt N+1; final code is saved at early stop",
    }
    write(root / "run.json", manifest)
    write_run_readme(root, manifest)
    write(root / "prepared.json", {"ready": True, "created": time.time()})
    return manifest


def validate_run(root, full_data=False):
    root = Path(root).resolve()
    plan = read(root / "run.json")
    if plan.get("schema") != "cam-run/1" or not (root / "prepared.json").is_file():
        raise ValueError("Expected a prepared Code-as-Memory run")
    expected = read(root / "inputs/environment.json")["packages"]
    runtime_packages = {
        "numpy",
        "scipy",
        "h5py",
        "pillow",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg",
        "torch",
        "torchvision",
        "sapien",
        "mani-skill",
        "mplib",
        "transforms3d",
        "trimesh",
        "toppra",
        "fast-kinematics",
    }
    for name, version in expected.items():
        if name.lower().replace("_", "-") in runtime_packages:
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            if actual != version:
                raise ValueError(
                    f"Pinned runtime dependency changed: {name} {version} -> {actual}"
                )
    for relative, key in [
        ("inputs/benchmark/" + plan["benchmark_package"], "benchmark_hash"),
        ("inputs/harness/cam", "harness_hash"),
        ("runtime", "runtime_hash"),
    ]:
        if tree_hash(root / relative) != plan[key]:
            raise ValueError(f"Pinned source changed: {relative}")
    for relative, expected in plan["inputs"].items():
        if relative.endswith(".h5") and not full_data:
            continue
        if sha256(root / relative) != expected:
            raise ValueError(f"Pinned input changed: {relative}")
    return plan


def pinned_environment(root):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(root).resolve() / "inputs/harness")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def use_pinned_cli(root, argv):
    expected = Path(root).resolve() / "inputs/harness/cam"
    if Path(package_file).resolve().parent != expected:
        os.execve(
            sys.executable,
            [sys.executable, str(Path(root).resolve() / "runner.py"), *argv],
            pinned_environment(root),
        )
