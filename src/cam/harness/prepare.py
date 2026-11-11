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
from cam.interface.base import pin_source, source_hash
from cam.interface.registry import checkout, get_adapter, simulator_python
from cam.config import COMPARISON_SUITES, RunConfig
from cam.core.budget import Ledger, program_hash
from cam.core.history import freeze, materialize, payload, snapshot
from cam.data.extract import load_dataset_manifest, raw_data_reference, validate_dataset
from cam.io import copy_tree, read, sha256, tree_hash, write
from cam.prompts import benchmark_mode_guidance, render, task_clarification
from cam.harness.run_readme import write_run_readme, write_runs_readme
from cam.harness.isolation import policy_python

STARTER_TEMPLATE = '''def perceive_decide(obs, goal, memory, last):
    """Replace this observation-only template with a general task program."""
    return {"memory": {}, "act": {"tool": "act_observe", "args": {}}}

def act_observe(robot):
    """Observe one native step; this template does not solve the task."""
    STEP
    return {"observed": True}
'''
STARTER = STARTER_TEMPLATE.replace("STEP", "robot.step(*robot.now().tcp, 1.0)")
SNAPSHOT = (
    "import importlib.metadata as m, json, sys\n"
    "import numpy, h5py, imageio, imageio_ffmpeg\n"
    "print(json.dumps({'python': sys.version, 'executable': sys.executable, "
    "'prefix': sys.prefix, 'packages': {d.metadata['Name']: d.version "
    "for d in m.distributions() if d.metadata['Name']}}))"
)


def starter(interface):
    """The blank policy: one observation step using the benchmark's declared action."""
    return STARTER_TEMPLATE.replace("STEP", interface["starter"])


def environment_snapshot():
    return {
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    }


def simulator_snapshot(python):
    """Pin a dedicated benchmark interpreter; it must also run the trusted broker."""
    result = subprocess.run(
        [str(python), "-I", "-c", SNAPSHOT], capture_output=True, text=True, timeout=300
    )
    if result.returncode:
        raise ValueError(
            f"Benchmark interpreter {python} is unusable (needs numpy, h5py, imageio, imageio-ffmpeg): "
            + result.stderr[-2000:]
        )
    value = json.loads(result.stdout.strip().splitlines()[-1])
    value["executable"] = str(python)
    return value


def simulator_build_files(adapter):
    """Preserve the simulator's own dependency lock and installation recipe."""
    if adapter.name in ("mikasa_robo", "mikasa_robo_v2"):
        folder = adapter.source
    else:
        root = checkout()
        if root is None:
            return []
        folder = root / "envs" / adapter.name
    return [
        folder / name
        for name in ("pyproject.toml", "uv.lock", "install.sh", "install_record.txt")
        if (folder / name).is_file()
    ]


def make_runtime(output, demonstrations=True, mikasa_motion_preview=False, rmbench_robot_checks=False):
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
    if mikasa_motion_preview:
        target = package / "runtime/mikasa_robo/dry_run.md"
        target.parent.mkdir()
        shutil.copy2(source / "interface/mikasa_robo/api/api.md", target)
        (target.parent / "__init__.py").write_text('"""Public MIKASA robot-only offline API."""\nfrom .offline import safety_check\n')
        for name in ("offline.py", "panda_model.json", "safety_check.md"):
            shutil.copy2(source / "interface/mikasa_robo/api" / name, target.parent / name)
    if rmbench_robot_checks:
        target = package / "runtime/rmbench"
        target.mkdir()
        for name in ("__init__.py", "offline.py", "aloha_model.json", "api.md"):
            shutil.copy2(source / "interface/rmbench/api" / name, target / name)
    return output


def training_schedule(adapter, task, demonstrations, config):
    """Select private interaction identities without using outcomes or test feedback."""
    if config.training_seed_policy == "demo_cycle":
        if config.training_seed_count is not None and config.training_seed_count > len(demonstrations):
            raise ValueError(f"Cannot select {config.training_seed_count} training seeds from {len(demonstrations)} demos for {task}")
        selected = adapter.select_training(demonstrations, config.training_seed_count)
        allowed = {(int(row["episode"]), int(row["seed"])) for row in demonstrations}
        identities = [(int(row["episode"]), int(row["seed"])) for row in selected]
        expected = len(demonstrations) if config.training_seed_count is None else config.training_seed_count
        if (len(identities) != expected or len(set(identities)) != expected
                or not set(identities) <= allowed):
            raise ValueError(f"demo_cycle must select distinct identities only from assigned demos: {task}")
        return selected
    from cam.data.demonstrations import random_select

    native = adapter.records(task, "train")
    demo_ids = {int(r["episode"]) for r in demonstrations}
    demo_seeds = {int(r["seed"]) for r in demonstrations}
    test_seeds = {int(r["seed"]) for r in adapter.records(task, "test")}
    if len({int(r["episode"]) for r in native}) != len(native) or len({int(r["seed"]) for r in native}) != len(native):
        raise ValueError(f"Duplicate native training identities: {task}")
    candidates = [r for r in native if int(r["episode"]) not in demo_ids and int(r["seed"]) not in demo_seeds | test_seeds]
    if len(candidates) < config.training_seed_count:
        raise ValueError(f"{task}: only {len(candidates)} disjoint training seeds for {config.training_seed_count} attempts")
    return random_select(candidates, config.training_seed_count, config.benchmark + "/interaction-disjoint-v1", task)


def preparation_inputs(dataset, config, tasks=None, benchmark_source=None, validate_data=False):
    dataset = Path(dataset).resolve()
    config = config if isinstance(config, RunConfig) else RunConfig.from_dict(config)
    adapter = get_adapter(config.benchmark, benchmark_source)
    if config.perceptual_variations:
        raise ValueError(
            "Perceptual variation protocols are not implemented in this RoboMME release"
        )
    selected = validate_dataset(dataset) if validate_data else load_dataset_manifest(dataset, adapter=adapter)
    from cam.interface.registry import benchmark_matches
    if not benchmark_matches(config.benchmark, selected["benchmark"]) or selected[
        "benchmark_hash"
    ] != source_hash(adapter):
        raise ValueError("Dataset and benchmark snapshot do not match")
    if selected["depth"] != config.depth:
        raise ValueError("Extraction and policy sensor settings must agree")
    if config.benchmark in ("mikasa_robo", "mikasa_robo_v2"):
        from cam.interface.mikasa_robo.data import DECLARED
        if not selected["depth"] or selected.get("fields") != DECLARED:
            raise ValueError("MIKASA requires the released 256x256 RGB-D schema; RGB-only data belongs in legacy")
    tasks = tasks or list(selected["tasks"])
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) - set(selected["tasks"])
    ):
        raise ValueError("Choose distinct extracted tasks")
    if config.builtin_demo and not getattr(adapter, "supports_builtin_demo", True):
        raise ValueError(f"{config.benchmark} has no reset video; set builtin_demo=false")
    for task in tasks:
        adapter.interface(task, config.depth)
        pinned = {}
        for split in ("train", "test"):
            try:
                rows = adapter.records(task, split)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"{config.benchmark}/{task}: pinned {split} metadata is missing; "
                    "complete metadata preparation before preparing a run"
                ) from exc
            if not rows:
                raise ValueError(f"{config.benchmark}/{task}: empty {split} metadata")
            pinned[split] = rows
        records = selected["tasks"][task]["episodes"]
        train = {int(row["episode"]): int(row["seed"]) for row in pinned["train"]}
        test = {int(row["seed"]) for row in pinned["test"]}
        if any(train.get(int(row["episode"])) != int(row["seed"]) or int(row["seed"]) in test
               for row in records):
            raise ValueError(f"{task}: assigned demonstration identity or train/test overlap")
        training_schedule(adapter, task, records, config)
    return dataset, config, tasks, adapter, selected


def initial_policy_input(source, config, tasks, selected):
    """Explicit same-task repair; never import feedback, rollouts or another task."""
    if source is None:
        return None
    if config.mode != "demos" or config.budget != 0 or len(tasks) != 1:
        raise ValueError("Initial-policy repairs require one demo-only task and zero rollouts")
    source = Path(source).resolve()
    prior = validate_run(source)
    task = tasks[0]
    if prior["tasks"] != tasks or RunConfig.from_dict(prior["config"]) != config:
        raise ValueError("Initial-policy repair must keep the same task and configuration")
    original = read(source / "inputs/dataset.json")["tasks"][task]
    assigned = selected["tasks"][task]
    if original["sha256"] != assigned["sha256"] or original["episodes"] != assigned["episodes"]:
        raise ValueError("Initial-policy repair must keep identical assigned demonstrations")
    folder = source / "tasks" / task
    if read(folder / "logs/completion.json")["status"] != "finished":
        raise ValueError("Initial policy learner has not finished")
    policy = folder / "policy"
    frozen = read(policy / "freeze.json")
    if frozen["training"]["attempts"] != 0 or program_hash(policy) != frozen["program_hash"]:
        raise ValueError("Initial policy is not an intact frozen demo-only policy")
    return {
        "provenance": {"source_run": str(source), "task": task,
                       "program_hash": frozen["program_hash"],
                       "source_freeze_sha256": sha256(policy / "freeze.json"),
                       "benchmark_hash": prior["benchmark_hash"],
                       "kind": "user_requested_code_repair", "test_feedback_imported": False},
        "files": payload(policy),
    }


def prepare(root, dataset, config, tasks=None, benchmark_source=None, initial_policy_run=None,
            validate_data=False):
    """Prepare an exact run path; retained for embedded callers and old runs."""
    dataset, config, tasks, adapter, selected = preparation_inputs(dataset, config, tasks, benchmark_source, validate_data)
    initial = initial_policy_input(initial_policy_run, config, tasks, selected)
    return _prepare(Path(root).resolve(), dataset, config, tasks, adapter, selected, initial, validate_data=validate_data)


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
    base, dataset, config, tasks=None, benchmark_source=None, timetag=None, initial_policy_run=None,
    validate_data=False,
):
    """Create one task per benchmark/task/client-model/setting/timetag leaf."""
    dataset, config, tasks, adapter, selected = preparation_inputs(
        dataset, config, tasks, benchmark_source, validate_data
    )
    initial = initial_policy_input(initial_policy_run, config, tasks, selected)
    timetag = timetag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if timetag in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", timetag):
        raise ValueError("Timetag must be a single alphanumeric path component")
    base = Path(base).resolve()
    client = coding_run_name(config.coding)
    roots = [
        base / (config.suite or config.benchmark) / task / client /
        (f"{config.setting}_{len(selected['tasks'][task]['episodes'])}" if config.suite else config.setting) / timetag
        for task in tasks
    ]
    for root in roots:
        if root.exists():
            raise FileExistsError(f"Run already exists; choose a new timetag: {root}")
    base.mkdir(parents=True, exist_ok=True)
    write_runs_readme(base)
    results = []
    for task, root in zip(tasks, roots):
        _prepare(root, dataset, config, [task], adapter, selected, initial, validate_data=validate_data)
        results.append({"task": task, "root": str(root)})
    return {
        "schema": "cam-preparation/1",
        "benchmark": config.benchmark,
        "setting": config.setting,
        "timetag": timetag,
        "runs": results,
    }


def _prepare(root, dataset, config, tasks, adapter, selected, initial=None, published=None,
             validate_data=False):
    if config.suite and set(tasks) - set(COMPARISON_SUITES[config.suite]["tasks"]):
        raise ValueError(f"Tasks are outside comparison suite {config.suite}")
    if root.exists():
        raise FileExistsError(
            "Use a new run directory; existing logs and budgets cannot be reset"
        )
    root.mkdir(parents=True)
    benchmark = root / "inputs/benchmark" / adapter.package
    assets = pin_source(adapter, benchmark)
    if initial and tree_hash(benchmark) != initial["provenance"]["benchmark_hash"]:
        raise ValueError("Initial-policy repair must keep the same native benchmark snapshot")
    package = root / "inputs/harness/cam"
    copy_tree(Path(package_file).parent, package, ("legacy",))
    runtime = make_runtime(
        root / "runtime", demonstrations=config.mode != "interaction",
        mikasa_motion_preview=adapter.name in ("mikasa_robo", "mikasa_robo_v2"),
        rmbench_robot_checks=adapter.name == "rmbench",
    )
    write(root / "inputs/dataset.json", selected)
    write(root / "inputs/environment.json", environment_snapshot())
    metadata = root / "inputs/metadata"
    inputs = {
        name: sha256(root / name)
        for name in ("inputs/dataset.json", "inputs/environment.json")
    }
    if initial:
        saved = root / "inputs/initial_policy"
        materialize(saved, initial["files"])
        for name in initial["files"]:
            path = saved / name
            inputs[path.relative_to(root).as_posix()] = sha256(path)
    if published:
        write(root / "inputs/published_solution.json", published["provenance"])
        inputs["inputs/published_solution.json"] = sha256(root / "inputs/published_solution.json")
        for task in tasks:
            saved = root / "inputs/published_policy" / task
            materialize(saved, published["files"][task])
            for name in published["files"][task]:
                path = saved / name
                inputs[path.relative_to(root).as_posix()] = sha256(path)
    simulator = simulator_python(adapter)
    if simulator is not None:
        write(root / "inputs/simulator_environment.json", simulator_snapshot(simulator))
        inputs["inputs/simulator_environment.json"] = sha256(
            root / "inputs/simulator_environment.json"
        )
        for source in simulator_build_files(adapter):
            target = root / "inputs/simulator_build" / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            inputs[target.relative_to(root).as_posix()] = sha256(target)
    interfaces = {task: adapter.interface(task, config.depth) for task in tasks}
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
        training_records = [] if published else training_schedule(adapter, task, records, config)
        counts[task] = len(records) if config.mode != "interaction" else 0
        train = {} if published else {int(row["episode"]): row for row in adapter.records(task, "train")}
        test = adapter.records(task, "test")
        for row in records:
            native = train[int(row["episode"])]
            if (
                int(native["seed"]) != int(row["seed"])
                or native.get("difficulty") != row.get("difficulty")
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
            "tools/policy.py": starter(interfaces[task]).encode(),
        }
        if initial:
            files = initial["files"]
        if published:
            files = published["files"][task]
        materialize(workspace, files)
        materialize(task_root / "program", files)
        (workspace / "scratch").mkdir()
        (task_root / "feedback").mkdir()
        (task_root / "logs").mkdir()
        prompt = "Evaluation of a published frozen policy. No learning is permitted.\n" if published else render(
            config, task, records, training_records, adapter, interfaces[task],
            initial_policy=initial["provenance"] if initial else None,
        )
        (workspace / "PROTOCOL.md").write_text(prompt)
        (task_root / "prompt.md").write_text(prompt)
        inputs[(task_root / "prompt.md").relative_to(root).as_posix()] = sha256(
            task_root / "prompt.md"
        )
        copy_tree(runtime / "cam", workspace / "api_reference")
        cfg = {
            "task": task,
            "mode": config.mode,
            "both_protocol": config.both_protocol,
            "budget": config.budget,
            "task_root": str(task_root),
            "ledger": str(task_root / "state/ledger.sqlite"),
            "train_episodes": [int(row["episode"]) for row in training_records],
            "train_records": training_records,
            "training_seed_policy": config.training_seed_policy,
            "depth": config.depth,
        }
        write(task_root / "state/config.json", cfg)
        inputs[(task_root / "state/config.json").relative_to(root).as_posix()] = sha256(
            task_root / "state/config.json"
        )
        Ledger(cfg["ledger"]).initialize(cfg)
        snapshot(cfg, "imported published final policy" if published else "initialized from user-authorized prior policy" if initial else "initialized")
        if published:
            freeze(cfg)
            write(task_root / "logs/completion.json", {
                "status": "finished", "source": "published_solution",
                "learning_performed": False, "model_calls": 0,
            })
        if config.mode != "interaction" and not published:
            target = task_root / "demonstrations" / Path(item["file"]).name
            target.parent.mkdir()
            try:
                os.link(dataset / item["file"], target)
            except OSError:
                shutil.copy2(dataset / item["file"], target)
            inputs[target.relative_to(root).as_posix()] = item["sha256"]
    # The guard matters: simulators that start multiprocessing "spawn" children
    # (e.g. RMBench's CuRobo planners) re-import the main script in each child.
    (root / "runner.py").write_text("""from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'inputs/harness'))

if __name__ == "__main__":
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
        "benchmark_mode_guidance": benchmark_mode_guidance(config),
        "task_clarifications": {
            task: value for task in tasks
            if (value := task_clarification(config.benchmark, task)) is not None
        },
        "dataset_reference": str(dataset / "manifest.json") if dataset is not None else None,
        "dataset_validation": {
            "mode": "none" if published else "full" if validate_data else "published_manifest",
            "h5_scanned": bool(validate_data and not published),
            "hashes_checked": bool(validate_data and not published),
        },
        "raw_data_reference": raw_data_reference(selected),
        "benchmark_package": adapter.package,
        "benchmark_hash": tree_hash(benchmark),
        "benchmark_source_reference": str(adapter.source),
        "benchmark_git_revision": revision.stdout.strip()
        if revision.returncode == 0
        else None,
        "harness_hash": tree_hash(package),
        "runtime_hash": tree_hash(runtime),
        "interfaces": interfaces,
        "policy_python": policy_python(),
        "simulator_python": str(simulator) if simulator is not None else sys.executable,
        "benchmark_assets": assets,
        "inputs": inputs,
        "test_episodes": {
            task: [int(row["episode"]) for row in adapter.records(task, "test")]
            for task in tasks
        },
        "evaluation_api_calls": 0,
        "test_feedback_to_coding": False,
        "fresh_policy": initial is None and published is None,
        "initial_policy": initial["provenance"] if initial else None,
        "checkpoint_rule": "R_N is saved before attempt N+1; final code is saved at early stop",
    }
    if published:
        manifest["published_solution"] = published["provenance"]
        manifest["evaluation_only"] = True
    if config.benchmark in ("mikasa_robo", "mikasa_robo_v2"):
        from cam.interface.mikasa_robo.adapter import physics_backend_for_task

        manifest["physics_backend_by_task"] = {
            task: physics_backend_for_task(task, config.physics_backend) for task in tasks
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
    snapshots = [("coding", read(root / "inputs/environment.json"))]
    if (root / "inputs/simulator_environment.json").exists():
        snapshots.append(
            ("simulator", read(root / "inputs/simulator_environment.json"))
        )
    current = Path(sys.prefix).resolve()
    matches = [
        (role, value)
        for role, value in snapshots
        if Path(value.get("prefix", sys.prefix)).resolve() == current
    ]
    if not matches:
        raise ValueError(f"Interpreter {sys.executable} is not pinned for this run")
    role, snapshot = matches[-1] if len(matches) > 1 else matches[0]
    expected = snapshot["packages"]
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
        if role == "simulator" or name.lower().replace("_", "-") in runtime_packages:
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
    if full_data:
        for link, expected in plan.get("benchmark_assets", {}).items():
            path = root / "inputs/benchmark" / plan["benchmark_package"] / link
            if tree_hash(path.resolve()) != expected:
                raise ValueError(f"Pinned benchmark assets changed: {link}")
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
