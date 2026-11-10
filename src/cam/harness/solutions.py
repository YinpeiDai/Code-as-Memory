"""Prepare published final code for held-out evaluation without a learning stage."""

from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from cam.interface.base import source_hash
from cam.interface.registry import get_adapter
from cam.config import COMPARISON_SUITES, RunConfig
from cam.core.workspace import FIELD_TYPES
from cam.harness.prepare import _prepare, write_runs_readme
from cam.io import read, sha256, task_name


def record_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_solutions(solution, catalog=None, tasks=None):
    """Read declarations and code bytes; never import a policy into the host."""
    solution = Path(solution).resolve()
    if catalog is None:
        folder = solution.parents[2] / "configs/solutions" / solution.parent.name
        catalog = folder / (solution.name + ".json")
        current = folder / (solution.name + "_observed_motion.json")
        if solution.parent.name == "rmbench" and current.is_file():
            catalog = current
    catalog = Path(catalog).resolve()
    spec = read(catalog)
    if spec.get("schema") != "cam-solutions/1":
        raise ValueError("Expected a cam-solutions/1 catalog")
    for key in ("benchmark", "experiment"):
        task_name(spec[key])
    config = RunConfig.from_dict(spec["evaluation_config"])
    if config.benchmark != spec["benchmark"] or config.mode != "demos" or config.budget != 0:
        raise ValueError("Published evaluation requires a zero-training configuration")
    if config.suite != spec.get("suite"):
        raise ValueError("Published catalog and evaluation configuration must name the same suite")
    if config.suite and set(spec["tasks"]) - set(COMPARISON_SUITES[config.suite]["tasks"]):
        raise ValueError(f"Published tasks are outside comparison suite {config.suite}")
    tasks = list(spec["tasks"]) if tasks is None else tasks
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - spec["tasks"].keys():
        raise ValueError("Choose distinct published tasks")
    files = {}
    for task in tasks:
        task_name(task)
        item = spec["tasks"][task]
        memory = item["task_spec"]
        names = [f["name"] for f in memory["memory"]]
        if (memory["task"] != task or memory.get("runtime", "python") != "python"
                or len(names) != len(set(names))
                or any(f["type"] not in FIELD_TYPES for f in memory["memory"])):
            raise ValueError(f"Invalid published memory schema: {task}")
        folder = solution / task
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"Expected a regular solution directory: {folder}")
        expected = item["code_sha256"]
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
                  if "__pycache__" not in p.parts and p.is_file()}
        if not expected or actual != set(expected):
            raise ValueError(f"Published policy file set changed: {task}")
        payload = {
            "task.json": (json.dumps(memory, indent=2) + "\n").encode(),
            "policy.md": b"Published final policy. Source provenance is pinned in the run manifest.\n",
        }
        for name, expected_hash in expected.items():
            path = folder / name
            if (Path(name).name != name or not name.endswith(".py") or path.is_symlink()
                    or path.stat().st_size > 2 * 1024**2):
                raise ValueError(f"Expected a regular Python policy file: {name}")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError(f"Published policy checksum mismatch: {task}/{name}")
            compile(data, str(path), "exec")  # Syntax only; no host execution.
            payload["tools/" + name] = data
        files[task] = payload
    provenance = {**spec, "tasks": {task: spec["tasks"][task] for task in tasks},
                  "catalog_sha256": sha256(catalog), "learning_performed": False}
    return spec, tasks, {"provenance": provenance, "files": files}


def prepare_solutions(base, solution, tasks=None, benchmark_source=None, timetag=None, catalog=None, recording=None):
    """Pin a new evaluation run per task, importing only final code and memory."""
    spec, tasks, published = load_solutions(solution, catalog, tasks)
    config = RunConfig.from_dict(spec["evaluation_config"])
    if recording is not None:
        config = replace(config, recording=recording)
    adapter = get_adapter(config.benchmark, benchmark_source)
    if source_hash(adapter) != spec["benchmark_source_sha256"]:
        raise ValueError("Published solution and installed benchmark source do not match")
    for task in tasks:
        item = spec["tasks"][task]
        records = adapter.records(task, "test")
        if (not records or len({int(r["episode"]) for r in records}) != len(records)
                or record_hash(records) != item["test_records_sha256"]):
            raise ValueError(f"Published test identities changed: {task}")
        if record_hash(adapter.interface(task, config.depth)) != item["interface_sha256"]:
            raise ValueError(f"Published policy interface changed: {task}")
    timetag = timetag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if timetag in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", timetag):
        raise ValueError("Timetag must be a single alphanumeric path component")
    base = Path(base).resolve()
    roots = [base / (config.suite or spec["benchmark"]) / task / "published" / spec["experiment"] / timetag for task in tasks]
    if any(root.exists() for root in roots):
        raise FileExistsError("Published evaluation run already exists; choose a new timetag")
    selected = {"schema": "cam-published-evaluation/1", "benchmark": spec["benchmark"],
                "raw_data": None, "tasks": {task: {"episodes": []} for task in tasks}}
    base.mkdir(parents=True, exist_ok=True)
    write_runs_readme(base)
    runs = []
    for task, root in zip(tasks, roots):
        one = {"provenance": {**published["provenance"], "tasks": {task: spec["tasks"][task]}},
               "files": {task: published["files"][task]}}
        _prepare(root, None, config, [task], adapter, selected, published=one)
        runs.append({"task": task, "root": str(root)})
    return {"schema": "cam-preparation/1", "benchmark": spec["benchmark"],
            "suite": config.suite, "setting": spec["experiment"], "evaluation_only": True,
            "timetag": timetag, "runs": runs}
