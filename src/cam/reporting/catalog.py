"""Portable run-directory index for published archives and live experiments.

Indexing reads manifests and JSON logs only. Images and videos are loaded on
demand. Historical snapshots, scored results and demonstration files stay intact.
"""

from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

from cam.io import read, write
from cam.config import COMPARISON_SUITES

BENCHMARKS = ("robomme", "mikasa_robo", "rmbench", "robomemarena")
BEST_SETTINGS = {"robomme": "demo_only_25", "mikasa_robo": "demo_interaction_50",
                 "rmbench": "demo_only_50", "robomemarena": "demo_interaction_50"}
SKIP = {"inputs", "runtime", "workspace", "history", "audit", "checkpoints", "policy",
        "frozen_policy", "benchmark", "harness", "runner", "sandbox_programs", "source_run",
        "demonstrations", "feedback", "rollouts", "test", "train", "demos", "eval", "solutions",
        "configs", "catalogs", "website", "dashboard", "__pycache__", ".venv", "infrastructure_retries"}


def load(path, default=None):
    try:
        return read(path)
    except (OSError, ValueError):
        return {} if default is None else default


def stable_id(*values):
    return hashlib.sha256("/".join(map(str, values)).encode()).hexdigest()[:20]


def canonical(name):
    return "mikasa_robo" if name == "mikasa_robo_v2" else name


def find_runs(root):
    """Prune simulator snapshots and artifact trees; never follow directory loops."""
    seen = set()
    for directory, children, files in os.walk(root, followlinks=True):
        path = Path(directory)
        resolved = path.resolve()
        if resolved in seen:
            children[:] = []
            continue
        seen.add(resolved)
        if "run.json" in files:
            yield path
            children[:] = []
        else:
            children[:] = sorted(c for c in children if c not in SKIP and not c.startswith((".", "_")))


def progress_batches(root):
    """Include queued work before its first task preparation exists."""
    result, seen = [], set()
    for directory, children, files in os.walk(root, followlinks=True):
        path = Path(directory)
        if path.resolve() in seen:
            children[:] = []
            continue
        seen.add(path.resolve())
        children[:] = sorted(c for c in children if c not in SKIP and not c.startswith((".", "_")))
        if "run.json" in files:
            children[:] = []
            continue
        if "status.json" not in files:
            continue
        state = load(path / "status.json")
        if str(state.get("schema", "")).startswith("cam-solution-check/"):
            result.append({"id": stable_id(_within(root, path)), "tag": path.name, "kind": "Sanity evaluation",
                           "status": state.get("status", "unknown"), "completed": state.get("completed", 0),
                           "expected": state.get("expected", 0), "unit": "episodes", "updated": state.get("updated"),
                           "episodes_per_policy": state.get("episodes_per_policy", 1),
                           "running": sum(v.get("status") == "running" for v in state.get("tasks", {}).values()),
                           "task_status": {k: v.get("status") for k, v in state.get("tasks", {}).items()}})
        elif (path / "plan.json").is_file():
            plan = load(path / "plan.json")
            if "benchmarks" in plan:
                jobs = state.get("jobs", state.get("tasks", {}))
                rows = list(jobs.values()) if isinstance(jobs, dict) else jobs
                result.append({"id": stable_id(_within(root, path)), "tag": path.name,
                               "kind": "Evaluation" if state.get("schema") == "cam-published-evaluation/1" else "Development",
                               "status": state.get("status", "unknown"),
                               "completed": sum(r.get("status") in ("finished", "complete", "frozen") for r in rows),
                               "expected": sum(len(b["tasks"]) for b in plan["benchmarks"].values()), "unit": "tasks",
                               "updated": state.get("updated"),
                               "running": sum(r.get("status") in ("running", "learning", "preparing", "extracting") for r in rows)})
    return result


def _within(root, path):
    """Archive-relative names remain portable, including its compatibility links."""
    path = Path(path).absolute()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _files(folder, patterns, root):
    found = set()
    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_file():
                relative = _within(root, path)
                if relative:
                    found.add(relative)
    return sorted(found)


def _state(folder, frozen):
    completion = load(folder / "logs/completion.json")
    if frozen:
        return "frozen"
    if completion:
        return completion.get("status", "unknown")
    sessions = sorted((folder / "logs").glob("session_*"))
    if sessions:
        session = sessions[-1]
        if (session / "completion.json").is_file():
            return load(session / "completion.json").get("status", "finished")
        pid = load(session / "process.json").get("pid")
        if pid:
            try:
                os.kill(int(pid), 0)
                return "running"
            except (OSError, ValueError):
                return "interrupted"
        return "starting"
    return "prepared"


def _episode(path, root, run_id, benchmark, setting, task, kind):
    raw = load(path)
    if not raw or "episode" not in raw:
        return None
    relative = _within(root, path)
    row = {k: raw.get(k) for k in ("episode", "status", "success", "steps", "seconds", "decisions", "goal", "attempt", "program_hash", "harness_hash", "benchmark_hash")}
    row.update(id=stable_id(relative), run_id=run_id, benchmark=benchmark, setting=setting,
               task=task, kind=kind, scorable=raw.get("scorable", raw.get("status") not in ("harness_error", "infrastructure_error")),
               result=relative, checkpoint=raw.get("maximum_rollouts", 0))
    for key, name in (("video", "episode.mp4"), ("observations", "observations.h5"),
                      ("actions", "actions.npy"), ("decisions_log", "decisions.jsonl"), ("stderr", "policy_stderr.txt")):
        file = path.parent / name
        row[key] = _within(root, file) if file.is_file() else None
    row["media_available"] = bool(row["video"])
    return row


def aggregate(rows, expected):
    scored = [r for r in rows if r.get("scorable")]
    successes = sum(bool(r.get("success")) for r in scored)
    complete = expected > 0 and len(rows) == len(scored) == expected
    return {"expected": expected, "recorded": len(rows), "scored": len(scored),
            "successes": successes, "infrastructure_errors": len(rows) - len(scored),
            "complete": complete, "success_rate": successes / expected if complete else None,
            "observed_success_rate": successes / len(scored) if scored else None}


def index_directory(data_root):
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    archive = (root / "eval").is_dir() and (root / "dev_logs").is_dir()
    batches = [] if archive else progress_batches(root)
    runs, episodes, demos, warnings = [], [], [], []
    run_map, seen_results = {}, set()
    # Optional release catalogs carry expected counts and source prompt provenance.
    catalogs = {}
    for path in sorted((root / "configs/solutions").glob("*/*.json")):
        spec = load(path)
        if spec.get("schema") == "cam-solutions/1":
            catalogs[(spec.get("suite") or canonical(spec["benchmark"]), spec["experiment"])] = spec

    def add_episodes(folder, patterns, run, kind):
        for pattern in patterns:
            for path in sorted(folder.glob(pattern)):
                identity = str(path.resolve())
                if identity in seen_results:
                    continue
                row = _episode(path, root, run["id"], run["benchmark"], run["setting"], run["task"], kind)
                if row:
                    seen_results.add(identity)
                    episodes.append(row)
                else:
                    warnings.append(f"Unreadable or incomplete result: {_within(root, path)}")

    for directory in find_runs(root / "dev_logs" if archive else root):
        plan = load(directory / "run.json")
        config = plan.get("config", {})
        simulator_benchmark = canonical(config.get("benchmark"))
        if simulator_benchmark not in BENCHMARKS:
            continue
        benchmark = config.get("suite") or simulator_benchmark
        if config.get("suite") and benchmark not in COMPARISON_SUITES:
            warnings.append(f"Unknown comparison suite: {benchmark}")
            continue
        for task in plan.get("tasks", []):
            folder = directory / "tasks" / task
            relative = _within(root, directory)
            setting = (Path(relative).parts[2] if archive else
                       plan.get("published_solution", {}).get("experiment") or
                       directory.parent.name)
            if not archive:
                published = load(directory / "inputs/published_solution.json")
                setting = published.get("experiment", setting)
            sanity = load(directory / "sanity_check.json")
            frozen = (folder / "policy/freeze.json").is_file()
            item = {"id": stable_id(relative, task), "benchmark": benchmark, "setting": setting,
                    "simulator_benchmark": simulator_benchmark, "suite": config.get("suite"),
                    "physics_backend": plan.get("physics_backend_by_task", {}).get(task, config.get("physics_backend")),
                    "task": task, "tag": directory.name, "path": relative,
                    "scope": "sanity" if sanity else "evaluation", "created": plan.get("created"),
                    "development_status": _state(folder, frozen), "frozen": frozen,
                    "evaluation_only": plan.get("evaluation_only", False),
                    "expected": sanity.get("expected", len(plan.get("test_episodes", {}).get(task, []))),
                    "assigned_demos": plan.get("demonstrations_per_task", {}).get(task, 0),
                    "budget": config.get("budget", 0), "model": config.get("coding", {}).get("model", ""),
                    "prompt": _within(root, folder / "prompt.md") if (folder / "prompt.md").is_file() else None,
                    "manifest": _within(root, directory / "run.json"),
                    "logs": _files(folder, ["logs/session_*/events.jsonl", "logs/session_*/*.txt", "logs/session_*/*.log"], root),
                    "code": _files(folder / "policy", ["tools/*.py", "task.json"], root),
                    "interface": plan.get("interfaces", {}).get(task), "demo_ids": []}
            item["group_tag"] = Path(sanity["batch"]).name if sanity.get("batch") else item["tag"]
            queue = load(directory / "queues/evaluation/status.json")
            item["evaluation_status"] = ("running" if any(j.get("status") == "running" for j in queue.get("jobs", {}).values())
                                          else queue.get("status", "pending"))
            for batch in batches:
                if batch["tag"] == item["group_tag"]:
                    # Current interface catalogs may append their revision to the setting name.
                    key = "/".join((benchmark, setting.removesuffix("_observed_motion"), task))
                    item["evaluation_status"] = batch.get("task_status", {}).get(key, item["evaluation_status"])
            if item["evaluation_only"]:
                item["development_status"] = "published policy"
            # A run's assigned demos are portable inside its own preparation.
            manifest = load(directory / "inputs/dataset.json")
            assignment = manifest.get("tasks", {}).get(task, {})
            if assignment.get("file") and not archive:
                h5 = folder / "demonstrations" / Path(assignment["file"]).name
                for e in assignment.get("episodes", []):
                    demo = _demo(root, benchmark, setting, task, h5, e, item["interface"], item["id"])
                    demos.append(demo)
                    item["demo_ids"].append(demo["id"])
            else:
                item["assigned_episode_ids"] = [e["episode"] for e in assignment.get("episodes", [])]
            runs.append(item)
            run_map[(benchmark, setting, task)] = item
            add_episodes(folder, ["rollouts/train/R*/result.json"], item, "train")
            if not archive:
                add_episodes(folder, ["rollouts/test/R*/ep*/result.json"], item, "test")

    if archive:
        # Public archives may have a native historical layout or results only.
        for benchmark in BENCHMARKS:
            for setting_dir in sorted((root / "eval" / benchmark).glob("*")):
                if not setting_dir.is_dir() or setting_dir.name.startswith("_"):
                    continue
                setting = setting_dir.name
                spec = catalogs.get((benchmark, setting), {})
                for task_dir in sorted(setting_dir.iterdir()):
                    if not task_dir.is_dir() or task_dir.name.startswith("_"):
                        continue
                    task = task_dir.name
                    run = run_map.get((benchmark, setting, task))
                    task_spec = spec.get("tasks", {}).get(task, {})
                    if run is None:
                        manifest = load(task_dir / "manifest.json")
                        historical = root / "dev_logs" / benchmark / setting / task
                        has_dev = (historical / "agents").exists()
                        run = {"id": stable_id(benchmark, setting, task), "benchmark": benchmark,
                               "setting": setting, "task": task, "tag": task_spec.get("source_run_tag", "published"),
                               "path": _within(root, task_dir), "scope": "evaluation", "frozen": True,
                               "development_status": "archived" if has_dev else "unavailable",
                               "evaluation_only": False, "assigned_demos": task_spec.get("assigned_demos"),
                               "budget": task_spec.get("training_attempts"),
                               "expected": task_spec.get("archived_result", {}).get("expected", manifest.get("episodes_per_task", 0)),
                               "model": spec.get("source_coding", {}).get("model", ""),
                               "prompt": _within(root, historical / "prompt.txt") if has_dev else None,
                               "manifest": _within(root, task_dir / "manifest.json") if manifest else None,
                               "logs": _files(historical, ["agents/agent.jsonl", "agents/agent.err"], root) if has_dev else [],
                               "code": _files(root / "solutions" / benchmark / setting / task, ["*.py"], root),
                               "interface": None, "demo_ids": []}
                        runs.append(run)
                        run_map[(benchmark, setting, task)] = run
                    add_episodes(task_dir, ["R*/ep*/result.json", "*/R*/ep*/result.json"], run, "test")
                    if not run["expected"]:
                        source = load(task_dir / "source_run.json")
                        run["expected"] = len(source.get("test_episodes", {}).get(task, []))
                    run["comparison_note"] = spec.get("comparison_note", spec.get("compatibility_note", ""))
                    run["provenance"] = {k: task_spec.get(k) for k in ("source_prompt_sha256", "source_harness_sha256", "source_program_hash")}

    # Dataset manifests already contain camera declarations and episode lists.
    for path in sorted((root / "demos").glob("*/manifest.json")) + sorted((root / "demos").glob("*/subsets/*/manifest.json")):
        manifest = load(path)
        benchmark = canonical(manifest.get("benchmark", path.parent.name))
        setting = path.parent.name if "subsets" in path.parts else "pool"
        for task, assignment in manifest.get("tasks", {}).items():
            h5 = path.parent / assignment["file"]
            fields = manifest.get("fields", {})
            cameras = [name for name in (fields.get("frame", {}) if isinstance(fields, dict) else fields) if name.endswith("_rgb")]
            interface = {"video": cameras or ["front_rgb", "wrist_rgb"]}
            for e in assignment.get("episodes", []):
                demo = _demo(root, benchmark, setting, task, h5, e, interface)
                if e.get("video") and (path.parent / e["video"]).is_file():
                    demo["video"] = _within(root, path.parent / e["video"])
                demos.append(demo)

    groups = {}
    # A queued solution is visible before its immutable task run is prepared.
    # This is a plan entry with zero results, never fabricated evaluation evidence.
    for batch in batches:
        if batch["status"] in ("stopped", "superseded"):
            continue
        for key, status in batch.get("task_status", {}).items():
            benchmark, setting, task = key.split("/", 2)
            if any(r["benchmark"] == benchmark and r["setting"].removesuffix("_observed_motion") == setting
                   and r["task"] == task and r.get("group_tag", r["tag"]) == batch["tag"] for r in runs):
                continue
            runs.append({"id": stable_id(batch["id"], key), "benchmark": benchmark, "setting": setting,
                         "task": task, "tag": batch["tag"], "group_tag": batch["tag"], "path": None,
                         "scope": "sanity", "created": None, "development_status": "published policy",
                         "evaluation_status": "queued" if status == "pending" else status,
                         "frozen": True, "evaluation_only": True, "queued": True,
                         "expected": batch["episodes_per_policy"], "assigned_demos": 0,
                         "budget": 0, "model": "", "prompt": None, "manifest": None,
                         "logs": [], "code": [], "interface": None, "demo_ids": []})
    for run in runs:
        test = [e for e in episodes if e["run_id"] == run["id"] and e["kind"] == "test"]
        # Multiple checkpoints are separate results; the final configured checkpoint is the headline.
        final = max((e["checkpoint"] for e in test), default=0)
        final_rows = [e for e in test if e["checkpoint"] == final]
        run["summary"] = aggregate(final_rows, run["expected"])
        if run["summary"]["complete"]:
            run["evaluation_status"] = "complete"
        elif run["summary"]["infrastructure_errors"]:
            run["evaluation_status"] = "needs attention"
        else:
            run.setdefault("evaluation_status", "partial" if test else "pending")
        run["training_attempts"] = sum(e["run_id"] == run["id"] and e["kind"] == "train" for e in episodes)
        run["media_count"] = sum(bool(e.get("video")) for e in test)
        if archive:
            assigned = set(run.get("assigned_episode_ids", []))
            candidates = [d for d in demos if d["benchmark"] == run["benchmark"] and d["task"] == run["task"]
                          and d["setting"] == ("demo_only_25" if run["setting"] == "demo_only_25" and run["benchmark"] != "robomme" else "pool")]
            run["demo_ids"] = [d["id"] for d in candidates if not assigned or d["episode"] in assigned]
        # Each live timetag remains separate; archives explicitly select one run per task.
        group_key = (run["benchmark"], run["setting"], run["scope"], "published" if archive else run.get("group_tag", run["tag"]))
        group = groups.setdefault(group_key, {"id": stable_id(*group_key), "benchmark": run["benchmark"],
                                  "setting": run["setting"], "scope": run["scope"], "tag": group_key[3],
                                  "tasks": [], "note": run.get("comparison_note", "")})
        group["tasks"].append(run["id"])
    for group in groups.values():
        selected = [r for r in runs if r["id"] in group["tasks"]]
        summaries = [r["summary"] for r in selected]
        group["summary"] = {k: sum(s[k] for s in summaries) for k in ("expected", "recorded", "scored", "successes", "infrastructure_errors")}
        s = group["summary"]
        s["complete"] = all(r["complete"] for r in summaries)
        s["success_rate"] = s["successes"] / s["expected"] if s["complete"] and s["expected"] else None
    newest = {}
    for run in runs:
        key = (run["benchmark"], run["setting"].removesuffix("_observed_motion"), run["task"])
        rank = (run.get("group_tag", run["tag"]), run["tag"])
        if key not in newest or rank > newest[key][0]:
            newest[key] = (rank, run["id"])
    for run in runs:
        key = (run["benchmark"], run["setting"].removesuffix("_observed_motion"), run["task"])
        run["latest"] = archive or newest[key][1] == run["id"]
    content = {"schema": "cam-directory/1", "name": root.name, "updated": time.time(), "archive": archive,
               "default_settings": BEST_SETTINGS,
               "runs": runs, "episodes": episodes, "demos": demos, "groups": list(groups.values()),
               "batches": batches, "warnings": warnings}
    dashboard = root / "dashboard"
    write(dashboard / "index.json", content)
    write(dashboard / "results.json", {"updated": content["updated"], "groups": content["groups"],
                                       "tasks": [{k: r[k] for k in ("id", "benchmark", "setting", "task", "scope", "summary")} for r in runs]})
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["benchmark", "setting", "task", "scope", "expected", "scored", "successes", "complete", "success_rate"])
    for r in runs:
        writer.writerow([r[k] for k in ("benchmark", "setting", "task", "scope")] + [r["summary"][k] for k in ("expected", "scored", "successes", "complete", "success_rate")])
    _text(dashboard / "results.csv", stream.getvalue())
    lines = ["# Robot Memory progress", "", f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", "",
             "| Benchmark | Setting | Task | Development | Scored / expected |", "|---|---|---|---|---|"]
    for batch in content["batches"]:
        lines.insert(4, f"{batch['kind']} {batch['tag']}: {batch['status']}; {batch['completed']}/{batch['expected']} {batch['unit']}.\n")
    for r in runs:
        lines.append(f"| {r['benchmark']} | {r['setting']} | {r['task']} | {r['development_status']} | {r['summary']['scored']}/{r['expected']} |")
    _text(dashboard / "PROGRESS.md", "\n".join(lines) + "\n")
    return content


def _text(path, text):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text)
    temporary.replace(path)


def _demo(root, benchmark, setting, task, h5, episode, interface, run_id=None):
    relative = _within(root, h5)
    return {"id": stable_id(relative, episode["episode"]), "benchmark": benchmark, "setting": setting,
            "task": task, "episode": episode["episode"], "frames": episode.get("frames"),
            "goal": episode.get("goal", ""), "h5": relative, "video": None, "kind": "demo",
            "cameras": (interface or {}).get("video", ["front_rgb", "wrist_rgb"]), "run_id": run_id}
