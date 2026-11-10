"""Shared website export and narrowly scoped artifact server for CaM runs."""

from functools import partial
import gzip
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import unquote, urlsplit

from cam.core.budget import program_hash
from cam.io import read, write
from cam.reporting.devlog import build_timeline
from cam.reporting.report import build, matched_efficiency
from cam.reporting.reference import encode_video, references
from cam.reporting.trace import cached_poster, cached_trace


def accepts_gzip(header):
    for item in header.split(','):
        parts = [part.strip() for part in item.split(';')]
        if parts[0].lower() == 'gzip':
            try:
                return next((float(p[2:]) for p in parts[1:] if p.startswith('q=')), 1.0) > 0
            except ValueError:
                return False
    return False


def task_interface(benchmark, task, depth):
    """The adapter's declared policy interface (sensors and action), if available."""
    try:
        from cam.interface.registry import get_adapter

        return get_adapter(benchmark).interface(task, depth)
    except Exception:  # an unavailable adapter must not break the site
        return None


def recorded_fields(manifest, path):
    """Recorded demonstration arrays: the v2 declaration, else read from the H5."""
    declared = manifest.get("fields")
    if declared and "frame" in declared:
        return {
            "frame": {k: v for k, v in declared["frame"].items()},
            "static": declared.get("static", {}),
        }
    try:
        import h5py

        with h5py.File(path, "r") as h5:
            group = h5[next(iter(h5))]
            length = len(group["step"])
            frame, static = {}, {}
            for name, data in group.items():
                target = frame if data.shape[:1] == (length,) and name not in ("front_K", "wrist_K") else static
                shape = list(data.shape[1:] if target is frame else data.shape)
                target[name] = {"dtype": str(data.dtype), "shape": shape, "description": ""}
            return {"frame": frame, "static": static}
    except Exception:
        return None


def messages(logs):
    result = []
    for path in sorted(logs.glob("session_*/events.jsonl")):
        with path.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "assistant":
                    for item in event.get("message", {}).get("content", []):
                        if item.get("type") in ("text", "thinking"):
                            result.append(
                                {
                                    "session": path.parent.name,
                                    "type": item["type"],
                                    "text": item.get("text", item.get("thinking", "")),
                                }
                            )
                item = event.get("item", {})
                if event.get("type") == "item.completed" and item.get("type") in (
                    "agent_message",
                    "reasoning",
                ):
                    result.append(
                        {
                            "session": path.parent.name,
                            "type": item["type"],
                            "text": item.get("text", ""),
                        }
                    )
    return result


def batch_status(batch):
    """Follow a queued experiment into its development batch without a new URL."""
    state = read(batch / "status.json")
    if "tasks" in state:
        return state
    launched = batch / "batch.json"
    if launched.is_file():
        target = Path(read(launched)["batch"]) / "status.json"
        if target.is_file():
            return read(target)
    experiment = read(batch / "experiment.json")
    plan = read(batch / "plan.json")
    status = "preparation_failed" if state["status"] == "needs_attention" else state["status"]
    tasks = {}
    for name, item in experiment["benchmarks"].items():
        for task in item["tasks"]:
            tasks[f"{name}/{task}"] = {"benchmark": name, "task": task, "status": status}
            if state.get("error"):
                tasks[f"{name}/{task}"]["error"] = state["error"]
    return {**state, "tasks": tasks, "workers": plan["workers"]}


def build_site(runs, datasets, output, batch=None, demo_limits=None, lazy_rollouts=False):
    """``demo_limits`` maps a dataset path to the number of demos shown per task."""
    demo_limits = {str(Path(k).resolve()): int(v) for k, v in (demo_limits or {}).items()}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    demo_sources = {}
    rollout_sources = {}
    batch_state = None
    if batch:
        batch = Path(batch).resolve()
        state = batch_status(batch)
        batch_state = {
            "name": batch.name, "status": state["status"], "workers": state.get("workers"),
            "tasks": [{k: row[k] for k in ("benchmark", "task", "status", "error") if k in row}
                      for row in state["tasks"].values()],
        }
        for item, row in zip(batch_state["tasks"], state["tasks"].values()):
            item["policy_frozen"] = bool(row.get("run")) and (
                Path(row["run"]) / "tasks" / row["task"] / "policy/freeze.json"
            ).is_file()
        # Append newly prepared tasks without changing existing browser run IDs.
        ready = [str(Path(row["run"]).resolve()) for row in state["tasks"].values()
                 if row.get("run") and (Path(row["run"]) / "prepared.json").is_file()]
        runs = list(dict.fromkeys([*(str(Path(r).resolve()) for r in runs), *ready]))

    def asset(path, suffix=None, deferred=False):
        # Recorded rollout paths can be checked when requested. Resolving and
        # statting every NFS artifact here otherwise dominates live refreshes.
        path = Path(path).absolute() if deferred else Path(path).resolve()
        if not deferred and not path.is_file():
            return None
        key = hashlib.sha256(str(path).encode()).hexdigest()[:24] + (suffix or path.suffix)
        artifacts[key] = str(path)
        return "artifact/" + key

    def deferred_rollout(folder, interface, kind):
        folder = Path(folder).absolute()
        key = hashlib.sha256((str(folder) + json.dumps(interface or {}, sort_keys=True)).encode()).hexdigest()[:24]
        rollout_sources[key] = {"folder": str(folder), "interface": interface}
        return f"rollout-{kind}/{key}." + ("json" if kind == "trace" else "jpg")

    def trace_url(folder, interface):
        if lazy_rollouts:
            return deferred_rollout(folder, interface, "trace")
        path = cached_trace(folder, output, interface)
        return path.relative_to(output).as_posix() if path else None

    def poster_url(folder, interface):
        if lazy_rollouts:
            return deferred_rollout(folder, interface, "poster")
        path = cached_poster(folder, output, interface)
        return path.relative_to(output).as_posix() if path else None

    content = {
        "schema": "cam-site/1",
        "updated": time.time(),
        "runs": [],
        "datasets": [],
        "batch": batch_state,
    }
    visibility_path = output / ".run_visibility.json"
    visibility = read(visibility_path) if visibility_path.is_file() else {}
    excluded_path = output / ".excluded_runs.json"
    excluded = {str(Path(source).resolve()) for source in read(excluded_path)} if excluded_path.is_file() else set()
    run_ids = {str(Path(source).resolve()): str(index) for index, source in enumerate(runs)}
    for index, source in enumerate(runs):
        root = Path(source).resolve()
        # Keep source indices stable while removing retired entries from the site.
        if str(root) in excluded:
            continue
        report = build(root)
        run = {
            "id": str(index),
            **report,
            "downloads": {
                "episodes": asset(root / "reports/episodes.csv"),
                "summary": asset(root / "reports/summary.json"),
            },
        }
        if str(root) in visibility:
            entry = visibility[str(root)]
            run.update(superseded=True, superseded_reason=entry["reason"],
                       replacement_run_id=run_ids.get(str(Path(entry["replacement"]).resolve())))
        plan = read(root / "run.json")
        interfaces = plan.get("interfaces") or {}
        for task in run["development"]:
            folder = root / "tasks" / task["task"]
            interface = interfaces.get(task["task"])
            task["reference_demos"] = references(root, plan, task["task"], output, asset, demo_sources)
            revisions = []
            for path in sorted((folder / "history/revisions").glob("*.json")):
                row = read(path)
                saved = Path(row["path"])
                revisions.append(
                    {
                        k: row[k]
                        for k in ("id", "digest", "attempts", "reason", "created")
                    }
                    | {
                        "diff": asset(path.with_suffix(".diff")),
                        "files": {
                            p.relative_to(saved).as_posix(): asset(p)
                            for p in sorted(saved.rglob("*"))
                            if p.is_file() and not p.is_symlink()
                        },
                    }
                )
            calls = []
            for path in sorted((folder / "audit").glob("call_*.json")):
                if not path.stem.removeprefix("call_").isdigit():
                    continue
                event = read(path)
                # Binary model images remain artifacts; avoid duplicating base64 in page JSON.
                response = event.get("result", {})
                text = [
                    r.get("text", "")
                    for r in response.get("content", [])
                    if r.get("type") == "text"
                ]
                calls.append(
                    {
                        k: event.get(k)
                        for k in ("id", "tool", "arguments", "started", "finished")
                    }
                    | {
                        "text": "\n".join(text),
                        "is_error": response.get("isError", False),
                        "stdout": asset(path.with_suffix(".stdout")),
                        "stderr": asset(path.with_suffix(".stderr")),
                    }
                )
            development = {
                "task": task["task"],
                "prompt": (folder / "prompt.md").read_text(),
                "messages": messages(folder / "logs"),
                "calls": calls,
                "revisions": revisions,
                "final_files": {
                    p.relative_to(folder / "policy").as_posix(): asset(p)
                    for p in sorted((folder / "policy").rglob("*"))
                    if p.is_file() and not p.is_symlink()
                },
            }
            destination = output / "development" / f"{index}_{task['task']}.json"
            write(destination, development)
            task["development_url"] = destination.relative_to(output).as_posix()
            task["rollouts"] = []
            for path in sorted((folder / "rollouts/train").glob("R*/result.json")):
                result = read(path)
                if "attempt" not in result:  # broker has not published its final provenance yet
                    continue
                checkpoint = folder / "checkpoints" / f"R{int(result['attempt']) - 1:03d}"
                frozen = read(checkpoint / "freeze.json") if (checkpoint / "freeze.json").exists() else {}
                matched = bool(frozen.get("program_hash")) and frozen["program_hash"] == result.get("program_hash")
                if matched:
                    try:
                        matched = program_hash(checkpoint) == result["program_hash"]
                    except (OSError, ValueError):
                        matched = False
                task["rollouts"].append(
                    {
                        **result,
                        "video": asset(path.parent / "episode.mp4"),
                        "poster": poster_url(path.parent, interface),
                        "observations": asset(path.parent / "observations.h5"),
                        "actions": asset(path.parent / "actions.npy"),
                        "decisions_log": asset(path.parent / "decisions.jsonl"),
                        "trace": trace_url(path.parent, interface),
                        "result_file": asset(path),
                        "policy_stderr": asset(path.parent / "policy_stderr.txt"),
                        "policy_files": {
                            p.relative_to(checkpoint).as_posix(): asset(p)
                            for p in sorted(checkpoint.rglob("*"))
                            if matched and p.is_file() and not p.is_symlink() and p.name != "freeze.json"
                        },
                        "policy_hash_verified": matched,
                    }
                )
            development["rollouts"] = task["rollouts"]
            development["timeline"] = build_timeline(folder, asset, task["rollouts"])
            write(destination, development)
        for row in run["episodes"]:
            folder = root / row["directory"]
            row.update(
                video=asset(folder / "episode.mp4", deferred=lazy_rollouts),
                poster=poster_url(folder, interfaces.get(row["task"])),
                observations=asset(folder / "observations.h5", deferred=lazy_rollouts),
                actions=asset(folder / "actions.npy", deferred=lazy_rollouts),
                decisions_log=asset(folder / "decisions.jsonl", deferred=lazy_rollouts),
                result_file=asset(folder / "result.json", deferred=lazy_rollouts),
                trace=trace_url(folder, interfaces.get(row["task"])),
            )
        content["runs"].append(run)
    for source in datasets:
        root = Path(source).resolve()
        # A dataset still being built shows its committed tasks (manifest.partial.json).
        in_progress = not (root / "manifest.json").exists() and (root / "manifest.partial.json").exists()
        manifest = read(root / ("manifest.partial.json" if in_progress else "manifest.json"))
        limit = demo_limits.get(str(root))
        if manifest["schema"] != "cam-dataset/1":
            raise ValueError("Only fresh CaM extractions are accepted")
        dataset = {
            "name": root.name,
            "benchmark": manifest["benchmark"],
            "depth": manifest["depth"],
            "selection": manifest["selection"],
            "in_progress": in_progress,
            "shown_per_task": limit,
            "tasks": [],
        }
        for name, item in manifest["tasks"].items():
            dataset["tasks"].append(
                {
                    "task": name,
                    "interface": task_interface(manifest["benchmark"], name, manifest["depth"]),
                    "recorded": recorded_fields(manifest, root / item["file"]),
                    "h5": asset(root / item["file"]),
                    "episodes": [
                        {
                            **row,
                            "video": asset(root / row["video"])
                            if row.get("video")
                            else None,
                        }
                        for row in item["episodes"][:limit]
                    ],
                    "episodes_total": len(item["episodes"]),
                }
            )
        content["datasets"].append(dataset)
    content["paired_comparisons"] = []
    # Cross-run comparisons require the identical native benchmark snapshot.
    current_runs = [run for run in content["runs"] if not run.get("superseded")]
    for identity in {r["benchmark_hash"] for r in current_runs}:
        groups = {}
        for run in current_runs:
            if run["benchmark_hash"] != identity:
                continue
            for setting in run["settings"]:
                if setting["complete"]:
                    groups[run["name"] + " / R" + str(setting["budget"])] = [
                        e
                        for e in run["episodes"]
                        if e["maximum_rollouts"] == setting["budget"]
                    ]
        content["paired_comparisons"].extend(matched_efficiency(groups))
    try:
        from cam.reporting.figures import export_figures

        figures = export_figures(current_runs, output / "figures")
        content["figures"] = ["figures/" + name for name in figures["files"]]
    except ImportError:
        content["figures"] = []  # Optional plotting extra; browser SVGs still work.
    source = Path(__file__).resolve().parents[1] / "web"
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    note = output / ".site_note.txt"
    if note.is_file():
        content["site_note"] = note.read_text().strip()
    presentation = output / '.presentation.json'
    if presentation.is_file():
        spec = read(presentation)
        if spec.get('kind') == 'robomemarena_archive':
            from cam.reporting.arena import compact_archive, summarize_archive
            content['archive'] = summarize_archive(content, spec, runs)
            compact_archive(content)
            shutil.copy2(source / 'arena.html', output / 'index.html')
    write(output / "data.json", content)
    if content.get('archive'):
        packed = output / '.data.json.gz.partial'
        packed.write_bytes(gzip.compress((output / 'data.json').read_bytes(), mtime=0))
        packed.replace(output / '.data.json.gz')
    write(output / ".artifacts.json", artifacts)
    write(output / ".demo_sources.json", demo_sources)
    write(output / ".rollout_sources.json", rollout_sources)
    write(
        output / ".sources.json",
        {
            "runs": [str(Path(p).resolve()) for p in runs],
            "datasets": [str(Path(p).resolve()) for p in datasets],
            "demo_limits": demo_limits,
            "batch": str(batch) if batch else None,
            "lazy_rollouts": lazy_rollouts,
        },
    )
    return {
        "site": str(output),
        "runs": len(content["runs"]),
        "datasets": len(datasets),
        "artifacts": len(artifacts),
    }


def serve(site, host="127.0.0.1", port=8910):
    site = Path(site).resolve()
    mutex = threading.Lock()
    demo_slots = threading.BoundedSemaphore(2)
    demo_guard = threading.Lock()
    demo_locks = {}
    rollout_slots = threading.BoundedSemaphore(2)
    rollout_guard = threading.Lock()
    rollout_locks = {}
    # The exported site is current at startup; later refreshes run in the
    # background so large sites never block a page load (writes are atomic).
    refreshed = [time.monotonic()]

    def refresh():
        try:
            sources = read(site / ".sources.json")
            if sources.get("data_root"):
                from cam.reporting.dashboard import build_dashboard
                build_dashboard(sources["data_root"], site)
            else:
                build_site(sources["runs"], sources["datasets"], site, sources.get("batch"), sources.get("demo_limits"), sources.get("lazy_rollouts", False))
        finally:
            refreshed[0] = time.monotonic()
            mutex.release()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if (
                route == "/data.json"
                and not ((site / '.presentation.json').is_file()
                         and read(site / '.presentation.json').get('live') is False)
                and time.monotonic() - refreshed[0] > 20
                and mutex.acquire(blocking=False)
            ):
                threading.Thread(target=refresh, daemon=True).start()
            self.serve_route(route, head=False)

        def do_HEAD(self):
            self.serve_route(unquote(urlsplit(self.path).path), head=True)

        def serve_route(self, route, head):
            compressed = False
            if route.startswith("/agent-log/"):
                key = route.removeprefix("/agent-log/").removesuffix(".json")
                registry = site / ".log_sources.json"
                sources = read(registry).get(key) if registry.is_file() else None
                if sources is None or not route.endswith(".json"):
                    self.send_error(404)
                    return
                from cam.reporting.dashboard import transcript
                payload = json.dumps(transcript(sources)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                if not head:
                    self.wfile.write(payload)
                return
            elif route.startswith("/preview/"):
                key = route.removeprefix("/preview/").removesuffix(".jpg")
                registry = site / ".preview_sources.json"
                spec = read(registry).get(key) if registry.is_file() else None
                if not spec or not route.endswith(".jpg"):
                    self.send_error(404)
                    return
                with demo_guard:
                    lock = demo_locks.setdefault("poster-" + key, threading.Lock())
                try:
                    with lock, demo_slots:
                        from cam.reporting.dashboard import preview
                        path = preview(spec, site, key)
                except Exception as error:
                    self.log_error("Preview %s: %s", key, error)
                    self.send_error(503, "Unable to prepare preview")
                    return
            elif route.startswith(("/rollout-trace/", "/rollout-poster/")):
                trace = route.startswith("/rollout-trace/")
                suffix = ".json" if trace else ".jpg"
                key = route.rsplit("/", 1)[-1].removesuffix(suffix)
                registry = site / ".rollout_sources.json"
                spec = read(registry).get(key) if registry.is_file() else None
                if not spec or not route.endswith(suffix):
                    self.send_error(404)
                    return
                with rollout_guard:
                    lock = rollout_locks.setdefault(key, threading.Lock())
                try:
                    with lock, rollout_slots:
                        builder = cached_trace if trace else cached_poster
                        path = builder(spec["folder"], site, spec.get("interface"))
                except Exception as error:
                    self.log_error("Rollout preview %s: %s", key, error)
                    self.send_error(503, "Unable to prepare rollout preview")
                    return
                if path is None:
                    self.send_error(404)
                    return
            elif route.startswith("/demo-video/"):
                key = route.removeprefix("/demo-video/").removesuffix(".mp4")
                registry = site / ".demo_sources.json"
                spec = read(registry).get(key) if registry.is_file() else None
                if not spec or not route.endswith(".mp4"):
                    self.send_error(404)
                    return
                with demo_guard:
                    lock = demo_locks.setdefault(key, threading.Lock())
                try:
                    with lock, demo_slots:
                        path = encode_video(spec, site, key)
                except Exception as error:
                    self.log_error("Demo preview %s: %s", key, error)
                    self.send_error(503, "Unable to prepare demonstration video")
                    return
            elif route.startswith("/artifact/"):
                key = route.removeprefix("/artifact/")
                value = read(site / ".artifacts.json").get(key)
                if not value:
                    self.send_error(404)
                    return
                path = Path(value)
            else:
                relative = route.lstrip("/") or "index.html"
                path = (site / relative).resolve()
                if not path.is_relative_to(site) or any(
                    p.startswith(".") for p in Path(relative).parts
                ):
                    self.send_error(404)
                    return
            if not path.is_file():
                self.send_error(404)
                return
            packed = site / '.data.json.gz'
            if (route == '/data.json' and accepts_gzip(self.headers.get('Accept-Encoding', '')) and not self.headers.get('Range')
                    and packed.is_file() and packed.stat().st_mtime_ns >= path.stat().st_mtime_ns):
                path, compressed = packed, True
            size = path.stat().st_size
            start, end, code = 0, size - 1, 200
            if self.headers.get("Range"):
                import re

                match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers["Range"])
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                left, right = match.groups()
                start = int(left) if left else max(0, size - int(right))
                end = min(int(right), size - 1) if left and right else size - 1
                if not 0 <= start <= end < size:
                    self.send_error(416)
                    return
                code = 206
            self.send_response(code)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(route.rsplit("/", 1)[-1])[0]
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
            )
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            if route == '/data.json':
                self.send_header('Vary', 'Accept-Encoding')
            if compressed:
                self.send_header('Content-Encoding', 'gzip')
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head:
                return
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = stream.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer((host, port), partial(Handler, directory=site))
    print(f"CaM website: http://{host}:{server.server_port}/", flush=True)
    server.serve_forever()
