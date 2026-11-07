"""Shared website export and narrowly scoped artifact server for CaM runs."""

from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import unquote, urlsplit

from cam.io import read, write
from cam.reporting.report import build, matched_efficiency


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


def build_site(runs, datasets, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    def asset(path):
        path = Path(path).resolve()
        if not path.is_file():
            return None
        key = hashlib.sha256(str(path).encode()).hexdigest()[:24] + path.suffix
        artifacts[key] = str(path)
        return "artifact/" + key

    content = {
        "schema": "cam-site/1",
        "updated": time.time(),
        "runs": [],
        "datasets": [],
    }
    for index, source in enumerate(runs):
        root = Path(source).resolve()
        report = build(root)
        run = {
            "id": str(index),
            **report,
            "downloads": {
                "episodes": asset(root / "reports/episodes.csv"),
                "summary": asset(root / "reports/summary.json"),
            },
        }
        for task in run["development"]:
            folder = root / "tasks" / task["task"]
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
                task["rollouts"].append(
                    {
                        **read(path),
                        "video": asset(path.parent / "episode.mp4"),
                        "observations": asset(path.parent / "observations.h5"),
                        "actions": asset(path.parent / "actions.npy"),
                    }
                )
        for row in run["episodes"]:
            folder = root / row["directory"]
            row.update(
                video=asset(folder / "episode.mp4"),
                observations=asset(folder / "observations.h5"),
                actions=asset(folder / "actions.npy"),
                decisions_log=asset(folder / "decisions.jsonl"),
            )
        content["runs"].append(run)
    for source in datasets:
        root = Path(source).resolve()
        manifest = read(root / "manifest.json")
        if manifest["schema"] != "cam-dataset/1":
            raise ValueError("Only fresh CaM extractions are accepted")
        dataset = {
            "name": root.name,
            "benchmark": manifest["benchmark"],
            "depth": manifest["depth"],
            "selection": manifest["selection"],
            "tasks": [],
        }
        for name, item in manifest["tasks"].items():
            dataset["tasks"].append(
                {
                    "task": name,
                    "h5": asset(root / item["file"]),
                    "episodes": [
                        {
                            **row,
                            "video": asset(root / row["video"])
                            if row.get("video")
                            else None,
                        }
                        for row in item["episodes"]
                    ],
                }
            )
        content["datasets"].append(dataset)
    content["paired_comparisons"] = []
    # Cross-run comparisons require the identical native benchmark snapshot.
    for identity in {r["benchmark_hash"] for r in content["runs"]}:
        groups = {}
        for run in content["runs"]:
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

        figures = export_figures(content["runs"], output / "figures")
        content["figures"] = ["figures/" + name for name in figures["files"]]
    except ImportError:
        content["figures"] = []  # Optional plotting extra; browser SVGs still work.
    source = Path(__file__).resolve().parents[1] / "web"
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    write(output / "data.json", content)
    write(output / ".artifacts.json", artifacts)
    write(
        output / ".sources.json",
        {
            "runs": [str(Path(p).resolve()) for p in runs],
            "datasets": [str(Path(p).resolve()) for p in datasets],
        },
    )
    return {
        "site": str(output),
        "runs": len(runs),
        "datasets": len(datasets),
        "artifacts": len(artifacts),
    }


def serve(site, host="127.0.0.1", port=8910):
    site = Path(site).resolve()
    mutex = threading.Lock()
    refreshed = [0.0]

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == "/data.json" and time.monotonic() - refreshed[0] > 20:
                with mutex:
                    if time.monotonic() - refreshed[0] > 20:
                        sources = read(site / ".sources.json")
                        build_site(sources["runs"], sources["datasets"], site)
                        refreshed[0] = time.monotonic()
            self.serve_route(route, head=False)

        def do_HEAD(self):
            self.serve_route(unquote(urlsplit(self.path).path), head=True)

        def serve_route(self, route, head):
            if route.startswith("/artifact/"):
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
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
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
