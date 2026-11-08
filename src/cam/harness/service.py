# Adapted from the Apache-2.0 CAP MCP server; see NOTICE.
"""Private MCP tool server. The coding model has no host Bash/Read/Web tools."""

import argparse
import base64
import fcntl
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from cam.harness.isolation import (
    command,
    limits,
    run,
    terminate,
    parent_death,
    watch_parent_process,
)
from cam.core import budget, history
from cam.harness.prepare import pinned_environment
from cam.harness.gpu import simulator_slot
from cam.io import read


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


class Service:
    def __init__(self, root, task, gpu):
        self.root = root
        self.task = task
        self.gpu = gpu
        self.ws = root / "tasks" / task
        if task not in read(root / "run.json")["tasks"]:
            raise ValueError("Unassigned task")
        self.public = self.ws / "workspace"
        self.runtime = root / "runtime"
        self.feedback = self.ws / "feedback"
        self.cfg = json.loads((self.ws / "state/config.json").read_text())
        self.ledger = budget.Ledger(self.cfg["ledger"])
        self.demonstrations = (
            self.ws / "demonstrations"
            if self.cfg["mode"] in ("demos", "both")
            else None
        )
        self.audit = self.ws / "audit"
        self.audit.mkdir(parents=True, exist_ok=True)
        self.lock = (self.audit / "service.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.counter = len(list(self.audit.glob("call_*.json")))

    def archive(self, reason):
        """Content-address all regular development files, including scratch code."""
        objects = self.audit / "objects"
        objects.mkdir(exist_ok=True)
        manifest = {}
        for parent, dirs, files in os.walk(self.public, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
            for name in files:
                p = Path(parent) / name
                rel = p.relative_to(self.public).as_posix()
                if p.is_symlink():
                    manifest[rel] = {"symlink": os.readlink(p)}
                    continue
                if not p.is_file():
                    continue
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                dest = objects / digest
                if not dest.exists():
                    shutil.copyfile(p, dest)
                manifest[rel] = {"sha256": digest, "bytes": p.stat().st_size}
        write(
            self.audit / "file_manifests" / f"{time.time_ns()}.json",
            {"time": time.time(), "reason": reason, "files": manifest},
        )

    def sync(self, reason):
        """Copy only regular policy files into the private checkpoint workspace."""
        if self.ledger.config().get("sealed"):
            raise ValueError("Development is sealed")
        files = ["task.json", "policy.md", "task_analysis.md", "CHANGELOG.md"]
        tools = self.public / "tools"
        if tools.is_symlink():
            raise ValueError("Policy tools must be a real directory")
        files += [p.relative_to(self.public).as_posix() for p in tools.glob("*.py")]
        payload = {}
        for name in files:
            p = self.public / name
            if not p.exists():
                continue
            if p.is_symlink() or not p.resolve().is_relative_to(self.public.resolve()):
                raise ValueError("Policy symlinks are not allowed")
            if p.stat().st_size > 2 * 1024**2:
                raise ValueError("Policy file exceeds 2 MiB")
            payload[name] = p.read_bytes()
        if json.loads(payload["task.json"])["task"] != self.task:
            raise ValueError("Assigned task is fixed")
        if json.loads(payload["task.json"]).get("runtime") != "python":
            raise ValueError("Runtime must remain python")
        for old in (self.ws / "program/tools").glob("*.py"):
            old.unlink()
        for name in ("task_analysis.md", "CHANGELOG.md"):
            if name not in payload:
                (self.ws / "program" / name).unlink(missing_ok=True)
        for name, data in payload.items():
            p = self.ws / "program" / name
            p.parent.mkdir(exist_ok=True, parents=True)
            p.write_bytes(data)
        history.snapshot(self.cfg, reason)

    def usage(self):
        rows = self.ledger.rows()
        seeds = self.cfg["train_episodes"]
        used = len(rows)
        return {
            "used": used,
            "maximum": self.cfg["budget"],
            "remaining": self.cfg["budget"] - used,
            "per_seed": {str(e): sum(r["episode"] == e for r in rows) for e in seeds},
            "next_episode": budget.next_episode(self.cfg, used),
            "attempts": [
                {k: r[k] for k in ("id", "episode", "status", "steps")} for r in rows
            ],
            "api_cost": self.api_cost(),
        }

    def api_cost(self):
        path = self.ws / "logs/usage.json"
        return (
            json.loads(path.read_text())
            if path.exists()
            else {"reported_cost_usd": None, "queries_logged": 0}
        )

    def call(self, name, args):
        self.counter += 1
        ident = f"call_{self.counter:06d}"
        start = time.time()
        event = {"id": ident, "tool": name, "arguments": args, "started": start}
        write(self.audit / (ident + ".json"), event)
        self.archive("before " + ident)
        try:
            result = self.dispatch(name, args, ident)
            event.update(finished=time.time(), result=result)
            return result
        except Exception as exc:
            result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
            event.update(finished=time.time(), result=result)
            return result
        finally:
            self.archive("after " + ident)
            write(self.audit / (ident + ".json"), event)

    def dispatch(self, name, args, ident):
        text = lambda value: {
            "content": [
                {
                    "type": "text",
                    "text": value if isinstance(value, str) else json.dumps(value),
                }
            ]
        }
        if name == "command":
            shell = args["command"]
            timeout = min(max(int(args.get("timeout", 120)), 1), 300)
            argv = command(
                self.public,
                self.runtime,
                ["bash", "--noprofile", "--norc", "-c", shell],
                self.feedback,
                demonstrations=self.demonstrations,
            )
            trace = self.audit / (ident + ".strace")
            proc = subprocess.Popen(
                [
                    "/usr/bin/strace",
                    "-f",
                    "-qq",
                    "-e",
                    "trace=%file,%network,%process",
                    "-o",
                    str(trace),
                    *argv,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=limits,
            )
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate(proc)
                out, err = proc.communicate()
                err += b"\nCommand timed out."
            (self.audit / (ident + ".stdout")).write_bytes(out)
            (self.audit / (ident + ".stderr")).write_bytes(err)
            # Full output remains in audit; a readable copy is available inside the workspace.
            saved = self.public / "scratch" / (ident + "_output.txt")
            if saved.parent.is_symlink() or saved.is_symlink():
                raise ValueError("Scratch output must not be a symlink")
            saved.parent.mkdir(exist_ok=True)
            saved.write_bytes(out + b"\nSTDERR:\n" + err)
            try:
                self.sync("command " + ident)
            except (ValueError, KeyError, json.JSONDecodeError):
                pass  # Incomplete edits are retained in the file archive.
            return text(
                {
                    "returncode": proc.returncode,
                    "stdout": out.decode(errors="replace")[:30000],
                    "stderr": err.decode(errors="replace")[:10000],
                    "full_output": "/workspace/scratch/" + saved.name,
                }
            )
        if name == "read_image":
            raw = Path(args["path"])
            base = self.public
            prefix = "/workspace/"
            if str(raw).startswith("/feedback/"):
                base = self.feedback
                prefix = "/feedback/"
            if str(raw).startswith("/demonstrations/") and self.demonstrations:
                base = self.demonstrations
                prefix = "/demonstrations/"
            relative = str(raw).removeprefix(prefix)
            p = (base / relative).resolve()
            if (
                not p.is_relative_to(base.resolve())
                or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp")
                or not p.is_file()
            ):
                raise ValueError("Image must be inside an allowed task directory")
            if p.stat().st_size > 8 * 1024**2:
                raise ValueError("Resize the image below 8 MiB first")
            return {
                "content": [
                    {
                        "type": "image",
                        "mimeType": mimetypes.guess_type(p.name)[0],
                        "data": base64.b64encode(p.read_bytes()).decode(),
                    }
                ]
            }
        if name == "usage":
            return text(self.usage())
        if name == "snapshot":
            self.sync("agent snapshot")
            return text({"saved": True, **self.usage()})
        if name == "validate":
            code, out, err = run(
                self.public,
                self.runtime,
                ["python3.11", "-m", "cam.runtime.policy_worker", "--validate"],
                timeout=30,
                readonly=True,
            )
            if not code:
                self.sync("validated")
            return text(
                {
                    "valid": code == 0,
                    "output": out.decode(errors="replace"),
                    "errors": err.decode(errors="replace"),
                }
            )
        if name == "rollout":
            usage = self.usage()
            if usage["remaining"] <= 0:
                raise ValueError(
                    f"R{self.cfg['budget']} exhausted; no simulator reset permitted"
                )
            self.sync("before rollout")
            code, out, err = run(
                self.public,
                self.runtime,
                ["python3.11", "-m", "cam.runtime.policy_worker", "--validate"],
                timeout=30,
                readonly=True,
            )
            if code:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Validation failed before simulator initialization:\n"
                            + err.decode(errors="replace"),
                        }
                    ],
                    "isError": True,
                }
            episode = usage["next_episode"]
            attempt = usage["used"] + 1
            folder = f"R{attempt:03d}"
            private = self.ws / "rollouts/train" / folder
            env = pinned_environment(self.root)
            env.update(
                CUDA_VISIBLE_DEVICES=str(self.gpu),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            with simulator_slot(self.gpu) as gpu, (self.audit / (ident + "_broker.log")).open("wb") as log:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                write(self.audit / (ident + "_gpu.json"), {"gpu": gpu, "acquired": time.time()})
                proc = subprocess.Popen(
                    [
                        read(self.root / "run.json").get(
                            "simulator_python", sys.executable
                        ),
                        str(self.root / "runner.py"),
                        "_episode",
                        str(self.root),
                        self.task,
                        str(episode),
                        "--split",
                        "train",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    preexec_fn=parent_death,
                )
                try:
                    code = proc.wait(
                        timeout=read(self.root / "run.json")["config"]["policy_timeout"]
                        + 600
                    )
                except subprocess.TimeoutExpired:
                    terminate(proc)
                    code = 124
            for row in self.ledger.rows():
                if row["status"] == "running":
                    self.ledger.update(
                        row["id"],
                        steps=row["steps"],
                        step_calls=row["step_calls"],
                        status="interrupted",
                    )
            if not (private / "result.json").exists():
                raise ValueError(
                    "Rollout worker failed; inspect private audit. "
                    + json.dumps(self.usage())
                )
            write(private / "api_cost_snapshot.json", self.api_cost())
            public = self.feedback / folder
            public.mkdir(exist_ok=False)
            for file in (
                "result.json",
                "observations.h5",
                "episode.mp4",
                "actions.npy",
                "dry_runs.jsonl",
                "policy_stderr.txt",
            ):
                if (private / file).exists():
                    shutil.copy2(private / file, public / file)
            result = json.loads((private / "result.json").read_text())
            return text(
                {
                    "episode": episode,
                    "attempt": attempt,
                    "status": result["status"],
                    "steps": result["steps"],
                    "feedback": "/feedback/" + folder,
                    "remaining": self.usage()["remaining"],
                    "api_cost": self.api_cost(),
                }
            )
        raise ValueError("Unknown tool")


TOOLS = [
    {
        "name": "command",
        "description": "Execute Bash/Python in the isolated workspace. No network, simulator source, other studies or unassigned datasets. Analyze allowed task data and create plots. Full commands and filesystem/network syscalls are audited.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "maximum": 300},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_image",
        "description": "Read one image from /workspace, your own /feedback, or assigned /demonstrations if enabled; returns pixels to the coding model.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    *[
        {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
        for name, description in [
            ("usage", "Show consumed attempts and the next round-robin training seed."),
            (
                "snapshot",
                "Save the current code iteration and complete development-file archive.",
            ),
            (
                "validate",
                "Check policy interface and code in isolation without any simulator reset.",
            ),
            (
                "rollout",
                "Run the current policy on its next assigned training episode, following the pinned seed schedule. Every attempt counts toward the fixed budget, including failures and early stops. Disabled in demos mode. Saves allowed RGB/depth/calibration/robot-state H5, video, actions and results.",
            ),
        ]
    ],
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("task")
    p.add_argument("--gpu", type=int, required=True)
    a = p.parse_args()
    watch_parent_process()
    service = Service(a.root.resolve(), a.task, a.gpu)
    for line in sys.stdin:
        req = json.loads(line)
        method = req.get("method")
        ident = req.get("id")
        if ident is None:
            continue
        try:
            if method == "initialize":
                value = {
                    "protocolVersion": req.get("params", {}).get(
                        "protocolVersion", "2024-11-05"
                    ),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "cam-isolated-study", "version": "1.0"},
                }
            elif method == "tools/list":
                value = {"tools": TOOLS}
            elif method == "tools/call":
                value = service.call(
                    req["params"]["name"], req["params"].get("arguments", {})
                )
            elif method == "ping":
                value = {}
            else:
                raise ValueError("Unsupported MCP method")
            response = {"jsonrpc": "2.0", "id": ident, "result": value}
        except Exception as e:
            response = {
                "jsonrpc": "2.0",
                "id": ident,
                "error": {"code": -32603, "message": str(e)},
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
