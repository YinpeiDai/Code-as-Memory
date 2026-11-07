"""Export coding commands and summarize observed tool/network/private-path access.

This is an evidence summary, not a general syscall sandbox verifier. The exact
commands and original traces remain authoritative; unresolved relative paths
and runtime-generated strings require manual review.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

from cam.io import read, sha256, write


def trace_summary(path):
    rows = []
    children = {}
    roots = set()
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"(\d+)\s+(.*)", line)
        if not match:
            continue
        pid, event = int(match[1]), match[2]
        rows.append((pid, event))
        if "execve(" in event and '"--noprofile"' in event and event.endswith("= 0"):
            roots.add(pid)
        if re.search(r"(?:clone3?|vfork|fork)(?:\(| resumed>)", event):
            child = re.search(r"= (\d+)$", event)
            if child:
                children.setdefault(pid, set()).add(int(child[1]))
    pids = set(roots)
    while True:
        more = (
            set().union(*(children.get(pid, set()) for pid in pids)) if pids else set()
        )
        if more <= pids:
            break
        pids |= more
    internet, private, unix, opened = [], [], Counter(), Counter()
    started = set()
    for pid, event in rows:
        if (
            pid in roots
            and "execve(" in event
            and '"--noprofile"' in event
            and event.endswith("= 0")
        ):
            started.add(pid)
        if pid not in pids or (pid in roots and pid not in started):
            continue
        if re.search(r"(?:socket\(AF_INET6?\b|sa_family=AF_INET6?\b)", event):
            internet.append(event)
        if "connect(" in event and "AF_UNIX" in event:
            unix[event] += 1
        if not re.match(r"(?:open|openat|openat2)\(", event) or not re.search(
            r"= \d+$", event
        ):
            continue
        quoted = re.search(r'"((?:[^"\\]|\\.)*)"', event)
        if not quoted:
            continue
        name = quoted[1]
        if name.startswith("/"):
            opened["/" + name.lstrip("/").split("/")[0]] += 1
            if name.startswith(
                ("/data/", "/home/daiyp/", "/root/", "/inputs/", "/rollouts/")
            ):
                private.append(event)
    return {
        "file": path.name,
        "sha256": sha256(path),
        "sandbox_shell_identified": bool(roots),
        "sandbox_process_count": len(pids),
        "internet_socket_events": internet,
        "successful_absolute_private_path_opens": private,
        "local_socket_connects": dict(unix),
        "successful_absolute_open_prefixes": dict(opened),
    }


def audit(root, task):
    root = Path(root).resolve()
    plan = read(root / "run.json")
    if task not in plan["tasks"]:
        raise ValueError("Task is not assigned to this run")
    folder = root / "tasks" / task
    calls = [read(f) for f in sorted((folder / "audit").glob("call_*.json"))]
    listing, traces, suspicious = [], [], []
    review = re.compile(
        r"https?://|\b(?:curl|wget|ssh|scp|git|pip|requests|urllib|socket)\b|"
        r"/data/|/home/|/inputs/|/rollouts/test|/proc/|/root/|cap_baseline|github|huggingface|robomme",
        re.I,
    )
    for call in calls:
        if call["tool"] != "command":
            continue
        command = call["arguments"]["command"]
        listing.append(
            f"{'=' * 70}\n{call['id']} started={call['started']} finished={call.get('finished')}\n{command}\n"
        )
        matches = [line for line in command.splitlines() if review.search(line)]
        if matches:
            suspicious.append({"call": call["id"], "review_lines": matches})
        path = folder / "audit" / (call["id"] + ".strace")
        if call.get("finished") and path.exists():
            traces.append(trace_summary(path))
    offered, used = [], Counter()
    for path in sorted((folder / "logs").glob("session_*/events.jsonl")):
        for line in path.open():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                offered.append(
                    {"session": path.parent.name, "tools": event.get("tools", [])}
                )
            if event.get("type") == "assistant":
                for content in event.get("message", {}).get("content", []):
                    if content.get("type") == "tool_use":
                        used[content["name"]] += 1
            item = event.get("item", {})
            if (
                event.get("type") == "item.completed"
                and item.get("type") == "mcp_tool_call"
            ):
                used[f"mcp__{item.get('server')}__{item.get('tool')}"] += 1
            elif event.get("type") == "item.completed" and item.get("type") in (
                "web_search",
                "command_execution",
                "file_change",
            ):
                used[item["type"]] += 1
    result = {
        "schema": "cam-access-audit/1",
        "created": time.time(),
        "task": task,
        "recorded_calls": len(calls),
        "tools": dict(Counter(c["tool"] for c in calls)),
        "offered_client_tools": offered,
        "used_client_tools": dict(used),
        "commands_flagged_for_manual_review": suspicious,
        "completed_command_traces": len(traces),
        "internet_socket_event_count": sum(
            len(t["internet_socket_events"]) for t in traces
        ),
        "successful_absolute_private_path_open_count": sum(
            len(t["successful_absolute_private_path_opens"]) for t in traces
        ),
        "traces": traces,
        "limits": "Snapshot of retained evidence. Bootstrap host mounts are excluded from sandbox subprocess analysis. Relative paths, dynamic strings, and syscalls outside successful open/openat or socket/connect checks require manual review. Provider API traffic belongs to the host coding client, not the command sandbox.",
    }
    out = root / "orchestration"
    out.mkdir(exist_ok=True)
    (out / "commands.txt").write_text("\n".join(listing))
    write(out / "access_audit.json", result)
    (out / "ACCESS_AUDIT.md").write_text(f"""# Coding-agent access audit: {task}

This snapshot covers {len(calls)} tool calls and {len(traces)} completed command traces.

- Recorded tools: `{result["tools"]}`.
- Observed Internet socket events in sandbox commands: **{result["internet_socket_event_count"]}**.
- Successful absolute opens under checked private host prefixes: **{result["successful_absolute_private_path_open_count"]}**.
- Commands flagged for manual review: **{len(suspicious)}** (a match is not automatically a violation).

Read [commands.txt](commands.txt) for every command and [access_audit.json](access_audit.json)
for the offered/used client tools and trace details. Original evidence remains in
`../tasks/{task}/audit/` and `../tasks/{task}/logs/`.

The trace includes Bubblewrap's host-side mount setup. Those host paths do not
mean the learner could read them; this summary follows the sandbox shell and
its subprocesses. Failed AF_UNIX `nscd` lookups are not Internet requests.

{result["limits"]}

Regenerate after learning finishes to cover the final calls:

```bash
.venv/bin/python scripts/audit_run.py RUN --task {task}
```
""")
    return {k: v for k, v in result.items() if k != "traces"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.task), indent=2))
