"""Chronological development transcript for the website.

Interleaves, in order, everything recorded for one task's coding sessions: the
learner's messages, each CaM tool call (arguments, timing, output, read images),
the code revisions saved by those calls (inline diffs), training rollouts, and
each session's closing summary and usage. Built from the run's own logs/audit.
"""

import json
from pathlib import Path

from cam.io import read

TOOLS = ("command", "read_image", "usage", "snapshot", "validate", "rollout")
TEXT_LIMIT = 6000
DIFF_LIMIT = 40000


def _clip(text, limit=TEXT_LIMIT):
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n… ({len(text) - limit} more characters in the full output)"


def _result_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def _manifests(audit):
    """Map 'after call_NNNNNN' → archived workspace manifest."""
    found = {}
    for path in sorted((audit / "file_manifests").glob("*.json")):
        try:
            value = read(path)
        except ValueError:
            continue
        if str(value.get("reason", "")).startswith("after "):
            found[value["reason"].removeprefix("after ")] = value.get("files", {})
    return found


def _image(call, folder, manifests, asset):
    raw = str((call.get("arguments") or {}).get("path", ""))
    if raw.startswith("/workspace/"):
        relative = raw.removeprefix("/workspace/")
        entry = manifests.get(call["id"], {}).get(relative)
        if entry and entry.get("sha256"):
            return asset(folder / "audit/objects" / entry["sha256"], suffix=Path(relative).suffix)
        return asset(folder / "workspace" / relative)
    for prefix, base in (("/feedback/", folder / "feedback"), ("/demonstrations/", folder / "demonstrations")):
        if raw.startswith(prefix):
            return asset(base / raw.removeprefix(prefix))
    return None


def build_timeline(folder, asset, rollouts=None):
    folder = Path(folder)
    audit = folder / "audit"
    calls = [read(p) for p in sorted(audit.glob("call_*.json")) if p.stem.removeprefix("call_").isdigit()]
    manifests = _manifests(audit) if any(c.get("tool") == "read_image" for c in calls) else {}
    revisions = []
    for path in sorted((folder / "history/revisions").glob("*.json")):
        row = read(path)
        diff_path = path.with_suffix(".diff")
        diff = diff_path.read_text(errors="replace") if diff_path.exists() else ""
        revisions.append({
            "kind": "revision", "id": row["id"], "reason": row["reason"], "created": row["created"],
            "attempts": row["attempts"], "digest": row["digest"][:12],
            "diff": diff if len(diff) <= DIFF_LIMIT else diff[:DIFF_LIMIT] + "\n… (diff truncated; open the full diff)",
            "diff_url": asset(diff_path), "changed": bool(diff.strip()),
        })
    by_call = {}
    loose = []
    for revision in revisions:
        reason = revision["reason"]
        if reason.startswith("command call_"):
            by_call.setdefault(reason.removeprefix("command "), []).append(revision)
            continue
        owner = next((c["id"] for c in calls if c.get("started", 0) <= revision["created"] <= (c.get("finished") or c.get("started", 0)) + 1), None)
        if owner and reason in ("validated", "agent snapshot", "before rollout"):
            by_call.setdefault(owner, []).append(revision)
        else:
            loose.append(revision)
    rollouts = {int(r.get("attempt") or 0): r for r in rollouts or []}
    entries, index = [], 0
    for path in sorted((folder / "logs").glob("session_*/events.jsonl")):
        session = path.parent.name
        entries.append({"kind": "session", "session": session})
        pending = {}
        for line in path.open():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            content = (event.get("message") or {}).get("content")
            if kind == "assistant" and isinstance(content, list):
                for item in content:
                    if item.get("type") == "text" and item.get("text", "").strip():
                        entries.append({"kind": "text", "text": item["text"]})
                    elif item.get("type") == "thinking":
                        text = item.get("thinking", "")
                        entries.append({"kind": "thinking", "text": text, "redacted": not text.strip()})
                    elif item.get("type") == "tool_use":
                        name = item.get("name", "").removeprefix("mcp__cam__")
                        entry = {"kind": "tool", "name": name, "input": item.get("input", {})}
                        if name in TOOLS and index < len(calls):
                            call = calls[index]
                            index += 1
                            entry |= {
                                "id": call.get("id"), "started": call.get("started"), "finished": call.get("finished"),
                                "stdout": asset(audit / f"{call.get('id')}.stdout"),
                                "stderr": asset(audit / f"{call.get('id')}.stderr"),
                                "revisions": by_call.get(call.get("id"), []),
                            }
                            if name == "read_image":
                                entry["image"] = _image(call, folder, manifests, asset)
                        pending[item.get("id")] = entry
                        entries.append(entry)
            elif kind == "user" and isinstance(content, list):
                for item in content:
                    if item.get("type") != "tool_result":
                        continue
                    entry = pending.get(item.get("tool_use_id"))
                    if entry is None:
                        continue
                    text = _result_text(item.get("content"))
                    entry["is_error"] = bool(item.get("is_error"))
                    try:
                        parsed = json.loads(text)
                    except ValueError:
                        parsed = None
                    if entry["name"] == "command" and isinstance(parsed, dict):
                        entry["returncode"] = parsed.get("returncode")
                        entry["output"] = _clip(parsed.get("stdout", ""))
                        entry["error_output"] = _clip(parsed.get("stderr", ""), 3000)
                    elif entry["name"] == "rollout" and isinstance(parsed, dict):
                        entry["rollout"] = parsed
                        attempt = rollouts.get(int(parsed.get("attempt") or 0))
                        if attempt:
                            entry["rollout_trace"] = attempt.get("trace")
                            entry["rollout_video"] = attempt.get("video")
                        entry["output"] = _clip(text, 3000)
                    else:
                        entry["output"] = _clip(text, 3000)
            elif kind == "result":
                entries.append({
                    "kind": "summary", "session": session, "text": event.get("result", ""),
                    "cost": event.get("total_cost_usd"), "turns": event.get("num_turns"),
                    "seconds": (event.get("duration_ms") or 0) / 1000, "status": event.get("subtype"),
                })
            elif kind == "item.completed":  # Codex exec
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    entries.append({"kind": "text", "text": item.get("text", "")})
                elif item.get("type") == "reasoning":
                    entries.append({"kind": "thinking", "text": item.get("text", ""), "redacted": False})
                elif item.get("type") == "mcp_tool_call":
                    entries.append({"kind": "tool", "name": item.get("tool", ""), "input": item.get("arguments", {}), "output": _clip(json.dumps(item.get("result"))[:TEXT_LIMIT])})
    for revision in loose:  # place by time among timed tool entries
        position = len(entries)
        for i, entry in enumerate(entries):
            if entry.get("started") and entry["started"] > revision["created"]:
                position = i
                break
        entries.insert(position, revision)
    return entries
