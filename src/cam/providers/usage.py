"""Normalize saved CLI usage. Missing dollar costs remain null, never zero."""

import json
from pathlib import Path
import time

from cam.reporting.costs import Usage as ClaudeUsage
from cam.io import write


def summarize_sessions(logs, agent):
    sessions = []
    for path in sorted(Path(logs).glob("session_*/events.jsonl")):
        events = []
        with path.open() as stream:
            for line in stream:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        if agent == "claude_code":
            usage = ClaudeUsage()
            for event in events:
                usage.feed(event)
            summary = usage.summary()
            row = {
                "session": path.parent.name,
                "reported_cost_usd": summary["reported_cost_usd"],
                "tokens": summary["final_session_token_totals"]
                or summary["query_token_sums"],
                "requests_logged": summary["queries_logged"],
                "final_usage_available": bool(summary["sessions_with_final_usage"]),
                "cost_basis": "Claude CLI reported estimate; excludes compute, not a verified bill",
            }
            write(path.parent / "requests.json", list(usage.requests.values()))
        else:
            # Codex exec emits one turn.completed usage object per completed turn.
            turns = [
                e["usage"]
                for e in events
                if e.get("type") == "turn.completed"
                and isinstance(e.get("usage"), dict)
            ]
            row = {
                "session": path.parent.name,
                "reported_cost_usd": None,
                "tokens": {
                    key: sum(t.get(key, 0) for t in turns)
                    for key in ("input_tokens", "cached_input_tokens", "output_tokens")
                },
                "requests_logged": None,
                "completed_turns": len(turns),
                "final_usage_available": bool(turns),
                "cost_basis": "Codex CLI token usage; dollar costs are not reported",
            }
        write(path.parent / "usage.json", row)
        sessions.append(row)
    known = [
        s["reported_cost_usd"] for s in sessions if s["reported_cost_usd"] is not None
    ]
    result = {
        "schema": "cam-usage/1",
        "agent": agent,
        "updated": time.time(),
        "sessions": sessions,
        "reported_cost_usd": sum(known) if known else None,
        "cost_complete": bool(sessions) and len(known) == len(sessions),
        "sessions_with_reported_cost": len(known),
        "session_count": len(sessions),
        "gpu_cost_included": False,
        "evaluation_api_cost_usd": 0,
        "tokens": {
            key: sum(s["tokens"].get(key, 0) for s in sessions)
            for key in {k for s in sessions for k in s["tokens"]}
        },
    }
    write(Path(logs) / "usage.json", result)
    return result
