"""Small progress files written alongside the authoritative scheduler JSON."""
from pathlib import Path
import time


def write_progress(folder, phase, state):
    rows = state.get("tasks", state.get("jobs", {}))
    lines = [f"# {phase} progress", "", f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", "",
             f"Status: {state.get('status', 'running')}", "",
             "| Task / episode | Development / job | Evaluation |", "|---|---|---|"]
    for key, value in rows.items():
        lines.append(f"| {str(key).replace('|', '/')} | {value.get('status', 'pending')} | {value.get('evaluation_status', '—')} |")
    destination = Path(folder) / "PROGRESS.md"
    pending = destination.with_suffix(".md.partial")
    pending.write_text("\n".join(lines) + "\n")
    pending.replace(destination)
