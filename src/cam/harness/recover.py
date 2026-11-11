"""Explicit recovery preserves failed attempts and never refunds training budget."""

from pathlib import Path
import shutil
import time
from cam.core.budget import Ledger
from cam.harness.jobs import locked
from cam.harness.prepare import validate_run
from cam.io import read, write


def recover_test(root, task, episode, maximum, reason):
    root = Path(root).resolve()
    plan = validate_run(root)
    if (
        task not in plan["tasks"]
        or episode not in plan["test_episodes"][task]
        or maximum < 0
    ):
        raise ValueError("Unassigned test identity")
    with locked(root / "locks/evaluation.lock"):
        folder = (
            root
            / "tasks"
            / task
            / "rollouts/test"
            / f"R{maximum:03d}"
            / f"ep{episode:03d}"
        )
        result = (
            read(folder / "result.json") if (folder / "result.json").exists() else None
        )
        if result and result["scorable"]:
            raise ValueError("Scored test episodes cannot be retried or replaced")
        destination = (
            root
            / "tasks"
            / task
            / "rollouts/infrastructure_retries"
            / f"R{maximum:03d}_ep{episode:03d}_{time.time_ns()}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if folder.exists():
            shutil.move(folder, destination)
        else:
            # A worker can die before execute creates its episode directory.
            destination.mkdir()
        write(
            destination / "recovery.json",
            {
                "reason": reason,
                "time": time.time(),
                "rule": "Identical pinned code, seed and limits; infrastructure retry only",
            },
        )
    return {"archived": str(destination), "next": "cam evaluate RUN --resume"}


def recover_learning(root, task, reason):
    root = Path(root).resolve()
    plan = validate_run(root)
    if task not in plan["tasks"]:
        raise ValueError("Unassigned task")
    folder = root / "tasks" / task
    with locked(folder / "state/learning.lock"), locked(folder / "audit/service.lock"):
        ledger = Ledger(folder / "state/ledger.sqlite")
        changed = []
        for row in ledger.rows():
            if row["status"] == "running":
                ledger.update(
                    row["id"],
                    steps=row["steps"],
                    step_calls=row["step_calls"],
                    status="interrupted",
                    error=reason,
                )
                changed.append(row["id"])
                write(
                    folder / "feedback" / f"R{row['id']:03d}" / "interrupted.json",
                    {
                        "status": "interrupted",
                        "attempt": row["id"],
                        "episode": row["episode"],
                        "steps": row["steps"],
                        "budget_refunded": False,
                    },
                )
        write(
            folder / "state" / f"recovery_{time.time_ns()}.json",
            {
                "reason": reason,
                "attempts": changed,
                "time": time.time(),
                "budget_refunded": False,
            },
        )
    return {"closed_attempts": changed, "budget_refunded": False}
