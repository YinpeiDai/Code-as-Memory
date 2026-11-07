"""Source revisions and checkpoints saved before later simulator feedback."""

import difflib
import hashlib
import json
from pathlib import Path
import shutil
import time

from cam.io import read, write
from .budget import Ledger, program_hash


def payload(root):
    root = Path(root)
    files = [
        root / name
        for name in ("task.json", "policy.md", "task_analysis.md", "CHANGELOG.md")
    ]
    files += sorted((root / "tools").glob("*.py"))
    result = {}
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Policy files must stay inside their workspace")
        if path.is_file():
            if path.stat().st_size > 2 * 1024**2:
                raise ValueError("Policy file exceeds 2 MiB")
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def materialize(root, files):
    root.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def snapshot_locked(config, db, reason):
    root = Path(config["task_root"])
    files = payload(root / "program")
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode() + b"\0" + value + b"\0")
    key = digest.hexdigest()
    target = root / "history/objects" / key
    if not target.exists():
        materialize(target, files)
    count = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    previous = db.execute("SELECT * FROM revisions ORDER BY id DESC LIMIT 1").fetchone()
    row = db.execute(
        "INSERT INTO revisions (digest,attempts,reason,created,path) VALUES (?,?,?,?,?)",
        (key, count, reason, time.time(), str(target)),
    )
    revision = dict(
        db.execute("SELECT * FROM revisions WHERE id=?", (row.lastrowid,)).fetchone()
    )
    name = root / "history/revisions" / f"{row.lastrowid:06d}"
    write(name.with_suffix(".json"), revision)
    old = payload(Path(previous["path"])) if previous else {}
    diff = []
    for filename in sorted(old.keys() | files.keys()):
        diff.extend(
            difflib.unified_diff(
                old.get(filename, b"").decode(errors="replace").splitlines(True),
                files.get(filename, b"").decode(errors="replace").splitlines(True),
                fromfile="before/" + filename,
                tofile="after/" + filename,
            )
        )
    name.with_suffix(".diff").write_text("".join(diff))
    return revision


def snapshot(config, reason):
    with Ledger(config["ledger"]).connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if json.loads(db.execute("SELECT config FROM protocol").fetchone()[0]).get(
            "sealed"
        ):
            raise ValueError("Cannot revise a sealed policy")
        return snapshot_locked(config, db, reason)


def checkpoint_locked(config, db):
    rows = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY id")]
    if any(row["status"] == "running" for row in rows):
        raise ValueError("Cannot checkpoint an active attempt")
    root = Path(config["task_root"])
    revision = snapshot_locked(config, db, "checkpoint")
    source = Path(revision["path"])
    count, key = len(rows), program_hash(source)
    target = root / "checkpoints" / f"R{count:03d}"
    if target.exists():
        if read(target / "freeze.json")["program_hash"] != key:
            raise ValueError("Checkpoint exists with different code")
        return target
    materialize(target, payload(source))
    write(
        target / "freeze.json",
        {
            "schema": "cam-policy/1",
            "task": config["task"],
            "checkpoint": count,
            "program_hash": key,
            "frozen_at": time.time(),
            "revision_id": revision["id"],
            "training": {
                "attempts": count,
                "attempt_records": rows,
                "simulator_steps": sum(row["steps"] for row in rows),
            },
        },
    )
    return target


def freeze(config):
    root = Path(config["task_root"])
    with Ledger(config["ledger"]).connect() as db:
        db.execute("BEGIN IMMEDIATE")
        saved = json.loads(db.execute("SELECT config FROM protocol").fetchone()[0])
        if saved.get("sealed"):
            raise ValueError("Policy already frozen")
        source = checkpoint_locked(config, db)
        shutil.copytree(source, root / "policy")
        saved["sealed"] = True
        db.execute("UPDATE protocol SET config=?", (json.dumps(saved),))
    return read(root / "policy/freeze.json")


def select_checkpoint(task_root, maximum):
    root = Path(task_root)
    final = read(root / "policy/freeze.json")
    number = min(maximum, final["checkpoint"])
    path = root / "checkpoints" / f"R{number:03d}"
    info = read(path / "freeze.json")
    if info["checkpoint"] != number or program_hash(path) != info["program_hash"]:
        raise ValueError("Saved checkpoint integrity mismatch")
    return path, info
