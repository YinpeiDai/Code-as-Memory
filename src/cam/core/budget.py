"""Transactional attempt ledger; simulator initialization consumes an attempt."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def program_hash(root):
    root = Path(root)
    value = hashlib.sha256()
    for path in [
        root / "task.json",
        root / "policy.md",
        *sorted((root / "tools").glob("*.py")),
    ]:
        if path.is_symlink():
            raise ValueError("Policy symlinks are forbidden")
        value.update(
            path.relative_to(root).as_posix().encode()
            + b"\0"
            + path.read_bytes()
            + b"\0"
        )
    return value.hexdigest()


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def initialize(self, config):
        if self.path.exists():
            raise FileExistsError("An existing attempt ledger cannot be reset")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE protocol (config TEXT NOT NULL);
                CREATE TABLE attempts (
                    id INTEGER PRIMARY KEY, episode INTEGER NOT NULL, program_hash TEXT NOT NULL,
                    started REAL NOT NULL, updated REAL NOT NULL, status TEXT NOT NULL,
                    steps INTEGER NOT NULL DEFAULT 0, step_calls INTEGER NOT NULL DEFAULT 0, error TEXT);
                CREATE TABLE revisions (
                    id INTEGER PRIMARY KEY, digest TEXT NOT NULL, attempts INTEGER NOT NULL,
                    reason TEXT NOT NULL, created REAL NOT NULL, path TEXT NOT NULL);
            """)
            db.execute("INSERT INTO protocol VALUES (?)", (json.dumps(config),))

    def rows(self):
        with self.connect() as db:
            return [
                dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY id")
            ]

    def config(self):
        with self.connect() as db:
            return json.loads(db.execute("SELECT config FROM protocol").fetchone()[0])

    def update(self, attempt, *, steps, step_calls, status="running", error=None):
        with self.connect() as db:
            db.execute(
                "UPDATE attempts SET steps=MAX(steps,?), step_calls=MAX(step_calls,?), "
                "status=?, error=COALESCE(?,error), updated=? WHERE id=?",
                (int(steps), int(step_calls), status, error, time.time(), attempt),
            )

    def reserve(self, config, episode):
        from .history import checkpoint_locked

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            saved = json.loads(db.execute("SELECT config FROM protocol").fetchone()[0])
            if saved.get("sealed"):
                raise ValueError("Development is sealed")
            for key in ("task", "mode", "budget", "train_episodes", "task_root"):
                if saved[key] != config[key]:
                    raise ValueError(f"Attempt configuration changed: {key}")
            used = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            if used >= saved["budget"]:
                raise ValueError("Attempt budget exhausted; simulator reset denied")
            if db.execute(
                "SELECT COUNT(*) FROM attempts WHERE status='running'"
            ).fetchone()[0]:
                raise ValueError("An attempt is already running")
            if episode != saved["train_episodes"][used % len(saved["train_episodes"])]:
                raise ValueError("Only the next assigned training episode is allowed")
            checkpoint_locked(config, db)
            now = time.time()
            return db.execute(
                "INSERT INTO attempts (episode,program_hash,started,updated,status) "
                "VALUES (?,?,?,?,'running')",
                (
                    episode,
                    program_hash(Path(config["task_root"]) / "program"),
                    now,
                    now,
                ),
            ).lastrowid
