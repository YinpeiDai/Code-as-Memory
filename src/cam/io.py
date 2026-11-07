"""Atomic manifests, content fingerprints and controlled artifact copying."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def tree_hash(root):
    root = Path(root)
    value = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            if path.is_symlink():
                raise ValueError(f"Pinned source must not contain symlinks: {path}")
            value.update(path.relative_to(root).as_posix().encode() + b"\0")
            value.update(bytes.fromhex(sha256(path)))
    return value.hexdigest()


def copy_tree(source, destination):
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )


def task_name(value):
    import re

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid task name: {value!r}")
    return value
