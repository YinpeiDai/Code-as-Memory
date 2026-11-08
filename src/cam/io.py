"""Atomic manifests, content fingerprints and controlled artifact copying."""

from fnmatch import fnmatch
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


def _ignored(relative, ignore):
    """Patterns with '/' match a root-relative path prefix; others match any name."""
    text = relative.as_posix()
    for pattern in ignore:
        if "/" in pattern:
            prefix = pattern.rstrip("/")
            if fnmatch(text, prefix) or text.startswith(prefix + "/"):
                return True
        elif any(fnmatch(part, pattern) for part in relative.parts):
            return True
    return False


def tree_hash(root, ignore=()):
    """Content hash of a source tree; directory symlinks contribute their target.

    ``ignore`` holds name patterns excluded at any depth, matching ``copy_tree``.
    File symlinks are rejected so pinned code cannot point outside its snapshot.
    """
    root = Path(root)
    value = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in path.parts or path.suffix == ".pyc" or _ignored(relative, ignore):
            continue
        if path.is_symlink():
            if not path.is_dir():
                raise ValueError(f"Pinned source must not contain file symlinks: {path}")
            value.update(
                relative.as_posix().encode() + b"\0->" + os.readlink(path).encode() + b"\0"
            )
            continue
        if path.is_file():
            value.update(relative.as_posix().encode() + b"\0")
            value.update(bytes.fromhex(sha256(path)))
    return value.hexdigest()


def copy_tree(source, destination, ignore=()):
    source = Path(source)
    patterns = ("__pycache__", "*.pyc", *ignore)

    def skip(directory, names):
        base = Path(directory).relative_to(source)
        return {name for name in names if _ignored(base / name, patterns)}

    shutil.copytree(source, destination, symlinks=True, ignore=skip)


def task_name(value):
    import re

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid task name: {value!r}")
    return value
