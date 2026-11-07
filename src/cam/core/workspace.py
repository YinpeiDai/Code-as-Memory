# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""A task's workspace: the files the coding agent writes and the harness reads.

<workspace>/
  task.json       {"task": "<Task>", "memory": [<field>, ...]}
                  a field: {"name", "type", "description"}
  policy.md       documentation of the executable policy and its memory
  tools/*.py      the perceive and act tools (see tools.py)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .tools import Toolkit

#: Field types, as the policy reads them.
FIELD_TYPES = {
    "position": "[x, y] or [x, y, z] in metres",
    "positions": "a list of positions",
    "number": "a number",
    "count": "an integer",
    "boolean": "true or false",
    "label": "a short string",
    "list": "a list of labels in order",
    "set": "a list of labels in any order",
    "text": "free text",
}


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class Workspace:
    root: Path
    task: str
    fields: tuple[Field, ...]
    policy: str
    runtime: str = "python"

    @classmethod
    def load(cls, root: str | Path) -> "Workspace":
        root = Path(root).resolve()
        spec = json.loads((root / "task.json").read_text())
        fields = tuple(
            Field(f["name"], f["type"], f.get("description", ""))
            for f in spec["memory"]
        )
        bad = [
            f"field {f.name}: unknown type {f.type!r} (types: {', '.join(FIELD_TYPES)})"
            for f in fields
            if f.type not in FIELD_TYPES
        ]
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            bad.append("field names repeat")
        if bad:
            raise ValueError("; ".join(bad))
        runtime = spec.get("runtime", "python")
        if runtime != "python":
            raise ValueError(f"Unknown policy runtime: {runtime}")
        return cls(
            root, spec["task"], fields, (root / "policy.md").read_text(), runtime
        )

    def tool_files(self) -> list[Path]:
        return sorted((self.root / "tools").glob("*.py"))

    def toolkit(self) -> Toolkit:
        return Toolkit.from_files(self.tool_files())
