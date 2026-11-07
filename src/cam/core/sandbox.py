# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""The restricted scope the coding agent's code runs in, as Code-as-Memory runs its programs: no imports, no names
that begin with __, and no exec, eval, open or compile. The scope holds numpy (np), math and safe builtins."""

from __future__ import annotations

import ast
import builtins as _py
import math
import sys

import numpy as np

FORBIDDEN_CALLS = {
    "exec",
    "eval",
    "open",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "setattr",
    "delattr",
}
ALLOWED_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hasattr",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "print",
    "ArithmeticError",
    "Exception",
    "IndexError",
    "KeyError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)


def check(source: str) -> list[str]:
    """What makes the code unsafe or invalid; empty when it may run."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad.append(f"line {node.lineno}: imports are not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            bad.append(f"line {node.lineno}: names beginning with __ are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            bad.append(
                f"line {node.lineno}: attributes beginning with __ are not allowed"
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        ):
            bad.append(f"line {node.lineno}: {node.func.id} is not allowed")
    return bad


def _import(name, *args, **kwargs):
    # numpy's C methods import their helpers through the calling frame's __import__; only modules already loaded are
    # given (the code cannot name __import__ itself, see check)
    if name in sys.modules or name.split(".")[0] in ("numpy", "math"):
        return _py.__import__(name, *args, **kwargs)
    raise ImportError(f"{name} is not available")


def scope(**names) -> dict:
    """A fresh global scope: np, math, the safe builtins and `names`."""
    return {
        "np": np,
        "math": math,
        "__builtins__": {
            **{k: getattr(_py, k) for k in ALLOWED_BUILTINS},
            "__import__": _import,
        },
        **names,
    }
