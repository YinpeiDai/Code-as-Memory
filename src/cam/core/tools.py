# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""Tools: Python functions the coding agent writes, which the policy calls by name with arguments it chooses.

Conventions of a tool file:
  - public functions are tools and are named perceive_<name> or act_<name>; helpers begin with _;
  - a perceive tool takes `obs` (observations.Observations) first, measures and returns JSON-able facts, moves
    nothing;
  - an act tool takes `robot` (robot.Robot) first, sees the scene from its start on, moves the robot and returns a
    JSON-able result (AGENT.md says what a result reports);
  - the remaining parameters are the policy's arguments; the docstring is what the policy reads;
  - the top level holds only functions, constants and docstrings; no imports (np and math are in scope).

Tools are stateless. Every call runs in a fresh namespace built from the sources, the objects a tool gets are
read-only (frames, arrays and intrinsics), and the source may not assign attributes, read private names, use global
or nonlocal, or catch every exception. All files of a workspace share one namespace, so helpers can be reused. A call
runs under a time limit.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import math
import signal
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import sandbox
from cam.runtime.robot import MotionError

PERCEIVE, ACT = "perceive", "act"
FIRST_PARAM = {PERCEIVE: "obs", ACT: "robot"}
TIME_LIMIT = {PERCEIVE: 120, ACT: 900}  # seconds per call
RESULT_CHARS = 3000  # characters of a result the policy reads
TOP_LEVEL = (ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)


class ToolError(Exception):
    """A call the policy made wrongly (unknown tool, bad arguments); the message goes back to the policy."""


class ToolTimeout(BaseException):
    """A tool ran past its time limit. A BaseException, so a tool cannot catch it."""


@dataclass(frozen=True)
class Tool:
    name: str
    kind: str
    signature: str
    doc: str
    origin: str

    def card(self) -> str:
        return f"{self.signature}\n    " + self.doc.strip().replace("\n", "\n    ")


def _kind(name: str) -> str | None:
    for kind in (PERCEIVE, ACT):
        if name.startswith(kind + "_"):
            return kind
    return None


def problems(source: str, origin: str) -> list[str]:
    """Why a tool file cannot be used; empty when it can."""
    out = [f"{origin}: {p}" for p in sandbox.check(source)]
    if out:
        return out
    tree = ast.parse(source)

    def at(node, msg):
        out.append(f"{origin} line {node.lineno}: {msg}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            at(node, "global and nonlocal are not allowed (tools are stateless)")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                at(node, f"private attribute {node.attr} is not readable")
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                at(node, "assigning attributes is not allowed (tools are stateless)")
        elif isinstance(node, ast.ExceptHandler):
            caught = node.type.id if isinstance(node.type, ast.Name) else None
            if node.type is None or caught == "BaseException":
                at(
                    node,
                    "catch specific exceptions (Exception at most), not everything",
                )
        elif isinstance(node, ast.Try):
            for sub in node.finalbody:
                for inner in ast.walk(sub):
                    if isinstance(inner, (ast.Return, ast.Break, ast.Continue)):
                        at(
                            inner,
                            "return, break or continue inside finally would swallow the episode's end",
                        )
    for node in tree.body:
        if not isinstance(node, TOP_LEVEL) or (
            isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant)
        ):
            at(node, "only functions, constants and docstrings at the top level")
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        kind = _kind(node.name)
        if kind is None:
            out.append(
                f"{origin}: public function {node.name} must be named perceive_* or act_* (helpers begin with _)"
            )
            continue
        args = [a.arg for a in node.args.args]
        if not args or args[0] != FIRST_PARAM[kind]:
            out.append(
                f"{origin}: {node.name} must take `{FIRST_PARAM[kind]}` as its first parameter"
            )
        if not ast.get_docstring(node):
            out.append(f"{origin}: {node.name} needs a docstring (the policy reads it)")
    return out


def _signature(node: ast.FunctionDef) -> str:
    a = node.args
    params, defaults = a.args[1:], a.defaults
    pad = [None] * (len(params) - len(defaults)) + list(defaults)
    parts = [
        p.arg if d is None else f"{p.arg}={ast.unparse(d)}" for p, d in zip(params, pad)
    ]
    if a.kwonlyargs:
        parts.append("*")
        parts += [
            p.arg if d is None else f"{p.arg}={ast.unparse(d)}"
            for p, d in zip(a.kwonlyargs, a.kw_defaults)
        ]
    return f"{node.name}({', '.join(parts)})"


def _public_getattr(obj, name: str, *default):
    """getattr for tools, which may read a frame field by name but no private attribute."""
    if not isinstance(name, str) or name.startswith("_"):
        raise AttributeError(f"{name!r} is not readable")
    return getattr(obj, name, *default)


@contextlib.contextmanager
def time_limit(seconds: float):
    """Raises ToolTimeout after `seconds` (only in the main thread, where signals are delivered)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(*_):
        raise ToolTimeout(f"the tool ran longer than {seconds} s")

    old = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class Toolkit:
    def __init__(self, sources: list[tuple[str, str]]):
        """`sources`: (origin, source) of every tool file."""
        bad = [p for origin, src in sources for p in problems(src, origin)]
        if bad:
            raise ToolError("; ".join(bad))
        self._codes = [compile(src, origin, "exec") for origin, src in sources]
        self.tools: dict[str, Tool] = {}
        for origin, src in sources:
            for node in ast.parse(src).body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    if node.name in self.tools:
                        raise ToolError(
                            f"{node.name} is defined in {self.tools[node.name].origin} and {origin}"
                        )
                    self.tools[node.name] = Tool(
                        node.name,
                        _kind(node.name),
                        _signature(node),
                        ast.get_docstring(node),
                        origin,
                    )
        probe = self._namespace()  # signatures only; calls always get a fresh namespace
        self._signatures = {n: inspect.signature(probe[n]) for n in self.tools}

    @classmethod
    def from_files(cls, paths: list[Path]) -> "Toolkit":
        return cls([(str(p), Path(p).read_text()) for p in paths])

    def names(self, kind: str) -> list[str]:
        return [n for n, t in self.tools.items() if t.kind == kind]

    def catalog(self, kind: str) -> str:
        return "\n".join(self.tools[n].card() for n in self.names(kind)) or "(none)"

    def _namespace(self) -> dict:
        ns = sandbox.scope()
        ns["__builtins__"]["getattr"] = _public_getattr
        for code in self._codes:
            exec(code, ns)
        return ns

    def validate(self, name, args, kind: str) -> None:
        """Raises ToolError when `name` is not a tool of `kind` or `args` do not fit its parameters."""
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None or tool.kind != kind:
            raise ToolError(
                f"No {kind} tool {name!r}. The {kind} tools are {', '.join(self.names(kind)) or 'none'}."
            )
        if not isinstance(args, dict):
            raise ToolError("args must be a JSON object.")
        try:
            self._signatures[name].bind(None, **args)
        except TypeError as e:
            raise ToolError(f"Bad arguments for {tool.signature}. {e}") from None

    def call(self, name: str, first, args: dict, kind: str):
        """Runs one tool in a fresh namespace under its time limit. Usage errors raise ToolError; errors inside the
        tool propagate (ToolTimeout included)."""
        self.validate(name, args, kind)
        fn = self._namespace()[name]
        with time_limit(TIME_LIMIT[kind]):
            return fn(first, **args)


def to_json(x, digits: int = 4):
    """A tool result as plain JSON values, floats rounded."""
    if isinstance(x, dict):
        return {str(k): to_json(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v, digits) for v in x]
    if isinstance(x, np.ndarray):
        return to_json(x.tolist(), digits)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), digits)
    if x is None or isinstance(x, str):
        return x
    return str(x)


def dumps(x, limit: int) -> str:
    text = json.dumps(to_json(x))
    return (
        text
        if len(text) <= limit
        else text[:limit] + f" ...(cut at {limit} characters)"
    )


def outcome(fn, *args) -> str:
    """Runs a tool call for the policy and returns its result, or the error, as text. EpisodeEnd propagates."""
    try:
        return dumps(fn(*args), RESULT_CHARS)
    except ToolError as e:
        return f"error: {e}"
    except MotionError as e:
        return f"error: {e}"
    except ToolTimeout as e:
        return f"the tool failed: {e}"
    except Exception as e:  # a fault inside the tool: shown, and kept for diagnosis
        return f"the tool failed: {type(e).__name__}: {str(e)[:400]}"
