"""The simulator boundary. Model drivers and statistics do not import a simulator.

Every benchmark adapter declares, per task, the public interface that crosses the
policy boundary: allowlisted sensors, the action vector, rollout video tiles and
the starter policy's one-step observation action. The shared harness validates
every frame and action against this declaration; nothing else is forwarded.

Adapters must be importable without their simulator. Import simulator packages
only inside ``make_environment`` (which runs in the benchmark's own interpreter).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

#: dtype names a sensor may declare; "float"/"int" accept any width of that kind.
DTYPE_KINDS = {"float": "f", "int": "iu", "bool": "b"}
FRAME_KEYS = ("phase", "step")
PHASES = ("demo", "exec")


@dataclass
class EpisodeStart:
    goal: str
    frames: list[dict]
    context: dict


@dataclass
class StepResult:
    frames: list[dict]
    status: str
    steps: int
    motion_error: str | None = None
    private_error: dict | None = None
    #: Native partial scores (e.g. stage completion). Recorded on held-out results;
    #: kept private for training attempts so they never become learner feedback.
    metrics: dict | None = None


class BenchmarkAdapter(Protocol):
    name: str
    package: str
    source: Path
    guide: str

    def tasks(self) -> list[str]: ...
    def records(self, task: str, split: str) -> list[dict]: ...
    def interface(self, task: str, depth: bool) -> dict: ...
    def make_environment(
        self, metadata: Path, task: str, episode: int, split: str, depth: bool
    ): ...
    def action_spec(self, task: str) -> dict: ...
    def begin(self, env, *, builtin_demo: bool, depth: bool) -> EpisodeStart: ...
    def advance(self, env, action, context: dict, *, depth: bool) -> StepResult: ...
    def select_training(
        self, extracted_records: list[dict], count: int | None
    ) -> list[dict]: ...


def source_hash(adapter):
    """Hash of the pinned simulator code, excluding the adapter's bulky asset trees."""
    from cam.io import tree_hash

    return tree_hash(adapter.source, getattr(adapter, "pin_ignore", ()))


def pin_source(adapter, destination):
    """Copy simulator code into a run and link (not copy) declared asset trees.

    Returns ``{link: sha-256 tree hash}`` for every linked asset tree.
    """
    from cam.io import copy_tree, tree_hash

    destination = Path(destination)
    copy_tree(adapter.source, destination, getattr(adapter, "pin_ignore", ()))
    assets = {}
    for link, target in getattr(adapter, "asset_links", {}).items():
        target = Path(target).resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"{adapter.name} assets missing: {target}")
        path = destination / link
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or path.exists():
            raise FileExistsError(f"Pinned source already contains {link}")
        path.symlink_to(target, target_is_directory=True)
        assets[link] = tree_hash(target)
    return assets


def sensor(dtype, shape, description, aligned_with=None):
    """One allowlisted per-frame array. ``None`` in ``shape`` accepts any length."""
    if dtype not in DTYPE_KINDS and np.dtype(dtype).kind not in "buif":
        raise ValueError(f"Unsupported sensor dtype: {dtype}")
    value = {"dtype": dtype, "shape": list(shape), "description": description}
    if aligned_with:
        value["aligned_with"] = aligned_with
    return value


def action(names, description, low=None, high=None, ignored=(), **extra):
    """A fixed-length action vector; ``None`` bounds are unbounded."""
    names = list(names)
    low = list(low) if low is not None else [None] * len(names)
    high = list(high) if high is not None else [None] * len(names)
    if not names or len(set(names)) != len(names) or not len(low) == len(high) == len(names):
        raise ValueError("Action names and bounds must be distinct and aligned")
    if set(ignored) - set(names):
        raise ValueError("Ignored action components must be named components")
    return {
        "names": names,
        "low": low,
        "high": high,
        "ignored": list(ignored),
        "description": description,
        **extra,
    }


def _shape_matches(actual, declared):
    return len(actual) == len(declared) and all(
        d is None or a == d for a, d in zip(actual, declared)
    )


def _dtype_matches(actual, declared):
    if declared in DTYPE_KINDS:
        return actual.kind in DTYPE_KINDS[declared]
    return actual == np.dtype(declared)


def check_frames(frames, sensors):
    """Reject any frame that is not exactly the declared, allowlisted sensors."""
    expected = set(sensors) | set(FRAME_KEYS)
    for frame in frames:
        if set(frame) != expected:
            extra, missing = set(frame) - expected, expected - set(frame)
            raise ValueError(
                f"Observation fields differ from the declared interface: extra {sorted(extra)}, missing {sorted(missing)}"
            )
        if frame["phase"] not in PHASES or int(frame["step"]) != frame["step"]:
            raise ValueError("Invalid frame phase or step")
        for name, spec in sensors.items():
            value = frame[name]
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Sensor {name} must be a NumPy array")
            if not _dtype_matches(value.dtype, spec["dtype"]):
                raise ValueError(f"Sensor {name} dtype {value.dtype} != {spec['dtype']}")
            if not _shape_matches(value.shape, spec["shape"]):
                raise ValueError(f"Sensor {name} shape {value.shape} != {spec['shape']}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"Sensor {name} contains nonfinite values")
            aligned = spec.get("aligned_with")
            if aligned and value.shape[:2] != frame[aligned].shape[:2]:
                raise ValueError(f"Sensor {name} must align with {aligned}")
    return frames


def normalize_action(spec, values):
    """Finite, bounded action in declared order; ignored components become zero."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (len(spec["names"]),) or not np.isfinite(values).all():
        raise ValueError(
            f"Action must contain {len(spec['names'])} finite numbers: {', '.join(spec['names'])}"
        )
    for i, name in enumerate(spec["names"]):
        low, high = spec["low"][i], spec["high"][i]
        if name in spec["ignored"]:
            values[i] = 0.0
            continue
        if low is not None:
            values[i] = max(values[i], low)
        if high is not None:
            values[i] = min(values[i], high)
    return values


def synthetic_value(spec, fill):
    """A constant array matching a declared sensor, for transport preflight."""
    shape = [64 if d is None else d for d in spec["shape"]]
    dtype = {"float": "float32", "int": "int32", "bool": "bool"}.get(
        spec["dtype"], spec["dtype"]
    )
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return np.full(shape, bool(fill % 2), dtype)
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        fill = int(min(max(fill, info.min), info.max))
    return np.full(shape, fill, dtype)


def zero_action(spec):
    """A bounded all-zero action, for transport checks only."""
    return [
        0.0 if lo is None and hi is None else float(min(max(0.0, lo if lo is not None else -math.inf), hi if hi is not None else math.inf))
        for lo, hi in zip(spec["low"], spec["high"])
    ]
