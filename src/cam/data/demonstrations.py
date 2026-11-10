"""Shared writer/validator for allowlisted ``cam-demonstrations/2`` H5 files.

Benchmark data protocols translate their native recordings into this schema:

  /                       attrs: schema, benchmark, task, split, depth, fields, source
  /episode_<id>/          attrs: episode, goal, difficulty, action_alignment, step_semantics
    step      (T,) int64  recording frame index 0..T-1
    is_demo   (T,) bool   task-provided video/cue frames before execution
    <field>   (T, ...)    each declared per-frame field
    <static>  (...)       each declared per-episode field (e.g. calibration)

``fields``/``static`` map names to {"dtype", "shape", "description"}. Values are
copied without resampling; nothing outside the declaration is written, and the
validator rejects links, virtual/external datasets and dataset attributes.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from cam.io import read, sha256

SCHEMA = "cam-demonstrations/2"
ROOT_ATTRS = {"schema", "benchmark", "task", "split", "depth", "fields", "source"}
EPISODE_ATTRS = {"episode", "goal", "difficulty", "action_alignment", "step_semantics"}
BUILTIN = {
    "step": {"dtype": "int64", "shape": [], "description": "recording frame index"},
    "is_demo": {
        "dtype": "bool",
        "shape": [],
        "description": "task-provided video/cue frame before execution",
    },
}


def declaration(fields, static=None):
    return {"frame": {**BUILTIN, **fields}, "static": dict(static or {})}


def create(path, benchmark, task, declared, depth, source):
    h5 = h5py.File(path, "x")
    h5.attrs.update(
        schema=SCHEMA,
        benchmark=benchmark,
        task=task,
        split="train",
        depth=bool(depth),
        fields=json.dumps(declared, sort_keys=True),
        source=source,
    )
    return h5


class Episode:
    """Stream declared arrays; length=None grows until the recorded episode ends."""

    def __init__(self, h5, declared, episode, length, attrs, static=None):
        if set(attrs) != EPISODE_ATTRS - {"episode"}:
            raise ValueError("Episode attributes must be exactly " + str(sorted(EPISODE_ATTRS)))
        self.declared, self.length, self.filled = declared, None if length is None else int(length), 0
        self.group = h5.create_group(f"episode_{int(episode)}")
        self.group.attrs.update(episode=int(episode), **{k: ("" if v is None else v) for k, v in attrs.items()})
        static = static or {}
        if set(static) != set(declared["static"]):
            raise ValueError("Static fields differ from the declaration")
        for name, value in static.items():
            spec = declared["static"][name]
            value = np.asarray(value)
            _check(name, value, spec, ())
            self.group.create_dataset(name, data=value)
        for name, spec in declared["frame"].items():
            shape = tuple(spec["shape"])
            self.group.create_dataset(
                name,
                shape=(self.length or 0, *shape),
                dtype=np.dtype(spec["dtype"]),
                chunks=(1, *shape) if shape else None,
                compression="lzf" if shape else None,
                **({"maxshape": (None, *shape)} if self.length is None else {}),
            )

    def append(self, arrays):
        if set(arrays) != set(self.declared["frame"]) - {"step"}:
            raise ValueError(
                f"Frame fields differ from the declaration: {sorted(set(arrays) ^ (set(self.declared['frame']) - {'step'}))}"
            )
        count = {len(v) for v in arrays.values()}
        if len(count) != 1:
            raise ValueError("Frame arrays have different lengths")
        count = count.pop()
        if self.length is not None and self.filled + count > self.length:
            raise ValueError("More frames than declared")
        arrays = {**arrays, "step": np.arange(self.filled, self.filled + count)}
        for name, value in arrays.items():
            spec = self.declared["frame"][name]
            value = np.asarray(value)
            _check(name, value, spec, (count,))
            if self.length is None:
                self.group[name].resize(self.filled + count, axis=0)
            self.group[name][self.filled : self.filled + count] = value
        self.filled += count

    def close(self):
        if self.length is not None and self.filled != self.length:
            raise ValueError(f"Episode has {self.filled} of {self.length} frames")


def _check(name, value, spec, prefix):
    if value.shape != (*prefix, *spec["shape"]):
        raise ValueError(f"{name}: shape {value.shape} != {(*prefix, *spec['shape'])}")
    if value.dtype != np.dtype(spec["dtype"]):
        raise ValueError(f"{name}: dtype {value.dtype} != {spec['dtype']}")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{name}: nonfinite values")


def validate_dataset(root, verify_hashes=True):
    """Validate a v2 extraction against its manifest; returns the manifest."""
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    if manifest.get("schema") != "cam-dataset/1" or manifest.get("demonstration_schema") != SCHEMA:
        raise ValueError("Expected a CaM cam-demonstrations/2 extraction")
    declared = manifest["fields"]
    for task, item in manifest["tasks"].items():
        path = root / item["file"]
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Extracted data must be an internal regular file")
        if verify_hashes and sha256(path) != item["sha256"]:
            raise ValueError(f"Extracted data hash mismatch: {task}")
        with h5py.File(path, "r") as h5:
            if (
                set(h5.attrs) != ROOT_ATTRS
                or h5.attrs["schema"] != SCHEMA
                or h5.attrs["task"] != task
                or h5.attrs["benchmark"] != manifest["benchmark"]
                or h5.attrs["split"] != "train"
                or bool(h5.attrs["depth"]) != manifest["depth"]
                or json.loads(h5.attrs["fields"]) != declared
            ):
                raise ValueError(f"Invalid public H5 attributes: {task}")
            expected = {f"episode_{int(row['episode'])}": row for row in item["episodes"]}
            if set(h5) != set(expected):
                raise ValueError(f"Unassigned demonstrations: {task}")
            names = set(declared["frame"]) | set(declared["static"])
            for key, row in expected.items():
                if not isinstance(h5.get(key, getlink=True), h5py.HardLink):
                    raise ValueError("H5 links cannot expose external evidence")
                group = h5[key]
                if set(group) != names or set(group.attrs) != EPISODE_ATTRS:
                    raise ValueError(f"Unexpected public H5 fields or annotations: {key}")
                length = int(row["frames"])
                for name in group:
                    data = group[name]
                    if (
                        not isinstance(group.get(name, getlink=True), h5py.HardLink)
                        or not isinstance(data, h5py.Dataset)
                        or data.is_virtual
                        or data.external
                        or len(data.attrs)
                    ):
                        raise ValueError("Only self-contained allowlisted H5 datasets are permitted")
                    spec = declared["frame"].get(name) or declared["static"][name]
                    prefix = (length,) if name in declared["frame"] else ()
                    if data.shape != (*prefix, *spec["shape"]) or data.dtype != np.dtype(spec["dtype"]):
                        raise ValueError(f"Field {key}/{name} shape or dtype changed")
                if int(group.attrs["episode"]) != int(row["episode"]) or not np.array_equal(
                    group["step"][:], np.arange(length)
                ):
                    raise ValueError(f"Episode identity or frame indices changed: {key}")
    return manifest


def selection_seed(benchmark, task):
    """Stable, documented RNG seed for random demonstration selection."""
    import hashlib

    return int(hashlib.sha256(f"cam/{benchmark}/{task}".encode()).hexdigest()[:8], 16)


def random_select(records, count, benchmark, task):
    """Uniform random subset without replacement, sorted by episode ID."""
    records = sorted(records, key=lambda row: int(row["episode"]))
    if count == "all":
        return records
    count = int(count)
    if not 1 <= count <= len(records):
        raise ValueError(f"Cannot select {count} of {len(records)} demonstrations for {task}")
    rng = np.random.default_rng(selection_seed(benchmark, task))
    chosen = sorted(rng.choice(len(records), size=count, replace=False).tolist())
    return [records[i] for i in chosen]


def allocate(sizes, count):
    """Proportional (largest-remainder) shares of ``count`` over strata ``sizes``.

    Every stratum gets at least one pick when ``count`` allows; ties go to the
    stratum listed first. Shares never exceed a stratum's size.
    """
    names = list(sizes)
    total = sum(sizes.values())
    if not 1 <= count <= total:
        raise ValueError(f"Cannot allocate {count} of {total}")
    exact = {name: count * sizes[name] / total for name in names}
    shares = {name: min(sizes[name], int(exact[name])) for name in names}
    if count >= len(names):
        for name in names:
            shares[name] = max(shares[name], 1)
    order = sorted(names, key=lambda n: (-(exact[n] - int(exact[n])), names.index(n)))
    while sum(shares.values()) < count:
        for name in order:
            if sum(shares.values()) == count:
                break
            if shares[name] < sizes[name]:
                shares[name] += 1
    while sum(shares.values()) > count:
        name = max(names, key=lambda n: (shares[n] - exact[n], -names.index(n)))
        shares[name] -= 1
    return shares


def stratified_select(records, count, benchmark, task, stratum):
    """Proportional random selection within host-side strata (e.g. scene variants).

    ``stratum(row)`` labels each record; labels are used only for sampling and are
    never written to the learner-visible demonstrations. Within each stratum the
    picks are uniform random with the same documented seed as ``random_select``.
    """
    records = sorted(records, key=lambda row: int(row["episode"]))
    if count == "all":
        return records
    groups = {}
    for row in records:
        groups.setdefault(str(stratum(row)), []).append(row)
    labels = sorted(groups)
    shares = allocate({label: len(groups[label]) for label in labels}, int(count))
    rng = np.random.default_rng(selection_seed(benchmark, task))
    chosen = []
    for label in labels:
        rows = groups[label]
        picks = rng.choice(len(rows), size=shares[label], replace=False).tolist()
        chosen.extend(rows[i] for i in picks)
    return sorted(chosen, key=lambda row: int(row["episode"]))
