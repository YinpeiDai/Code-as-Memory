"""Materialize assigned demos from a larger validated pool without exposing the pool."""

from copy import deepcopy
import os
from pathlib import Path
import shutil
import time

import h5py

from cam.data.demonstrations import SCHEMA, random_select
from cam.data.extract import load_dataset_manifest, validate_dataset
from cam.io import read, sha256, write


def choose(manifest, task, count):
    rows = manifest["tasks"][task]["episodes"]
    if manifest["benchmark"] == "robomemarena":
        from cam.interface.robomemarena.data import release, select

        return select(rows, count, "robomemarena", task, release()["tasks"][task])
    return random_select(rows, count, manifest["benchmark"], task)


def selected_rows(pool, selection, task, count):
    """Match both episode and seed; never silently replace a published assignment."""
    from cam.interface.registry import benchmark_matches
    if selection.get("schema") != "cam-demo-selection/1" or not benchmark_matches(pool["benchmark"], selection.get("benchmark")):
        raise ValueError("Selection benchmark or schema differs from the pool")
    requested = selection.get("tasks", {}).get(task, [])
    identities = [(int(row["episode"]), int(row["seed"])) for row in requested]
    if len(identities) != count or len(set(identities)) != count:
        raise ValueError(f"{task}: selection must contain exactly {count} distinct identities")
    available = {(int(row["episode"]), int(row["seed"])): row for row in pool["tasks"][task]["episodes"]}
    if set(identities) - set(available):
        raise ValueError(f"{task}: selected episode/seed pair is absent from the pool")
    return [available[key] for key in identities]


def subset(source, output, count=25, tasks=None, allow_fewer=False, resume=False,
           selection=None, validate_source=True):
    """Copy selected groups into self-contained H5s; originals remain unchanged.

    Selection is deterministic within this pool. It need not equal a selection
    of the same size made directly from the raw release. Counts are never padded.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Subset output must be outside the source dataset")
    if output.exists() and not resume:
        raise FileExistsError("Use a fresh subset output or --resume")
    count = int(count)
    if count < 1:
        raise ValueError("Subset count must be positive")
    manifest_hash = sha256(source / "manifest.json")
    pool = validate_dataset(source) if validate_source else load_dataset_manifest(source)
    selection_path = Path(selection) if selection is not None else None
    assignment = read(selection_path) if selection_path else None
    if pool.get("demonstration_schema") != SCHEMA:
        raise ValueError("Pool subsets require cam-demonstrations/2")
    tasks = list(tasks or pool["tasks"])
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(pool["tasks"]):
        raise ValueError("Choose distinct tasks from the pool")
    selected = {}
    for task in tasks:
        available = len(pool["tasks"][task]["episodes"])
        if not available or (available < count and not allow_fewer):
            raise ValueError(f"{task}: requested {count} demos, only {available} available")
        selected[task] = (selected_rows(pool, assignment, task, count) if assignment is not None
                          else choose(pool, task, min(count, available)))
    if sha256(source / "manifest.json") != manifest_hash:
        raise ValueError("Pool manifest changed during validation")
    manifest = deepcopy(pool)
    manifest.update(created=time.time(), requested_tasks=tasks, tasks={})
    manifest["selection"] = {
        "count": count,
        "allow_fewer": bool(allow_fewer),
        "rule": "deterministic selection within the prepared pool; proportional scene-variant selection for RoboMemArena, uniform otherwise",
        "source_dataset": str(source),
        "source_manifest_sha256": manifest_hash,
        "source_selection": pool.get("selection"),
        "actual_counts": {task: len(rows) for task, rows in selected.items()},
    }
    if selection_path:
        manifest["selection"].update(rule="Exact published episode/seed list", selection_sha256=sha256(selection_path))
    partial = output / "manifest.partial.json"
    if resume:
        previous = read(output / "manifest.json" if (output / "manifest.json").exists() else partial)
        for key in ("benchmark", "benchmark_hash", "fields", "depth", "selection", "requested_tasks"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Subset resume inputs changed: {key}")
        manifest["created"] = previous["created"]
    output.mkdir(parents=True, exist_ok=resume)
    if selection_path:
        shutil.copy2(selection_path, output / "selected_25.json")
    write(partial, manifest)
    for task, rows in selected.items():
        item = deepcopy(pool["tasks"][task])
        destination = output / item["file"]
        audit = output / "audit" / f"{task}.json"
        if resume and audit.exists():
            saved = read(audit)
            if sha256(destination) != saved["sha256"]:
                raise ValueError(f"Completed subset changed: {task}")
            manifest["tasks"][task] = saved
            write(partial, manifest)
            continue
        pending = destination.with_suffix(".h5.partial")
        for path in (pending, destination):
            if path.exists():
                archived = output / "interrupted" / str(time.time_ns()) / path.name
                archived.parent.mkdir(parents=True)
                shutil.move(path, archived)
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = source / item["file"]
        before = original.stat()
        with h5py.File(original, "r") as src, h5py.File(pending, "x") as dst:
            dst.attrs.update(dict(src.attrs))
            for row in rows:
                key = f"episode_{int(row['episode'])}"
                src.copy(key, dst, name=key)
        after = original.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Pool H5 changed during subset copying: {task}")
        os.replace(pending, destination)
        item["episodes"] = deepcopy(rows)
        item["sha256"] = sha256(destination)
        if "replay" in item:
            item["replay"]["selected"] = len(rows)
            if pool["benchmark"] == "robomemarena":
                from cam.interface.robomemarena.data import release, strata

                item["replay"]["strata"] = strata(pool["tasks"][task]["episodes"], rows, release()["tasks"][task])
        for row in rows:
            if not row.get("video"):
                continue
            video = source / row["video"]
            if not video.resolve().is_relative_to(source):
                raise ValueError("Preview video must be inside the source pool")
            if not video.is_file():
                # Downloads may omit optional previews; H5 contains the frames.
                continue
            target = output / row["video"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, target)
        write(audit, item)
        manifest["tasks"][task] = item
        write(partial, manifest)
    if sha256(source / "manifest.json") != manifest_hash:
        raise ValueError("Pool manifest changed during copying")
    write(output / "manifest.json", manifest)
    validate_dataset(output, verify_hashes=False)
    partial.unlink(missing_ok=True)
    return manifest
