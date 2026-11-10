"""Derive RoboMemArena demonstrations by simulator replay of the released expert actions.

The pinned HF release (``subtask_data``; one demonstration = all subtask files of one
seed, concatenated in subtask order) supplies expert actions and RGB, but no depth. For every released
training identity the benchmark interpreter performs the official reset at that seed
(``np.random.seed``/``env.seed``/``reset`` and 10 settling steps) and replays the
recorded actions open loop, clipped to the action bounds as ``robot.step`` would,
under the pinned official scorer. Only replays the official scorer marks successful
are eligible; ``count`` are drawn with ``stratified_select`` (host-only strata: the
scene-variant signature of the release file names). In the historical two-pass
mode, selected replays are re-run and
recorded with exactly the online sensors (RGB, optional aligned depth, calibration,
TCP, joints, fingers) plus the replayed command, up to the official decision point.
The opt-in single-pass mode captures those sensors and checks success together,
reuses existing RGB-D, and independently replays only the requested validation sample.

Released files are verified (size and Git LFS sha256) before use. Per-identity replay
outcomes and image agreement with the released recordings stay host-side in the
manifest and audit; no release annotation, file name, seed or variant reaches the H5.
The CaM environment orchestrates simulator-interpreter workers and never imports the
simulator itself.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np

import cam
from cam.interface.base import source_hash
from cam.interface.registry import get_adapter, simulator_python
from cam.data import demonstrations as demos
from cam.io import read, sha256, write

from cam.interface.robomemarena.adapter import IMAGE, METADATA, TASKS, task_id, task_prompts

RAW_KEYS = {"actions", "dones", "keyframe_indices", "obs", "rewards"}
#: Datasets present in some task folders that CaM neither needs nor exports.
RAW_OPTIONAL = {"absolute_actions"}
RAW_OBS = {"agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states"}
RAW_ATTRS = {"language_instruction", "num_samples"}
CHUNK = 64
IDENTITIES_PER_WORKER = 13
ACTION_ALIGNMENT = (
    "action[t] is the replayed command sent with robot.step semantics after frame t; "
    "frame t+1 is its result. The last action completed the task under the official "
    "scorer; its result frame is not recorded."
)
STEP_SEMANTICS = (
    "Native policy step: frame 0 is the first online frame after the 10 settling "
    "steps, exactly as in live episodes."
)
METHOD = (
    "CaM simulator replay of released expert actions at each demonstration's seed: "
    "official reset and settling, open-loop actions clipped to [-1, 1], pinned official "
    "scorer; frames re-rendered with the online sensors; successful replays only"
)


def fields(adapter, task, depth):
    """Online sensors (calibration K static per episode) plus the replayed command."""
    online = adapter.interface(task, depth)["sensors"]
    frame, static = {}, {}
    for name, spec in online.items():
        target = static if name.endswith("_K") else frame
        target[name] = {key: spec[key] for key in ("dtype", "shape", "description", "aligned_with") if key in spec}
    frame["action"] = {
        "dtype": "float64",
        "shape": [7],
        "description": "replayed command (dx, dy, dz, drx, dry, drz, gripper) sent after this frame "
        "with robot.step semantics: the released expert action clipped to [-1, 1]",
    }
    return demos.declaration(frame, static)


def snapshot_root(raw_data):
    """``data/raw/robomemarena`` (containing ``hf/``) or the HF snapshot itself."""
    raw_data = Path(raw_data).resolve()
    return raw_data / "hf" if (raw_data / "hf").is_dir() else raw_data


def release():
    return read(METADATA / "release.json")


def variant(item, seed):
    """Host-only scene-variant signature: the ordered subtask primitives of one seed.

    Used solely to stratify selection; never written to learner-visible files.
    """
    names = [row["name"] for row in sorted(item["seeds"][str(seed)], key=lambda r: r["order"])]
    return "|".join(re.sub(r"_\d+_seed\d+_task\d+\.hdf5$", "", name) for name in names)


def select(records, count, benchmark, task, item):
    """Proportional selection across scene variants (``stratified_select``)."""
    return demos.stratified_select(records, count, benchmark, task, lambda row: variant(item, row["seed"]))


def strata(records, selected, item):
    """Anonymous per-variant pool sizes and selected counts for the host manifest."""
    labels = sorted({variant(item, r["seed"]) for r in records})
    chosen = [variant(item, r["seed"]) for r in selected]
    return [
        {"stratum": i, "size": sum(variant(item, r["seed"]) == label for r in records), "selected": chosen.count(label)}
        for i, label in enumerate(labels)
    ]


def verify_files(root, item, seeds):
    """Stat every listed file of ``seeds`` against the pinned release sizes."""
    directory = root / item["directory"]
    listed = {row["name"] for rows in item["seeds"].values() for row in rows}
    present = {p.name for p in directory.glob("*.hdf5")}
    if present - listed:
        raise ValueError(f"Files outside the pinned release in {directory}: {sorted(present - listed)[:5]}")
    stats = {}
    for seed in seeds:
        for row in item["seeds"][str(seed)]:
            path = directory / row["name"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing release file: {path}")
            stat = path.stat()
            if stat.st_size != row["size"]:
                raise ValueError(f"Size mismatch (incomplete download?): {path}")
            stats[path.name] = {"path": path.relative_to(root).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return stats


def _check_raw(group, path):
    if not RAW_KEYS <= set(group) <= RAW_KEYS | RAW_OPTIONAL or set(group["obs"]) != RAW_OBS or set(group.attrs) != RAW_ATTRS:
        raise ValueError(f"Unexpected raw fields or annotations: {path}")
    length = int(group.attrs["num_samples"])
    expected = {
        "actions": ("float64", (length, 7)),
        "obs/agentview_rgb": ("uint8", (length, *IMAGE)),
        "obs/eye_in_hand_rgb": ("uint8", (length, *IMAGE)),
    }
    for key, (dtype, shape) in expected.items():
        if group[key].dtype != np.dtype(dtype) or group[key].shape != shape:
            raise ValueError(f"{path}:{key} is {group[key].dtype}{group[key].shape}")
    if not np.isfinite(group["actions"][:]).all():
        raise ValueError(f"Nonfinite released action: {path}")
    return length


def load_recording(root, item, seed, images=True, verify=False):
    """Released actions and optional RGB; verify=True also hashes action-only reads.

    With ``images`` every file is read once into memory, checked against its Git LFS
    sha256 and parsed from the verified bytes. Without images only the actions are
    read; callers compare their sha256 with verified earlier evidence or request
    full-file verification for a previously unchecked identity.
    """
    actions, agentview, wrist, files = [], [], [], []
    for row in sorted(item["seeds"][str(seed)], key=lambda r: r["order"]):
        path = root / item["directory"] / row["name"]
        if images or verify:
            data = path.read_bytes()
            if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"LFS sha256 mismatch: {path}")
            handle = h5py.File(io.BytesIO(data), "r")
        else:
            handle = h5py.File(path, "r")
        with handle:
            if set(handle) != {"data"} or set(handle["data"]) != {"demo_0"}:
                raise ValueError(f"Expected one data/demo_0 recording: {path}")
            group = handle["data/demo_0"]
            _check_raw(group, path)
            actions.append(group["actions"][:])
            if images:
                agentview.append(group["obs/agentview_rgb"][:])
                wrist.append(group["obs/eye_in_hand_rgb"][:])
        files.append(row["name"])
    actions = np.concatenate(actions)
    value = {"actions": actions, "actions_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(), "files": files}
    if images:
        value.update(agentview=np.concatenate(agentview), wrist=np.concatenate(wrist))
    return value


def replay(env, seed, actions, observe=None):
    """Official reset at ``seed`` then open-loop clipped actions until the scorer decides.

    ``observe(k, frame, command)`` sees frame k (before command k). Returns the last
    StepResult (or None) and the number of commands sent.
    """
    frames = env.restart(seed)
    frame, result, sent = frames[-1], None, 0
    for k, values in enumerate(actions):
        command = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
        if observe is not None:
            observe(k, frame, command)
        result = env.step(command)
        sent += 1
        if result.status != "running":
            break
        frame = result.frames[-1]
    return result, sent


# ----------------------------------------------------------------- simulator workers


def native_environment(job):
    """One reusable native environment for a worker (benchmark interpreter only)."""
    adapter = get_adapter("robomemarena", job["source"])
    task = job["task"]
    records = adapter.records(task, "train")
    folder = Path(tempfile.mkdtemp(prefix="cam-rma-replay-"))
    write(folder / f"record_dataset_{task}_metadata.json", {"records": records})
    try:
        return adapter.make_environment(folder, task, int(records[0]["episode"]), "train", job["depth"])
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def score_identities(job, environment=native_environment):
    """Phase 1: replay every listed identity once; report the official outcome."""
    root, item = Path(job["root"]), release()["tasks"][job["task"]]
    env, results = environment(job), []
    try:
        for row in job["identities"]:
            seed = int(row["seed"])
            recorded = load_recording(root, item, seed, images=True)
            errors = {"agentview": [], "wrist": []}

            def observe(k, frame, command, reference=recorded):
                if k < len(reference["agentview"]):
                    for key in errors:
                        errors[key].append(float(np.abs(frame[key + "_rgb"].astype(np.int16) - reference[key][k]).mean()))

            result, sent = replay(env, seed, recorded["actions"], observe)
            status = "actions_exhausted" if result is None or result.status == "running" else result.status
            metrics = (result.metrics or {}) if result is not None else {}
            results.append(
                {
                    "episode": int(row["episode"]),
                    "seed": seed,
                    "status": status,
                    "steps": sent,
                    "tsr": bool(metrics.get("tsr", False)) and status == "success",
                    "csr": metrics.get("csr"),
                    "failure_reason": metrics.get("failure_reason"),
                    "released_frames": int(len(recorded["actions"])),
                    "actions_sha256": recorded["actions_sha256"],
                    "agentview_mae_mean": float(np.mean(errors["agentview"])),
                    "agentview_mae_max": float(np.max(errors["agentview"])),
                    "wrist_mae_mean": float(np.mean(errors["wrist"])),
                    "wrist_mae_max": float(np.max(errors["wrist"])),
                    "private_error": result.private_error if result is not None else None,
                }
            )
            del observe, recorded
    finally:
        env.close()
    return {"results": results}


def record_task(job, environment=native_environment):
    """Phase 2: re-run the selected successful replays and write the task H5."""
    root, item = Path(job["root"]), release()["tasks"][job["task"]]
    adapter = get_adapter("robomemarena", job["source"])
    declared = fields(adapter, job["task"], job["depth"])
    frame_names = [n for n in declared["frame"] if n not in ("step", "is_demo", "action")]
    output, episodes = Path(job["output"]), []
    env = None
    try:
        with demos.create(Path(job["h5"]), "robomemarena", job["task"], declared, job["depth"], job["source_text"]) as h5:
            for row in job["episodes"]:
                seed, episode, steps = int(row["seed"]), int(row["episode"]), int(row["steps"])
                reused = job.get("reuse", {}).get(str(episode))
                if reused:
                    source = Path(reused["h5"])
                    before = source.stat()
                    if (before.st_size, before.st_mtime_ns) != tuple(reused["stat"]):
                        raise ValueError(f"Reusable H5 changed: {source}")
                    with h5py.File(source, "r") as original:
                        key = f"episode_{episode}"
                        if len(original[key]["step"]) != steps:
                            raise ValueError(f"Reusable episode length changed: {episode}")
                        original.copy(key, h5, name=key)
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f"Reusable H5 changed during copying: {source}")
                    movie = None
                    if job["videos"]:
                        if not reused.get("video"):
                            raise ValueError("Requested preview video is absent from reusable episode")
                        movie = output / "rollouts" / job["task"] / f"episode_{episode}.mp4"
                        movie.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(reused["video"], movie)
                    episodes.append({"episode": episode, "frames": steps,
                        "video": movie.relative_to(output).as_posix() if movie else None,
                        "reused_from": reused["dataset"]})
                    continue
                if env is None:
                    env = environment(job)
                recorded = load_recording(root, item, seed, images=False)
                if recorded["actions_sha256"] != row["actions_sha256"]:
                    raise ValueError(f"Released actions changed since verification: seed {seed}")
                target, buffer, static, movie, video = None, [], {}, None, None
                try:
                    if job["videos"]:
                        import imageio.v2 as imageio

                        movie = output / "rollouts" / job["task"] / f"episode_{episode}.mp4"
                        movie.parent.mkdir(parents=True, exist_ok=True)
                        video = imageio.get_writer(
                            movie, fps=20, codec="libx264", quality=7,
                            ffmpeg_params=["-threads", "1"], ffmpeg_log_level="error",
                        )

                    def flush():
                        arrays = {name: np.stack([f[name] for f, _ in buffer]) for name in frame_names}
                        arrays["action"] = np.stack([c for _, c in buffer])
                        arrays["is_demo"] = np.zeros(len(buffer), dtype=bool)
                        target.append(arrays)
                        buffer.clear()

                    def observe(k, frame, command):
                        nonlocal target
                        if target is None:
                            static.update({name: frame[name] for name in declared["static"]})
                            target = demos.Episode(
                                h5, declared, episode, steps,
                                {"goal": job["goal"], "difficulty": row.get("difficulty"),
                                 "action_alignment": ACTION_ALIGNMENT, "step_semantics": STEP_SEMANTICS},
                                static=static,
                            )
                        for name in declared["static"]:
                            if not np.array_equal(frame[name], static[name]):
                                raise ValueError(f"{name} changed within an episode")
                        if int(frame["step"]) != k:
                            raise ValueError("Online frame step does not match the replay index")
                        buffer.append((frame, command))
                        if video is not None:
                            video.append_data(np.hstack([frame["agentview_rgb"], frame["wrist_rgb"]]))
                        if len(buffer) == CHUNK:
                            flush()

                    result, sent = replay(env, seed, recorded["actions"], observe)
                    if result is None or result.status != "success" or sent != steps:
                        raise ValueError(f"Replay of seed {seed} is not reproducible: {getattr(result, 'status', None)} after {sent}")
                    if buffer:
                        flush()
                    target.close()
                finally:
                    if video is not None:
                        video.close()
                episodes.append(
                    {"episode": episode, "frames": steps, "video": movie.relative_to(output).as_posix() if movie else None}
                )
    finally:
        if env is not None:
            env.close()
    return {"episodes": episodes}


def worker_main(path):
    """Benchmark-interpreter entry: ``python -m cam.interface.robomemarena.data JOB.json``."""
    job = read(path)
    from cam.interface.robomemarena.capture import capture_task
    value = {"score": score_identities, "record": record_task,
             "capture": capture_task, "validate": score_identities}[job["mode"]](job)
    write(Path(job["result"]), value)


def run_worker(job, python, folder):
    """Run one job in the benchmark interpreter; returns its JSON result."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{job['mode']}_{job['task']}_{time.time_ns()}"
    job = {**job, "result": str(folder / f"{name}.result.json")}
    write(folder / f"{name}.job.json", job)
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(cam.__file__).resolve().parents[1]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
    with (folder / f"{name}.log").open("wb") as log:
        code = subprocess.run(
            [str(python), "-B", "-m", "cam.interface.robomemarena.data", str(folder / f"{name}.job.json")],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=folder,
        ).returncode
    if code:
        raise RuntimeError(f"Replay worker failed ({code}); see {folder / (name + '.log')}")
    return read(job["result"])


# ----------------------------------------------------------------- orchestration


def extract(
    raw_data,
    output,
    benchmark="robomemarena",
    count=25,
    tasks=None,
    depth=True,
    videos=True,
    workers=2,
    benchmark_source=None,
    resume=False,
    reuse=None,
    max_successful=None,
    replay_cache=None,
    single_pass=False,
    validation_episodes=0,
):
    raw_data, output = Path(raw_data).resolve(), Path(output).resolve()
    adapter = get_adapter(benchmark, benchmark_source)
    tasks = tasks or adapter.tasks()
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS.values()) or workers < 1:
        raise ValueError("Choose distinct native tasks and positive extraction workers")
    count = "all" if count == "all" else int(count)
    if max_successful is not None and (max_successful < 1 or count != 'all'):
        raise ValueError('A successful-replay cap requires count=all and a positive maximum')
    if replay_cache is not None and max_successful is None:
        raise ValueError('Replay-cache adoption requires an explicit successful-replay cap')
    if validation_episodes < 0 or (validation_episodes and not single_pass):
        raise ValueError('Sampled replay validation requires single-pass capture and a nonnegative sample size')
    if single_pass and (max_successful is None or not depth or videos):
        raise ValueError('Single-pass capture requires a success cap, depth, and --no-videos')
    root, info, depth = snapshot_root(raw_data), release(), bool(depth)
    python = simulator_python(adapter)
    if python is None:
        raise FileNotFoundError("RoboMemArena replay needs its benchmark interpreter")
    from cam.harness.prepare import simulator_snapshot

    manifest = {
        "schema": "cam-dataset/1",
        "demonstration_schema": demos.SCHEMA,
        "created": time.time(),
        "benchmark": benchmark,
        "benchmark_hash": source_hash(adapter),
        "raw_data": str(raw_data),
        "depth": depth,
        "fields": fields(adapter, tasks[0], depth),
        "selection": {
            "count": count,
            "rule": "cam.data.demonstrations.stratified_select over the successful replays of all "
            "released identities: proportional largest-remainder shares per host-only scene "
            "variant (ordered subtask primitives of the release file names), uniform without "
            "replacement within each, RNG seed = int(sha256('cam/robomemarena/<task>')[:8], 16)",
        },
        "requested_tasks": tasks,
        "videos": videos,
        "release": {
            "repo": info["repo"],
            "revision": info["revision"],
            "listing": "cam/interface/robomemarena/metadata/release.json",
            "listing_sha256": sha256(METADATA / "release.json"),
            "verification": "every replayed identity: size + Git LFS sha256 of all its files",
        },
        "derivation": {
            "method": METHOD,
            "benchmark_commit": info["benchmark_source"]["commit"],
            "simulator": simulator_snapshot(python),
        },
        "tasks": {},
    }
    if any(fields(adapter, task, depth) != manifest["fields"] for task in tasks):
        raise ValueError("Field declarations differ across tasks")
    cache_jobs = {}
    if max_successful is not None:
        manifest['selection'].update(max_successful=max_successful,
            rule='Reuse verified outcomes; score remaining training identities in pinned order until the success cap or release exhaustion. Retain existing successful captures, then stratify remaining selections up to the cap.')
    if single_pass:
        manifest['selection'].update(single_pass=True, validation_episodes=validation_episodes,
            rule='Retain existing RGB-D captures; capture known successes first, then unchecked training identities in pinned order, scoring during capture. Skip known failures. Stop at the success cap or exhaustion; independently replay a deterministic sample only.')
    if replay_cache is not None:
        from cam.interface.robomemarena.replay_cache import open_cache
        cache_jobs, manifest['replay_cache'] = open_cache(replay_cache, manifest)
    reusable = None
    if reuse is not None:
        reuse = Path(reuse).resolve()
        reusable = validate_dataset(reuse, benchmark_source=adapter.source)
        for key in ("benchmark", "benchmark_hash", "raw_data", "depth", "fields", "release"):
            if manifest[key] != reusable[key]:
                raise ValueError(f"Reusable demonstration protocol differs: {key}")
        manifest["reuse"] = {"dataset": str(reuse), "manifest_sha256": sha256(reuse / "manifest.json"),
                             "rule": "Copy existing successful RGB-D episodes verbatim; reuse verified replay outcomes only when raw files are unchanged"}
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        previous = read(
            output / "manifest.json" if (output / "manifest.json").exists() else output / "manifest.partial.json"
        )
        for key in ("benchmark", "benchmark_hash", "raw_data", "depth", "fields", "selection", "requested_tasks", "videos", "release"):
            if manifest[key] != previous[key]:
                raise ValueError(f"Extraction resume configuration changed: {key}")
        if previous.get("reuse") != manifest.get("reuse"):
            raise ValueError("Extraction reuse source changed")
        if previous.get('replay_cache', {}).get('source_plan_sha256') != manifest.get('replay_cache', {}).get('source_plan_sha256'):
            raise ValueError('Extraction replay cache source changed')
        manifest["created"] = previous["created"]
    work = output / "work"
    write(output / "manifest.partial.json", manifest)

    def unchanged(stats):
        for stat in stats.values():
            now = (root / stat["path"]).stat()
            if (now.st_size, now.st_mtime_ns) != (stat["size"], stat["mtime_ns"]):
                raise ValueError(f"Raw data changed since verification: {stat['path']}")

    pending, scored, prepared_stats = [], {}, {}
    for task in tasks:
        audit = output / "audit" / f"{task}.json"
        if resume and audit.exists():
            item = read(audit)
            unchanged(item["stats"])
            if sha256(output / item["file"]) != item["sha256"]:
                raise ValueError(f"Completed extraction changed: {task}")
            manifest["tasks"][task] = {k: v for k, v in item.items() if k != "stats"}
            continue
        if resume and not single_pass:
            for path in [output / "h5" / f"record_dataset_{task}{s}" for s in (".h5", ".h5.partial")] + [output / "rollouts" / task]:
                if path.exists():
                    target = output / "interrupted" / str(time.time_ns()) / path.name
                    target.parent.mkdir(parents=True)
                    shutil.move(path, target)
        item = info["tasks"][task]
        if item["task_id"] != task_id(task):
            raise ValueError("Release listing and task mapping disagree")
        records = adapter.records(task, "train")
        test_seeds = {int(r["seed"]) for r in adapter.records(task, "test")}
        if {int(r["seed"]) for r in records} & test_seeds:
            raise ValueError("Training/test seed overlap")
        if sorted(int(r["seed"]) for r in records) != sorted(int(s) for s in item["seeds"]):
            raise ValueError("Pinned train identities disagree with the release listing")
        stats = verify_files(root, item, [int(r["seed"]) for r in records])
        prepared_stats[task] = stats
        phase = output / "audit" / f"{task}.replay.json"
        if resume and (output / 'audit' / f'{task}.pre_capture.json').exists() and single_pass:
            previous = read(output / 'audit' / f'{task}.pre_capture.json')
            unchanged(previous['stats'])
            scored[task] = previous
        elif resume and phase.exists():
            previous = read(phase)
            unchanged(previous["stats"])
            scored[task] = previous
            if max_successful and sum(r['status'] == 'success' for r in previous['results']) < max_successful and len(previous['results']) < len(records):
                pending.append((task, records, stats))
        elif reusable and task in reusable["tasks"]:
            original = reusable["tasks"][task]
            audit = read(reuse / "audit" / f"{task}.json")
            if audit["sha256"] != original["sha256"] or audit["replay"] != original["replay"]:
                raise ValueError(f"Reusable audit differs from manifest: {task}")
            unchanged(audit["stats"])
            outcomes = original["replay"]["identities"]
            if [(int(r["episode"]), int(r["seed"])) for r in outcomes] != [(int(r["episode"]), int(r["seed"])) for r in records]:
                raise ValueError(f"Reusable replay outcomes do not cover pinned training identities: {task}")
            scored[task] = {"stats": stats, "results": outcomes, "reused_from": str(reuse)}
            write(phase, scored[task])
        else:
            pending.append((task, records, stats))

    if single_pass:
        from cam.interface.robomemarena.capture import extract_captures
        return extract_captures(adapter=adapter, root=root, output=output, manifest=manifest,
            scored=scored, stats=prepared_stats, cache_jobs=cache_jobs, reusable=reusable,
            reuse=Path(reuse) if reuse else None, python=python, workers=workers,
            unchanged=unchanged)

    common = {"source": str(adapter.source), "root": str(root), "depth": False}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = []
        def capped_score(task, records, stats):
            from cam.interface.robomemarena.replay_cache import collect
            inherited, provenance = collect(cache_jobs.get(task, []), records)
            initial = scored.get(task, {}).get('results', [])
            inherited.update({int(r['episode']): r for r in initial})
            phase = output / 'audit' / f'{task}.replay.json'

            def checkpoint():
                value = {'stats': stats, 'results': [inherited[k] for k in sorted(inherited)],
                         'reused_jobs': provenance, 'success_cap': max_successful}
                write(phase, value)
                return value

            checkpoint()
            remaining = [r for r in records if int(r['episode']) not in inherited]
            while remaining:
                needed = max_successful - sum(r['status'] == 'success' for r in inherited.values())
                if needed <= 0:
                    break
                chunk, remaining = remaining[:min(IDENTITIES_PER_WORKER, needed)], remaining[min(IDENTITIES_PER_WORKER, needed):]
                result = run_worker({**common, 'mode': 'score', 'task': task,
                    'identities': [{'episode': int(r['episode']), 'seed': int(r['seed'])} for r in chunk]}, python, work)
                expected = {(int(r['episode']), int(r['seed'])) for r in chunk}
                if len(result['results']) != len(chunk) or {(int(r['episode']), int(r['seed'])) for r in result['results']} != expected:
                    raise ValueError('Native replay result identities differ from the assigned chunk')
                inherited.update({int(r['episode']): r for r in result['results']})
                checkpoint()
            return checkpoint()

        if max_successful is not None:
            capped = {task: pool.submit(capped_score, task, records, stats) for task, records, stats in pending}
            for task, future in capped.items():
                scored[task] = future.result()
        for task, records, stats in ([] if max_successful is not None else pending):
            size = max(1, math.ceil(len(records) / max(1, math.ceil(len(records) / IDENTITIES_PER_WORKER))))
            for start in range(0, len(records), size):
                chunk = [{"episode": int(r["episode"]), "seed": int(r["seed"])} for r in records[start:start + size]]
                job = {**common, "mode": "score", "task": task, "identities": chunk}
                jobs.append((task, pool.submit(run_worker, job, python, work)))
        results, failures = {task: [] for task, _, _ in pending}, {}
        for task, future in jobs:
            try:
                results[task] += future.result()["results"]
            except Exception as exc:  # keep other tasks' completed replays for --resume
                failures.setdefault(task, exc)
        for task, records, stats in ([] if max_successful is not None else pending):
            if task in failures:
                continue
            value = {"stats": stats, "results": sorted(results[task], key=lambda r: r["episode"])}
            if [r["episode"] for r in value["results"]] != [int(r["episode"]) for r in records]:
                raise ValueError(f"Replay results do not cover every identity: {task}")
            write(output / "audit" / f"{task}.replay.json", value)
            scored[task] = value
        if failures:
            raise next(iter(failures.values()))

        recordings = {}
        for task in scored:
            item, records = info["tasks"][task], adapter.records(task, "train")
            outcome = {r["episode"]: r for r in scored[task]["results"]}
            successful = [r for r in records if outcome.get(int(r["episode"]), {}).get("status") == "success"]
            if count != "all" and len(successful) < count:
                raise ValueError(f"Only {len(successful)} of {len(records)} replays succeed for {task}")
            if max_successful is None:
                selected = select(successful, count, benchmark, task, item)
            else:
                existing = {int(r['episode']) for r in (reusable or {}).get('tasks', {}).get(task, {}).get('episodes', [])}
                retained = [r for r in successful if int(r['episode']) in existing]
                if len(retained) > max_successful:
                    raise ValueError('Existing captures exceed the new cap; preserve the source and choose a separate subset')
                candidates = [r for r in successful if int(r['episode']) not in existing]
                needed = min(max_successful - len(retained), len(candidates))
                chosen = select(candidates, needed, benchmark, task, item) if needed else []
                selected = sorted(retained + chosen, key=lambda r: int(r['episode']))
            goal = task_prompts(adapter.source)[f"task{task_id(task)}"]
            partial = output / "h5" / f"record_dataset_{task}.h5.partial"
            partial.parent.mkdir(parents=True, exist_ok=True)
            job = {
                **common,
                "mode": "record",
                "task": task,
                "depth": depth,
                "videos": videos,
                "goal": goal,
                "output": str(output),
                "h5": str(partial),
                "source_text": f"{METHOD}. Release {info['repo']}@{info['revision']}",
                "episodes": [
                    {**r, "steps": outcome[int(r["episode"])]["steps"], "actions_sha256": outcome[int(r["episode"])]["actions_sha256"]}
                    for r in selected
                ],
            }
            if reusable and task in reusable["tasks"]:
                original = reusable["tasks"][task]
                h5_source = reuse / original["file"]
                stat = h5_source.stat()
                eligible = {int(r["episode"]) for r in selected}
                job["reuse"] = {str(r["episode"]): {"dataset": str(reuse), "h5": str(h5_source),
                    "stat": [stat.st_size, stat.st_mtime_ns],
                    "video": str(reuse / r["video"]) if r.get("video") else None}
                    for r in original["episodes"] if int(r["episode"]) in eligible}
            recordings[task] = (pool.submit(run_worker, job, python, work), selected, successful, outcome, goal)
        for task, (future, selected, successful, outcome, goal) in recordings.items():
            written = {row["episode"]: row for row in future.result()["episodes"]}
            unchanged(scored[task]["stats"])
            item = info["tasks"][task]
            destination = output / "h5" / f"record_dataset_{task}.h5"
            (output / "h5" / f"record_dataset_{task}.h5.partial").replace(destination)
            value = {
                "task": task,
                "task_id": task_id(task),
                "file": destination.relative_to(output).as_posix(),
                "sha256": sha256(destination),
                "episodes": [
                    {
                        **r,
                        "frames": written[int(r["episode"])]["frames"],
                        "demo_frames": 0,
                        "goal": goal,
                        "video": written[int(r["episode"])]["video"],
                    }
                    for r in selected
                ],
                "source": {
                    "method": METHOD,
                    "repo": info["repo"],
                    "revision": info["revision"],
                    "directory": item["directory"],
                    "snapshot": str(root),
                    "bddl": item["bddl"],
                    "bddl_sha256": item["bddl_sha256"],
                    "benchmark_commit": info["benchmark_source"]["commit"],
                },
                "replay": {
                    "released": len(adapter.records(task, 'train')),
                    "replayed": len(outcome),
                    "successful": len(successful),
                    "selected": len(selected),
                    "strata": strata(successful, selected, item),
                    "identities": [outcome[e] for e in sorted(outcome)],
                },
            }
            reused_episodes = sorted(e for e, r in written.items() if r.get("reused_from"))
            if reused_episodes:
                value["reuse"] = {"dataset": str(reuse), "episodes": reused_episodes,
                                  "source_h5_sha256": reusable["tasks"][task]["sha256"]}
            write(output / "audit" / f"{task}.json", {**value, "stats": scored[task]["stats"]})
            manifest["tasks"][task] = value
            write(output / "manifest.partial.json", manifest)
    manifest["tasks"] = {task: manifest["tasks"][task] for task in tasks}
    write(output / "manifest.json", manifest)
    (output / "manifest.partial.json").unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    validate_dataset(output, verify_hashes=False, benchmark_source=adapter.source)
    return manifest


def validate_dataset(root, verify_hashes=True, *, benchmark_source=None):
    """Shared v2 validation plus RoboMemArena identity, field and attribute checks."""
    manifest = demos.validate_dataset(root, verify_hashes)
    root = Path(root).resolve()
    if manifest["benchmark"] != "robomemarena":
        raise ValueError("Expected a RoboMemArena extraction")
    adapter = get_adapter("robomemarena", benchmark_source)
    if source_hash(adapter) != manifest['benchmark_hash']:
        raise ValueError('Validation benchmark differs from the captured benchmark')
    for task, item in manifest["tasks"].items():
        if manifest["fields"] != fields(adapter, task, manifest["depth"]):
            raise ValueError(f"Declared fields changed: {task}")
        train = {int(r["episode"]): r for r in adapter.records(task, "train")}
        test = {int(r["seed"]) for r in adapter.records(task, "test")}
        goal = task_prompts(adapter.source)[f"task{task_id(task)}"]
        outcome = {int(r["episode"]): r for r in item["replay"]["identities"]}
        with h5py.File(root / item["file"], "r") as h5:
            for row in item["episodes"]:
                episode = int(row["episode"])
                native = train.get(episode)
                if native is None or int(native["seed"]) != int(row["seed"]) or int(row["seed"]) in test:
                    raise ValueError(f"Episode identity is not a pinned training identity: {task}/{episode}")
                if outcome[episode]["status"] != "success" or outcome[episode]["steps"] != row["frames"]:
                    raise ValueError(f"Demonstration is not a verified successful replay: {task}/{episode}")
                attrs = h5[f"episode_{episode}"].attrs
                if attrs["goal"] != goal or attrs["goal"] != row["goal"] or attrs["difficulty"] != "":
                    raise ValueError(f"Episode attributes changed: {task}/{episode}")
    return manifest


if __name__ == "__main__":
    worker_main(sys.argv[1])
