"""Pin MIKASA-Robo-VLA episode identities and release hashes for CaM.

Writes ``src/cam/benchmarks/metadata/mikasa_robo/``:

* ``release.json``: Hugging Face repo, revision and, for every file of the 90
  task folders, its size and SHA-256 (LFS object id) or git blob SHA-1.
* ``train/record_dataset_<Task>_metadata.json``: released training episode
  ``k`` (RLDS ``episode_metadata/file_path`` ``.../train_data_{k}.npz``) with
  seed ``metadata.json["episode_seeds"][k]``.
* ``test/record_dataset_<Task>_metadata.json``: the official 50 episodes,
  seeds ``4242424242 + i`` (``mikasa_robo_suite/vla/benchmarking.py``).

The registered native step limit comes from the pinned simulator registry, not
from the CSV (FindImposterShapeAndColor5x3 registers 40; the CSV says 25).
Run in the MIKASA-Robo interpreter from the CaM checkout:

    envs/mikasa_robo/.venv/bin/python -I scripts/pin_mikasa_robo_metadata.py \
        --raw-data data/raw/mikasa_robo/rlds
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

REPO = "mikasa-robo/mikasa-robo-vla-rlds"
REVISION = "8575624b7e054fc1bceccedc7a8c1b533b4f15d2"
TEST_START_SEED, TEST_EPISODES = 4242424242, 50
ROOT = Path(__file__).resolve().parents[1]


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def write(path, value):
    """Indented JSON with one line per record/file entry (compact, diff-friendly)."""
    parts = []
    for key, item in value.items():
        if key in ("records", "files"):
            if isinstance(item, dict):
                body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in item.items())
                parts.append(f" {json.dumps(key)}: {{\n{body}\n }}")
            else:
                body = ",\n".join("  " + json.dumps(v) for v in item)
                parts.append(f" {json.dumps(key)}: [\n{body}\n ]")
        else:
            parts.append(f" {json.dumps(key)}: " + json.dumps(item, indent=1).replace("\n", "\n "))
    text = "{\n" + ",\n".join(parts) + "\n}\n"
    if json.loads(text) != value:
        raise AssertionError("JSON layout changed the value")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def listing(revision):
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(REPO, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"Hugging Face resolved {info.sha}, expected {revision}")
    files = {}
    for item in info.siblings:
        if "/" not in item.rfilename:
            continue
        entry = {"size": int(item.size)}
        if item.lfs:
            entry["sha256"] = item.lfs.sha256
        else:
            entry["git_sha1"] = item.blob_id
        files[item.rfilename] = entry
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "third_party/mikasa_robo")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/cam/benchmarks/metadata/mikasa_robo"
    )
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(ROOT / "src"))
    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401 - registers env IDs
    from mikasa_robo_suite.vla.utils.apply_wrappers import VLA_WRAPPER_CONFIGS
    from mikasa_robo_suite.vla.utils.dataset_naming import env_id_to_dataset_name
    from cam.benchmarks.mikasa_robo import dataset_name

    with (source / "mikasa_robo_vla_envs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    tasks = [row["Name"] for row in rows]
    if len(tasks) != 90 or set(tasks) != set(VLA_WRAPPER_CONFIGS):
        raise ValueError("CSV tasks differ from the canonical wrapper registry")
    files = listing(REVISION)
    folders = {path.split("/")[0] for path in files}
    if {env_id_to_dataset_name(t) for t in tasks} != folders:
        raise ValueError("Release folders differ from the 90 canonical tasks")
    if any(dataset_name(t) != env_id_to_dataset_name(t) for t in tasks):
        raise ValueError("CaM release folder naming differs from MIKASA naming")
    write(
        args.output / "release.json",
        {"repo": REPO, "repo_type": "dataset", "revision": REVISION, "files": files},
    )
    test_seeds = {TEST_START_SEED + i for i in range(TEST_EPISODES)}
    missing = []
    for row in rows:
        task = row["Name"]
        folder = dataset_name(task) + "/1.0.0"
        raw = args.raw_data / folder / "metadata.json"
        if not raw.is_file():
            missing.append(task)
            continue
        expected = files[folder + "/metadata.json"]
        if raw.stat().st_size != expected["size"] or git_blob_sha1(raw) != expected["git_sha1"]:
            raise ValueError(f"metadata.json differs from the pinned release: {task}")
        native = json.loads(raw.read_text())
        seeds = [int(s) for s in native["episode_seeds"]]
        if (
            native["env_id"] != task
            or native["num_episodes"] != len(seeds)
            or len(set(seeds)) != len(seeds)
            or len(native["episode_lengths"]) != len(seeds)
            or not all(native["success_once"])
            or len(native["success_once"]) != len(seeds)
        ):
            raise ValueError(f"Inconsistent release metadata: {task}")
        if test_seeds & set(seeds):
            raise ValueError(f"Released training seeds overlap test seeds: {task}")
        horizon = int(gym.spec(task).max_episode_steps)
        common = {
            "env_id": task,
            "max_episode_steps": horizon,
            "horizon_split": row["Horizon Split"],
            "memory_type": row["Memory Type"],
            "data_source": row["Data Source"],
            # The canonical wrapper executes zero commands during the cue phase.
            "noop_cue_phase": VLA_WRAPPER_CONFIGS[task].curriculum_wrapper is not None,
        }
        write(
            args.output / "train" / f"record_dataset_{task}_metadata.json",
            {
                **common,
                "records": [
                    {"task": task, "episode": k, "seed": seed} for k, seed in enumerate(seeds)
                ],
                "provenance": {
                    "repo": REPO,
                    "revision": REVISION,
                    "file": folder + "/metadata.json",
                    "git_sha1": expected["git_sha1"],
                    "identity": "episode k is RLDS episode_metadata/file_path "
                    ".../train_data_{k}.npz; seed = metadata.json episode_seeds[k]",
                    "seed_note": "PPO/MP collection seeds identify demonstrations; they are "
                    "not single-env replay keys",
                },
            },
        )
        write(
            args.output / "test" / f"record_dataset_{task}_metadata.json",
            {
                **common,
                "records": [
                    {"task": task, "episode": i, "seed": TEST_START_SEED + i}
                    for i in range(TEST_EPISODES)
                ],
                "provenance": {
                    "source": "mikasa_robo_suite/vla/benchmarking.py START_SEED, "
                    "NUM_EPISODES_PER_TASK; docs/source/evaluation_protocol.rst",
                    "csv_max_length": int(row["Max Length"]),
                    "max_episode_steps_source": "registered gym spec (env.max_episode_steps)",
                },
            },
        )
    print(json.dumps({"tasks": len(tasks), "pinned": len(tasks) - len(missing), "missing": missing}))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
