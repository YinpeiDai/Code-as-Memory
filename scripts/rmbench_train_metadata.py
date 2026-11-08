"""Pin RMBench training identities from the preserved release zips (no simulator).

For each task, reads ``data/<task>/demo_clean.zip`` (as downloaded from
``TianxingChen/RMBench`` at a fixed revision), checks that its sha256 equals the
Hugging Face LFS oid, that the episode files are contiguous and match the
``seed.txt`` order (position i -> seed of episode i), and writes
``src/cam/benchmarks/metadata/rmbench/train/record_dataset_<task>_metadata.json``.
The loose ``demo_clean/`` copies on the hub are not used.

  envs/rmbench/.venv/bin/python -I scripts/rmbench_train_metadata.py \
      --zips data/raw/rmbench/hf/data --revision 5d8ff21038fba32ea549e9eee8497aaba9aab5cf
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = "TianxingChen/RMBench"
TASKS = (
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "observe_and_pickup",
    "place_block_mat",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "swap_T",
    "swap_blocks",
)


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/cam/benchmarks/metadata/rmbench/train"
    )
    args = parser.parse_args()
    from huggingface_hub import HfApi

    paths = [f"data/{task}/demo_clean.zip" for task in args.tasks]
    remote = {
        item.path: item
        for item in HfApi().get_paths_info(REPO, paths, repo_type="dataset", revision=args.revision)
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for task, path in zip(args.tasks, paths):
        archive = args.zips.resolve() / task / "demo_clean.zip"
        lfs = remote[path].lfs
        local = digest(archive)
        if lfs is None or local != lfs.sha256 or archive.stat().st_size != lfs.size:
            raise SystemExit(f"{task}: local zip does not match the {REPO}@{args.revision} LFS object")
        prefix = f"{task}/aloha_agilex/"
        with zipfile.ZipFile(archive) as z:
            seeds = [int(s) for s in z.read(prefix + "seed.txt").decode().split()]
            episodes = sorted(
                name[len(prefix + "data/") :]
                for name in z.namelist()
                if name.startswith(prefix + "data/") and name.endswith(".hdf5")
            )
        if episodes != [f"episode_{i:07d}.hdf5" for i in range(len(seeds))]:
            raise SystemExit(f"{task}: episode files do not match seed.txt")
        if len(set(seeds)) != len(seeds) or max(seeds) >= 100000:
            raise SystemExit(f"{task}: duplicate or evaluation-range training seeds")
        value = {
            "env_id": task,
            "split": "train",
            "records": [{"episode": i, "seed": s} for i, s in enumerate(seeds)],
            "provenance": {
                "rule": "released demo_clean zip; episode_%07d.hdf5 index i has seed seed.txt[i]",
                "repo": REPO,
                "repo_type": "dataset",
                "revision": args.revision,
                "path": path,
                "zip_sha256": local,
                "hf_lfs_sha256": lfs.sha256,
                "zip_size": lfs.size,
                "task_config": "demo_clean",
                "embodiment": "aloha_agilex",
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
        target = args.output / f"record_dataset_{task}_metadata.json"
        target.write_text(json.dumps(value, indent=2) + "\n")
        print(f"{task}: {len(seeds)} training seeds {min(seeds)}..{max(seeds)} -> {target}")


if __name__ == "__main__":
    main()
