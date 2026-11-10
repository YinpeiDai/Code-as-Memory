"""Pin RoboMemArena episode identities and release provenance into CaM metadata.

Lists the gated Hugging Face release at one revision (``HF_TOKEN``), groups the
``subtask_data`` files by task and seed, and writes

  src/cam/interface/robomemarena/metadata/release.json        per-file size + LFS sha256
  src/cam/interface/robomemarena/metadata/train/record_dataset_<Task>_metadata.json
  src/cam/interface/robomemarena/metadata/test/record_dataset_<Task>_metadata.json

Train identities are the released demonstrations (seed parsed from file names),
ranked by seed as episodes 0..N-1. Test identities are the official sweep
(``run_all_tasks1_26.py``: ``--seed 50``, ``--num-trials-per-task 50``, seed = 50 + ep).
Run with the CaM interpreter; it never downloads data files.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.request

from cam.interface.robomemarena import adapter as rma
from cam.io import write

REPO = "RoboMemArenaBenchmark/RoboMemArena"
REVISION = "93d765af20efa13f7b5190b0f35319a86f7ab202"
UPSTREAM = "https://github.com/OpenHelix-Team/RoboMemArena"
PATTERN = re.compile(
    r"^(?P<category>[^/]+)/(?P<directory>(?P<tid>\d+)_[^/]+_dataset)/subtask_data/"
    r"(?P<name>(?P<primitive>.+)_(?P<order>\d+)_seed(?P<seed>\d+)_task(?P<task>\d+)\.hdf5)$"
)


def api(url, token):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response), response.headers.get("Link", "")
        except OSError:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def listing(revision, token):
    url = (
        f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}?"
        + urllib.parse.urlencode({"recursive": "true", "limit": 1000})
    )
    files = []
    while url:
        page, link = api(url, token)
        files += [row for row in page if row.get("type") == "file"]
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, default=rma.METADATA)
    args = parser.parse_args()
    token = os.environ["HF_TOKEN"]
    commits, _ = api(f"https://huggingface.co/api/datasets/{REPO}/commits/{args.revision}", token)
    if commits[0]["id"] != args.revision:
        raise ValueError("Revision is not a full commit ID")
    files = listing(args.revision, token)
    source = rma.default_source()
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    category_of = {tid: cat for cat, ids in rma.CATEGORIES.items() for tid in ids}
    grouped = defaultdict(lambda: defaultdict(list))
    directories, other = {}, []
    for row in files:
        match = PATTERN.match(row["path"])
        if not match:
            other.append({"path": row["path"], "size": row["size"], "sha256": (row.get("lfs") or {}).get("oid")})
            continue
        tid = int(match["task"])
        if int(match["tid"]) != tid or category_of[tid] != match["category"]:
            raise ValueError(f"Unexpected release layout: {row['path']}")
        if "lfs" not in row or not re.fullmatch(r"[0-9a-f]{64}", row["lfs"]["oid"]):
            raise ValueError(f"Missing LFS sha256 (is HF_TOKEN valid?): {row['path']}")
        directory = f"{match['category']}/{match['directory']}/subtask_data"
        if directories.setdefault(tid, directory) != directory:
            raise ValueError(f"Task {tid} spans several directories")
        grouped[tid][int(match["seed"])].append(
            {
                "name": match["name"],
                "order": int(match["order"]),
                "size": int(row["size"]),
                "sha256": row["lfs"]["oid"],
            }
        )
    if set(grouped) != set(rma.TASKS):
        raise ValueError("Release does not cover tasks 1-26")
    test_seeds = {rma.TEST_SEED_START + k for k in range(rma.TEST_TRIALS)}
    release = {
        "schema": "cam-robomemarena-release/1",
        "repo": REPO,
        "repo_type": "dataset",
        "revision": args.revision,
        "revision_date": commits[0]["date"],
        "revision_title": commits[0]["title"],
        "benchmark_source": {"repository": UPSTREAM, "commit": commit, "subdirectory": "evaluation_benchmark"},
        "layout": "<category>/<id>_<name>_dataset/subtask_data/<primitive>_<order>_seed<seed>_task<id>.hdf5",
        "trajectory_rule": "one demonstration = all subtask files of one seed, concatenated by <order>",
        "other_files": sorted(other, key=lambda row: row["path"]),
        "tasks": {},
    }
    for tid, task in sorted(rma.TASKS.items()):
        seeds = grouped[tid]
        for seed, rows in seeds.items():
            rows.sort(key=lambda row: row["order"])
            if [row["order"] for row in rows] != list(range(len(rows))):
                raise ValueError(f"Task {tid} seed {seed}: subtask orders are not contiguous")
        if set(seeds) & test_seeds:
            raise ValueError(f"Task {tid}: released demonstration seeds overlap the test seeds")
        bddl = rma.bddl_file(source, tid)
        common = {
            "env_id": task,
            "task_id": tid,
            "bddl": bddl.relative_to(source).as_posix(),
            "bddl_sha256": rma.file_sha256(bddl),
            "benchmark_commit": commit,
        }
        release["tasks"][task] = {
            "task_id": tid,
            "category": category_of[tid],
            "directory": directories[tid],
            "bddl": common["bddl"],
            "bddl_sha256": common["bddl_sha256"],
            "seeds": {str(seed): seeds[seed] for seed in sorted(seeds)},
        }
        train = [
            {"episode": index, "seed": seed, "difficulty": None}
            for index, seed in enumerate(sorted(seeds))
        ]
        test = [
            {"episode": k, "seed": rma.TEST_SEED_START + k, "difficulty": None}
            for k in range(rma.TEST_TRIALS)
        ]
        write(
            args.output / "train" / f"record_dataset_{task}_metadata.json",
            {
                **common,
                "split": "train",
                "identity": f"released demonstrations of {REPO}@{args.revision}, ranked by seed",
                "records": train,
            },
        )
        write(
            args.output / "test" / f"record_dataset_{task}_metadata.json",
            {
                **common,
                "split": "test",
                "identity": "official sweep run_all_tasks1_26.py: seed = 50 + ep",
                "protocol": {
                    "seed_start": rma.TEST_SEED_START,
                    "trials": rma.TEST_TRIALS,
                    "max_steps": rma.MAX_STEPS,
                    "num_steps_wait": rma.SETTLE_STEPS,
                    "post_goal_steps": rma.POST_GOAL_STEPS,
                    "extra_pour_monitor_steps": rma.EXTRA_POUR_MONITOR_STEPS,
                    "fail_on_extra_pour": rma.FAIL_ON_EXTRA_POUR,
                    "resize_size": rma.IMAGE[0],
                    "render": list(rma.RENDER),
                },
                "records": test,
            },
        )
        print(task, len(train), "demonstrations; seeds", min(seeds), "-", max(seeds))
    write(args.output / "release.json", release)
    print("files", sum(len(v) for s in grouped.values() for v in s.values()), "other", len(other))


if __name__ == "__main__":
    main()
