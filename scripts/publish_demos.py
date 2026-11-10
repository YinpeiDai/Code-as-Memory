"""Publish the released demo pools, selected-25 identities and existing videos.

Usage: python scripts/publish_demos.py --output public_runs/dataset_uploads/TAG
Each repository has one writer, durable per-file receipts and checksum verification.
Interrupted uploads can resume using the same output directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import CommitOperationAdd, HfApi, RepoFile
from cam.io import read, sha256, write

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"robomme": 25, "mikasa_robo": 50, "rmbench": 50, "robomemarena": 50}


def portable_manifest(original):
    keys = ("schema", "benchmark", "benchmark_hash", "demonstration_schema", "fields",
            "depth", "depth_description", "action_alignment", "waypoint_storage", "count_per_task")
    result = {key: original[key] for key in keys if key in original}
    if original.get("replay", {}).get("protocol"):
        result["replay"] = {"protocol": original["replay"]["protocol"]}
    result["tasks"] = {}
    for task, item in original["tasks"].items():
        result["tasks"][task] = {key: item[key] for key in
            ("task", "task_id", "file", "sha256", "episodes", "train_metadata_sha256", "test_metadata_sha256") if key in item}
        if "replay" in item:
            replay = {key: item["replay"][key] for key in ("protocol", "selected") if key in item["replay"]}
            if "identities" in item["replay"]:
                assigned = {int(r["episode"]) for r in item["episodes"]}
                public = ("episode", "seed", "status", "success", "steps", "frames", "observations_sha256",
                          "capture_sha256", "actions_sha256")
                replay["identities"] = [{k: row[k] for k in public if k in row}
                    for row in item["replay"]["identities"] if int(row["episode"]) in assigned]
            result["tasks"][task]["replay"] = replay
    result["total_episodes"] = sum(len(item["episodes"]) for item in result["tasks"].values())
    result["selection"] = {"rule": "Exact released pool; selected_25.json records the separate 25-demo assignment.",
                           "actual_counts": {task: len(item["episodes"]) for task, item in result["tasks"].items()}}
    return result


def publish(benchmark, output, metadata_parent=None):
    api = HfApi()
    source = ROOT / "public_runs/demos" / benchmark
    folder = output / benchmark
    folder.mkdir(parents=True, exist_ok=True)
    original = read(source / "manifest.json")
    manifest = portable_manifest(original)
    selection = read(ROOT / "configs" / benchmark / "datasets/demo_only_25_seeds.json")
    payload = []
    for task, item in manifest["tasks"].items():
        available = {(int(row["episode"]), int(row["seed"])) for row in item["episodes"]}
        chosen = [(int(row["episode"]), int(row["seed"])) for row in selection["tasks"][task]]
        if len(chosen) != 25 or len(set(chosen)) != 25 or not set(chosen) <= available:
            raise ValueError(f"Invalid selected-25 assignment: {benchmark}/{task}")
        payload.append({"path": item["file"], "source": str(source / item["file"]), "sha256": item["sha256"]})
        for row in item["episodes"]:
            for key in ("video", "poster"):
                if row.get(key):
                    path = (source / row[key]).resolve()
                    if not path.is_relative_to(source.resolve()) or not path.is_file():
                        raise ValueError(f"Missing or external demo preview: {row[key]}")
                    payload.append({"path": row[key], "source": str(path), "sha256": sha256(path)})
    repo = f"Yinpei/{benchmark}-{COUNTS[benchmark]}-demos"
    for name, value in (("manifest.json", manifest), ("selected_25.json", selection)):
        path = folder / name
        write(path, value)
        payload.append({"path": name, "source": str(path), "sha256": sha256(path)})
    payload.sort(key=lambda row: (row["path"].endswith('.h5'), row["path"]))
    for row in payload:
        row["bytes"] = Path(row["source"]).stat().st_size
    inventory = {"repo": repo, "source_manifest_sha256": sha256(source / "manifest.json"), "files": payload}
    plan = folder / "inventory.json"
    if plan.exists() and read(plan) != inventory:
        raise ValueError("Upload inputs changed; use a fresh output directory")
    write(plan, inventory)
    receipt = folder / "results.json"
    state = read(receipt) if receipt.exists() else {"repo": repo, "status": "running", "completed": {}}
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    remote = {f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if isinstance(f, RepoFile)}
    unexpected = set(remote) - {r["path"] for r in payload} - {".gitattributes"}
    if unexpected:
        raise ValueError(f"Repository contains unrelated files: {sorted(unexpected)}")
    state.update(status="running", expected=len(payload), bytes=sum(r["bytes"] for r in payload), updated=time.time())
    write(receipt, state)
    pending = []
    for row in payload:
        existing = remote.get(row["path"])
        if existing:
            if existing.lfs:
                matches = existing.lfs.sha256 == row["sha256"]
            else:
                data = Path(row["source"]).read_bytes()
                matches = existing.blob_id == hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if not matches and row["path"] == "manifest.json" and metadata_parent:
                previous = Path(metadata_parent) / benchmark / "manifest.json"
                data = previous.read_bytes()
                original_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                if existing.blob_id != original_blob:
                    raise ValueError("Remote metadata differs from the recorded preceding publication")
                commit = api.create_commit(repo, repo_type="dataset", commit_message="Keep only portable demonstration validation metadata",
                    operations=[CommitOperationAdd(path_in_repo=row["path"], path_or_fileobj=row["source"])])
                write(folder / "metadata_revision.json", {"previous_sha256": sha256(previous), "sha256": row["sha256"],
                      "commit": commit.oid, "reason": "Remove local audit paths, private errors, planner flags and unassigned replay identities"})
                matches = True
                existing = None
            if not matches or (existing and existing.size != row["bytes"]):
                raise ValueError(f"Refusing to overwrite a different file: {repo}/{row['path']}")
            state["completed"][row["path"]] = {"sha256": row["sha256"], "bytes": row["bytes"], "verified": True}
        else:
            pending.append(row)
    # Small previews are grouped; large H5s each get a resumable commit.
    batches = []
    small = [row for row in pending if not row["path"].endswith('.h5')]
    batches.extend(small[i:i + 100] for i in range(0, len(small), 100))
    batches.extend([row] for row in pending if row["path"].endswith('.h5'))
    for batch in batches:
        operations = []
        for row in batch:
            operation = CommitOperationAdd(path_in_repo=row["path"], path_or_fileobj=row["source"])
            if operation.upload_info.sha256.hex() != row["sha256"]:
                raise ValueError(f"Source checksum differs from published evidence: {row['path']}")
            operations.append(operation)
        for attempt in range(4):
            try:
                commit = api.create_commit(repo, repo_type="dataset", operations=operations,
                    commit_message=f"Add {len(batch)} training demonstration files", num_threads=2)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))
        for row in batch:
            state["completed"][row["path"]] = {"sha256": row["sha256"], "bytes": row["bytes"], "commit": commit.oid}
        state.update(updated=time.time(), revision=commit.oid)
        write(receipt, state)
        (folder / "PROGRESS.md").write_text(f"# {benchmark} demonstration upload\n\n{len(state['completed'])}/{len(payload)} files uploaded.\n")
        print(f"{benchmark}: {len(state['completed'])}/{len(payload)} files", flush=True)
    remote = {f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if isinstance(f, RepoFile)}
    for row in payload:
        item = remote[row["path"]]
        if item.size != row["bytes"] or (item.lfs and item.lfs.sha256 != row["sha256"]):
            raise ValueError(f"Remote verification failed: {row['path']}")
        state["completed"][row["path"]]["verified"] = True
    state.update(status="complete", revision=api.repo_info(repo, repo_type="dataset").sha, updated=time.time())
    write(receipt, state)
    release = {"schema": "cam-demo-release/1", "repo_id": repo, "revision": state["revision"],
               "manifest_sha256": sha256(folder / "manifest.json"), "selection_sha256": sha256(folder / "selected_25.json"),
               "tasks": len(manifest["tasks"]), "episodes": manifest["total_episodes"]}
    write(ROOT / "configs" / benchmark / "datasets/release.json", release)
    (folder / "PROGRESS.md").write_text(f"# {benchmark} demonstration upload\n\nComplete: {len(payload)} files, verified against remote checksums.\n")
    return release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=list(COUNTS), default=list(COUNTS))
    parser.add_argument("--metadata-parent", type=Path, help="Prior publication whose exact manifest may be replaced with portable metadata")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(publish, name, args.output.resolve(), args.metadata_parent): name for name in args.benchmarks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                write(args.output / name / "error.json", {"type": type(error).__name__, "message": str(error), "time": time.time()})
                print(f"{name}: {type(error).__name__}: {error}", flush=True)
