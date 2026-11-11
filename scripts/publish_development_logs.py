"""Publish recorded development evidence without H5 data or external asset trees.

Regular file bytes are unchanged. Internal aliases are made portable; omitted
H5 data and external links are inventoried separately. Originals are read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars

BENCHMARKS = ("mikasa_robo_vla_suite", "robomme", "rmbench", "robomemarena")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(root, benchmark, output):
    source = root / benchmark
    destination = output / f"{benchmark}.tar.zst"
    inventory_path = output / f"{benchmark}.inventory.json"
    if destination.exists() and inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        if sha256(destination) != inventory["archive_sha256"]:
            raise ValueError(f"Archive checksum changed: {destination}")
        return inventory
    inventory = {"benchmark": benchmark, "files": [], "internal_links": [],
                 "excluded_h5": [], "external_links": []}

    def select(info):
        path = root / info.name
        target = path.resolve() if path.is_symlink() else path
        if path.suffix.lower() in (".h5", ".hdf5") or (path.is_symlink() and target.suffix.lower() in (".h5", ".hdf5")):
            inventory["excluded_h5"].append({"path": info.name, "bytes": path.stat().st_size if path.exists() and path.is_file() else None})
            return None
        if info.issym():
            if not target.is_relative_to(source):
                inventory["external_links"].append({"path": info.name, "target": os.readlink(path)})
                return None
            info.linkname = os.path.relpath(target, path.parent)
            inventory["internal_links"].append({"path": info.name, "target": info.linkname})
        elif info.isfile() or info.islnk():
            inventory["files"].append({"path": info.name, "bytes": path.stat().st_size,
                                       **({"hardlink": info.linkname} if info.islnk() else {})})
        elif not info.isdir():
            raise ValueError(f"Unexpected special file: {path}")
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    temporary = destination.with_suffix(".zst.partial")
    with (output / f"{benchmark}.compression.log").open("w") as log:
        process = subprocess.Popen(["zstd", "-q", "-f", "-3", "-T4", "-o", str(temporary)],
                                   stdin=subprocess.PIPE, stderr=log)
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                archive.add(source, arcname=benchmark, recursive=True, filter=select)
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"Compression failed: {benchmark}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    temporary.rename(destination)
    inventory.update(archive=destination.name, archive_bytes=destination.stat().st_size,
                     archive_sha256=sha256(destination), file_count=len(inventory["files"]),
                     uncompressed_file_bytes=sum(row["bytes"] for row in inventory["files"]))
    write_json(inventory_path, inventory)
    return inventory


def main():
    disable_progress_bars()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("public_runs/dev_logs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", default="Yinpei/CaM_develop_logs")
    args = parser.parse_args()
    root, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    if api.whoami()["name"] != args.repo.split("/")[0]:
        raise ValueError("The active Hugging Face account does not match the requested namespace")
    for benchmark in BENCHMARKS:
        if not (root / benchmark).is_dir():
            raise ValueError(f"Missing source: {benchmark}")
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    receipt_path = output / "release.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        "schema": "cam-development-log-release/1", "repository": args.repo,
        "source": "public_runs/dev_logs", "created": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {}, "excluded_benchmark": "mikasa_robo", "h5_included": False,
    }

    def progress(phase, benchmark=""):
        receipt["status"], receipt["current_benchmark"] = phase, benchmark
        write_json(receipt_path, receipt)
        (output / "PROGRESS.md").write_text(
            f"# Development-log publication\n\nStatus: {phase}\n\nCurrent benchmark: {benchmark or 'all complete'}\n\n"
            + "\n".join(f"- {name}: uploaded and SHA-256 verified" for name in receipt["benchmarks"]) + "\n")
        print(f"{phase}: {benchmark}", flush=True)

    for benchmark in BENCHMARKS:
        if benchmark in receipt["benchmarks"]:
            continue
        progress("packaging", benchmark)
        inventory = package(root, benchmark, output)
        progress("uploading", benchmark)
        destination = output / inventory["archive"]
        remote_path = "archives/" + destination.name
        commit = api.upload_file(path_or_fileobj=destination, path_in_repo=remote_path,
                                 repo_id=args.repo, repo_type="dataset",
                                 commit_message=f"Publish {benchmark} development evidence without H5")
        remote = api.get_paths_info(args.repo, [remote_path], repo_type="dataset", revision=commit.oid)[0]
        remote_hash = remote.lfs.sha256 if remote.lfs else None
        if remote.size != inventory["archive_bytes"] or remote_hash != inventory["archive_sha256"]:
            raise ValueError(f"Remote transfer checksum mismatch: {benchmark}")
        api.upload_file(path_or_fileobj=output / f"{benchmark}.inventory.json",
                        path_in_repo=f"manifests/{benchmark}.json", repo_id=args.repo, repo_type="dataset",
                        commit_message=f"Record {benchmark} archive contents and excluded references")
        receipt["benchmarks"][benchmark] = {key: inventory[key] for key in
            ("archive", "archive_bytes", "archive_sha256", "file_count", "uncompressed_file_bytes")}
        receipt["benchmarks"][benchmark]["verified_commit"] = commit.oid
        progress("uploaded", benchmark)
    progress("documenting")
    (output / "SHA256SUMS").write_text("".join(
        f"{row['archive_sha256']}  archives/{row['archive']}\n" for row in receipt["benchmarks"].values()))
    card = """---
pretty_name: CaM Development Logs
license: apache-2.0
task_categories:
- robotics
---
# CaM development logs

Recorded agent transcripts, prompts, tool audits, code revisions, frozen policies,
results and available videos. Publishing generates no new simulator data.

Includes RoboMME, RMBench, RoboMemArena and the five-task **MIKASA VLA suite**.
The separate historical `mikasa_robo/` collection and **all H5 files** are excluded.
Manifests list omitted H5 paths and external simulator/evaluation links. Evaluation
summaries inside the development directories remain included. No external data
trees are copied. Training-observation H5 files are not part of this release.

Original regular-file bytes are retained. Internal symbolic links are made
relative; hard-linked copies are stored once per archive. Historical absolute
paths inside logs and prepared manifests remain unchanged. These are evidence
archives: use current CaM preparation commands to reproduce development.

Download demonstration pools separately from the four benchmark datasets under
[Yinpei](https://huggingface.co/Yinpei), and prepare only the recorded assignment.
See [Code-as-Memory](https://github.com/YinpeiDai/Code-as-Memory) for reproduction.
Upstream benchmark code retains its original notices and licenses.

## Download and extract

Install `zstd` and the Hugging Face CLI, then:

```bash
hf download Yinpei/CaM_develop_logs --repo-type dataset --local-dir CaM_develop_logs
cd CaM_develop_logs
sha256sum -c SHA256SUMS
mkdir -p ../public_runs/dev_logs
for archive in archives/*.tar.zst; do
  tar --zstd -xf "$archive" -C ../public_runs/dev_logs
done
```

The four benchmark directories retain their settings and task layout.

| Archive | Files | Compressed GiB |
|---|---:|---:|
"""
    for name, row in receipt["benchmarks"].items():
        card += f"| [{name}](archives/{row['archive']}) | {row['file_count']:,} | {row['archive_bytes']/2**30:.2f} |\n"
    (output / "README.md").write_text(card)
    for name in ("README.md", "SHA256SUMS"):
        api.upload_file(path_or_fileobj=output / name, path_in_repo=name,
                        repo_id=args.repo, repo_type="dataset", commit_message="Document verified development-log release")
    progress("complete")
    api.upload_file(path_or_fileobj=receipt_path, path_in_repo="release.json", repo_id=args.repo,
                    repo_type="dataset", commit_message="Record completed development-log release")
    api.update_repo_settings(args.repo, repo_type="dataset", private=False)
    print(f"Complete: https://huggingface.co/datasets/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
