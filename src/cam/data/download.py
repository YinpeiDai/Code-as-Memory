"""Download immutable published demo pools and their experiment assignments."""
from pathlib import Path
from cam.io import read, sha256, write


def download(benchmark, output, tasks=None, videos=False):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install download support with: uv sync --extra data") from error
    from cam.interface.registry import checkout
    repository = checkout()
    if repository is None:
        raise ValueError("Run download-demos from an editable CaM checkout containing configs/")
    release_path = repository / "configs" / benchmark / "datasets/release.json"
    if not release_path.is_file():
        raise ValueError(f"{benchmark}: the verified public upload is not yet available")
    release = read(release_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    options = dict(repo_id=release["repo_id"], repo_type="dataset", revision=release["revision"], local_dir=output)
    # Fetch metadata in a separate directory so a partial download never appears ready.
    metadata = output / ".download"
    for filename, key in (("manifest.json", "manifest_sha256"), ("selected_25.json", "selection_sha256")):
        path = Path(hf_hub_download(filename=filename, **{**options, "local_dir": metadata}))
        if sha256(path) != release[key]:
            raise ValueError(f"Published metadata checksum mismatch: {filename}")
    manifest = read(metadata / "manifest.json")
    tasks = tasks or list(manifest["tasks"])
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(manifest["tasks"]):
        raise ValueError("Choose distinct published tasks")
    if (output / "manifest.json").exists():
        previous = read(output / "manifest.json")
        if previous.get("download", {}).get("revision") != release["revision"]:
            raise ValueError("Destination contains another dataset revision; use a new directory")
        tasks = list(dict.fromkeys([*previous["tasks"], *tasks]))
    for task in tasks:
        item = manifest["tasks"][task]
        hf_hub_download(filename=item["file"], **options)
        if videos:
            for row in item["episodes"]:
                for key in ("video", "poster"):
                    if row.get(key):
                        hf_hub_download(filename=row[key], **options)
    manifest["tasks"] = {task: manifest["tasks"][task] for task in tasks}
    manifest["download"] = {"repo_id": release["repo_id"], "revision": release["revision"]}
    manifest["total_episodes"] = sum(len(item["episodes"]) for item in manifest["tasks"].values())
    write(output / "selected_25.json", read(metadata / "selected_25.json"))
    write(output / "manifest.json", manifest)
    return {"dataset": str(output), "tasks": len(tasks), "episodes": manifest["total_episodes"], **manifest["download"]}
