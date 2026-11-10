"""Downloads pin a revision and finalize only a complete requested assignment."""
import sys
from types import SimpleNamespace

import pytest

from cam.data.download import download
from cam.harness.jobs import development_slots
from cam.io import read, sha256, write


def test_pinned_download_optional_videos_and_failure_recovery(tmp_path, monkeypatch):
    from cam.interface import registry
    repository = tmp_path / "repo"
    remote = tmp_path / "remote"
    rows = [{"episode": 0, "seed": 5, "video": "rollouts/A/episode_0.mp4"}]
    write(remote / "manifest.json", {"schema": "cam-dataset/1", "benchmark": "rmbench", "tasks": {
        "A": {"file": "h5/A.h5", "episodes": rows, "sha256": "a" * 64},
        "B": {"file": "h5/B.h5", "episodes": rows, "sha256": "b" * 64}}})
    write(remote / "selected_25.json", {"schema": "cam-demo-selection/1", "tasks": {"A": rows, "B": rows}})
    write(repository / "configs/rmbench/datasets/release.json", {"repo_id": "owner/pool", "revision": "fixed-revision",
        "manifest_sha256": sha256(remote / "manifest.json"), "selection_sha256": sha256(remote / "selected_25.json")})
    monkeypatch.setattr(registry, "checkout", lambda: repository)
    calls, fail = [], [True]
    def fetch(filename, *, repo_id, repo_type, revision, local_dir):
        assert (repo_id, repo_type, revision) == ("owner/pool", "dataset", "fixed-revision")
        calls.append(filename)
        if filename.endswith('.h5') and fail[0]:
            raise OSError("transfer interrupted")
        target = local_dir / filename
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes((remote / filename).read_bytes() if filename.endswith('.json') else b"fixture")
        return str(target)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fetch))
    output = tmp_path / "download"
    with pytest.raises(OSError):
        download("rmbench", output, ["A"])
    assert not (output / "manifest.json").exists()
    fail[0] = False
    download("rmbench", output, ["A"])
    assert set(read(output / "manifest.json")["tasks"]) == {"A"}
    assert not any(path.endswith('.mp4') for path in calls)
    download("rmbench", output, ["B"], videos=True)
    assert set(read(output / "manifest.json")["tasks"]) == {"A", "B"}
    assert "rollouts/A/episode_0.mp4" in calls


def test_development_default_is_eight_jobs_for_any_gpu_count():
    assert development_slots([0]) == [0] * 8
    assert development_slots([0, 1]) == [0, 1] * 4
    assert len(development_slots([0, 1, 2])) == 8
    with pytest.raises(ValueError):
        development_slots([0, 0])
