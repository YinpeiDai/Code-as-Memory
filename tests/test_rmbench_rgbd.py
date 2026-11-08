"""Storage, identity and scheduling safeguards for regenerated RGB-D data."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from cam.benchmarks import rmbench, rmbench_rgbd_data as rgbd
from cam.data import demonstrations
from cam.io import read, sha256, write
from test_rmbench import native_obs


def capture(root, record, success):
    task, ep = "put_back_block", record["episode"]
    folder = root / "jobs" / task / f"ep{ep:05d}"
    folder.mkdir(parents=True)
    recorder = rgbd.Recorder(folder / "observations.h5", task, ep, "public goal")
    obs = native_obs()
    for camera in obs["observation"].values():
        camera["depth"][:] = 750.25
    recorder.append(obs)
    obs["joint_action"]["left_gripper"] = 0.0
    recorder.append(obs)
    recorder.close()
    result = {**record, "success": success, "status": "success" if success else "expert_failed",
              "frames": 2, "observations_sha256": sha256(folder / "observations.h5")}
    write(folder / "result.json", result)
    return folder


def test_successful_only_dataset_keeps_alignment_and_failure_evidence(tmp_path):
    adapter = rmbench.RMBench()
    records = adapter.records("put_back_block", "train")[:2]
    first = capture(tmp_path, records[0], True)
    failed = capture(tmp_path, records[1], False)
    item = rgbd.assemble(tmp_path, "put_back_block", records)
    assert len(item["episodes"]) == 1 and item["replay"]["shortfall"] == 1
    assert (failed / "observations.h5").exists()
    manifest = {"schema": "cam-dataset/1", "demonstration_schema": demonstrations.SCHEMA,
                "benchmark": "rmbench", "depth": True, "fields": rgbd.fields(),
                "replay": {"protocol": rgbd.PROTOCOL}, "tasks": {"put_back_block": item}}
    write(tmp_path / "manifest.json", manifest)
    rgbd.validate_dataset(tmp_path)
    with h5py.File(tmp_path / item["file"]) as h5, h5py.File(first / "observations.h5") as original:
        key = f"episode_{records[0]['episode']}"
        assert list(h5) == [key]
        assert np.array_equal(h5[key]["head_rgb"][:], original[key]["head_rgb"][:])
        assert h5[key]["head_depth"].shape == (2, 240, 320)
        assert np.all(h5[key]["head_depth"][:] == 750)
        assert list(h5[key]["left_gripper"][:]) == [1, 0]
        assert "action" not in h5[key]
    manifest["tasks"]["put_back_block"]["episodes"][0]["seed"] = 100000
    write(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="identity"):
        rgbd.validate_dataset(tmp_path)


def test_assembly_rejects_changed_capture(tmp_path):
    records = rmbench.RMBench().records("put_back_block", "train")[:1]
    folder = capture(tmp_path, records[0], True)
    with h5py.File(folder / "observations.h5", "r+") as h5:
        h5[f"episode_{records[0]['episode']}/left_gripper"][0] = 0.25
    with pytest.raises(ValueError, match="changed"):
        rgbd.assemble(tmp_path, "put_back_block", records)


def test_waiting_for_gpu_never_starts_on_existing_compute_job(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_RGBD_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(rgbd, "gpu_processes", lambda gpu: [{"pid": 123, "name": "mikasa worker"}])

    def stop_wait(_):
        raise InterruptedError("test stops waiting")

    monkeypatch.setattr(rgbd.time, "sleep", stop_wait)
    with pytest.raises(InterruptedError), rgbd.exclusive_gpu(tmp_path, 1):
        pytest.fail("must not enter while GPU is occupied")
    assert read(tmp_path / "status.json")["status"] == "waiting_for_gpu"


def test_gpu_accounting_parses_process_names_and_propagates_errors(monkeypatch):
    monkeypatch.setattr(rgbd.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="123, /env/bin/python\n"))
    assert rgbd.gpu_processes(1) == [{"pid": 123, "name": "/env/bin/python"}]
    monkeypatch.setattr(rgbd.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="not a process row"))
    with pytest.raises(ValueError):
        rgbd.gpu_processes(1)


def test_two_gpu_allocation_has_two_slots_per_device():
    assert rgbd.gpu_allocation(4, (0, 1)) == {0: 2, 1: 2}
    assert rgbd.gpu_allocation(10, (0, 1)) == {0: 5, 1: 5}
    for count, devices in [(3, (0, 1)), (4, (0, 0)), (1, (0, 1)), (7, (0,))]:
        with pytest.raises(ValueError):
            rgbd.gpu_allocation(count, devices)


def test_rgbd_subset_preserves_replay_provenance(tmp_path):
    from cam.data.subset import subset

    pool = tmp_path / "pool"
    records = rmbench.RMBench().records("put_back_block", "train")[:2]
    for record in records:
        capture(pool, record, True)
    item = rgbd.assemble(pool, "put_back_block", records)
    manifest = {"schema": "cam-dataset/1", "demonstration_schema": demonstrations.SCHEMA,
                "benchmark": "rmbench", "depth": True, "fields": rgbd.fields(),
                "replay": {"protocol": rgbd.PROTOCOL}, "tasks": {"put_back_block": item}}
    write(pool / "manifest.json", manifest)
    result = subset(pool, tmp_path / "selected", count=1)
    rgbd.validate_dataset(tmp_path / "selected")
    assert len(result["tasks"]["put_back_block"]["episodes"]) == 1
    assert result["tasks"]["put_back_block"]["replay"]["identities"] == item["replay"]["identities"]
    assert "strata" not in result["tasks"]["put_back_block"]["replay"]


def test_capture_reuse_verifies_identity_and_preserves_original(tmp_path):
    record = rmbench.RMBench().records("put_back_block", "train")[0]
    old = capture(tmp_path / "old", record, True)
    spec = {**record, "task": "put_back_block", "raw_crc32": 123,
            "protocol": rgbd.PROTOCOL, "source": "/pinned/original", "folder": str(old)}
    write(old / "job.json", spec)
    result = read(old / "result.json")
    result["protocol"] = rgbd.PROTOCOL
    write(old / "result.json", result)
    digest = sha256(old / "observations.h5")
    new = tmp_path / "new/jobs/put_back_block" / old.name
    write(new / "job.json", {**spec, "folder": str(new), "source": "/pinned/new"})
    rgbd.reuse_capture(new / "job.json", tmp_path / "old")
    assert sha256(new / "observations.h5") == sha256(old / "observations.h5") == digest
    assert read(new / "result.json")["reused_from"]["source"] == "/pinned/original"
    bad = tmp_path / "bad/jobs/put_back_block" / old.name
    write(bad / "job.json", {**spec, "folder": str(bad), "seed": -1})
    with pytest.raises(ValueError, match="identity changed"):
        rgbd.reuse_capture(bad / "job.json", tmp_path / "old")
