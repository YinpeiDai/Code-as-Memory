"""Offline RoboMemArena checks: pinned identities, interface, extraction and scoring.

No simulator is imported. The scoring tests drive the pinned official episode loops
and the CaM scorers with the same scripted stage events and compare outcomes.
"""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from cam.interface.robomemarena import adapter as rma
from cam.interface.robomemarena import data as rmd
from cam.interface.base import StepResult, check_frames, normalize_action, synthetic_value
from cam.interface.registry import get_adapter
from cam.data.demonstrations import random_select
from cam.io import sha256, write

try:
    SOURCE = rma.default_source()
except FileNotFoundError:
    SOURCE = None
pytestmark = pytest.mark.skipif(SOURCE is None, reason="third_party/robomemarena not initialized")
TASK = "T10_PourWineMugTwice"


@pytest.fixture(scope="module")
def adapter():
    return get_adapter("robomemarena")


def test_pinned_identities(adapter):
    release = json.loads((rma.METADATA / "release.json").read_text())
    assert release["revision"] == "93d765af20efa13f7b5190b0f35319a86f7ab202"
    assert adapter.tasks() == [rma.TASKS[i] for i in range(1, 27)]
    prompts = rma.task_prompts(adapter.source)
    for task in adapter.tasks():
        tid = rma.task_id(task)
        train, test = adapter.records(task, "train"), adapter.records(task, "test")
        assert [r["episode"] for r in train] == list(range(100))
        assert [r["seed"] for r in train] == sorted(int(s) for s in release["tasks"][task]["seeds"])
        assert [(r["episode"], r["seed"]) for r in test] == [(k, 50 + k) for k in range(50)]
        assert not {r["seed"] for r in train} & {r["seed"] for r in test}
        for split in ("train", "test"):
            meta = adapter.metadata(task, split)
            assert meta["bddl"] == rma.bddl_file(adapter.source, tid).relative_to(adapter.source).as_posix()
            assert meta["bddl_sha256"] == rma.file_sha256(adapter.source / meta["bddl"])
        for seed, rows in release["tasks"][task]["seeds"].items():
            assert [r["order"] for r in rows] == list(range(len(rows)))
            assert all(r["name"].endswith(f"_seed{seed}_task{tid}.hdf5") for r in rows)
        assert prompts[f"task{tid}"]


def test_interface_frames_and_actions(adapter):
    task = "T04_ButterNonEmptyDrawer"
    interface = adapter.interface(task, False)
    sensors = interface["sensors"]
    frame = {name: synthetic_value(spec, 1) for name, spec in sensors.items()}
    frame.update(phase="exec", step=0)
    check_frames([frame], sensors)
    for leak in ({"butter_1_pos": np.zeros(3)}, {"object-state": np.zeros(70)}):
        with pytest.raises(ValueError):
            check_frames([{**frame, **leak}], sensors)
    with pytest.raises(ValueError):
        check_frames([{**frame, "agentview_rgb": np.zeros((480, 640, 3), np.uint8)}], sensors)
    rgbd = adapter.interface(task, True)["sensors"]
    assert set(rgbd) - set(sensors) == {"agentview_depth", "wrist_depth"}
    assert rgbd["wrist_depth"]["dtype"] == "int16" and rgbd["wrist_depth"]["aligned_with"] == "wrist_rgb"
    depth_frame = {**frame, **{k: synthetic_value(rgbd[k], 3) for k in ("agentview_depth", "wrist_depth")}}
    check_frames([depth_frame], rgbd)
    with pytest.raises(ValueError):
        check_frames([frame], rgbd)
    with pytest.raises(ValueError):
        adapter.interface("BinFill", False)
    spec = interface["action"]
    assert spec["names"] == ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    assert normalize_action(spec, [2.0] * 7).tolist() == [1.0] * 7
    assert normalize_action(spec, [-2.0] * 7).tolist() == [-1.0] * 7
    assert interface["video"] == ["agentview_rgb", "wrist_rgb"]
    rows = [{"episode": e} for e in (7, 3, 5)]
    assert [r["episode"] for r in adapter.select_training(rows, 2)] == [3, 5]
    assert [r["episode"] for r in adapter.select_training(rows, None)] == [3, 5, 7]
    with pytest.raises(ValueError):
        adapter.select_training(rows, 4)


def test_depth_resize_is_centre_aligned():
    metres = np.zeros(rma.RENDER)
    metres[:] = np.arange(rma.RENDER[0])[::-1, None] + 1.0  # bottom-up render rows
    mm = rma.depth_millimetres(metres[..., None] / 1000.0)
    assert mm.dtype == np.int16 and mm.shape == rma.IMAGE[:2]
    # after flipud, row r holds r + 1 mm; destination row i samples floor((i + 0.5) * 480 / 256)
    assert np.array_equal(mm[:, 0], np.floor((np.arange(256) + 0.5) * 480 / 256).astype(int) + 1)
    bad = np.full(rma.RENDER, np.inf)
    bad[0, 0] = 40.0
    assert not rma.depth_millimetres(bad).any()


def test_calibration_conventions():
    K = rma.intrinsics(45.0)
    focal = 240 / np.tan(np.deg2rad(22.5))
    assert np.allclose(K, [[focal * 0.4, 0, 127.5], [0, focal * 256 / 480, 127.5], [0, 0, 1]])
    # MuJoCo camera at the origin with identity axes looks along world -z, y up.
    E = rma.extrinsic([0, 0, 0], np.eye(3))
    for point, expect in [((0, 0, -1), "centre"), ((0, 0.1, -1), "up"), ((0.1, 0, -1), "right")]:
        camera = E @ np.r_[point, 1.0]
        u, v, w = K @ camera
        assert camera[2] > 0
        u, v = u / w, v / w
        if expect == "centre":
            assert np.allclose((u, v), (127.5, 127.5))
        elif expect == "up":
            assert v < 127.5 and np.isclose(u, 127.5)
        else:
            assert u > 127.5 and np.isclose(v, 127.5)
    moved = rma.extrinsic([1.0, 2.0, 3.0], np.eye(3)) @ np.r_[1.0, 2.0, 2.0, 1.0]
    assert np.allclose(moved, [0, 0, 1])


def test_constructor_retries_only_placement_errors_without_changing_assignment():
    class PlacementError(Exception):
        pass

    kwargs = dict(bddl_file_name='official.bddl', camera_depths=True, **rma.ENV_KWARGS)
    calls = []
    result = object()
    def factory(**values):
        calls.append(values)
        if len(calls) < 3:
            raise PlacementError('Cannot place all objects')
        return result
    assert rma.construct_environment(factory, kwargs, PlacementError, attempts=3) is result
    assert calls == [kwargs, kwargs, kwargs]
    calls.clear()
    with pytest.raises(PlacementError):
        rma.construct_environment(factory, kwargs, PlacementError, attempts=2)
    assert len(calls) == 2
    def broken(**values):
        raise ValueError('unrelated failure')
    with pytest.raises(ValueError, match='unrelated failure'):
        rma.construct_environment(broken, kwargs, PlacementError)


# --------------------------------------------------------------------------- extraction


def _raw_file(path, start, length, rng, gap=0.0):
    """One synthetic subtask recording with the released raw layout."""
    joints = (start + np.arange(length))[:, None] * 0.01 + gap + np.zeros((1, 7))
    ee_pos = rng.normal(size=(length, 3))
    ee_ori = rng.normal(size=(length, 3))
    with h5py.File(path, "w") as h5:
        group = h5.create_group("data/demo_0")
        group.attrs.update(language_instruction="pick up the wine bottle", num_samples=length)
        group["actions"] = rng.uniform(-1, 1, (length, 7))
        group["dones"] = np.zeros(length, np.uint8)
        group["rewards"] = np.zeros(length, np.uint8)
        group["keyframe_indices"] = np.array([length - 1], np.uint8)
        obs = group.create_group("obs")
        obs["agentview_rgb"] = rng.integers(0, 255, (length, 256, 256, 3), dtype=np.uint8)
        obs["eye_in_hand_rgb"] = rng.integers(0, 255, (length, 256, 256, 3), dtype=np.uint8)
        obs["ee_pos"], obs["ee_ori"] = ee_pos, ee_ori
        obs["ee_states"] = np.concatenate([ee_pos, ee_ori], axis=1)
        obs["gripper_states"] = rng.normal(size=(length, 2))
        obs["joint_states"] = joints


@pytest.fixture
def synthetic(tmp_path, monkeypatch, adapter):
    """Four released identities of TASK with two subtask files each."""
    metadata, raw = tmp_path / "metadata", tmp_path / "raw"
    directory = "Multi-Counting/10_pour_wine_bottle_into_mug_dataset/subtask_data"
    (raw / "hf" / directory).mkdir(parents=True)
    rng = np.random.default_rng(0)
    seeds, listing = [100, 101, 103, 107], {}
    for seed in seeds:
        rows, start = [], 0
        for order, (name, length) in enumerate([("pick_wine_bottle", 3), ("pour_wine_into_mug_1st", 4)]):
            filename = f"{name}_{order}_seed{seed}_task10.hdf5"
            _raw_file(raw / "hf" / directory / filename, start, length, rng)
            start += length
            path = raw / "hf" / directory / filename
            rows.append({"name": filename, "order": order, "size": path.stat().st_size, "sha256": sha256(path)})
        listing[str(seed)] = rows
    pinned = adapter.metadata(TASK, "train")
    common = {k: pinned[k] for k in ("env_id", "task_id", "bddl", "bddl_sha256")}
    write(metadata / "train" / f"record_dataset_{TASK}_metadata.json",
          {**common, "records": [{"episode": i, "seed": s, "difficulty": None} for i, s in enumerate(seeds)]})
    write(metadata / "test" / f"record_dataset_{TASK}_metadata.json", {**common, **adapter.metadata(TASK, "test") | common})
    write(metadata / "release.json", {
        "repo": "RoboMemArenaBenchmark/RoboMemArena", "revision": "synthetic",
        "benchmark_source": {"commit": "synthetic"},
        "tasks": {TASK: {"task_id": 10, "directory": directory, "seeds": listing,
                         "bddl": common["bddl"], "bddl_sha256": common["bddl_sha256"]}},
    })
    monkeypatch.setattr(rma, "METADATA", metadata)
    monkeypatch.setattr(rmd, "METADATA", metadata)
    return SimpleNamespace(raw=raw, directory=raw / "hf" / directory, seeds=seeds, listing=listing, tmp=tmp_path)


class FakeReplayEnv:
    """Stands in for NativeEnvironment: declared sensors, success after ``need`` steps."""

    def __init__(self, adapter, task, depth, need):
        self.sensors, self.need = adapter.interface(task, depth)["sensors"], need
        self.commands = []

    def frame(self):
        row = {
            name: synthetic_value(spec, 7 if name.endswith("_K") else (self.seed + 3 * self.k) % 200)
            for name, spec in self.sensors.items()
        }
        row.update(phase="exec", step=self.k)
        return row

    def restart(self, seed):
        self.seed, self.k = int(seed), 0
        return [self.frame()]

    def step(self, command):
        self.k += 1
        self.commands.append(np.asarray(command))
        done = self.need.get(self.seed) == self.k
        metrics = {"tsr": done, "csr": 100.0 if done else 50.0, "failure_reason": None if done else "incomplete_stage"}
        return StepResult([self.frame()], "success" if done else "running", self.k, metrics=metrics)

    def close(self):
        pass


@pytest.fixture
def replaying(synthetic, monkeypatch, adapter):
    """Run the replay workers in-process against FakeReplayEnv (no simulator)."""
    need = {100: 5, 101: 6, 103: 4}  # seed 107 never succeeds: its actions run out

    def run(job, python, folder):
        from cam.interface.robomemarena.capture import capture_task
        env = lambda j: FakeReplayEnv(adapter, j["task"], j["depth"], need)
        return {"score": rmd.score_identities, "record": rmd.record_task,
                "capture": capture_task, "validate": rmd.score_identities}[job["mode"]](job, environment=env)

    import cam.harness.prepare as prepare

    monkeypatch.setattr(rmd, "run_worker", run)
    monkeypatch.setattr(rmd, "simulator_python", lambda adapter: Path("/simulator/python"))
    monkeypatch.setattr(prepare, "simulator_snapshot", lambda python: {"executable": str(python)})
    synthetic.need = need
    return synthetic


@pytest.mark.parametrize("depth", [True, False])
def test_replay_extraction_records_successful_online_frames(replaying, adapter, depth):
    out = replaying.tmp / f"out{depth}"
    manifest = rmd.extract(replaying.raw, out, "robomemarena", 2, [TASK], depth, True, 2)
    item = manifest["tasks"][TASK]
    records = adapter.records(TASK, "train")
    successful = [r for r in records if r["seed"] in replaying.need]
    expected = rmd.select(successful, 2, "robomemarena", TASK, rmd.release()["tasks"][TASK])
    assert [r["episode"] for r in item["episodes"]] == [r["episode"] for r in expected]
    assert {k: item["replay"][k] for k in ("released", "replayed", "successful", "selected")} == {
        "released": 4, "replayed": 4, "successful": 3, "selected": 2,
    }
    status = {r["seed"]: r["status"] for r in item["replay"]["identities"]}
    assert status == {100: "success", 101: "success", 103: "success", 107: "actions_exhausted"}
    assert manifest["depth"] is depth and manifest["demonstration_schema"] == "cam-demonstrations/2"
    rmd.validate_dataset(out)
    from cam.data.extract import validate_dataset

    validate_dataset(out)
    sensors = adapter.interface(TASK, depth)["sensors"]
    with h5py.File(out / item["file"], "r") as h5:
        assert set(h5) == {f"episode_{r['episode']}" for r in expected}
        text = str(dict(h5.attrs))
        assert not any(str(seed) in text for seed in replaying.seeds)
        assert "hdf5" not in text and "pick_wine_bottle" not in text
        for row in item["episodes"]:
            group = h5[f"episode_{row['episode']}"]
            assert set(group) == set(sensors) | {"step", "is_demo", "action"}
            assert ("agentview_depth" in group) is depth
            assert group["agentview_K"].shape == (3, 3) and row["frames"] == replaying.need[row["seed"]]
            assert len(group["step"]) == row["frames"] and not group["is_demo"][:].any()
            raw = [h5py.File(replaying.directory / f["name"], "r") for f in replaying.listing[str(row["seed"])]]
            released = np.concatenate([r["data/demo_0/actions"][:] for r in raw])
            for r in raw:
                r.close()
            assert np.array_equal(group["action"][:], np.clip(released[: row["frames"]], -1, 1))
            assert group.attrs["goal"] == rma.task_prompts(adapter.source)["task10"]
            assert "language_instruction" not in str(dict(group.attrs))
    for row in item["episodes"]:
        assert (out / row["video"]).stat().st_size > 0
    assert "pick_wine_bottle" not in json.dumps(item["episodes"])
    assert not (out / "work").exists()


def test_replay_pool_subset_keeps_success_provenance_and_depth(replaying):
    from cam.data.subset import subset

    pool = replaying.tmp / "pool"
    original = rmd.extract(replaying.raw, pool, "robomemarena", "all", [TASK], True, False, 2)
    output = replaying.tmp / "initial"
    result = subset(pool, output, 2)
    rmd.validate_dataset(output)
    item = result["tasks"][TASK]
    assert len(original["tasks"][TASK]["episodes"]) == 3 and len(item["episodes"]) == 2
    assert item["replay"]["selected"] == 2 and item["replay"]["successful"] == 3
    assert item["replay"]["identities"] == original["tasks"][TASK]["replay"]["identities"]
    with h5py.File(output / item["file"]) as h5:
        assert len(h5) == 2
        assert all("agentview_depth" in h5[key] for key in h5)


def test_replay_extraction_rejects_bad_release_files(replaying):
    with pytest.raises(ValueError, match="Only 3 of 4"):
        rmd.extract(replaying.raw, replaying.tmp / "many", "robomemarena", 4, [TASK], True, False, 1)
    name = replaying.listing["101"][1]["name"]
    path = replaying.directory / name
    backup = replaying.tmp / "backup.hdf5"
    shutil.copy2(path, backup)
    data = bytearray(path.read_bytes())  # same size, different content
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(Exception, match="sha256"):
        rmd.extract(replaying.raw, replaying.tmp / "tampered", "robomemarena", 2, [TASK], True, False, 1)
    shutil.copy2(backup, path)
    (replaying.directory / "extra_9_seed100_task10.hdf5").write_bytes(b"x")
    with pytest.raises(Exception, match="outside the pinned release"):
        rmd.extract(replaying.raw, replaying.tmp / "extra", "robomemarena", 2, [TASK], True, False, 1)


def test_reuse_copies_existing_rgbd_and_only_replays_missing_capture(replaying, monkeypatch):
    source = replaying.tmp / "already_recorded"
    original = rmd.extract(replaying.raw, source, "robomemarena", 2, [TASK], True, False, 1)
    before = sha256(source / original["tasks"][TASK]["file"])
    existing = {r["seed"] for r in original["tasks"][TASK]["episodes"]}
    seen = []
    restart = FakeReplayEnv.restart

    def tracked(self, seed):
        seen.append(seed)
        return restart(self, seed)

    monkeypatch.setattr(FakeReplayEnv, "restart", tracked)
    monkeypatch.setattr(rmd, "score_identities", lambda *a, **kw: pytest.fail("Verified outcomes must not be replayed"))
    target = replaying.tmp / "expanded"
    result = rmd.extract(replaying.raw, target, "robomemarena", "all", [TASK], True, False, 1, reuse=source)
    assert seen == sorted(set(replaying.need) - existing)
    item = result["tasks"][TASK]
    assert len(item["episodes"]) == 3 and len(item["reuse"]["episodes"]) == 2
    assert before == sha256(source / original["tasks"][TASK]["file"])
    with h5py.File(source / original["tasks"][TASK]["file"]) as src, h5py.File(target / item["file"]) as dst:
        for key in src:
            assert dict(src[key].attrs) == dict(dst[key].attrs)
            for name in src[key]:
                np.testing.assert_array_equal(src[key][name][:], dst[key][name][:])
    rmd.validate_dataset(target)


def test_reuse_rejects_changed_raw_files(replaying):
    source = replaying.tmp / "already_recorded"
    rmd.extract(replaying.raw, source, "robomemarena", 2, [TASK], True, False, 1)
    raw = replaying.directory / replaying.listing["100"][0]["name"]
    import os
    stat = raw.stat()
    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="Raw data changed"):
        rmd.extract(replaying.raw, replaying.tmp / "reject_changed", "robomemarena", "all", [TASK], True, False, 1, reuse=source)


# --------------------------------------------------------------------------- scoring


def test_success_cap_stops_early_and_reports_unchecked_identities(replaying, monkeypatch):
    original = rmd.run_worker
    checked = []
    def track(job, python, folder):
        if job['mode'] == 'score':
            checked.extend(r['seed'] for r in job['identities'])
        return original(job, python, folder)
    monkeypatch.setattr(rmd, 'run_worker', track)
    out = replaying.tmp / 'capped'
    value = rmd.extract(replaying.raw, out, count='all', tasks=[TASK], videos=False, max_successful=2)
    assert checked == [100, 101]
    assert value['tasks'][TASK]['replay']['released'] == 4
    assert value['tasks'][TASK]['replay']['replayed'] == 2
    assert len(value['tasks'][TASK]['episodes']) == 2
    rmd.validate_dataset(out)


def test_success_cap_reports_release_shortfall_without_padding(replaying):
    value = rmd.extract(replaying.raw, replaying.tmp / 'shortfall', count='all', tasks=[TASK], videos=False, max_successful=5)
    row = value['tasks'][TASK]
    assert row['replay']['replayed'] == 4 and row['replay']['successful'] == 3
    assert len(row['episodes']) == 3


def test_capped_pool_preserves_every_existing_capture(replaying, monkeypatch):
    source = replaying.tmp / 'existing'
    old = rmd.extract(replaying.raw, source, count=2, tasks=[TASK], videos=False)
    monkeypatch.setattr(FakeReplayEnv, 'restart', lambda *a: pytest.fail('Existing captures must not be replayed'))
    new = rmd.extract(replaying.raw, replaying.tmp / 'retained', count='all', tasks=[TASK], videos=False,
                      max_successful=2, reuse=source)
    assert new['tasks'][TASK]['episodes'] == old['tasks'][TASK]['episodes']


def test_capped_resume_keeps_completed_checks(replaying, monkeypatch):
    original = rmd.run_worker
    monkeypatch.setattr(rmd, 'IDENTITIES_PER_WORKER', 1)
    def interrupt(job, python, folder):
        if job['mode'] == 'score' and job['identities'][0]['seed'] == 101:
            raise RuntimeError('test interruption')
        return original(job, python, folder)
    monkeypatch.setattr(rmd, 'run_worker', interrupt)
    out = replaying.tmp / 'resumed'
    with pytest.raises(RuntimeError, match='test interruption'):
        rmd.extract(replaying.raw, out, count='all', tasks=[TASK], videos=False, max_successful=3)
    checked = []
    def resumed(job, python, folder):
        if job['mode'] == 'score':
            checked.extend(r['seed'] for r in job['identities'])
        return original(job, python, folder)
    monkeypatch.setattr(rmd, 'run_worker', resumed)
    new = rmd.extract(replaying.raw, out, count='all', tasks=[TASK], videos=False, max_successful=3, resume=True)
    assert checked == [101, 103]
    assert len(new['tasks'][TASK]['episodes']) == 3


def test_inherited_results_are_reused_before_new_checks(replaying, monkeypatch, adapter):
    from cam.interface.robomemarena import replay_cache as cache
    records = adapter.records(TASK, 'train')[:2]
    job = {'mode': 'score', 'task': TASK, 'depth': False, 'root': str(replaying.raw / 'hf'), 'identities': records}
    value = rmd.run_worker(job, None, None)
    path, result, dispatch = [replaying.tmp / name for name in ('old.job.json', 'old.result.json', 'dispatch.json')]
    write(path, job); write(result, value); write(dispatch, {'finished': 1, 'returncode': 0})
    entry = {'job': str(path), 'job_sha256': sha256(path), 'result': str(result),
             'result_sha256': sha256(result), 'dispatch': str(dispatch), 'identities': records,
             'producer': None, 'wrapper': None}
    monkeypatch.setattr(cache, 'open_cache', lambda *a: ({TASK: [entry]}, {'source_plan_sha256': 'test'}))
    monkeypatch.setattr(rmd, 'score_identities', lambda *a, **kw: pytest.fail('Inherited checks must not be repeated'))
    out = replaying.tmp / 'adopted'
    new = rmd.extract(replaying.raw, out, count='all', tasks=[TASK], videos=False, max_successful=2, replay_cache=replaying.tmp)
    assert len(new['tasks'][TASK]['episodes']) == 2
    with pytest.raises(ValueError, match='unassigned'):
        cache.collect([entry], [{'episode': 0, 'seed': 999}])
    write(result, {'results': []})
    with pytest.raises(ValueError, match='result changed'):
        cache.collect([entry], records)


def test_single_pass_captures_once_and_only_replays_validation_sample(replaying, monkeypatch):
    monkeypatch.setattr(rmd, 'CHUNK', 2)  # Exercise multiple streaming writes before success.
    seen, jobs = [], []
    original_restart, original_worker = FakeReplayEnv.restart, rmd.run_worker
    def restart(self, seed):
        seen.append(seed)
        return original_restart(self, seed)
    def worker(job, *args):
        jobs.append(job)
        assert job['mode'] in ('capture', 'validate')
        return original_worker(job, *args)
    monkeypatch.setattr(FakeReplayEnv, 'restart', restart)
    monkeypatch.setattr(rmd, 'run_worker', worker)
    out = replaying.tmp / 'single'
    result = rmd.extract(replaying.raw, out, count='all', tasks=[TASK], videos=False,
                         max_successful=2, single_pass=True, validation_episodes=1)
    assert seen[:2] == [100, 101] and len(seen) == 3
    assert seen[2] in (100, 101)
    assert [j['mode'] for j in jobs] == ['capture', 'validate']
    item = result['tasks'][TASK]
    assert len(item['episodes']) == 2 and item['validation']['checked'] == 1
    assert len(list((out/'captures'/TASK).glob('episode_*.json'))) == 2
    with h5py.File(out/item['file']) as h5:
        for row in item['episodes']:
            g = h5[f"episode_{row['episode']}"]
            assert len(g['step']) == replaying.need[row['seed']]
            assert g['agentview_depth'].shape == (row['frames'], 256, 256)
            raw = rmd.load_recording(replaying.raw/'hf', rmd.release()['tasks'][TASK], row['seed'], images=False)
            np.testing.assert_array_equal(g['action'][:], np.clip(raw['actions'][:row['frames']], -1, 1))
    rmd.validate_dataset(out)


def test_single_pass_keeps_failed_evidence_out_of_demo_pool(replaying):
    out = replaying.tmp / 'single_shortfall'
    result = rmd.extract(replaying.raw, out, count='all', tasks=[TASK], videos=False,
                         max_successful=5, single_pass=True)
    assert len(result['tasks'][TASK]['episodes']) == 3
    receipt = json.loads((out/'captures'/TASK/'episode_003.json').read_text())
    assert receipt['status'] == 'actions_exhausted' and not receipt['tsr']
    assert (out/'captures'/TASK/receipt['capture_file']).exists()
    with h5py.File(out/result['tasks'][TASK]['file']) as h5:
        assert 'episode_3' not in h5
    rmd.validate_dataset(out)


def test_single_pass_reuses_rgbd_and_skips_known_failures(replaying, monkeypatch):
    source = replaying.tmp/'source_rgbd'
    old = rmd.extract(replaying.raw, source, count=2, tasks=[TASK], videos=False)
    existing = {r['seed'] for r in old['tasks'][TASK]['episodes']}
    seen, restart = [], FakeReplayEnv.restart
    def tracked(self, seed):
        seen.append(seed)
        return restart(self, seed)
    monkeypatch.setattr(FakeReplayEnv, 'restart', tracked)
    out = replaying.tmp/'combined_reuse'
    result = rmd.extract(replaying.raw, out, count='all', tasks=[TASK], videos=False,
                         max_successful=5, single_pass=True, reuse=source)
    assert set(seen) == set(replaying.need)-existing
    assert len(result['tasks'][TASK]['episodes']) == 3
    with h5py.File(source/old['tasks'][TASK]['file']) as a, h5py.File(out/result['tasks'][TASK]['file']) as b:
        for ep in a:
            for key in a[ep]:
                np.testing.assert_array_equal(a[ep][key][:], b[ep][key][:])


def test_single_pass_resume_does_not_repeat_completed_captures(replaying, monkeypatch):
    seen, restart = [], FakeReplayEnv.restart
    interrupted = False
    def tracked(self, seed):
        nonlocal interrupted
        seen.append(seed)
        if seed == 101 and not interrupted:
            interrupted = True
            raise RuntimeError('simulated interruption')
        return restart(self, seed)
    monkeypatch.setattr(FakeReplayEnv, 'restart', tracked)
    out = replaying.tmp/'capture_resume'
    args = dict(count='all', tasks=[TASK], videos=False, max_successful=3, single_pass=True)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        rmd.extract(replaying.raw, out, **args)
    result = rmd.extract(replaying.raw, out, resume=True, **args)
    assert seen == [100, 101, 101, 103]
    assert len(result['tasks'][TASK]['episodes']) == 3
    assert list((out/'captures'/TASK/'interrupted').glob('*/*.h5.partial'))


def test_single_pass_validation_failure_blocks_completion(replaying, monkeypatch):
    worker = rmd.run_worker
    def fail_validation(job, *args):
        result = worker(job, *args)
        if job['mode'] == 'validate':
            result['results'][0].update(status='actions_exhausted', tsr=False)
        return result
    monkeypatch.setattr(rmd, 'run_worker', fail_validation)
    out = replaying.tmp/'validation_fail'
    with pytest.raises(RuntimeError, match='Sampled replay validation failed'):
        rmd.extract(replaying.raw, out, count='all', tasks=[TASK], videos=False,
                    max_successful=2, single_pass=True, validation_episodes=1)
    assert not (out/'manifest.json').exists()
    assert (out/'audit'/f'{TASK}.validation.json').exists()
    assert len(list((out/'captures'/TASK).glob('episode_*.json'))) == 2


def test_capture_cache_rejects_wrong_identities_and_changed_raw_audit(replaying, adapter):
    from cam.interface.robomemarena.capture import historical_outcomes
    job = replaying.tmp/'cache_job'
    pool = job/'pool'
    write(job/'plan.json', {'pool': str(pool)})
    stats = {'pinned': 1}
    records = adapter.records(TASK, 'train')
    row = {'episode': 0, 'seed': records[0]['seed'], 'status': 'success'}
    path = pool/'audit'/f'{TASK}.replay.json'
    write(path, {'stats': stats, 'results': [row]})
    args = ({'replay_cache': {'source_job': str(job)}}, TASK, records, {}, {}, stats)
    assert historical_outcomes(*args)[0] == {0: row}
    write(path, {'stats': stats, 'results': [{**row, 'seed': 9999}]})
    with pytest.raises(ValueError, match='unassigned'):
        historical_outcomes(*args)
    write(path, {'stats': {'changed': 1}, 'results': [row]})
    with pytest.raises(ValueError, match='raw-file audit changed'):
        historical_outcomes(*args)


def test_independent_validation_selects_five_distinct_reproducible_identities():
    from cam.interface.robomemarena.capture import validation_sample
    episodes = [{'episode': ep, 'seed': 100 + ep} for ep in range(25)]
    sample = validation_sample(episodes, 5, TASK)
    assert len(sample) == len({r['episode'] for r in sample}) == 5
    assert sample == validation_sample(list(reversed(episodes)), 5, TASK)


def test_single_pass_rejects_modified_unchecked_release(replaying):
    path = replaying.directory / replaying.listing['100'][0]['name']
    with h5py.File(path, 'r+') as h5:
        h5['data/demo_0/actions'][0, 0] += 0.01
    with pytest.raises(ValueError, match='sha256'):
        rmd.extract(replaying.raw, replaying.tmp/'bad_single', count='all', tasks=[TASK],
                    videos=False, max_successful=2, single_pass=True)


@pytest.fixture(scope="module")
def official():
    """Import the pinned official evaluation scripts without writing bytecode."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for path in (SOURCE / "libero_fork", SOURCE / "scripts"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import eval_common
        import eval_task1_only
        import eval_tasks2_26
        import task2_26_reference_stage
    finally:
        sys.dont_write_bytecode = previous
    return SimpleNamespace(ec=eval_common, t1=eval_task1_only, t26=eval_tasks2_26, rs=task2_26_reference_stage)


class FakeEnv:
    """Counts native steps; scripted checks read the policy-step counter."""

    def __init__(self):
        self.k = 0

    @property
    def policy_step(self):
        return self.k - rma.SETTLE_STEPS

    def obs(self):
        image = np.zeros((8, 8, 3), np.uint8)
        return {"agentview_image": image, "robot0_eye_in_hand_image": image, "robot0_eef_quat": np.array([1.0, 0, 0, 0])}

    def reset(self):
        self.k = 0
        return self.obs()

    def step(self, action):
        self.k += 1
        return self.obs(), 0.0, False, {}


def _adapter(official):
    class Zero(official.ec.BasePolicyAdapter):
        def infer_actions(self, obs, prompt, resize_size):
            return np.zeros((10, 7), np.float32)

    return Zero()


def _drive(scorer_class, modules, tid, specs_env):
    env = specs_env
    env.reset()
    for _ in range(rma.SETTLE_STEPS):
        env.step(None)
    scorer = scorer_class(tid, modules, env)
    steps = 0
    while True:
        steps += 1
        obs, _, done, _ = env.step(None)
        if scorer.update(env, obs, rma.SETTLE_STEPS + steps - 1, done) or steps >= rma.MAX_STEPS:
            return scorer.result(), steps


def _patch_stages(monkeypatch, official, tid, at, extra=None):
    real = official.rs._task_specs(tid)
    assert len(real) == len(at)

    def specs(task_id):
        return [
            official.rs.StageSpec(spec.name, lambda env, state, start, a=a: a is not None and env.policy_step >= a)
            for spec, a in zip(real, at)
        ]

    def extra_check(task_id):
        if not official.rs._is_counting_pour_task(task_id):
            return None
        return lambda env, state, start: extra is not None and env.policy_step >= extra

    initial = lambda env: {"step_idx": 0, "tilt_angles": [], "shared_pour_counters": {}, "last_obs": None}
    for module in (official.rs, official.t26):
        monkeypatch.setattr(module, "_task_specs", specs)
        monkeypatch.setattr(module, "_extra_pour_check", extra_check)
        monkeypatch.setattr(module, "_build_initial_state", initial)


@pytest.mark.parametrize(
    "tid, at, extra, status, cam_steps",
    [
        (4, [5, 10, 15, 20, 25, 30, 35, 40, None], None, "success", 40),
        (4, [5, 10, 15, 20, 25, 30, 35, 40, 45], None, "success", 40),
        (4, [5, 10, None, None, None, None, None, None, None], None, "timeout", 2500),
        (10, [5, 20, 40], None, "success", 70),
        (10, [5, 20, 40], 50, "fail", 50),
        (10, [5, 20, 40], 80, "success", 70),
        (10, [5, 20, None], None, "timeout", 2500),
        (22, [5, 20, 40, 60, 80, 100, None], None, "success", 100),
        (22, [5, 20, 40, 45, 50, 55, None], None, "success", 70),
        (22, [5, 20, 40, 60, 80, 100, None], 90, "fail", 90),
        (22, [5, 20, 40, None, None, None, None], 300, "fail", 300),
    ],
)
def test_stateful_scorer_matches_official_loop(monkeypatch, official, tid, at, extra, status, cam_steps):
    _patch_stages(monkeypatch, official, tid, at, extra)
    env = FakeEnv()
    score, done, tsr, diagnostics, _, _ = official.t26.run_episode_with_stateful_stages(
        tid, env, _adapter(official), "goal", 256, 10, rma.SETTLE_STEPS, rma.MAX_STEPS,
        rma.POST_GOAL_STEPS, official.t26._task_specs(tid), {}, None, True, rma.EXTRA_POUR_MONITOR_STEPS,
    )
    official_steps = env.policy_step
    result, steps = _drive(rma.StatefulScorer, {"task2_26_reference_stage": official.rs}, tid, FakeEnv())
    counted = official.rs._counted_stage_names(tid, done)
    assert result["tsr"] == tsr == (status == "success")
    assert result["csr"] == score
    assert {k: result["stages"][k] for k in counted} == {k: done[k] for k in counted}
    assert result["failure_reason"] == diagnostics["failure_reason"]
    assert steps == cam_steps
    if official.rs._is_counting_pour_task(tid) or status == "timeout":
        assert official_steps == steps
    else:  # official runner continues POST_GOAL_STEPS after the latched success
        assert official_steps == steps + rma.POST_GOAL_STEPS


@pytest.mark.parametrize("a, b, cam_steps, tsr", [(5, 9, 9, True), (5, 3, 5, True), (None, 3, 2500, False)])
def test_task1_scorer_matches_official_loop(official, a, b, cam_steps, tsr):
    checks = [
        ("01_Place_Cookies_Basket", lambda env: a is not None and env.policy_step >= a),
        ("02_Place_Tomato_Basket", lambda env: env.policy_step >= b),
    ]
    env = FakeEnv()
    score, done, goal, _, _ = official.ec.run_episode_with_stages(
        env, _adapter(official), "goal", 256, 10, rma.SETTLE_STEPS, rma.MAX_STEPS, rma.POST_GOAL_STEPS, checks, {}, True,
    )
    official_steps = env.policy_step
    modules = {"eval_task1_only": SimpleNamespace(STAGE_CHECKS=checks)}
    result, steps = _drive(rma.Task1Scorer, modules, 1, FakeEnv())
    assert result["tsr"] == (bool(done) and all(done.values())) == tsr
    assert result["csr"] == score and result["stages"] == done and steps == cam_steps
    assert official_steps == (steps + rma.POST_GOAL_STEPS if tsr else steps)


def test_official_task1_checks_are_stateless(official):
    assert [name for name, _ in official.t1.STAGE_CHECKS] == [s.name for s in official.rs._task_specs(1)]
    assert official.rs._stage_success_from_stage_done(4, {"a": True, "09_Close_Top_Drawer_Final": False})


def test_selection_balances_scene_variants(adapter):
    info = rmd.release()
    for task in adapter.tasks():
        records = adapter.records(task, "train")
        selected = rmd.select(records, 25, "robomemarena", task, info["tasks"][task])
        counts = sorted(row["selected"] for row in rmd.strata(records, selected, info["tasks"][task]))
        if task == "T04_ButterNonEmptyDrawer":
            assert counts == [8, 8, 9]
        elif task == "T05_ButterEmptyDrawer":
            assert counts == [5, 20]
        else:  # one variant: exactly the plain uniform selection
            assert counts == [25]
            assert selected == random_select(records, 25, "robomemarena", task)


@pytest.mark.parametrize('failed_validation', [False, True])
def test_local_assembly_reuses_capture_and_waits_for_validation(replaying, adapter, monkeypatch, failed_validation):
    from cam.interface.robomemarena import local as local
    from cam.io import read

    original = rmd.run_worker
    calls = []
    def record(job, python, folder):
        result = original(job, python, folder)
        calls.append((job, result))
        return result
    monkeypatch.setattr(rmd, 'run_worker', record)
    source, destination = replaying.tmp/'capture_pool', replaying.tmp/'local_pool'
    manifest = rmd.extract(replaying.raw, source, 'robomemarena', 'all', [TASK], True, False, 1,
                           max_successful=3, single_pass=True, validation_episodes=1)
    capture_job, capture_result = next((j, r) for j, r in calls if j['mode'] == 'capture')
    before = {p: sha256(p) for p in source.rglob('*.h5')}
    validation_path = source/'audit'/f'{TASK}.validation.json'
    if failed_validation:
        value = read(validation_path)
        value['results'][0].update(status='failure', tsr=False)
        write(validation_path, value)
    def adopt(job, python, folder):
        assert job == capture_job  # original assignment; no duplicate capture or validation
        return capture_result
    monkeypatch.setattr(rmd, 'run_worker', adopt)
    stage = local.stage_file
    def checked_stage(*args):
        assert (destination/'capture_validation_complete.json').exists()
        assert read(destination/'audit'/f'{TASK}.validation.json')['passed']
        return stage(*args)
    monkeypatch.setattr(local, 'stage_file', checked_stage)
    # A scheduler snapshot has an explicit benchmark path but no checkout-local
    # third_party directory. Validation must not fall back to that directory.
    def missing_default():
        raise FileNotFoundError('No implicit benchmark in the scheduler snapshot')
    monkeypatch.setattr(rma, 'default_source', missing_default)
    options = dict(adapter=adapter, root=replaying.raw/'hf', output=source, destination=destination,
                   manifest=manifest, scored={}, stats={TASK: read(source/'audit'/f'{TASK}.json')['stats']},
                   cache_jobs={}, reusable=None, reuse=None, python='/unused', workers=1, unchanged=lambda _: None)
    if failed_validation:
        with pytest.raises(RuntimeError, match='Sampled replay validation failed'):
            local.capture_and_assemble_local(**options)
        assert not (destination/'manifest.json').exists()
        assert not list(destination.rglob('*.h5'))
    else:
        result = local.capture_and_assemble_local(**options)
        assert len(result['tasks'][TASK]['episodes']) == 3
        assert result['tasks'][TASK]['validation']['checked'] == 1
        assert rmd.validate_dataset(destination, benchmark_source=adapter.source)['assembly']['storage'] == 'local'
        assert not list((destination/'.staging').rglob('*.h5'))
        written = {p: sha256(p) for p in destination.rglob('*.h5')}
        saved_manifest = (destination/'manifest.json').read_bytes()
        (destination/'manifest.partial.json').write_bytes(saved_manifest)
        (destination/'manifest.json').rename(destination/'manifest.previous.json')
        monkeypatch.setattr(rmd, 'run_worker', lambda *a, **kw: pytest.fail('Final validation cannot replay'))
        monkeypatch.setattr(local, 'stage_file', lambda *a, **kw: pytest.fail('Final validation cannot reassemble'))
        local_validation = destination/'audit'/f'{TASK}.validation.json'
        saved_validation = local_validation.read_bytes()
        value = read(local_validation)
        value['results'][0]['tsr'] = False
        write(local_validation, value)
        with pytest.raises(ValueError, match='Sampled replay validation changed'):
            local.finalize_assembled(destination, adapter, [TASK], 3, 1)
        assert not (destination/'manifest.json').exists()
        local_validation.write_bytes(saved_validation)
        local.finalize_assembled(destination, adapter, [TASK], 3, 1)
        assert (destination/'manifest.json').read_bytes() == saved_manifest
        assert not (destination/'manifest.partial.json').exists()
        assert {p: sha256(p) for p in written} == written
    assert {p: sha256(p) for p in before} == before


def test_local_staging_rejects_corruption(tmp_path):
    from cam.interface.robomemarena.local import stage_file
    source, dest = tmp_path/'source.h5', tmp_path/'staged.h5'
    source.write_bytes(b'capture')
    with pytest.raises(ValueError, match='hash does not match'):
        stage_file(source, 'bad', dest)
    assert not dest.exists() and source.read_bytes() == b'capture'
    assert stage_file(source, sha256(source), dest).read_bytes() == b'capture'
    dest.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Staged source file changed'):
        stage_file(source, sha256(source), dest)
