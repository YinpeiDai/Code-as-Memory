"""MIKASA-Robo interface, released demonstrations and the dry-run API (no simulator)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

import cam.interface.mikasa_robo.adapter as mikasa
import cam.interface.mikasa_robo.api.kinematics as kin
import cam.interface.mikasa_robo.data as data
from cam.interface.base import check_frames
from cam.interface.registry import get_adapter
from cam.config import RunConfig
from cam.prompts import render
from cam.runtime.robot import MotionError, Robot
from cam.runtime.wire import receive, send

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/mikasa_robo"
TASK = "TakeItBack-VLA-v0"


def test_physics_backend_is_explicit_and_mikasa_only():
    default = RunConfig(benchmark="mikasa_robo", mode="demos")
    assert default.physics_backend is None
    assert mikasa.MAKE_KWARGS_RGBD["sim_backend"] == "cpu"
    for backend in ("cpu", "gpu"):
        cfg = RunConfig(benchmark="mikasa_robo", mode="demos", physics_backend=backend)
        assert RunConfig.from_dict(cfg.to_dict()).physics_backend == backend
    for benchmark, backend in (("robomme", "cpu"), ("mikasa_robo", "auto")):
        with pytest.raises(ValueError, match="physics_backend"):
            RunConfig(benchmark=benchmark, mode="demos", physics_backend=backend)


@pytest.fixture
def adapter():
    if not (SOURCE / "mikasa_robo_suite/vla").is_dir():
        pytest.skip("MIKASA-Robo submodule missing")
    return get_adapter("mikasa_robo")


@pytest.mark.parametrize("requested", [None, "cpu", "gpu"])
def test_task_physics_routing_and_prompt_agree_for_all_ninety_tasks(adapter, requested):
    cfg = RunConfig(benchmark="mikasa_robo", mode="demos", physics_backend=requested)
    tasks = adapter.tasks()
    assert len(tasks) == 90 and "ShellGameTouch-VLA-v0" in tasks
    for task in tasks:
        expected = "gpu" if task == "ShellGameTouch-VLA-v0" else requested or "cpu"
        assert mikasa.physics_backend_for_task(task, requested) == expected
        assert f"Physics backend: {expected} for both training and held-out execution" in render(
            cfg, task, [], adapter=adapter
        )
    with pytest.raises(ValueError, match="physics_backend"):
        mikasa.physics_backend_for_task("ShellGameTouch-VLA-v0", "auto")


def _raw(index_shift=0):
    rgb = np.random.default_rng(0).integers(0, 255, (1, 256, 256, 3), dtype=np.uint8)
    wrist = np.random.default_rng(1).integers(0, 255, (1, 256, 256, 3), dtype=np.uint8)
    camera = lambda image: {"rgb": image, "depth": np.full((1, 256, 256, 1), 500, np.int16)}
    param = {"intrinsic_cv": np.eye(3)[None].astype(np.float32), "extrinsic_cv": np.zeros((1, 3, 4), np.float32)}
    raw = {
        "agent": {"qpos": np.arange(9, dtype=np.float32)[None] / 10, "qvel": np.zeros((1, 9), np.float32)},
        "sensor_data": {"base_camera": camera(rgb), "hand_camera": camera(wrist)},
        "sensor_param": {"base_camera": param, "hand_camera": param},
    }
    obs = {"rgb": np.concatenate([rgb, wrist], -1), "proprio": np.arange(7, dtype=np.float32)[None]}
    return raw, obs


def test_interface_matches_robomme_style_and_frames_are_allowlisted(adapter):
    interface = adapter.interface(TASK, True)
    sensors = interface["sensors"]
    assert set(sensors) == {
        f"{c}_{k}" for c in ("front", "wrist") for k in ("rgb", "depth", "K", "extrinsic")
    } | {"tcp", "joints", "fingers"}
    assert sensors["front_rgb"]["shape"] == [256, 256, 3] and sensors["front_depth"]["dtype"] == "int16"
    assert "J4 [-3.0718, -0.0698]" in sensors["joints"]["description"]
    assert interface["video"] == ["front_rgb", "wrist_rgb"]
    with pytest.raises(ValueError, match="requires depth=true"):
        adapter.interface(TASK, False)
    raw, obs = _raw()
    row = mikasa.frame(raw, obs, 3)
    check_frames([row], sensors)
    assert row["joints"].tolist() == pytest.approx(np.arange(7) / 10) and row["fingers"].shape == (2,)
    obs["rgb"] = obs["rgb"].copy()
    obs["rgb"][0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="disagree"):
        mikasa.frame(raw, obs, 3)
    with pytest.raises(ValueError):
        mikasa.frame(raw, {**obs, "oracle_info": np.zeros(1)}, 3)


@pytest.mark.parametrize("split", ["train", "test"])
def test_both_splits_reject_rgb_only_before_simulator_creation(adapter, split):
    with pytest.raises(ValueError, match="requires depth=true"):
        adapter.make_environment("unused", TASK, 0, split, False)


@pytest.mark.parametrize("camera", ["base_camera", "hand_camera"])
@pytest.mark.parametrize("field", ["rgb", "depth"])
def test_wrong_resolution_is_rejected_instead_of_resized(camera, field):
    raw, obs = _raw()
    raw["sensor_data"][camera][field] = raw["sensor_data"][camera][field][:, :128, :128]
    with pytest.raises(ValueError, match="Expected.*256,256"):
        mikasa.frame(raw, obs, 0)


def test_reset_and_every_step_preserve_both_rgbd_cameras(adapter):
    from types import SimpleNamespace

    raw, obs = _raw()
    for camera in ("base_camera", "hand_camera"):
        raw["sensor_data"][camera]["depth"][0, 4, 7, 0] = 1234

    class Env:
        recorder = SimpleNamespace(last=raw)
        steps, horizon = 0, 60

        def reset(self):
            return obs, {"language_instruction": ["Interface test"]}

        def step(self, action):
            self.steps += 1
            return obs, False, False, False

    env = Env()
    start = adapter.begin(env, builtin_demo=False, depth=True)
    for step in range(3):
        row = start.frames[0] if step == 0 else adapter.advance(env, np.zeros(7), {}, depth=True).frames[0]
        check_frames([row], adapter.interface(TASK, True)["sensors"])
        assert row["step"] == step
        for prefix, camera in mikasa.CAMERAS.items():
            assert np.array_equal(row[prefix + "_rgb"], raw["sensor_data"][camera]["rgb"][0])
            assert np.array_equal(row[prefix + "_depth"], raw["sensor_data"][camera]["depth"][0, ..., 0])
            assert row[prefix + "_depth"][4, 7] == 1234
    with pytest.raises(ValueError, match="requires depth=true"):
        adapter.advance(env, np.zeros(7), {}, depth=False)
    assert env.steps == 2


def test_rotation_helpers_match_tcp_convention():
    roll, pitch, yaw = 0.3, -0.4, 1.2
    R = kin.rpy_matrix(roll, pitch, yaw)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.isclose(np.arctan2(R[2, 1], R[2, 2]), roll) and np.isclose(np.arcsin(-R[2, 0]), pitch)
    axis = np.array([0.2, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    for angle in (0.0, 0.7, 3.0):
        Rk = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        assert np.allclose(kin.rotation_vector(Rk), axis * angle, atol=1e-6)
    assert np.allclose(kin.quat_matrix([1, 0, 0, 0]), np.eye(3))




def _robot(reply):
    reader, writer = io.BytesIO(), io.BytesIO()
    send(reader, reply)
    reader.seek(0)
    return Robot({}, reader, writer, max_steps=1), writer


def test_dry_run_is_a_query_with_validation_and_limits():
    robot, writer = _robot({"dry_run": {"reachable": True}})
    assert robot.dry_run(0.1, 0, 0.2, np.pi, 0, 0) == {"reachable": True}
    writer.seek(0)
    message = receive(writer)
    assert message == {"kind": "dry_run", "pose": [0.1, 0.0, 0.2, float(np.pi), 0.0, 0.0]}
    assert robot.frames() == []  # nothing stepped
    with pytest.raises(MotionError, match="limit"):
        robot.dry_run(x=0, y=0, z=0, roll=0, pitch=0, yaw=0)
    for bad in ((0, 0, 0), (0, 0, 0, 0, 0, float("nan")), ("a", 0, 0, 0, 0, 0)):
        with pytest.raises(MotionError):
            _robot({"dry_run": {}})[0].dry_run(*bad)
    robot, _ = _robot({"error": "robot.dry_run is not available for this benchmark"})
    with pytest.raises(MotionError, match="not available"):
        robot.dry_run(0, 0, 0, 0, 0, 0)


def test_demonstration_declaration_and_no_public_collector():
    frame = data.DECLARED["frame"]
    assert {"front_rgb", "front_depth", "joints", "fingers", "action", "step", "is_demo"} <= set(frame)
    assert set(data.DECLARED["static"]) == {"front_K", "wrist_K"}
    with pytest.raises(ValueError):
        data.extract("unused", "unused", depth=False)


def test_configs_and_prompt(adapter):
    config = RunConfig.from_dict(json.loads((ROOT / "configs/mikasa_robo/both_claude_code.json").read_text()))
    assert (config.benchmark, config.budget, config.training_seed_count, config.training_seed_policy, config.depth) == (
        "mikasa_robo", 50, 50, "demo_cycle", True
    )
    episodes = [{"episode": 3, "seed": 3}]
    prompt = render(config, TASK, episodes, [{"episode": 7}], adapter=adapter, interface=adapter.interface(TASK, True))
    assert "robot.dry_run(x, y, z, roll, pitch, yaw)" in prompt and "Approaching MIKASA-Robo tasks" in prompt
    assert "Native step limit for this task: 60" in prompt
    shared = (ROOT / "src/cam/prompts/core.md").read_text()
    assert shared in prompt  # the shared core prompt is unchanged
    for forbidden in ("0.05 m", "0.2 rad/s", "0.085", "success_tests"):
        assert forbidden not in (ROOT / "src/cam/interface/mikasa_robo/guide.md").read_text()
