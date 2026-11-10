"""MIKASA-Robo: the official 90-task MIKASA-Robo-VLA protocol with a RoboMME-style
observation interface.

By default, environment, wrappers, control mode, seeds, step limits and scoring are the
shared native protocol (``mikasa_common.py``: ``mikasa_robo_suite.vla.benchmarking`` settings and
``apply_mikasa_vla_wrappers(include_overlays=False)``). Only the sensors differ:
``obs_mode="rgbd"`` with both cameras rendered at 256x256 (ManiSkill
``sensor_configs``), and the raw observation is read beneath the canonical wrappers
(which drop depth, calibration and joints) by a pass-through recorder. Rendering
settings do not touch physics. An explicit physics_backend="cpu" uses native CPU
physics and CPU IK for a separate comparison; it is not the official GPU protocol.

Policy frames carry RoboMME's names: ``front_*`` (MIKASA ``base_camera``) and
``wrist_*`` (``hand_camera``) RGB, int16 optical-axis depth in millimetres, 3x3
intrinsics, 3x4 world-to-camera extrinsics, plus TCP pose, 7 arm joints and 2
finger positions. ``preview`` evaluates the controller's own delta-IK joint target
for a command from the measured state without stepping (joint-limit dry-run).
"""

import numpy as np

from cam.interface.base import sensor
from cam.interface.mikasa_robo.api.kinematics import (
    ArmKinematics, JOINT_NAMES, JOINT_LOWER, JOINT_UPPER, limits_text,
)
from cam.interface.mikasa_robo.common import MAKE_KWARGS, MikasaProtocol, NativeEnvironment, to_numpy, METADATA, dataset_name, default_assets

IMAGE = (256, 256, 3)
CAMERAS = {"front": "base_camera", "wrist": "hand_camera"}
MAKE_KWARGS_RGBD = {
    **MAKE_KWARGS,
    "obs_mode": "rgbd",
    "sensor_configs": {"width": IMAGE[1], "height": IMAGE[0]},
}


def require_rgbd(depth):
    if not depth:
        raise ValueError("MIKASA-Robo requires depth=true: development and evaluation use 256x256 RGB-D")


def recorder_class():
    import gymnasium as gym

    class RawObservation(gym.ObservationWrapper):
        """Pass-through: keeps the latest raw ManiSkill observation (shallow copy,
        before the canonical flatten wrapper pops sensor data and calibration)."""

        last = None

        def observation(self, observation):
            self.last = dict(observation)
            return observation

    return RawObservation


def frame(raw, obs, step):
    """Allowlisted RoboMME-style frame from the raw and canonical observations."""
    if raw is None or set(obs) - {"rgb", "proprio", "task_cue"} or not {"rgb", "proprio"} <= set(obs):
        raise ValueError("Unexpected MIKASA-Robo observation structure")
    proprio = to_numpy(obs["proprio"])
    qpos = to_numpy(raw["agent"]["qpos"])
    if proprio.shape != (1, 7) or qpos.shape != (1, 9):
        raise ValueError(f"Unexpected proprio {proprio.shape} or qpos {qpos.shape}")
    row = {"phase": "exec", "step": int(step)}
    canonical = to_numpy(obs["rgb"])
    for key, camera in CAMERAS.items():
        data, param = raw["sensor_data"][camera], raw["sensor_param"][camera]
        rgb, depth = to_numpy(data["rgb"]), to_numpy(data["depth"])
        if rgb.shape != (1, *IMAGE) or rgb.dtype != np.uint8:
            raise ValueError(f"Expected (1,256,256,3) uint8 {camera} rgb, got {rgb.shape} {rgb.dtype}")
        if depth.shape != (1, *IMAGE[:2], 1) or depth.dtype != np.int16:
            raise ValueError(f"Expected (1,256,256,1) int16 {camera} depth, got {depth.shape} {depth.dtype}")
        row[key + "_rgb"] = np.ascontiguousarray(rgb[0])
        row[key + "_depth"] = np.ascontiguousarray(depth[0, ..., 0])
        row[key + "_K"] = np.ascontiguousarray(to_numpy(param["intrinsic_cv"])[0], dtype=np.float32)
        row[key + "_extrinsic"] = np.ascontiguousarray(to_numpy(param["extrinsic_cv"])[0], dtype=np.float32)
    # The canonical image is exactly the raw cameras concatenated (base, hand).
    if canonical.shape != (1, *IMAGE[:2], 6) or not (
        np.array_equal(canonical[0, ..., :3], row["front_rgb"]) and np.array_equal(canonical[0, ..., 3:], row["wrist_rgb"])
    ):
        raise ValueError("Raw camera images disagree with the canonical observation")
    row["tcp"] = np.ascontiguousarray(proprio[0, :6], dtype=np.float32)
    row["joints"] = np.ascontiguousarray(qpos[0, :7], dtype=np.float32)
    row["fingers"] = np.ascontiguousarray(qpos[0, 7:9], dtype=np.float32)
    return row


class RGBDEnvironment(NativeEnvironment):
    def __init__(self, env, recorder, seed, horizon, device):
        super().__init__(env, seed, horizon, device)
        self.recorder = recorder
        self.kinematics = None
        self.physics_backend = "gpu" if env.unwrapped.gpu_sim_enabled else "cpu"

    @property
    def unwrapped(self):
        return self.env.unwrapped


class MikasaRobo(MikasaProtocol):
    name = "mikasa_robo"
    guide = "mikasa_robo/guide.md"

    def interface(self, task, depth):
        require_rgbd(depth)
        shape = "x".join(map(str, IMAGE))
        sensors = {}
        for key, label in (("front", "fixed scene camera (MIKASA base_camera)"), ("wrist", "hand-mounted camera (MIKASA hand_camera)")):
            sensors[key + "_rgb"] = sensor("uint8", IMAGE, f"{label} RGB {shape}, row 0 = top")
        for key in CAMERAS:
            sensors[key + "_depth"] = sensor(
                "int16", IMAGE[:2], f"{key} depth, int16 optical-axis millimetres, 0 unknown", aligned_with=key + "_rgb"
            )
        for key in CAMERAS:
            sensors[key + "_K"] = sensor("float32", (3, 3), f"{key} 3x3 intrinsics of the {shape} image (OpenCV)")
            sensors[key + "_extrinsic"] = sensor(
                "float32", (3, 4), f"{key} 3x4 world-to-camera [R t], OpenCV axes (x right, y down, z forward), metres"
            )
        sensors["tcp"] = sensor(
            "float32",
            (6,),
            "measured TCP XYZ metres in the world (scene) frame and roll/pitch/yaw radians "
            "(R = Rz(yaw) Ry(pitch) Rx(roll))",
        )
        sensors["joints"] = sensor("float32", (7,), "Panda arm joint positions J1..J7, radians; limits: " + limits_text())
        sensors["fingers"] = sensor("float32", (2,), "Panda finger joint positions, metres (0.04 each open, 0 closed)")
        return {
            "sensors": sensors,
            "action": self.action_spec(task),
            "video": ["front_rgb", "wrist_rgb"],
            "starter": "robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)",
        }

    def sensor_summary(self, task, depth):
        require_rgbd(depth)
        return (
            "front (scene) and wrist RGB-D 256x256 with intrinsics/extrinsics; "
            "TCP pose, 7 arm joints and 2 finger positions"
        )

    def begin(self, env, *, builtin_demo, depth):
        require_rgbd(depth)
        if builtin_demo:
            raise ValueError("MIKASA-Robo has no reset video; set builtin_demo=false")
        from cam.interface.mikasa_robo.common import instruction

        obs, info = env.reset()
        return self._start(instruction(info), env, obs, depth)

    def _start(self, goal, env, obs, depth):
        from cam.interface.base import EpisodeStart

        return EpisodeStart(goal, [self._frame(env, obs, 0, depth)], {})

    def _frame(self, env, obs, step, depth):
        require_rgbd(depth)
        return frame(env.recorder.last, obs, step)

    def advance(self, env, action, context, *, depth):
        require_rgbd(depth)
        from cam.interface.base import StepResult

        obs, terminated, truncated, success = env.step(action)
        status = (
            "success"
            if success
            else "fail"
            if terminated
            else "timeout"
            if truncated or env.steps >= env.horizon
            else "running"
        )
        return StepResult([self._frame(env, obs, env.steps, depth)], status, env.steps)

    def dry_run(self, env, pose, context):
        """Kinematic dry-run of an absolute TCP pose (world frame, same convention as
        the ``tcp`` sensor). Uses the controller's own Panda chain and joint limits;
        never touches the simulator state. No collision checking."""
        if env.kinematics is None:
            env.kinematics = ArmKinematics(env.unwrapped)
        return env.kinematics.dry_run(pose)

    def make_environment(self, metadata, task, episode, split, depth, *, physics_backend=None):
        """Official GPU backend by default; CPU is an explicitly recorded variant."""
        require_rgbd(depth)
        if physics_backend not in (None, "cpu", "gpu"):
            raise ValueError("physics_backend must be cpu, gpu, or null")
        import importlib
        import os
        from pathlib import Path
        import sys

        from cam.io import read
        from cam.interface.mikasa_robo.common import default_assets

        pinned = {int(row["episode"]): row for row in self.records(task, split)}
        given = {
            int(row["episode"]): row
            for row in read(Path(metadata) / f"record_dataset_{task}_metadata.json")["records"]
        }
        if episode not in given or episode not in pinned:
            raise ValueError("Episode is outside the pinned split")
        seed = int(pinned[episode]["seed"])
        if int(given[episode]["seed"]) != seed:
            raise ValueError("Run metadata seed disagrees with pinned metadata")
        horizon = self.max_episode_steps(task)
        assets = self.source / self.asset_dir
        assets = assets if assets.is_dir() else default_assets()
        if "mani_skill" in sys.modules:
            loaded = Path(sys.modules["mani_skill"].ASSET_DIR).resolve()
            if assets is not None and loaded != assets.resolve():
                raise ValueError("mani_skill was imported with a different asset directory")
        elif assets is not None:
            os.environ["MS_ASSET_DIR"] = str(assets.resolve())
        sys.path.insert(0, str(self.source))
        for module in ("mikasa_robo_suite", "baselines"):
            path = Path(importlib.import_module(module).__file__).resolve()
            if not path.is_relative_to(self.source):
                raise ValueError(f"A different {module} package is already imported; use an episode subprocess")
        import gymnasium as gym

        importlib.import_module("mikasa_robo_suite.vla.memory_envs")  # registers env IDs
        wrappers = importlib.import_module("mikasa_robo_suite.vla.utils.apply_wrappers")
        if not Path(wrappers.__file__).resolve().is_relative_to(self.source):
            raise ValueError("Canonical wrappers were not loaded from the pinned source")
        kwargs = dict(MAKE_KWARGS_RGBD)
        if physics_backend is not None:
            kwargs["sim_backend"] = physics_backend
        recorder = recorder_class()(gym.make(task, **kwargs))
        env = wrappers.apply_mikasa_vla_wrappers(recorder, include_overlays=False)
        if int(env.max_episode_steps) != horizon:
            env.close()
            raise ValueError("Registered step limit disagrees with pinned metadata")
        robot = env.unwrapped.agent.robot
        names = [j.name for j in robot.get_active_joints()][:7]
        limits = to_numpy(robot.get_qlimits())[0, :7]
        if tuple(names) != JOINT_NAMES or not np.allclose(limits, np.stack([JOINT_LOWER, JOINT_UPPER], 1), atol=1e-4):
            env.close()
            raise ValueError("Panda joint names or limits differ from the documented interface")
        return RGBDEnvironment(env, recorder, seed, horizon, env.unwrapped.device)
