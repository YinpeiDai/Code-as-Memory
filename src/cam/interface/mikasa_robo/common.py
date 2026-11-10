"""Shared MIKASA native task identities, controller, scoring and robot utilities."""

import importlib
import os
from pathlib import Path
import re
import sys

import numpy as np

from cam.io import read, task_name
from cam.interface.base import EpisodeStart, StepResult, action, sensor

METADATA = Path(__file__).resolve().parent / "metadata"
TEST_START_SEED = 4242424242  # mikasa_robo_suite/vla/benchmarking.py START_SEED
TEST_EPISODES = 50  # NUM_EPISODES_PER_TASK
#: Native observation keys after the canonical wrappers; task_cue is dropped.
NATIVE_KEYS = {"rgb", "proprio"}
PRIVATE_KEYS = {"task_cue"}
MAKE_KWARGS = {
    "num_envs": 1,
    "obs_mode": "rgb",
    "control_mode": "pd_ee_delta_pose",
    "reward_mode": "normalized_dense",
    "render_mode": "all",
    "sim_backend": "gpu",
}


def checkout():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/cam").is_dir():
            return parent
    return None


def default_source():
    root = checkout()
    path = root / "third_party/mikasa_robo" if root else None
    if path is None or not (path / "mikasa_robo_suite").is_dir():
        raise FileNotFoundError(
            "Pass --benchmark-source or initialize third_party/mikasa_robo"
        )
    return path


def default_assets():
    """ManiSkill asset root (``MS_ASSET_DIR``) holding the YCB cups for ShellGame*."""
    override = os.environ.get("CAM_MIKASA_ROBO_ASSETS")
    if override:
        return Path(override).resolve()
    root = checkout()
    return (root / "envs/mikasa_robo/assets") if root else None


def dataset_name(task):
    """Release folder name; mirrors mikasa_robo_suite/vla/utils/dataset_naming.py."""

    def token(value):
        value = re.sub(r"([a-z])([A-Z])", r"\1_\2", value)
        value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
        value = re.sub(r"(?<=[A-Za-z])([A-Za-z])([0-9])", r"\1_\2", value)
        return value.lower()

    return "_".join(token(part) for part in task.split("-") if part)


def to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def first(value):
    return to_numpy(value).reshape(-1)[0].item()


def instruction(info):
    value = info.get("language_instruction")
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Native language instruction is missing")
    return value




class NativeEnvironment:
    """One canonical wrapped env bound to one pinned seed."""

    def __init__(self, env, seed, horizon, device):
        self.env, self.seed, self.horizon, self.device = env, int(seed), int(horizon), device
        self.steps = 0

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        self.steps = 0
        if int(first(info["elapsed_steps"])) != 0:
            raise ValueError("Native reset did not restart the step counter")
        return obs, info

    def step(self, values):
        import torch

        command = torch.as_tensor(
            np.asarray(values, dtype=np.float32).reshape(1, 7), device=self.device
        )
        obs, _, terminated, truncated, info = self.env.step(command)
        self.steps += 1
        if int(first(info["elapsed_steps"])) != self.steps:
            raise ValueError("Native step counter disagrees with issued actions")
        return obs, bool(first(terminated)), bool(first(truncated)), bool(first(info["success"]))

    def close(self):
        self.env.close()


class MikasaProtocol:
    package = "mikasa_robo"
    describe_interface = True
    supports_builtin_demo = False
    #: Root-relative patterns ('/') or names at any depth. ``baselines/`` stays:
    #: the canonical wrapper chain imports ``baselines.ppo.ppo_memtasks``.
    pin_ignore = (".git", ".github", "assets/", "docs/", "utils/", "examples/", "*.egg-info")
    asset_dir = "maniskill_data"

    def __init__(self, source=None):
        self.source = Path(source).resolve() if source else default_source()
        if not (self.source / "mikasa_robo_suite/vla").is_dir():
            raise FileNotFoundError(f"MIKASA-Robo source missing: {self.source}")
        if not (METADATA / "train").is_dir():
            raise FileNotFoundError(f"Pinned MIKASA-Robo metadata missing: {METADATA}")

    @property
    def asset_links(self):
        assets = default_assets()
        return {self.asset_dir: assets} if assets is not None else {}

    def tasks(self):
        return sorted(
            p.name.removeprefix("record_dataset_").removesuffix("_metadata.json")
            for p in (METADATA / "train").glob("record_dataset_*_metadata.json")
        )

    def metadata(self, task, split):
        task_name(task)
        if split not in ("train", "test"):
            raise ValueError("Choose train or test")
        path = METADATA / split / f"record_dataset_{task}_metadata.json"
        if not path.is_file():
            raise ValueError(f"Unknown MIKASA-Robo task: {task}")
        value = read(path)
        if value["env_id"] != task:
            raise ValueError("Pinned metadata names a different task")
        return value

    def records(self, task, split):
        records = self.metadata(task, split)["records"]
        ids = [int(row["episode"]) for row in records]
        if len(ids) != len(set(ids)) or len({int(row["seed"]) for row in records}) != len(ids):
            raise ValueError("Duplicate episode or seed in metadata")
        if any(row["task"] != task for row in records):
            raise ValueError("Pinned metadata mixes tasks")
        if split == "test" and [(int(r["episode"]), int(r["seed"])) for r in records] != [
            (i, TEST_START_SEED + i) for i in range(TEST_EPISODES)
        ]:
            raise ValueError("Test metadata is not the official seed stream")
        return records

    def max_episode_steps(self, task):
        return int(self.metadata(task, "test")["max_episode_steps"])

    def action_spec(self, task):
        from cam.interface.mikasa_robo.api import PREVIEW_SPEC

        return action(
            ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"),
            "One native pd_ee_delta_pose command for one 20 Hz control step. "
            "Translation 0.1*(dx, dy, dz) m along the robot-base axes (parallel to the "
            "scene x, y, z). With r = (drx, dry, drz) scaled to unit norm if |r| > 1, "
            "the signed angular offset is -0.1*r rad. The default GPU controller "
            "uses this as a rotation-vector offset about the base axes in its "
            "first-order joint target (q + pinv(J) * offset). With explicit CPU "
            "physics, native iterative IK instead targets a pose whose rotation "
            "is Rx(-0.1*r[0]) Ry(-0.1*r[1]) Rz(-0.1*r[2]) left-multiplied onto the "
            "measured orientation. These agree to first order for small offsets. "
            "Both use the measured arm state; one step covers only part of the "
            "commanded offset. gripper is an absolute finger target: -1 closed, +1 open. "
            f"Native step limit for this task: {self.max_episode_steps(task)}.",
            low=[-1.0] * 7,
            high=[1.0] * 7,
            type="normalized_pd_ee_delta_pose",
            control_frequency=20,
            native_max_steps=self.max_episode_steps(task),
            motion_preview=PREVIEW_SPEC.copy(),
        )

    def preview_motion(self, env, values, mode, context):
        from cam.interface.mikasa_robo.api.preview import preview

        return preview(env.env.unwrapped, values, mode)






    def select_training(self, extracted_records, count):
        records = sorted(extracted_records, key=lambda row: int(row["episode"]))
        return records if count is None else records[: int(count)]
