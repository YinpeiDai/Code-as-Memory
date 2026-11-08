"""Official RoboMME adapter; native seed resolution, control counters and scoring."""

import importlib
from pathlib import Path
import sys


from cam.io import read, task_name
from .robomme_observations import array, observations
from .base import EpisodeStart, StepResult, action, sensor

STICK_TASKS = ("PatternLock", "RouteStick")


def default_source():
    for parent in Path(__file__).resolve().parents:
        path = parent / "third_party/robomme_benchmark/src/robomme"
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "Pass --benchmark-source or initialize third_party/robomme_benchmark"
    )


class NativeEnvironment:
    def __init__(self, env):
        self.env, self.baseline = env, 0

    def decorate(self, info):
        count = max(
            0,
            int(self.env.get_wrapper_attr("steps_without_demonstration"))
            - self.baseline,
        )
        return {**info, "execution_total_control_steps": count}

    def reset(self):
        obs, info = self.env.reset()
        self.baseline = int(self.env.get_wrapper_attr("steps_without_demonstration"))
        return obs, self.decorate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            obs,
            reward,
            bool(array(terminated).reshape(-1)[-1]),
            bool(array(truncated).reshape(-1)[-1]),
            self.decorate(info or {}),
        )

    def close(self):
        self.env.close()


class RoboMME:
    name = "robomme"
    package = "robomme"
    guide = "robomme.md"
    simulator = None  # the CaM environment's `robomme` extra serves the simulator
    observations = staticmethod(observations)

    def __init__(self, source=None):
        self.source = Path(source).resolve() if source else default_source()
        if not (self.source / "env_metadata/train").is_dir():
            raise FileNotFoundError(f"RoboMME metadata missing: {self.source}")

    def tasks(self):
        return sorted(
            p.name.removeprefix("record_dataset_").removesuffix("_metadata.json")
            for p in (self.source / "env_metadata/train").glob(
                "record_dataset_*_metadata.json"
            )
        )

    def records(self, task, split):
        task_name(task)
        if split not in ("train", "test"):
            raise ValueError("Choose train or test")
        records = read(
            self.source
            / "env_metadata"
            / split
            / f"record_dataset_{task}_metadata.json"
        )["records"]
        ids = [int(row["episode"]) for row in records]
        if len(ids) != len(set(ids)) or len(
            {int(row["seed"]) for row in records}
        ) != len(ids):
            raise ValueError("Duplicate episode or seed in metadata")
        return records

    def action_spec(self, task):
        gripper = task not in STICK_TASKS
        return action(
            ("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
            "One absolute world TCP target (metres, XYZ roll/pitch/yaw radians) for one "
            "native control step; gripper -1 closes and +1 opens"
            + ("." if gripper else "; ignored for stick tasks."),
            low=[None] * 6 + [-1.0],
            high=[None] * 6 + [1.0],
            ignored=() if gripper else ("gripper",),
            type="absolute_world_tcp_rpy",
            gripper=gripper,
            control_frequency=20,
            native_max_steps_argument=1300,
        )

    def interface(self, task, depth):
        sensors = {
            "front_rgb": sensor("uint8", (None, None, 3), "front camera RGB"),
            "wrist_rgb": sensor("uint8", (None, None, 3), "wrist camera RGB"),
            "front_K": sensor("float", (3, 3), "front camera 3x3 intrinsics"),
            "wrist_K": sensor("float", (3, 3), "wrist camera 3x3 intrinsics"),
            "front_extrinsic": sensor("float", (3, 4), "front camera 3x4 world-to-camera extrinsics"),
            "wrist_extrinsic": sensor("float", (3, 4), "wrist camera 3x4 world-to-camera extrinsics"),
            "tcp": sensor("float", (6,), "measured native TCP XYZ metres and roll/pitch/yaw radians"),
            "joints": sensor("float", (None,), "arm joint positions"),
            "fingers": sensor("float", (None,), "gripper finger widths"),
        }
        if depth:
            for camera in ("front", "wrist"):
                sensors[camera + "_depth"] = sensor(
                    "int16",
                    (None, None),
                    camera + " depth, int16 optical-axis millimetres, 0 unknown",
                    aligned_with=camera + "_rgb",
                )
        return {
            "sensors": sensors,
            "action": self.action_spec(task),
            "video": ["front_rgb", "wrist_rgb"],
            "starter": "robot.step(*robot.now().tcp, 1.0)",
        }

    def begin(self, env, *, builtin_demo, depth):
        obs, info = env.reset()
        value = info["task_goal"]
        goal = str(value if isinstance(value, str) else value[0])
        calibration = {
            k: info[k] for k in ("front_camera_intrinsic", "wrist_camera_intrinsic")
        }
        frames = self.observations(
            obs,
            calibration,
            0,
            reset=True,
            builtin_demo=builtin_demo,
            include_depth=depth,
        )
        return EpisodeStart(
            goal,
            frames,
            {
                "calibration": calibration,
                "start": int(info["execution_total_control_steps"]),
            },
        )

    def advance(self, env, action, context, *, depth):
        obs, _, terminated, truncated, info = env.step(action)
        steps = max(0, int(info["execution_total_control_steps"]) - context["start"])
        native = str(info.get("status", "ongoing"))
        status = (
            native
            if native in ("success", "fail", "timeout", "error")
            else "timeout"
            if terminated or truncated
            else "running"
        )
        count = len((obs or {}).get("front_rgb_list", []))
        frames = (
            self.observations(
                obs, context["calibration"], steps - count + 1, include_depth=depth
            )
            if count
            else []
        )
        private_error = (
            {
                k: str(info[k])
                for k in ("status", "error_message", "exception_type", "failure_reason")
                if k in info
            }
            if status == "error"
            else None
        )
        motion_error = (
            "Pose could not be reached; native step penalty applied."
            if info.get("action_error")
            else None
        )
        return StepResult(frames, status, steps, motion_error, private_error)

    def select_training(self, extracted_records, count):
        from .robomme_data import select

        return select(extracted_records, "all" if count is None else count)

    def make_environment(self, metadata, task, episode, split, depth):
        sys.path.insert(0, str(self.source.parent))
        package = importlib.import_module(self.package)
        if not Path(package.__file__).resolve().is_relative_to(self.source):
            raise ValueError(
                "A different RoboMME package is already imported; use an episode subprocess"
            )
        importlib.import_module("robomme.robomme_env")
        builder_class = importlib.import_module(
            "robomme.env_record_wrapper"
        ).BenchmarkEnvBuilder
        builder = builder_class(
            task,
            dataset=split,
            action_space="ee_pose",
            gui_render=False,
            max_steps=1300,
            override_metadata_path=metadata,
        )
        records = read(metadata / f"record_dataset_{task}_metadata.json")["records"]
        selected = {int(row["episode"]): row for row in records}
        if episode not in selected:
            raise ValueError("Episode is outside the pinned split")
        expected = selected[episode]
        if builder.resolve_episode(episode) != (
            int(expected["seed"]),
            expected["difficulty"],
        ):
            raise ValueError("Native seed/difficulty disagrees with pinned metadata")
        return NativeEnvironment(
            builder.make_env_for_episode(
                episode,
                include_front_depth=depth,
                include_wrist_depth=depth,
                include_front_camera_intrinsic=True,
                include_wrist_camera_intrinsic=True,
                include_front_camera_extrinsic=True,
                include_wrist_camera_extrinsic=True,
            )
        )
