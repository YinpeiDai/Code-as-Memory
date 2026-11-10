"""Official RMBench adapter (RoboTwin 2.0, dual-arm aloha-agilex, ``demo_clean``).

One ``robot.step`` is one native ``take_action(action, action_type="ee")`` call, the
unit counted against ``env_cfg/task_config/_eval_step_limit.yml``. Success is the
native ``eval_success`` flag that ``take_action`` sets after checking every
simulator step; the adapter never calls ``check_success`` (it mutates task state).
Only the declared camera, calibration and robot-state arrays cross the boundary.
"""

import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from cam.io import read, task_name
from cam.interface.base import EpisodeStart, StepResult, action, sensor

TASKS = (
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "observe_and_pickup",
    "place_block_mat",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "swap_T",
    "swap_blocks",
)
TASK_CONFIG = "demo_clean"
EMBODIMENT = "aloha-agilex"
#: CaM camera name -> (native observation key, released-demo vision group).
CAMERAS = {
    "head": ("head_camera", "cam_head"),
    "front": ("front_camera", "cam_third_view"),
    "left_wrist": ("left_camera", "cam_left_wrist"),
    "right_wrist": ("right_camera", "cam_right_wrist"),
}
HEIGHT, WIDTH = 240, 320
ARMS = ("left", "right")
POSE = ("x", "y", "z", "qw", "qx", "qy", "qz")
METADATA = Path(__file__).with_name("metadata")


def default_source():
    for parent in Path(__file__).resolve().parents:
        path = parent / "third_party/rmbench"
        if (path / "envs/_base_task.py").is_file():
            return path
    raise FileNotFoundError("Pass --benchmark-source or initialize third_party/rmbench")


def default_assets():
    override = os.environ.get("CAM_RMBENCH_ASSETS")
    if override:
        return Path(override).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "envs/rmbench/pyproject.toml").is_file():
            return parent / "envs/rmbench/assets"
    raise FileNotFoundError("Set CAM_RMBENCH_ASSETS; see envs/rmbench/README.md")


def step_limits(source):
    """Parse the flat ``task: int`` step-limit file without a YAML dependency."""
    limits = {}
    path = Path(source) / "env_cfg/task_config/_eval_step_limit.yml"
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        value = value.split("#")[0].strip()
        if key.strip() and value not in ("", "null", "~"):
            limits[key.strip()] = int(value)
    return limits


def instruction(source, task):
    """The single ``seen`` instruction (no placeholders) used by demo_clean eval."""
    data = json.loads(
        (Path(source) / "description/task_instruction" / f"{task}.json").read_text()
    )
    if len(data["seen"]) != 1 or "{" in data["seen"][0]:
        raise ValueError(f"Unexpected instruction templates for {task}")
    return data["seen"][0]


class NativeEnvironment:
    """One RMBench task instance at one seed; holds the native objects privately."""

    def __init__(self, task_env, goal, step_lim):
        self.task, self.goal, self.step_lim = task_env, goal, step_lim
        self.applied = 0

    def observe(self):
        return self.task.get_obs()

    def step(self, values):
        # eval_policy_xpolicylab.xpolicylab_action_to_robotwin sends float32 actions.
        self.task.take_action(np.asarray(values, dtype=np.float32), action_type="ee")
        self.applied += 1
        return self.observe()

    @property
    def count(self):
        return int(self.task.take_action_cnt)

    @property
    def success(self):
        return bool(self.task.eval_success)

    def close(self):
        robot = getattr(self.task, "robot", None)
        try:
            self.task.close_env(clear_cache=True)
        finally:
            for arm in ARMS:
                proc = getattr(robot, f"{arm}_proc", None)
                if proc is not None and proc.is_alive():
                    try:
                        getattr(robot, f"{arm}_conn").send({"cmd": "exit"})
                    except (OSError, ValueError):
                        pass
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.terminate()


def depth_millimetres(value):
    """Native SAPIEN optical-axis depth is already in mm; 0 means invalid."""
    value = np.asarray(value)
    if value.shape != (HEIGHT, WIDTH):
        raise ValueError(f"Unexpected depth shape {value.shape}")
    valid = np.isfinite(value) & (value > 0) & (value <= np.iinfo(np.int16).max)
    result = np.zeros(value.shape, dtype=np.int16)
    result[valid] = np.rint(value[valid]).astype(np.int16)
    return result


def frame(obs, step, depth=False):
    """Rebuild one allowlisted frame; raw observation dictionaries never leave."""
    row = {"phase": "exec", "step": int(step)}
    cameras = obs["observation"]
    for name, (native, _) in CAMERAS.items():
        camera = cameras[native]
        rgb = np.asarray(camera["rgb"])
        if rgb.dtype != np.uint8 or rgb.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Unexpected {native} RGB {rgb.dtype} {rgb.shape}")
        row[name + "_rgb"] = np.ascontiguousarray(rgb)
        if depth:
            row[name + "_depth"] = depth_millimetres(camera["depth"])
        row[name + "_K"] = np.asarray(camera["intrinsic_cv"], dtype=np.float32).reshape(3, 3)
        row[name + "_extrinsic"] = np.asarray(
            camera["extrinsic_cv"], dtype=np.float32
        ).reshape(3, 4)
    joints, endpose = obs["joint_action"], obs["endpose"]
    for arm in ARMS:
        row[arm + "_ee_pose"] = np.asarray(endpose[arm + "_endpose"], dtype=np.float64).reshape(7)
        row[arm + "_joints"] = np.asarray(joints[arm + "_arm"], dtype=np.float64).reshape(6)
        row[arm + "_gripper"] = np.asarray(joints[arm + "_gripper"], dtype=np.float64).reshape(())
    return row


class RMBench:
    name = "rmbench"
    package = "rmbench"
    guide = "rmbench/guide.md"
    describe_interface = True
    supports_builtin_demo = False
    #: Root-relative ("/") patterns: the submodule gitlink, the XPolicyLab policy
    #: server, released/raw data and the asset trees (linked separately).
    pin_ignore = (
        ".git",
        "XPolicyLab/",
        "assets/",
        "data/demo_clean",
        "data/download_cache",
        "eval_result/",
        "*.zip",
    )

    def __init__(self, source=None):
        self.source = Path(source).resolve() if source else default_source()
        if not (self.source / "envs/_base_task.py").is_file():
            raise FileNotFoundError(f"RMBench source missing: {self.source}")
        self.limits = step_limits(self.source)

    @property
    def asset_links(self):
        return {"assets": default_assets()}

    def tasks(self):
        return [
            task
            for task in TASKS
            if (self.source / "envs" / f"{task}.py").is_file()
            and (METADATA / "train" / f"record_dataset_{task}_metadata.json").is_file()
        ]

    def records(self, task, split):
        task_name(task)
        if task not in TASKS or split not in ("train", "test"):
            raise ValueError("Choose an RMBench task and train or test")
        records = read(METADATA / split / f"record_dataset_{task}_metadata.json")["records"]
        ids = [int(row["episode"]) for row in records]
        if len(ids) != len(set(ids)) or len({int(r["seed"]) for r in records}) != len(ids):
            raise ValueError("Duplicate episode or seed in metadata")
        return records

    def goal(self, task):
        return instruction(self.source, task)

    def action_spec(self, task):
        names = [f"{arm}_{c}" for arm in ARMS for c in (*POSE, "gripper")]
        low = ([None] * 7 + [0.0]) * 2
        high = ([None] * 7 + [1.0]) * 2
        return action(
            names,
            "One native RMBench end-effector action for both arms: absolute world-frame "
            "pose targets (metres; unit quaternion w,x,y,z) in the same frame as the "
            "*_ee_pose sensors, submitted to native planning/execution; grippers 0 closed, "
            f"1 open. Each call is one of this task's {self.limits[task]} native actions.",
            low=low,
            high=high,
            type="absolute_world_ee_pose_quaternion_dual_arm",
            action_type="ee",
            native_max_steps=self.limits[task],
            planner_failure_feedback=False,
            motion_feedback={
                "schema": "cam-rmbench-observed-motion/1",
                "source": "public_post_action_ee_pose",
                "position_tolerance_m": 0.02,
                "rotation_tolerance_rad": 0.1,
            },
            offline_robot={
                "schema": "cam-rmbench-offline-robot/1",
                "module": "cam.runtime.rmbench",
                "reference": "api_reference/runtime/rmbench/api.md",
            },
        )

    def interface(self, task, depth):
        sensors = {}
        labels = {
            "head": "head camera",
            "front": "front camera",
            "left_wrist": "left wrist camera",
            "right_wrist": "right wrist camera",
        }
        for name, label in labels.items():
            sensors[name + "_rgb"] = sensor("uint8", (HEIGHT, WIDTH, 3), label + " RGB")
        if depth:
            for name, label in labels.items():
                sensors[name + "_depth"] = sensor(
                    "int16", (HEIGHT, WIDTH),
                    label + " aligned optical-axis depth in millimetres; 0 invalid/out of range",
                    aligned_with=name + "_rgb",
                )
        for name, label in labels.items():
            sensors[name + "_K"] = sensor("float32", (3, 3), label + " 3x3 OpenCV intrinsics (pixels)")
            sensors[name + "_extrinsic"] = sensor(
                "float32", (3, 4), label + " 3x4 OpenCV world-to-camera extrinsics (metres)"
            )
        for arm in ARMS:
            sensors[arm + "_ee_pose"] = sensor(
                "float", (7,), f"{arm} end-effector pose, world x,y,z metres and quaternion w,x,y,z"
            )
            sensors[arm + "_joints"] = sensor("float", (6,), f"{arm} arm commanded joint targets (radians)")
            sensors[arm + "_gripper"] = sensor("float", (), f"{arm} commanded gripper opening, 0 closed to 1 open")
        hold = ", ".join(
            f"*robot.now().{arm}_ee_pose, float(robot.now().{arm}_gripper)" for arm in ARMS
        )
        return {
            "sensors": sensors,
            "action": self.action_spec(task),
            "video": ["head_rgb", "front_rgb", "left_wrist_rgb", "right_wrist_rgb"],
            "starter": f"robot.step({hold})",
        }

    def sensor_summary(self, task, depth):
        return (
            ("RGB-D" if depth else "RGB (no depth)") +
            " 320x240 from head, front, left-wrist and right-wrist cameras with "
            "intrinsics/extrinsics; per-arm end-effector pose, joint targets and gripper"
        )

    def begin(self, env, *, builtin_demo, depth):
        if builtin_demo:
            raise ValueError("RMBench has no reset video; set builtin_demo=false")
        return EpisodeStart(env.goal, [frame(env.observe(), 0, depth)], {"start": env.count})

    def advance(self, env, values, context, *, depth):
        # Official eval_policy_xpolicylab.eval_remote_policy counts an action only
        # after take_action returns, and does not score a crash before that:
        #     task_env.take_action(flat_action, action_type=robotwin_action_type)
        #     rollout_steps += 1
        #     ...
        #     if rollout_failed and rollout_steps == 0 and not succ:
        #         # Env broke before the policy could act (e.g. renderer buffer errors).
        #         # Don't burn an episode on it; ...
        # Here such a crash is an unscored harness_error (retry the same pinned seed
        # with cam recover-test); any later exception is a scored native failure.
        try:
            obs = env.step(values)
        except Exception as exc:
            if env.applied == 0:
                raise
            return StepResult(
                [],
                "error",
                env.count - context["start"],
                private_error={"exception_type": type(exc).__name__, "error_message": str(exc)},
            )
        steps = env.count - context["start"]
        status = (
            "success"
            if env.success
            else "timeout"
            if env.count >= env.step_lim
            else "running"
        )
        # Preserve native motion/scoring, without exposing internal planner status.
        # Policies observe the resulting public frame and can use offline robot math.
        return StepResult([frame(obs, steps, depth)], status, steps)

    def select_training(self, extracted_records, count):
        rows = sorted(extracted_records, key=lambda row: int(row["episode"]))
        return rows if count is None else rows[: int(count)]

    def task_arguments(self, task):
        """Native ``load_task_args`` for demo_clean/aloha-agilex, without XPolicyLab."""
        import yaml

        configs = self.source / "env_cfg/task_config"
        args = yaml.safe_load((configs / f"{TASK_CONFIG}.yml").read_text())
        if args["embodiment"] != [EMBODIMENT]:
            raise ValueError("Pinned demo_clean embodiment changed")
        embodiments = yaml.safe_load((configs / "_embodiment_config.yml").read_text())
        robot_file = embodiments[EMBODIMENT]["file_path"]
        config = yaml.safe_load((Path(robot_file) / "config.yml").read_text())
        cameras = yaml.safe_load((configs / "_camera_config.yml").read_text())
        args.update(
            task_name=task,
            task_config=TASK_CONFIG,
            left_robot_file=robot_file,
            right_robot_file=robot_file,
            dual_arm_embodied=True,
            left_embodiment_config=config,
            right_embodiment_config=config,
            head_camera_h=cameras[args["camera"]["head_camera_type"]]["h"],
            head_camera_w=cameras[args["camera"]["head_camera_type"]]["w"],
            eval_mode=True,
            render_freq=0,
        )
        args["data_type"].update(rgb=True, qpos=True, endpose=True)
        for key in ("depth", "pointcloud", "mesh_segmentation", "actor_segmentation"):
            if args["data_type"].get(key):
                raise ValueError(f"demo_clean unexpectedly enables privileged {key}")
        args.pop("eval_video_save_dir", None)
        return args

    def import_native(self, task):
        """Import the pinned simulator modules (chdir: RMBench opens ./assets/...)."""
        os.chdir(self.source)
        if str(self.source) not in sys.path:
            sys.path.insert(0, str(self.source))
        for name in ("envs", "description.utils.generate_episode_instructions"):
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(self.source):
                raise ValueError(f"A different {name} package is already imported")
        module = importlib.import_module(f"envs.{task}")
        if not Path(module.__file__).resolve().is_relative_to(self.source):
            raise ValueError("Imported RMBench task module is outside the pinned source")
        return module

    def make_environment(self, metadata, task, episode, split, depth):
        records = read(Path(metadata) / f"record_dataset_{task}_metadata.json")["records"]
        selected = {int(row["episode"]): row for row in records}
        pinned = {int(row["episode"]): row for row in self.records(task, split)}
        if episode not in selected or episode not in pinned:
            raise ValueError("Episode is outside the pinned split")
        seed = int(selected[episode]["seed"])
        if seed != int(pinned[episode]["seed"]):
            raise ValueError("Run metadata seed disagrees with pinned metadata")
        module = self.import_native(task)
        args = self.task_arguments(task)
        args["data_type"]["depth"] = bool(depth)
        task_env = getattr(module, task)()
        task_env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **args)
        if task_env.step_lim != self.limits[task]:
            raise ValueError("Native step limit disagrees with the pinned limit file")
        # Same instruction draw as eval_policy_xpolicylab.build_instruction (it also
        # consumes one global NumPy draw after setup, which we keep for parity).
        generate = importlib.import_module(
            "description.utils.generate_episode_instructions"
        ).generate_episode_descriptions
        goal = str(np.random.choice(generate(task, [{}], 100)[0][args["eval_instruction"]]))
        if goal != self.goal(task):
            raise ValueError("Native instruction differs from the pinned task instruction")
        task_env.set_instruction(instruction=goal)
        return NativeEnvironment(task_env, goal, int(task_env.step_lim))
