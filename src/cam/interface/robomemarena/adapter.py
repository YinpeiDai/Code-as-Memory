"""RoboMemArena adapter: the official 26-task LIBERO/MuJoCo evaluation protocol.

The native environment is built as ``evaluation_benchmark/scripts/run_all_tasks1_26.py``
does: ``OffScreenRenderEnv(bddl, ignore_done=True, reward_shaping=True,
control_freq=20, initialization_noise=None)`` rendered at 480x640, seeded with
``np.random.seed(s); env.seed(s)`` immediately before ``env.reset()``, then 10
settling steps ``[0]*6 + [-1]``. Stage checks are the pinned official functions
(``eval_task1_only.STAGE_CHECKS`` for task 1, ``task2_26_reference_stage`` for
tasks 2-26) driven by a line-by-line port of the official per-step loops.

Only the official 256x256 policy images (``np.flipud`` + ``cv2.INTER_AREA``),
optional aligned depth (``camera_depths=True``; metric z-buffer, ``np.flipud``,
centre-aligned nearest-neighbour resize, int16 millimetres), camera calibration
and robot proprioception cross the policy boundary; object poses and every other
raw observation key stay private. The episode ends at the
official decision point: success when the counted stages complete (non-counting
tasks; the official 200 post-goal steps cannot change latched stages, TSR or CSR),
the official counting-task break (30 steps after the second pour, or a third pour)
or 2500 policy steps (``timeout``). Native exceptions are scored as failures, as
the official loop does, with status ``error``.
"""

import ast
import gc
import hashlib
import importlib
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from cam.io import read, task_name
from cam.interface.base import EpisodeStart, StepResult, action, sensor

METADATA = Path(__file__).resolve().parent / "metadata"
#: Official task ID -> CaM task name (readable, no hidden-state words from BDDL stems).
TASKS = {
    1: "T01_CookiesTomatoBasket",
    2: "T02_ButterPopcornBasket",
    3: "T03_CreamPuddingBasket",
    4: "T04_ButterNonEmptyDrawer",
    5: "T05_ButterEmptyDrawer",
    6: "T06_PourSauceCookiesTwice",
    7: "T07_PourSauceFrypanTwice",
    8: "T08_ChocolateFrypanPourSauceTwice",
    9: "T09_ButterFrypanPourSauceTwice",
    10: "T10_PourWineMugTwice",
    11: "T11_CookiesTopButterMiddle",
    12: "T12_CookiesChocolateMiddleDrawer",
    13: "T13_CookiesButterMiddleDrawer",
    14: "T14_CookiesChocolateDrawers",
    15: "T15_ButterFrypanPourMilkTwice",
    16: "T16_PourMilkMugTwice",
    17: "T17_ButterChocolateMiddleDrawer",
    18: "T18_ChocolateButterCabinet",
    19: "T19_SauceMilkJuiceCabinet",
    20: "T20_CookiesChocolateMicrowave",
    21: "T21_ButterChocolateMicrowave",
    22: "T22_PourSauceCookiesMicrowave",
    23: "T23_CreamPopcornMicrowave",
    24: "T24_CookiesPopcornMicrowave",
    25: "T25_ButterCreamPlate",
    26: "T26_ChocolateCreamPlate",
}
TASK_IDS = {name: tid for tid, name in TASKS.items()}
#: Release category folders (README task table).
CATEGORIES = {
    "Multi-Sequence": (1, 2, 3, 22),
    "Multi-Occlusion": (4, 5, 11, 12, 13, 14, 17, 20, 21, 23, 24),
    "Multi-Counting": (6, 7, 8, 9, 10, 15, 16),
    "Multi-Transferring": (18, 19, 25, 26),
}
IMAGE = (256, 256, 3)  # resize_size of the official sweep
RENDER = (480, 640)  # eval_common.patch_env_resolution(480, 640)
SETTLE_STEPS = 10  # --num-steps-wait
SETTLE_ACTION = [0.0] * 6 + [-1.0]  # eval_common.LIBERO_DUMMY_ACTION
MAX_STEPS = 2500  # --max-steps (policy steps after settling)
POST_GOAL_STEPS = 200  # --post-goal-steps (cannot change latched outcomes)
EXTRA_POUR_MONITOR_STEPS = 30  # --extra-pour-monitor-steps
FAIL_ON_EXTRA_POUR = True
TEST_SEED_START = 50  # --seed
TEST_TRIALS = 50  # --num-trials-per-task
ENV_KWARGS = {
    "camera_heights": RENDER[0],
    "camera_widths": RENDER[1],
    "ignore_done": True,
    "reward_shaping": True,
    "control_freq": 20,
    "initialization_noise": None,
}
CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
NATIVE_IMAGES = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
NATIVE_DEPTHS = {"agentview": "agentview_depth", "wrist": "robot0_eye_in_hand_depth"}
#: Centre-aligned nearest-neighbour source pixels for the 480x640 -> 256x256 resize; the
#: same pixel-centre geometry as INTER_AREA, so depth aligns with RGB and its K.
DEPTH_ROWS = np.floor((np.arange(IMAGE[0]) + 0.5) * RENDER[0] / IMAGE[0]).astype(int)
DEPTH_COLS = np.floor((np.arange(IMAGE[1]) + 0.5) * RENDER[1] / IMAGE[1]).astype(int)
#: CV (x right, y down, z forward) from MuJoCo camera axes (x right, y up, looks along -z).
GL_TO_CV = np.diag([1.0, -1.0, -1.0])
PINNED_MODULES = ("eval_common", "policy_adapter", "task2_26_reference_stage", "shared_pour_counter", "eval_task1_only", "libero")


def construct_environment(factory, kwargs, randomization_error, attempts=30):
    """Retry only unseeded construction; assigned episode reset is unchanged.

    LIBERO retries placement failures during reset but its constructor does not.
    Its released retry runner also retries construction up to 30 times. No task
    seed, region, geometry, action or scorer is changed by these retries.
    """
    for attempt in range(1, attempts + 1):
        try:
            return factory(**kwargs)
        except randomization_error:
            if attempt == attempts:
                raise
            logging.getLogger(__name__).warning(
                "Environment construction placement failed (%s/%s); retrying before episode reset",
                attempt, attempts,
            )
        # Release partially constructed environments and renderer resources.
        gc.collect()


def checkout():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/cam").is_dir():
            return parent
    return None


def default_source():
    root = checkout()
    path = root / "third_party/robomemarena/evaluation_benchmark" if root else None
    if path is None or not (path / "libero_fork").is_dir():
        raise FileNotFoundError(
            "Pass --benchmark-source or initialize third_party/robomemarena"
        )
    return path


def task_id(task):
    task_name(task)
    if task not in TASK_IDS:
        raise ValueError(f"Unknown RoboMemArena task: {task}")
    return TASK_IDS[task]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def task_prompts(source):
    """``TASK_PROMPTS`` from the pinned ``eval_common.py``, read without importing it."""
    tree = ast.parse((Path(source) / "scripts/eval_common.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "TASK_PROMPTS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("TASK_PROMPTS not found in eval_common.py")


def bddl_file(source, tid):
    """Same resolution as ``eval_common._resolve_bddl_path`` for integer task IDs."""
    matches = sorted(Path(source, "bddl").glob(f"{tid}_*.bddl"))
    if not matches:
        raise FileNotFoundError(f"No BDDL for task {tid} under {source}")
    return matches[0]


def intrinsics(fovy_degrees):
    """3x3 K of the 256x256 policy image resized from the 480x640 render.

    MuJoCo renders square pixels with vertical field of view ``fovy`` and the
    principal point at the image centre. OpenCV pixel-centre convention: pixel
    (u, v) spans [u - 0.5, u + 0.5]; INTER_AREA maps destination centres as
    ``u_dst = (u_src + 0.5) * scale - 0.5``.
    """
    height, width = RENDER
    focal = 0.5 * height / np.tan(np.deg2rad(float(fovy_degrees)) / 2)
    sx, sy = IMAGE[1] / width, IMAGE[0] / height
    cx, cy = (width / 2) * sx - 0.5, (height / 2) * sy - 0.5
    return np.array([[focal * sx, 0.0, cx], [0.0, focal * sy, cy], [0.0, 0.0, 1.0]])


def depth_millimetres(metres):
    """Upright 256x256 int16 optical-axis millimetres from a bottom-up 480x640 metric map."""
    metres = np.asarray(metres, dtype=np.float64)
    if metres.ndim == 3:
        metres = metres[..., 0]
    if metres.shape != RENDER:
        raise ValueError(f"Unexpected depth render {metres.shape}")
    metres = np.flipud(metres)[DEPTH_ROWS][:, DEPTH_COLS]
    valid = np.isfinite(metres) & (metres > 0) & (np.rint(metres * 1000) <= np.iinfo(np.int16).max)
    return np.where(valid, np.rint(metres * 1000), 0).astype(np.int16)


def extrinsic(position, xmat):
    """3x4 world-to-camera [R | t] in OpenCV axes from MuJoCo ``cam_xpos``/``cam_xmat``."""
    rotation = (np.asarray(xmat, dtype=np.float64).reshape(3, 3) @ GL_TO_CV).T
    return np.hstack([rotation, -(rotation @ np.asarray(position, dtype=np.float64))[:, None]])


class Task1Scorer:
    """``eval_common.run_episode_with_stages`` (sequential, stateless) per policy step."""

    def __init__(self, task_id, modules, env):
        self.task_id = task_id
        self.checks = list(modules["eval_task1_only"].STAGE_CHECKS)
        self.done = {name: False for name, _ in self.checks}
        self.goal_reached_t = None
        self.failure = None

    def update(self, env, obs, t, done):
        for i, (name, check) in enumerate(self.checks):
            if self.done[name]:
                continue
            if all(self.done[n] for n, _ in self.checks[:i]) and check(env):
                self.done[name] = True
        if bool(self.done) and all(self.done.values()) and self.goal_reached_t is None:
            self.goal_reached_t = t
        return self.goal_reached_t is not None or bool(done)

    def result(self):
        tsr = bool(self.done) and all(self.done.values())
        score = 100.0 * sum(self.done.values()) / len(self.checks)
        return {
            "tsr": tsr,
            "csr": float(score),
            "stages": dict(self.done),
            "counted_stages": list(self.done),
            "failure_reason": None if tsr else "incomplete_stage",
        }


class StatefulScorer:
    """``eval_tasks2_26.run_episode_with_stateful_stages`` per policy step."""

    def __init__(self, task_id, modules, env):
        rs = self.rs = modules["task2_26_reference_stage"]
        self.task_id = task_id
        self.specs = rs._task_specs(task_id)
        self.done = {spec.name: False for spec in self.specs}
        self.index = 0
        self.counting = rs._is_counting_pour_task(task_id)
        self.extra_check = rs._extra_pour_check(task_id)
        # Captured after the settling steps, exactly as the official loop does.
        self.state = rs._build_initial_state(env)
        self.stage_start = self.state["step_idx"]
        self.goal_reached_t = None
        self.monitor_start = self.deadline = None
        self.extra_pour = False
        self.pour_1 = self.pour_2 = None
        self.last_t = -1

    def update(self, env, obs, t, done):
        rs = self.rs
        rs._update_state(obs, self.state)
        self.last_t = t
        if self.index < len(self.specs):
            spec = self.specs[self.index]
            if spec.check_fn(env, self.state, self.stage_start):
                self.done[spec.name] = True
                if spec.name.endswith("_Pour_One"):
                    self.pour_1 = t
                elif spec.name.endswith("_Pour_Two"):
                    self.pour_2 = t
                    if self.counting and FAIL_ON_EXTRA_POUR:
                        self.monitor_start = int(self.state["step_idx"])
                        self.deadline = t + EXTRA_POUR_MONITOR_STEPS
                self.index += 1
                self.stage_start = self.state["step_idx"]
        if (
            self.counting
            and FAIL_ON_EXTRA_POUR
            and self.extra_check is not None
            and self.monitor_start is not None
            and self.deadline is not None
            and self.pour_2 is not None
            and self.pour_2 < t
            and (self.task_id == 22 or t <= self.deadline)
            and self.extra_check(env, self.state, self.monitor_start)
        ):
            self.extra_pour = True
        required = rs._stage_success_from_stage_done(self.task_id, self.done)
        if not self.counting and self.goal_reached_t is None and required:
            self.goal_reached_t = t
        monitor = not FAIL_ON_EXTRA_POUR or (self.deadline is not None and t >= self.deadline)
        if done:
            return True
        if not self.counting:
            # The official loop breaks POST_GOAL_STEPS later. Verified in the pinned
            # code: stage_done entries are only ever set True, the goal monitor and
            # goal-override arguments are unused by the loop, extra-pour logic is
            # gated on counting tasks, and exceptions keep stage_done. Those steps can
            # only latch the optional final stage, which _counted_stage_names excludes,
            # so TSR and CSR are final at this step.
            return self.goal_reached_t is not None
        return self.extra_pour or (required and monitor)

    def result(self):
        rs = self.rs
        required = rs._stage_success_from_stage_done(self.task_id, self.done)
        monitor = not FAIL_ON_EXTRA_POUR or (
            self.deadline is not None and self.last_t >= self.deadline
        )
        tsr = required and (not self.counting or (monitor and not self.extra_pour))
        if self.extra_pour:
            reason = "extra_pour"
        elif not required:
            reason = "incomplete_stage"
        elif self.counting and not monitor:
            reason = "monitor_incomplete"
        else:
            reason = None
        value = {
            "tsr": bool(tsr),
            "csr": float(rs._stage_score_pct(self.task_id, self.done)),
            "stages": dict(self.done),
            "counted_stages": rs._counted_stage_names(self.task_id, self.done),
            "failure_reason": reason,
        }
        if self.counting:
            # Official loop counter t includes the settling steps; report policy steps.
            value.update(
                extra_pour_detected=bool(self.extra_pour),
                pour_1_policy_step=None if self.pour_1 is None else self.pour_1 - SETTLE_STEPS + 1,
                pour_2_policy_step=None if self.pour_2 is None else self.pour_2 - SETTLE_STEPS + 1,
            )
        return value


class NativeEnvironment:
    def __init__(self, adapter, env, modules, task, seed, config_dir, depth=False):
        self.adapter, self.env, self.modules, self.depth = adapter, env, modules, bool(depth)
        self.task, self.task_id, self.seed = task, task_id(task), int(seed)
        self.config_dir = config_dir
        self.scorer = None
        self.steps = 0
        self.finished = False
        model = env.sim.model
        self.camera_ids = {
            key: model.camera_name2id(name) for key, name in CAMERAS.items()
        }
        self.K = {
            key: intrinsics(model.cam_fovy[cid]) for key, cid in self.camera_ids.items()
        }

    def frame(self, obs, step):
        process = self.modules["policy_adapter"]._process_image_match_eval26
        quat2axisangle = self.modules["transform_utils"].quat2axisangle
        data = self.env.sim.data
        row = {"phase": "exec", "step": int(step)}
        for key, native in NATIVE_IMAGES.items():
            image = np.ascontiguousarray(process(np.asarray(obs[native]), IMAGE[0]))
            if image.shape != IMAGE or image.dtype != np.uint8:
                raise ValueError(f"Unexpected processed {key} image {image.shape} {image.dtype}")
            cid = self.camera_ids[key]
            row[key + "_rgb"] = image
            row[key + "_K"] = self.K[key].copy()
            row[key + "_extrinsic"] = extrinsic(data.cam_xpos[cid], data.cam_xmat[cid])
            if self.depth:
                metric = self.modules["camera_utils"].get_real_depth_map(self.env.sim, obs[NATIVE_DEPTHS[key]])
                row[key + "_depth"] = depth_millimetres(metric)
        row["tcp"] = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                np.asarray(quat2axisangle(np.array(obs["robot0_eef_quat"], dtype=np.float64)), dtype=np.float64),
            ]
        )
        row["joints"] = np.array(obs["robot0_joint_pos"], dtype=np.float64)
        row["fingers"] = np.array(obs["robot0_gripper_qpos"], dtype=np.float64)
        return row

    def restart(self, seed):
        """Host-side demonstration replay only: reuse this environment for another seed.

        Equivalent to a fresh environment (verified: identical first frame and state),
        as in the official sweep, which reuses one environment for all trials.
        """
        self.seed, self.scorer, self.steps, self.finished = int(seed), None, 0, False
        return self.reset()

    def reset(self):
        # Nothing may touch the global NumPy RNG between seeding and reset.
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        obs = self.env.reset()
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = self.env.step(list(SETTLE_ACTION))
        scorer = Task1Scorer if self.task_id == 1 else StatefulScorer
        self.scorer = scorer(self.task_id, self.modules, self.env)
        return [self.frame(obs, 0)]

    def step(self, values):
        if self.finished or self.scorer is None:
            raise ValueError("Episode is not running")
        self.steps += 1
        t = SETTLE_STEPS + self.steps - 1  # official loop counter
        try:
            obs, _, done, _ = self.env.step([float(v) for v in values])
            decided = self.scorer.update(self.env, obs, t, done)
        except Exception as exc:  # official loop: logged, episode scored as failure
            self.finished = True
            metrics = {**self.scorer.result(), "tsr": False, "policy_steps": self.steps}
            metrics["failure_reason"] = "native_exception"
            return StepResult(
                [],
                "error",
                self.steps,
                private_error={"exception_type": type(exc).__name__, "error_message": str(exc)[:2000]},
                metrics=metrics,
            )
        frames = [self.frame(obs, self.steps)]  # a CaM observation failure is a harness error
        metrics = {**self.scorer.result(), "policy_steps": self.steps}
        if decided:
            status = "success" if metrics["tsr"] else "fail"
        elif self.steps >= MAX_STEPS:
            status = "success" if metrics["tsr"] else "timeout"
        else:
            status = "running"
        self.finished = status != "running"
        return StepResult(frames, status, self.steps, metrics=metrics)

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            shutil.rmtree(self.config_dir, ignore_errors=True)


class RoboMemArena:
    name = "robomemarena"
    package = "robomemarena"
    guide = "robomemarena/guide.md"
    describe_interface = True
    supports_builtin_demo = False
    #: Root-relative ("/") excludes; LIBERO assets are linked, not copied.
    pin_ignore = (
        "docs/",
        "tests/",
        "reference_evaluation/",
        "async_vlm26_reference/",
        "openpi_minimal_runtime/",
        "libero_fork/libero/assets",
        "libero_fork/libero/init_files",
        "libero_fork/libero/bddl_files",
        "libero_fork/libero/datasets",
    )

    def __init__(self, source=None):
        self.source = Path(source).resolve() if source else default_source()
        if not (self.source / "scripts/task2_26_reference_stage.py").is_file():
            raise FileNotFoundError(f"RoboMemArena evaluation_benchmark missing: {self.source}")

    @property
    def asset_links(self):
        return {"libero_fork/libero/assets": self.source / "libero_fork/libero/assets"}

    def tasks(self):
        return [TASKS[tid] for tid in sorted(TASKS)]

    def metadata(self, task, split):
        task_id(task)
        if split not in ("train", "test"):
            raise ValueError("Choose train or test")
        value = read(METADATA / split / f"record_dataset_{task}_metadata.json")
        if value.get("env_id") != task or int(value.get("task_id", -1)) != task_id(task):
            raise ValueError(f"Pinned metadata does not describe {task}")
        return value

    def records(self, task, split):
        records = self.metadata(task, split)["records"]
        ids = [int(row["episode"]) for row in records]
        seeds = [int(row["seed"]) for row in records]
        if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate episode or seed in metadata")
        return records

    def action_spec(self, task):
        task_id(task)
        return action(
            ("dx", "dy", "dz", "drx", "dry", "drz", "gripper"),
            "One native 20 Hz OSC_POSE delta step, each component normalized to [-1, 1]: "
            "dx..dz offset the end-effector position target by up to 0.05 m and drx..drz "
            "rotate the orientation target by an axis-angle of up to 0.5 rad, both in the "
            "world frame; gripper uses only its sign: +1 (>0) closes, -1 (<0) opens, 0 holds.",
            low=[-1.0] * 7,
            high=[1.0] * 7,
            type="osc_pose_delta_normalized",
            gripper=True,
            control_frequency=20,
            native_max_steps_argument=MAX_STEPS,
            settling_steps=SETTLE_STEPS,
        )

    def interface(self, task, depth):
        task_id(task)
        shape = "x".join(map(str, IMAGE))
        sensors = {}
        for key, label in (("agentview", "front agentview"), ("wrist", "eye-in-hand wrist")):
            sensors[key + "_rgb"] = sensor(
                "uint8",
                IMAGE,
                f"{label} RGB {shape}, upright (row 0 = top), the official policy image",
            )
        for key in CAMERAS:
            sensors[key + "_K"] = sensor(
                "float64", (3, 3), f"{key} 3x3 intrinsics of the {IMAGE[0]}x{IMAGE[1]} image (OpenCV pixel centres)"
            )
        for key in CAMERAS:
            sensors[key + "_extrinsic"] = sensor(
                "float64", (3, 4), f"{key} 3x4 world-to-camera matrix [R t], OpenCV axes (x right, y down, z forward), metres"
            )
        sensors["tcp"] = sensor(
            "float64",
            (6,),
            "measured end-effector XYZ metres (world) + robosuite quat2axisangle(eef_quat xyzw) radians; "
            "angle not wrapped (norm can exceed pi)",
        )
        sensors["joints"] = sensor("float64", (7,), "Panda arm joint positions, radians")
        sensors["fingers"] = sensor(
            "float64", (2,), "Panda finger joint positions, metres (about +/-0.04 open, 0 closed)"
        )
        if depth:
            for key in CAMERAS:
                sensors[key + "_depth"] = sensor(
                    "int16",
                    IMAGE[:2],
                    f"{key} depth, int16 optical-axis millimetres, 0 unknown",
                    aligned_with=key + "_rgb",
                )
        return {
            "sensors": sensors,
            "action": self.action_spec(task),
            "video": ["agentview_rgb", "wrist_rgb"],
            "starter": "robot.step(0, 0, 0, 0, 0, 0, -1)",
        }

    def sensor_summary(self, task, depth):
        task_id(task)
        cameras = "RGB-D" if depth else "RGB (no depth)"
        return (
            f"{cameras} agentview and wrist cameras 256x256, camera calibration, "
            "end-effector pose, arm joints and finger positions"
        )

    def select_training(self, extracted_records, count):
        records = sorted(extracted_records, key=lambda row: int(row["episode"]))
        if count is None:
            return records
        count = int(count)
        if not 1 <= count <= len(records):
            raise ValueError(f"Cannot select {count} of {len(records)} training episodes")
        return records[:count]

    def begin(self, env, *, builtin_demo, depth):
        if builtin_demo:
            raise ValueError("RoboMemArena has no task-provided video; set builtin_demo false")
        if bool(depth) != env.depth:
            raise ValueError("Environment and interface depth settings differ")
        frames = env.reset()
        ec = env.modules["eval_common"]
        tid = env.task_id
        goal = ec.get_prompt(f"task{tid}", bddl_file(self.source, tid).stem)
        if goal != task_prompts(self.source)[f"task{tid}"]:
            raise ValueError("Official prompt resolution changed")
        return EpisodeStart(goal, frames, {"seed_private": True})

    def advance(self, env, action, context, *, depth):
        return env.step(action)

    def _import_pinned(self):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        sys.dont_write_bytecode = True  # never write into the pinned snapshot
        for name in PINNED_MODULES:
            module = sys.modules.get(name)
            if module is not None and not Path(module.__file__).resolve().is_relative_to(self.source):
                raise ValueError(f"A different {name} is already imported; use an episode subprocess")
        for path in (self.source / "libero_fork", self.source / "scripts"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        modules = {
            name: importlib.import_module(name)
            for name in ("eval_common", "policy_adapter", "task2_26_reference_stage", "eval_task1_only")
        }
        modules["libero_envs"] = importlib.import_module("libero.libero.envs")
        for name, module in modules.items():
            if not Path(module.__file__).resolve().is_relative_to(self.source):
                raise ValueError(f"{name} was not imported from the pinned source")
        modules["libero"] = sys.modules["libero"]
        modules["transform_utils"] = importlib.import_module("robosuite.utils.transform_utils")
        modules["camera_utils"] = importlib.import_module("robosuite.utils.camera_utils")
        return modules

    def make_environment(self, metadata, task, episode, split, depth):
        tid = task_id(task)
        pinned = self.metadata(task, split)
        expected = {int(row["episode"]): row for row in pinned["records"]}
        supplied = {
            int(row["episode"]): row
            for row in read(Path(metadata) / f"record_dataset_{task}_metadata.json")["records"]
        }
        if episode not in supplied or episode not in expected:
            raise ValueError("Episode is outside the pinned split")
        if int(supplied[episode]["seed"]) != int(expected[episode]["seed"]):
            raise ValueError("Run metadata seed disagrees with the pinned metadata")
        seed = int(expected[episode]["seed"])
        bddl = bddl_file(self.source, tid)
        if file_sha256(bddl) != pinned["bddl_sha256"]:
            raise ValueError(f"BDDL changed: {bddl}")
        config_dir = tempfile.mkdtemp(prefix="cam-libero-")
        try:
            libero_root = self.source / "libero_fork/libero"
            import yaml  # simulator environment only

            with open(os.path.join(config_dir, "config.yaml"), "w") as handle:
                yaml.dump(
                    {
                        "benchmark_root": str(libero_root),
                        "bddl_files": str(libero_root / "bddl_files"),
                        "init_states": str(libero_root / "init_files"),
                        "datasets": str(libero_root.parent / "datasets"),
                        "assets": str(libero_root / "assets"),
                    },
                    handle,
                )
            # Without a config file libero/__init__.py blocks on input().
            os.environ["LIBERO_CONFIG_PATH"] = config_dir
            modules = self._import_pinned()
            env_class = modules["eval_common"]._get_env_class()
            # camera_depths only adds depth renders; physics, RNG and scoring are unchanged.
            from robosuite.utils.errors import RandomizationError

            env = construct_environment(env_class,
                dict(bddl_file_name=str(bddl), camera_depths=bool(depth), **ENV_KWARGS),
                RandomizationError)
            return NativeEnvironment(self, env, modules, task, seed, config_dir, depth)
        except BaseException:
            shutil.rmtree(config_dir, ignore_errors=True)
            raise
