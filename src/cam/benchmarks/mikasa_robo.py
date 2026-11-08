"""MIKASA-Robo-VLA adapter: the official 90-task single-env evaluation protocol.

The native environment is built exactly as ``mikasa_robo_suite.vla.benchmarking``
does (``num_envs=1``, ``obs_mode="rgb"``, ``control_mode="pd_ee_delta_pose"``,
``reward_mode="normalized_dense"``, ``render_mode="all"``, GPU simulation, then
``apply_mikasa_vla_wrappers(include_overlays=False)``) and reset with the pinned
seed. Only the two cameras and ``obs["proprio"]`` cross the policy boundary;
``task_cue``, ``oracle_info``, rewards and every ``info`` field except the
language instruction stay private. The episode ends at the first native
``info["success"]`` (equivalent to the official ``success_once`` latch), on
native termination (``fail``) or at the registered step limit (``timeout``).
"""

import ast
from functools import lru_cache
from typing import NamedTuple
import importlib
import inspect
import os
from pathlib import Path
import re
import sys

import numpy as np

from cam.io import read, task_name
from .base import EpisodeStart, StepResult, action, sensor

METADATA = Path(__file__).resolve().parent / "metadata" / "mikasa_robo"
TEST_START_SEED = 4242424242  # mikasa_robo_suite/vla/benchmarking.py START_SEED
TEST_EPISODES = 50  # NUM_EPISODES_PER_TASK
IMAGE = (128, 128, 3)
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
    return (root / "data/raw/mikasa_robo/maniskill_data") if root else None


def dataset_name(task):
    """Release folder name; mirrors mikasa_robo_suite/vla/utils/dataset_naming.py."""

    def token(value):
        value = re.sub(r"([a-z])([A-Z])", r"\1_\2", value)
        value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
        value = re.sub(r"(?<=[A-Za-z])([A-Za-z])([0-9])", r"\1_\2", value)
        return value.lower()

    return "_".join(token(part) for part in task.split("-") if part)


#: Docstring sections that state the task contract; "How to customize" etc. are dropped.
CONTRACT = ("Episode flow", "Success", "Failure")
HEADER = re.compile(r"^([A-Z][A-Za-z ]{0,30}?)(?: \([^)]*\))?:(.*)$")
ITEM = re.compile(r"^(?:- |\d+\. |\s)")
PARAMETER = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*\b")


def docstring_sections(doc):
    """[(label, lines)] for each headed section (header line included), in order."""
    sections, current = [], None
    for line in inspect.cleandoc(doc or "").splitlines():
        match = HEADER.match(line)
        if match:
            current = (match.group(1), [line])
            sections.append(current)
        elif current is not None:
            if line.strip() and not ITEM.match(line):
                current = None  # a free paragraph after the list ends the section
            else:
                current[1].append(line)
    return [(label, [l for l in lines]) for label, lines in sections]


def contract_sections(doc, parameters=()):
    """Verbatim Episode flow / Success / Failure sections that name no class parameter."""
    kept = []
    for label, lines in docstring_sections(doc):
        if not label.startswith(CONTRACT):
            continue
        text = "\n".join(lines).rstrip()
        if set(PARAMETER.findall(text)) & set(parameters):
            continue
        kept.append("\n".join(l for i, l in enumerate(text.splitlines()) if l.strip() or i == 0))
    return "\n\n".join(kept)


class EnvClass(NamedTuple):
    """One MIKASA-Robo env class parsed from source (no simulator import)."""

    name: str
    doc: str | None
    names: frozenset  # class-level attribute names
    values: dict  # class-level attribute -> AST expression
    methods: dict  # method name -> exact source text


@lru_cache(maxsize=4)
def _task_classes(source):
    classes, registered = {}, {}
    for path in sorted((Path(source) / "mikasa_robo_suite/vla/memory_envs").glob("*.py")):
        text = path.read_text()
        for node in ast.parse(text).body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", None) for b in node.bases]
            values, names, methods = {}, set(), {}
            for item in node.body:
                if isinstance(item, (ast.Assign, ast.AnnAssign)):
                    for target in item.targets if isinstance(item, ast.Assign) else [item.target]:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
                            if item.value is not None:
                                values[target.id] = item.value
                elif isinstance(item, ast.FunctionDef):
                    methods[item.name] = ast.get_source_segment(text, item)
            if node.name in classes:
                raise ValueError(f"Ambiguous MIKASA-Robo class name: {node.name}")
            classes[node.name] = (
                EnvClass(node.name, ast.get_docstring(node, clean=False), frozenset(names), values, methods),
                bases,
            )
            for deco in node.decorator_list:
                if (
                    isinstance(deco, ast.Call)
                    and getattr(deco.func, "id", None) == "register_env"
                    and deco.args
                    and isinstance(deco.args[0], ast.Constant)
                ):
                    registered[deco.args[0].value] = node.name
    chains = {}
    for task, name in registered.items():
        chain = []
        while name in classes and name not in [c.name for c in chain]:
            chain.append(classes[name][0])
            name = next((b for b in classes[name][1] if b in classes), None)
        chains[task] = tuple(chain)
    return chains


def task_classes(source):
    """Registered env ID -> class chain (leaf first) by AST, without importing the simulator."""
    return _task_classes(str(Path(source).resolve()))


def task_contract(chain):
    """Contract text from the first class in the chain whose docstring has contract sections."""
    parameters = set().union(*(c.names for c in chain)) if chain else set()
    for env_class in chain:
        if any(label.startswith(CONTRACT) for label, _ in docstring_sections(env_class.doc)):
            return contract_sections(env_class.doc, parameters)
    return ""


def constant(chain, expr, seen=()):
    """Numeric value of a class-constant expression, resolved leaf-first like Python."""
    if isinstance(expr, str):
        expr = ast.parse(expr, mode="eval").body
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
        return expr.value
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        return -constant(chain, expr.operand, seen)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = constant(chain, expr.left, seen), constant(chain, expr.right, seen)
        return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b if b else None}[type(expr.op)]
    if (
        isinstance(expr, ast.Call)
        and getattr(expr.func, "id", None) == "int"
        and len(expr.args) == 1
        and not expr.keywords
    ):
        return int(constant(chain, expr.args[0], seen))
    if isinstance(expr, ast.Name) and expr.id not in seen:
        owner = next((c for c in chain if expr.id in c.values), None)
        if owner is not None:
            return constant(chain, owner.values[expr.id], (*seen, expr.id))
    raise ValueError(f"Unsupported constant expression: {ast.dump(expr)}")


def number(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def evaluation_digest(chain, owner):
    """SHA-256 of every evaluate() in the chain plus the first step() and __init__ overrides.

    ``owner`` must define evaluate(); parents it may call through super() are included.
    """
    import hashlib

    if "evaluate" not in next(c for c in chain if c.name == owner).methods:
        raise ValueError(f"{owner} does not define evaluate()")
    parts = [c.methods["evaluate"] for c in chain if "evaluate" in c.methods]
    for name in ("step", "__init__"):
        parts.append(next((c.methods[name] for c in chain if name in c.methods), ""))
    return hashlib.sha256("\n#--\n".join(parts).encode()).hexdigest()


def success_test(chain, tests):
    """Pinned physical success/failure summary for the class that defines evaluate()."""
    owner = next((c.name for c in chain if "evaluate" in c.methods), None)
    entry = tests.get(owner) if owner else None
    if entry is None:
        return ""
    if entry["sha256"] != evaluation_digest(chain, owner):
        raise ValueError(f"MIKASA-Robo evaluate()/step() changed for {owner}; re-derive its success test")
    return re.sub(r"\{([^{}]+)\}", lambda m: number(constant(chain, m.group(1))), entry["text"])


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


def frame(obs, step):
    """Rebuild one allowlisted frame; never forward raw obs or info."""
    keys = set(obs)
    if not NATIVE_KEYS <= keys or keys - NATIVE_KEYS - PRIVATE_KEYS:
        raise ValueError(f"Unexpected native observation keys: {sorted(keys)}")
    rgb, proprio = to_numpy(obs["rgb"]), to_numpy(obs["proprio"])
    if rgb.shape != (1, *IMAGE[:2], 6) or rgb.dtype != np.uint8:
        raise ValueError(f"Expected (1,128,128,6) uint8 rgb, got {rgb.shape} {rgb.dtype}")
    if proprio.shape != (1, 7) or proprio.dtype != np.float32:
        raise ValueError(f"Expected (1,7) float32 proprio, got {proprio.shape} {proprio.dtype}")
    return {
        "phase": "exec",
        "step": int(step),
        "base_rgb": np.ascontiguousarray(rgb[0, :, :, :3]),
        "wrist_rgb": np.ascontiguousarray(rgb[0, :, :, 3:]),
        "tcp": np.ascontiguousarray(proprio[0, :6]),
        "gripper_opening": np.ascontiguousarray(proprio[0, 6:]),
    }


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


class MikasaRobo:
    name = "mikasa_robo"
    package = "mikasa_robo"
    guide = "mikasa_robo.md"
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
        from .mikasa_robo_api import PREVIEW_SPEC

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
        from .mikasa_robo_api.preview import preview

        return preview(env.env.unwrapped, values, mode)

    def interface(self, task, depth):
        if depth:
            raise ValueError("The pinned MIKASA-Robo RLDS release has no depth; use depth=false")
        return {
            "sensors": {
                "base_rgb": sensor("uint8", IMAGE, "fixed base (scene) camera RGB"),
                "wrist_rgb": sensor("uint8", IMAGE, "wrist (hand) camera RGB"),
                "tcp": sensor(
                    "float32",
                    (6,),
                    "measured TCP XYZ metres in the scene frame and roll/pitch/yaw radians "
                    "(R = Rz(yaw) Ry(pitch) Rx(roll))",
                ),
                "gripper_opening": sensor(
                    "float32", (1,), "sum of both finger positions, metres: 0 closed, 0.08 open"
                ),
            },
            "action": self.action_spec(task),
            "video": ["base_rgb", "wrist_rgb"],
            "starter": "robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)",
        }

    def diagnostic_task_contract(self, task):
        """Host-only historical diagnostic; never included in learner prompts."""
        if task not in self.tasks():
            raise ValueError(f"Unknown MIKASA-Robo task: {task}")
        chain = task_classes(self.source).get(task, ())
        contract = task_contract(chain)
        test = success_test(chain, read(METADATA / "success_tests.json")["classes"])
        if not contract and not test:
            return None
        parts = ["## Task contract"]
        if contract:
            parts.append("From the official MIKASA-Robo environment description of this task:\n\n" + contract)
        if test:
            parts.append(test)
        return "\n\n".join(parts) + "\n"

    def sensor_summary(self, task, depth):
        return "base and wrist RGB (128x128), TCP pose and gripper opening; no depth"

    def begin(self, env, *, builtin_demo, depth):
        if builtin_demo:
            raise ValueError("MIKASA-Robo has no reset video; set builtin_demo=false")
        obs, info = env.reset()
        return EpisodeStart(instruction(info), [frame(obs, 0)], {})

    def advance(self, env, action, context, *, depth):
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
        return StepResult([frame(obs, env.steps)], status, env.steps)

    def select_training(self, extracted_records, count):
        records = sorted(extracted_records, key=lambda row: int(row["episode"]))
        return records if count is None else records[: int(count)]

    def make_environment(self, metadata, task, episode, split, depth):
        if depth:
            raise ValueError("The pinned MIKASA-Robo RLDS release has no depth")
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
                raise ValueError(
                    f"A different {module} package is already imported; use an episode subprocess"
                )
        import gymnasium as gym

        importlib.import_module("mikasa_robo_suite.vla.memory_envs")  # registers env IDs
        wrappers = importlib.import_module("mikasa_robo_suite.vla.utils.apply_wrappers")
        if not Path(wrappers.__file__).resolve().is_relative_to(self.source):
            raise ValueError("Canonical wrappers were not loaded from the pinned source")
        env = wrappers.apply_mikasa_vla_wrappers(
            gym.make(task, **MAKE_KWARGS), include_overlays=False
        )
        if int(env.max_episode_steps) != horizon:
            env.close()
            raise ValueError("Registered step limit disagrees with pinned metadata")
        return NativeEnvironment(env, seed, horizon, env.unwrapped.device)
