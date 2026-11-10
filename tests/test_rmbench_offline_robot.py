"""Robot-model numerics, native frame parity, and simulator-free isolation."""

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cam.interface.rmbench.api import RobotModel, dry_run, safety_check
from cam.interface.rmbench.api.offline import _pose
from cam.harness.prepare import make_runtime


ROOT = Path(__file__).resolve().parents[1]
Q = np.array([.2, -.7, 1.1, -.5, .3, .6])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_joint_solution_matches_target_pose_and_shared_safety_core(arm):
    model = RobotModel(arm)
    target = model.forward_kinematics(Q)
    joints = Q + .08
    state = {"joints": joints, "joint_source": "commanded"}
    before = joints.copy()
    result = dry_run(state, target, arm=arm)
    assert result["ik_found"] and result["status"] == "ik_found"
    assert result["joint_source"] == "commanded"
    predicted = _pose(model.forward_kinematics(result["joint_solution"]))
    desired = _pose(target)
    assert np.linalg.norm(predicted[:3, 3] - desired[:3, 3]) <= .002
    assert Rotation.from_matrix(predicted[:3, :3] @ desired[:3, :3].T).magnitude() <= .02
    checks = safety_check(state, target, arm=arm)
    assert checks["dry_run"] == result
    assert checks["target"]["within_limits"]
    assert np.array_equal(joints, before)
    assert checks["supplied_torque"]["status"] == "unavailable"
    assert checks["supplied_velocity"]["status"] == "unavailable"
    assert checks["native_plan_status"] == "not_checked"
    for key in ("path_checked", "collision_checked", "dynamics_checked", "simulator_used"):
        assert result[key] is False and checks[key] is False
    assert "safe" not in result and "safe" not in checks
    json.dumps(checks, allow_nan=False)


def test_impossible_pose_is_search_failure_not_infeasibility_claim():
    result = safety_check({"joints": Q}, [10, 10, 10, 1, 0, 0, 0], arm="left", max_nfev=15)
    assert result["target"] is None
    preview = result["dry_run"]
    assert preview["status"] == "ik_not_found" and preview["joint_solution"] is None
    assert "not proof" in preview["message"]
    assert preview["candidate"]["within_limits"]
    json.dumps(result, allow_nan=False)


def test_current_limits_and_supplied_torque_velocity_are_not_future_safety():
    result = safety_check({
        "joints": [11, -11, 0, 0, 0, 0], "joint_target": Q,
        "joint_torques": [101, -102, 0, 0, 0, 0],
        "joint_velocities": [0, 0, 1001, 0, 0, 0],
    }, arm="right")
    assert result["current"]["violations"] == ["fr_joint1", "fr_joint2"]
    assert result["current"]["margins_rad"][:2] == [-1, -1]
    assert result["supplied_torque"]["violations"] == ["fr_joint1", "fr_joint2"]
    assert result["supplied_velocity"]["violations"] == ["fr_joint3"]
    assert result["target"]["within_limits"]
    assert result["dry_run"] is None and not result["actual_future_torque_checked"]
    assert "not manufacturer" in result["limit_reference"]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_geometric_jacobian_matches_independent_finite_differences(arm):
    model = RobotModel(arm)
    rng = np.random.default_rng(4)
    for q in rng.uniform(-2, 2, (4, 6)):
        transform, jacobian = model._kinematics(q)
        for index in range(6):
            offset = q.copy()
            offset[index] += 1e-7
            moved, _ = model._kinematics(offset)
            derivative = np.r_[(moved[:3, 3] - transform[:3, 3]) / 1e-7,
                                Rotation.from_matrix(moved[:3, :3] @ transform[:3, :3].T).as_rotvec() / 1e-7]
            assert np.allclose(derivative, jacobian[:, index], atol=1e-7)


def test_quaternion_sign_and_scale_do_not_change_target():
    target = RobotModel().forward_kinematics(Q)
    a = dry_run({"joints": Q}, target, arm="left")
    target[3:] = [-3 * x for x in target[3:]]
    b = dry_run({"joints": Q}, target, arm="left")
    assert a["ik_found"] and b["ik_found"]
    assert b["joint_solution"] == pytest.approx(a["joint_solution"])


@pytest.mark.parametrize("state,target,options", [
    ({"joints": [0]*5}, [0, 0, 0, 1, 0, 0, 0], {}),
    ({"joints": [np.nan]*6}, [0, 0, 0, 1, 0, 0, 0], {}),
    ({"joints": Q, "objects": []}, [0, 0, 0, 1, 0, 0, 0], {}),
    ({"joints": Q, "joint_source": "hidden"}, [0, 0, 0, 1, 0, 0, 0], {}),
    ({"joints": Q}, [0]*7, {}),
    ({"joints": Q}, [0]*6, {}),
    ({"joints": Q}, [0, 0, np.inf, 1, 0, 0, 0], {}),
    ({"joints": Q, "joint_target": Q}, [0, 0, 0, 1, 0, 0, 0], {}),
    ({"joints": Q}, [0, 0, 0, 1, 0, 0, 0], {"max_attempts": 0}),
    ({"joints": Q}, [0, 0, 0, 1, 0, 0, 0], {"max_nfev": True}),
    ({"joints": Q}, [0, 0, 0, 1, 0, 0, 0], {"rotation_tolerance_rad": float("nan")}),
])
def test_invalid_inputs_rejected(state, target, options):
    with pytest.raises(ValueError):
        dry_run(state, target, arm="left", **options)


def test_export_is_reproducible_and_contains_only_robot_data():
    pytest.importorskip("yaml")
    assets = ROOT / "envs/rmbench/assets/embodiments/aloha-agilex"
    if not (assets / "config.yml").exists():
        pytest.skip("Optional native robot assets not installed")
    spec = importlib.util.spec_from_file_location("export_rmbench_model", ROOT / "scripts/export_rmbench_robot_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.export_model(assets, ROOT / "third_party/rmbench")
    actual = json.loads((ROOT / "src/cam/interface/rmbench/api/aloha_model.json").read_text())
    assert actual == expected
    assert set(actual) == {"schema", "robot", "setting", "world_root_pose_wxyz", "gripper_bias_m",
                           "action_offset_m", "delta_matrix", "limit_reference", "arms", "provenance"}
    assert "/home/" not in json.dumps(actual)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_target_transform_matches_native_source_without_importing_simulator(arm):
    """Execute only native robot pose arithmetic, stopping before the planner call."""
    folder = ROOT / "third_party/rmbench/envs/robot"
    if not (folder / "planner.py").exists():
        pytest.skip("Optional RMBench source not installed")

    class Pose:
        def __init__(self, p, q):
            self.p, self.q = np.asarray(p), np.asarray(q)

    def compose(p, r, scale):
        t = np.eye(4)
        t[:3, 3], t[:3, :3] = p, r @ np.diag(scale)
        return t

    t3d = SimpleNamespace(
        quaternions=SimpleNamespace(
            quat2mat=lambda q: Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix(),
            mat2quat=lambda r: Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]],
        ),
        affines=SimpleNamespace(compose=compose),
        axangles=SimpleNamespace(axangle2mat=lambda axis, angle: Rotation.from_rotvec(np.asarray(axis)*angle).as_matrix()),
    )
    namespace = {"np": np, "t3d": t3d, "deepcopy": deepcopy, "sapien": SimpleNamespace(Pose=Pose)}
    functions = []
    for filename, names in (
        ("robot.py", ["_trans_from_gripper_to_endlink"]),
        ("planner.py", ["_trans_from_world_to_base", "plan_path"]),
    ):
        tree = ast.parse((folder / filename).read_text())
        if filename == "planner.py":
            tree = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CuroboPlanner")
        for name in names:
            function = deepcopy(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name))
            if name == "plan_path":
                stop = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "goal_pose_of_ee")
                function.body = function.body[:stop] + [ast.parse("return target_pose_p, target_pose_q").body[0]]
            functions.append(function)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "native_robot_math", "exec"), namespace)
    model_json = json.loads((ROOT / "src/cam/interface/rmbench/api/aloha_model.json").read_text())
    root_pose = model_json["world_root_pose_wxyz"]
    native = SimpleNamespace(
        yml_path="aloha-agilex/curobo.yml", robot_origion_pose=Pose(root_pose[:3], root_pose[3:]),
        frame_bias=model_json["arms"][arm]["frame_bias_m"],
        left_gripper_bias=.12, right_gripper_bias=.12,
        left_inv_delta_matrix=np.eye(3), right_inv_delta_matrix=np.eye(3),
    )
    native._trans_from_world_to_base = lambda *args: namespace["_trans_from_world_to_base"](native, *args)
    model = RobotModel(arm)
    for angles in ([.1, .3, -.8], [2, -.3, 1.5]):
        quat = Rotation.from_euler("xyz", angles).as_quat()[[3, 0, 1, 2]]
        pose = np.r_[[.15, -.1, .95], quat]
        endlink = namespace["_trans_from_gripper_to_endlink"](native, pose, arm_tag=arm)
        p, q = namespace["plan_path"](native, Q, endlink, arms_tag=arm)
        expected = _pose(np.r_[p, q])
        assert np.allclose(model.to_arm @ _pose(pose) @ model.action_to_link, expected, atol=1e-12)


@pytest.mark.isolation
def test_public_helpers_and_robot_wrappers_work_without_rpc_or_simulator(tmp_path):
    from cam.harness.isolation import run

    runtime = make_runtime(tmp_path / "runtime", rmbench_robot_checks=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code = r'''
from io import BytesIO
import sys, importlib.util, json
from types import SimpleNamespace
from cam.runtime.rmbench import RobotModel, dry_run, safety_check
from cam.runtime.robot import Robot, MotionError
from cam.core.sandbox import scope, check
q = [.2,-.7,1.1,-.5,.3,.6]
state = {"joints": q, "joint_source": "commanded"}
target = RobotModel("left").forward_kinematics(q)
expected = dry_run(state, target, arm="left")
writer = BytesIO()
robot = Robot(SimpleNamespace(left_joints=q), BytesIO(), writer,
              {"offline_robot": {"schema": "cam-rmbench-offline-robot/1"}}, max_steps=2)
assert robot.dry_run(*target, arm="left") == expected
assert robot.dry_run(*target, arm="left", robot_state=state) == expected
assert robot.safety_check(state, target, arm="left") == safety_check(state, target, arm="left")
try:
    robot.dry_run(*target, arm="left")
except MotionError as e:
    assert "limit" in str(e)
else:
    raise AssertionError("query cap not enforced")
assert robot._count == 0 and robot.frames() == [] and not writer.getvalue()
source = 'def act_check(robot, state, target):\n    return robot.safety_check(state, target, arm="left")'
assert check(source) == []
namespace = scope()
exec(source, namespace)
assert namespace["act_check"](robot, state, target)["dry_run"] == expected
assert importlib.util.find_spec("cam.interface") is None
assert not any(x in sys.modules for x in ["sapien", "torch", "curobo", "mani_skill", "gymnasium"])
print(json.dumps(expected, allow_nan=False))
'''
    status, out, err = run(workspace, runtime, ["python", "-c", code], timeout=40)
    assert status == 0, (out, err)
    assert json.loads(out)["simulator_used"] is False
    assert sorted(p.name for p in (runtime / "cam/runtime/rmbench").iterdir()) == [
        "__init__.py", "aloha_model.json", "api.md", "offline.py"]
    other = make_runtime(tmp_path / "other", mikasa_motion_preview=True)
    assert not (other / "cam/runtime/rmbench").exists()
