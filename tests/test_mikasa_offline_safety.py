"""Offline robot math, torque-reference separation, and the public sandbox API."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cam.interface.mikasa_robo.api.offline import _kinematics, safety_check
from cam.harness.prepare import make_runtime

Q = np.array([0, .3, 0, -1.9, 0, 2.2, .7])


def test_missing_torque_information_never_means_safe():
    result = safety_check({"joints": Q}, [0]*6)
    assert result["current"]["within_limits"] and result["predicted_target"]["within_limits"]
    assert result["supplied_torque"]["status"] == "unavailable"
    assert result["estimated_drive_torque"]["status"] == "unavailable"
    assert not result["simulator_used"] and not result["actual_future_torque_checked"]
    json.dumps(result, allow_nan=False)


def test_supplied_effort_and_estimated_drive_caps_remain_distinct():
    qtarget = Q + .2
    result = safety_check({"joints": Q, "joint_target": qtarget, "joint_velocities": [.5]*7,
                           "joint_torques": [0, 0, 0, 0, 11, 13, -12]})
    supplied = result["supplied_torque"]
    assert supplied["limits_nm"] == [87]*4 + [12]*3
    assert supplied["margin_nm"][-3:] == [1, -1, 0]
    assert supplied["exceeds_limits"] == ["panda_joint6"]
    drive = result["estimated_drive_torque"]
    assert drive["limits_nm"] == [100]*7
    assert drive["values_nm"] == pytest.approx([150]*7)
    assert drive["margin_nm"] == pytest.approx([-50]*7)
    assert drive["utilization"] == pytest.approx([1.5]*7)
    assert not drive["within_limits"]


def test_pd_damping_sign_and_position_limit_margin():
    result = safety_check({"joints": Q, "joint_target": Q, "joint_velocities": [-.1]*7})
    assert result["estimated_drive_torque"]["values_nm"] == pytest.approx([10]*7)
    outside = Q.copy()
    outside[3] = 0
    current = safety_check({"joints": outside})["current"]
    assert current["violations"] == ["panda_joint4"]
    assert current["margins_rad"][3] == pytest.approx(-.0698)


def test_geometric_jacobian_matches_forward_kinematic_derivatives():
    pose, jacobian = _kinematics(Q)
    for i in range(7):
        offset = Q.copy()
        offset[i] += 1e-7
        moved, _ = _kinematics(offset)
        derivative = np.r_[(moved[:3, 3] - pose[:3, 3]) / 1e-7,
                            Rotation.from_matrix(moved[:3, :3] @ pose[:3, :3].T).as_rotvec() / 1e-7]
        assert np.allclose(jacobian[:, i], derivative, atol=1e-7)


def test_absolute_world_input_matches_equivalent_returned_delta_command():
    fk, _ = _kinematics(Q)
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("xyz", [.1, .2, .7]).as_matrix()
    base[:3, 3] = [1, 2, .1]
    world = base @ fk
    tcp = np.r_[world[:3, 3], Rotation.from_matrix(world[:3, :3]).as_euler("xyz")]
    state = {"joints": Q, "tcp": tcp}
    same = safety_check(state, tcp, mode="absolute")
    assert same["predicted_target"]["joints_rad"] == pytest.approx(Q)
    target = tcp + [.02, -.01, .01, 0, .03, 0]
    absolute = safety_check(state, target, mode="absolute")
    delta = safety_check(state, absolute["command"])
    assert absolute["predicted_target"] == delta["predicted_target"]
    with pytest.raises(ValueError, match="measured world tcp"):
        safety_check({"joints": Q}, target, mode="absolute")


@pytest.mark.parametrize("state,action", [
    ({"joints": [0]*6}, None), ({"joints": [np.nan]*7}, None),
    ({"joints": Q, "joint_velocities": [0]*6}, None),
    ({"joints": Q, "joint_torques": [np.inf]*7}, None),
    ({"joints": Q}, [0]*8), ({"joints": Q}, [np.nan]*6),
    ({"joints": Q, "joint_target": Q}, [0]*6),
])
def test_invalid_inputs_rejected(state, action):
    with pytest.raises(ValueError):
        safety_check(state, action)


@pytest.mark.isolation
def test_offline_api_runs_in_command_sandbox_without_simulator(tmp_path):
    from cam.harness.isolation import run

    runtime = make_runtime(tmp_path / "runtime", mikasa_motion_preview=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = '''
import json, sys, importlib.util
from cam.runtime.mikasa_robo import safety_check
from cam.runtime.robot import Robot
from io import BytesIO
state = {"joints": [0,.3,0,-1.9,0,2.2,.7], "joint_velocities": [0]*7}
result = safety_check(state, [0]*6)
assert result["estimated_drive_torque"]["values_nm"] == [0]*7
assert not any(m in sys.modules for m in ["mani_skill", "sapien", "torch", "gymnasium"])
assert importlib.util.find_spec("cam.interface") is None
writer = BytesIO()
robot = Robot(None, BytesIO(), writer, {"motion_preview": {"schema": "test"}})
assert robot.safety_check(state, [0]*6) == result
assert not writer.getvalue()
print(json.dumps({"passed": True, "simulator_used": result["simulator_used"]}))
'''
    code, out, err = run(workspace, runtime, ["python", "-c", script])
    assert code == 0, (out, err)
    assert json.loads(out) == {"passed": True, "simulator_used": False}
    other = make_runtime(tmp_path / "other")
    assert not (other / "cam/runtime/mikasa_robo").exists()


def test_exported_robot_geometry_matches_installed_urdf():
    import hashlib
    from cam.interface.mikasa_robo.api.offline import MODEL

    root = Path(__file__).resolve().parents[1]
    urdf = root / "envs/mikasa_robo/.venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf"
    if not urdf.exists():
        pytest.skip("Simulator environment not installed")
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == MODEL["urdf_sha256"]
