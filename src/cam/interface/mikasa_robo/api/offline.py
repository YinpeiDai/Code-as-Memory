"""Public MIKASA robot-state safety checks. NumPy/SciPy only; no simulator or RPC.

Only the adjacent, sanitized robot model is read. No task, dataset, seed, scorer,
assets, contacts, or environment state are accessed. Results are estimates, not
a guarantee of safe physical motion. See safety_check.md for units and scope.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MODEL = json.loads(Path(__file__).with_name("panda_model.json").read_text())
ARM = [j for j in MODEL["chain"] if j["type"] == "revolute"]
NAMES = [j["name"] for j in ARM]
LOWER = np.array([j["limits"]["lower"] for j in ARM])
UPPER = np.array([j["limits"]["upper"] for j in ARM])
EFFORT = np.array([j["limits"]["effort"] for j in ARM])


def _vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return value


def _pose(tcp):
    pose = np.eye(4)
    pose[:3, 3] = tcp[:3]
    pose[:3, :3] = Rotation.from_euler("xyz", tcp[3:]).as_matrix()
    return pose


def _kinematics(q):
    """URDF forward kinematics and geometric Jacobian in the robot base frame."""
    transform = np.eye(4)
    origins, axes = [], []
    for joint in MODEL["chain"]:
        transform = transform @ _pose(np.array(joint["xyz"] + joint["rpy"]))
        if joint["type"] == "revolute":
            axis = np.asarray(joint["axis"])
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ axis)
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * q[len(axes) - 1]).as_matrix()
            transform = transform @ rotation
    jacobian = np.column_stack([
        np.r_[np.cross(axis, transform[:3, 3] - origin), axis]
        for axis, origin in zip(axes, origins)
    ])
    return transform, jacobian


def _position_report(q, margin):
    margins = np.minimum(q - LOWER, UPPER - q)
    return {"joints_rad": q.tolist(), "margins_rad": margins.tolist(),
            "min_margin_rad": float(margins.min()),
            "within_limits": bool(np.all(margins >= 0)),
            "near_limits": [NAMES[i] for i in np.flatnonzero(margins <= margin)],
            "violations": [NAMES[i] for i in np.flatnonzero(margins < 0)]}


def _torque_report(torque, limits, fraction):
    ratio = np.abs(torque) / limits
    return {"values_nm": torque.tolist(), "limits_nm": limits.tolist(),
            "margin_nm": (limits - np.abs(torque)).tolist(),
            "utilization": ratio.tolist(), "max_utilization": float(ratio.max()),
            "within_limits": bool(np.all(ratio <= 1)),
            "near_limits": [NAMES[i] for i in np.flatnonzero(ratio >= fraction)],
            "exceeds_limits": [NAMES[i] for i in np.flatnonzero(ratio > 1)]}


def safety_check(robot_state, action=None, *, mode="delta", margin_rad=0.1, torque_fraction=0.8):
    """Check supplied Panda state and optional candidate action, entirely offline.

    State: joints[7] radians; optional tcp[6] world XYZ/RPY,
    joint_velocities[7] rad/s, joint_torques[7] Nm, joint_target[7] radians.
    Action: six normalized EEF offsets (delta) or world XYZ/RPY (absolute).
    A seventh gripper argument is accepted and excluded from this arm-only check.
    Supplied torques and estimated PD drive demand are reported separately.
    """
    if not isinstance(robot_state, dict) or set(robot_state) - {
        "joints", "tcp", "joint_velocities", "joint_torques", "joint_target"
    }:
        raise ValueError("robot_state must contain only the documented robot fields")
    if mode not in ("delta", "absolute"):
        raise ValueError("mode must be 'delta' or 'absolute'")
    if not np.isfinite(margin_rad) or margin_rad < 0 or not 0 < torque_fraction <= 1:
        raise ValueError("Use margin_rad >= 0 and 0 < torque_fraction <= 1")
    q = _vector(robot_state.get("joints"), 7, "joints")
    velocity = _vector(robot_state["joint_velocities"], 7, "joint_velocities") if "joint_velocities" in robot_state else None
    torque = _vector(robot_state["joint_torques"], 7, "joint_torques") if "joint_torques" in robot_state else None
    tcp = _vector(robot_state["tcp"], 6, "tcp") if "tcp" in robot_state else None
    target = _vector(robot_state["joint_target"], 7, "joint_target") if "joint_target" in robot_state else None
    if action is not None and target is not None:
        raise ValueError("Provide action or joint_target, not both")
    current, jacobian = _kinematics(q)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    command, clipped = None, False
    controller = MODEL["controller"]
    if action is not None:
        action = np.asarray(action, dtype=float)
        if action.shape == (7,):
            _vector(action, 7, "action")
            action = action[:6]
        action = _vector(action, 6, "action")
        if mode == "absolute":
            if tcp is None:
                raise ValueError("Absolute actions require the measured world tcp in robot_state")
            # Derive the world-to-base transform entirely from supplied q and TCP.
            base = _pose(tcp) @ np.linalg.inv(current)
            desired = np.linalg.inv(base) @ _pose(action)
            translation = (desired[:3, 3] - current[:3, 3]) / controller["position_scale_m"]
            rotation = Rotation.from_matrix(desired[:3, :3] @ current[:3, :3].T).as_rotvec() / controller["rotation_scale_rad"]
            command = np.r_[translation, rotation]
            clipped = bool(np.any(np.abs(command[:3]) > 1) or np.linalg.norm(command[3:]) > 1)
            command[:3] = np.clip(command[:3], -1, 1)
            command[3:] /= max(1.0, np.linalg.norm(command[3:]))
        else:
            command = np.clip(action, -1, 1)
            clipped = bool(np.any(command != action) or np.linalg.norm(command[3:]) > 1)
        command = command.astype(np.float32).astype(float)
        rotation = command[3:] / max(1.0, np.linalg.norm(command[3:]))
        offset = np.r_[command[:3] * controller["position_scale_m"], rotation * controller["rotation_scale_rad"]]
        # Native GPU IK uses an undamped pseudoinverse; match its float32 cutoff.
        # CPU float64 geometry/SVD still make this an estimate, not an exact replay.
        target = q + np.linalg.pinv(jacobian, rcond=7 * np.finfo(np.float32).eps) @ offset
        if not np.isfinite(target).all():
            raise ValueError("Candidate joint target is non-finite")
    supplied = {"status": "unavailable", "reason": "joint_torques not supplied"}
    if torque is not None:
        supplied = {"status": "supplied", "reference": "URDF effort limits", **_torque_report(torque, EFFORT, torque_fraction)}
    drive = {"status": "unavailable", "reason": "Requires joint_velocities and action or joint_target"}
    if velocity is not None and target is not None:
        demand = np.array(controller["stiffness_nm_per_rad"]) * (target - q) - np.array(controller["damping_nm_s_per_rad"]) * velocity
        if not np.isfinite(demand).all():
            raise ValueError("PD torque estimate is non-finite; check state units and values")
        drive = {"status": "estimated_initial_pd_demand", "reference": "simulator drive force limits",
                 **_torque_report(demand, np.array(controller["drive_force_limit_nm"]), torque_fraction),
                 "urdf_effort_comparison": _torque_report(demand, EFFORT, torque_fraction)}
    return {
        "schema": "cam-mikasa-offline-safety/1", "joint_names": list(NAMES),
        "position_limits_rad": np.column_stack([LOWER, UPPER]).tolist(),
        "current": _position_report(q, margin_rad),
        "predicted_target": _position_report(target, margin_rad) if target is not None else None,
        "command": command.tolist() if command is not None else None, "command_clipped": clipped,
        "jacobian_singular_values": singular_values.tolist(),
        "supplied_torque": supplied, "estimated_drive_torque": drive,
        "torque_estimate_includes_contacts_or_payload": False,
        "actual_future_torque_checked": False, "collision_checked": False,
        "simulator_used": False, "model_schema": MODEL["schema"],
    }
