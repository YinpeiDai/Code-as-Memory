"""RMBench robot-only calculations on explicit inputs, using NumPy/SciPy.

Only the adjacent sanitized model is read. There are no simulator, GPU, RPC,
task, scene, scoring or dataset dependencies. IK search failure is not proof
of infeasibility; IK success is not a collision-free native motion plan.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return value


def _transform(xyz, rotation):
    result = np.eye(4)
    result[:3, 3], result[:3, :3] = xyz, rotation
    return result


def _pose(value):
    value = _vector(value, 7, "pose (x,y,z,qw,qx,qy,qz)")
    norm = np.linalg.norm(value[3:])
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Pose quaternion must be nonzero and finite")
    return _transform(value[:3], Rotation.from_quat(value[[4, 5, 6, 3]] / norm).as_matrix())


def _pose_values(transform):
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat(canonical=True)
    return np.r_[transform[:3, 3], quaternion[[3, 0, 1, 2]]].tolist()


def _state(robot_state):
    allowed = {"joints", "joint_source", "joint_target", "joint_velocities", "joint_torques"}
    if not isinstance(robot_state, dict) or set(robot_state) - allowed:
        raise ValueError("robot_state must contain only the documented robot fields")
    result = {"joints": _vector(robot_state.get("joints"), 6, "joints")}
    for name in ("joint_target", "joint_velocities", "joint_torques"):
        if name in robot_state:
            result[name] = _vector(robot_state[name], 6, name)
    result["joint_source"] = robot_state.get("joint_source", "unspecified")
    if result["joint_source"] not in ("commanded", "measured", "unspecified"):
        raise ValueError("joint_source must be commanded, measured or unspecified")
    return result


def _positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _count(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return int(value)


class RobotModel:
    """One released aloha-agilex arm and its native planner frame calibration.

    World poses use RMBench action units/conventions: XYZ metres, quaternion
    wxyz. The model's root pose is the public fixed demo_clean calibration.
    forward_kinematics returns the native planner's corresponding action pose;
    it does not read or predict the simulator's measured end-effector sensor.
    """

    def __init__(self, arm="left"):
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        model = json.loads(Path(__file__).with_name("aloha_model.json").read_text())
        cfg = model["arms"][arm]
        chain = cfg["chain"]
        self.arm, self.schema = arm, model["schema"]
        self.names = [j["name"] for j in chain]
        self.lower = np.array([j["limits"]["lower"] for j in chain])
        self.upper = np.array([j["limits"]["upper"] for j in chain])
        self.effort = np.array([j["limits"]["effort"] for j in chain])
        self.velocity = np.array([j["limits"]["velocity"] for j in chain])
        self.origins = [_transform(j["xyz"], Rotation.from_euler("xyz", j["rpy"]).as_matrix()) for j in chain]
        self.axes = np.array([j["axis"] for j in chain])
        self.root = _pose(model["world_root_pose_wxyz"])
        # Native planner: T_arm_target = Rz(yaw) @ T_bias @ inv(T_root) @ T_endlink.
        self.to_arm = (
            _transform([0, 0, 0], Rotation.from_euler("z", cfg["planner_yaw_rad"]).as_matrix())
            @ _transform(cfg["frame_bias_m"], np.eye(3)) @ np.linalg.inv(self.root)
        )
        self.to_world = np.linalg.inv(self.to_arm)
        # Native robot._trans_from_gripper_to_endlink, including the action-frame offset.
        self.action_to_link = _transform(
            [model["action_offset_m"] - model["gripper_bias_m"], 0, 0],
            np.linalg.inv(np.asarray(model["delta_matrix"], dtype=float)),
        )

    def _kinematics(self, q):
        transform = np.eye(4)
        origins, axes = [], []
        for origin, axis, angle in zip(self.origins, self.axes, q):
            transform = transform @ origin
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ axis)
            transform = transform @ _transform([0, 0, 0], Rotation.from_rotvec(axis * angle).as_matrix())
        jacobian = np.column_stack([
            np.r_[np.cross(axis, transform[:3, 3] - origin), axis]
            for origin, axis in zip(origins, axes)
        ])
        return transform, jacobian

    def forward_kinematics(self, joints):
        """Modeled world action pose corresponding to six supplied joint angles."""
        transform, _ = self._kinematics(_vector(joints, 6, "joints"))
        return _pose_values(self.to_world @ transform @ np.linalg.inv(self.action_to_link))

    def _position_report(self, q, margin):
        margins = np.minimum(q - self.lower, self.upper - q)
        return {
            "joints_rad": q.tolist(), "margins_rad": margins.tolist(),
            "within_limits": bool(np.all(margins >= 0)),
            "near_limits": [self.names[i] for i in np.flatnonzero(margins <= margin)],
            "violations": [self.names[i] for i in np.flatnonzero(margins < 0)],
        }

    def dry_run(self, robot_state, target_pose, *, position_tolerance_m=0.002,
                rotation_tolerance_rad=0.02, max_attempts=4, max_nfev=150):
        """Bounded deterministic IK search; never calls the native planner.

        Returns ik_found/ik_not_found, residuals and candidate joints. A candidate
        is not an executable trajectory. No collisions, gripper, timing, torque
        or dynamics are checked. max_nfev bounds evaluations per solver attempt.
        """
        state = _state(robot_state)
        if "joint_target" in state:
            raise ValueError("Provide target_pose or joint_target, not both")
        ptol = _positive(position_tolerance_m, "position_tolerance_m")
        rtol = _positive(rotation_tolerance_rad, "rotation_tolerance_rad")
        max_attempts = _count(max_attempts, "max_attempts", 8)
        max_nfev = _count(max_nfev, "max_nfev", 500)
        target = self.to_arm @ _pose(target_pose) @ self.action_to_link
        q0 = state["joints"]

        def error(q):
            actual, _ = self._kinematics(q)
            return np.r_[target[:3, 3] - actual[:3, 3],
                         Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()]

        def residual(q):
            e = error(q)
            return np.r_[e[:3] / ptol, e[3:] / rtol]

        def accepted(e):
            return bool(np.linalg.norm(e[:3]) <= ptol and np.linalg.norm(e[3:]) <= rtol)

        start = np.clip(q0, self.lower, self.upper)
        seeds = [start, np.zeros(6)]
        rng = np.random.default_rng(0)  # Local generator; no shared random-state mutation.
        seeds.extend(np.clip(start + rng.uniform(-np.pi, np.pi, 6), self.lower, self.upper)
                     for _ in range(max(0, max_attempts - 2)))
        best, best_error, attempted, evaluations = start, error(start), 0, 0
        for seed in seeds[:max_attempts]:
            if accepted(best_error):
                break
            fit = least_squares(residual, seed, bounds=(self.lower, self.upper),
                                max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            attempted += 1
            evaluations += int(fit.nfev)
            candidate_error = error(fit.x)
            if accepted(candidate_error) or np.linalg.norm(residual(fit.x)) < np.linalg.norm(residual(best)):
                best, best_error = fit.x, candidate_error
        found = accepted(best_error)
        return {
            "schema": "cam-rmbench-offline-dry-run/1", "model_schema": self.schema,
            "arm": self.arm, "joint_source": state["joint_source"],
            "status": "ik_found" if found else "ik_not_found", "ik_found": found,
            "joint_solution": best.tolist() if found else None,
            "candidate": self._position_report(best, 0.1),
            "current": self._position_report(q0, 0.1),
            "position_error_m": float(np.linalg.norm(best_error[:3])),
            "rotation_error_rad": float(np.linalg.norm(best_error[3:])),
            "position_tolerance_m": ptol, "rotation_tolerance_rad": rtol,
            "solver_attempts": attempted, "solver_nfev": evaluations,
            "native_plan_status": "not_checked", "path_checked": False,
            "collision_checked": False, "dynamics_checked": False, "simulator_used": False,
            "message": (
                "IK solution found within the supplied tolerances and released joint limits; native planning and execution remain unchecked."
                if found else
                "Bounded IK search found no solution within the supplied tolerances; this is not proof that the pose is unreachable."
            ),
        }

    def safety_check(self, robot_state, action=None, *, margin_rad=0.1, **ik_options):
        """Check supplied joint state and optional seven-value world pose.

        Uses exactly the same dry_run core for an action. Optional explicit
        joint_target is checked without IK. Supplied velocities/torques are
        compared with released URDF limits, never treated as predicted forces.
        """
        state = _state(robot_state)
        if not np.isscalar(margin_rad) or not np.isfinite(margin_rad) or margin_rad < 0:
            raise ValueError("margin_rad must be finite and nonnegative")
        if action is None and ik_options:
            raise ValueError("IK options require an action")
        preview = self.dry_run(robot_state, action, **ik_options) if action is not None else None
        target = state.get("joint_target")
        if preview is not None and preview["ik_found"]:
            target = np.asarray(preview["joint_solution"])

        def supplied(name, limit, unit):
            if name not in state:
                return {"status": "unavailable", "reason": f"{name} not supplied"}
            margins = limit - np.abs(state[name])
            return {
                "status": "supplied", "reference": "released URDF", "unit": unit,
                "values": state[name].tolist(), "limits": limit.tolist(), "margins": margins.tolist(),
                "within_limits": bool(np.all(margins >= 0)),
                "violations": [self.names[i] for i in np.flatnonzero(margins < 0)],
            }

        return {
            "schema": "cam-rmbench-offline-safety/1", "model_schema": self.schema,
            "arm": self.arm, "joint_source": state["joint_source"],
            "joint_names": list(self.names),
            "position_limits_rad": np.column_stack([self.lower, self.upper]).tolist(),
            "limit_reference": "Released RMBench URDF; not manufacturer safety limits",
            "current": self._position_report(state["joints"], margin_rad),
            "target": self._position_report(target, margin_rad) if target is not None else None,
            "dry_run": preview,
            "supplied_torque": supplied("joint_torques", self.effort, "Nm"),
            "supplied_velocity": supplied("joint_velocities", self.velocity, "rad/s"),
            "actual_future_torque_checked": False, "native_plan_status": "not_checked",
            "path_checked": False, "collision_checked": False, "dynamics_checked": False,
            "simulator_used": False,
        }


@lru_cache(maxsize=2)
def _model(arm):
    return RobotModel(arm)


def dry_run(robot_state, target_pose, *, arm, **options):
    """Simulator-independent IK query on explicitly supplied robot state."""
    return _model(arm).dry_run(robot_state, target_pose, **options)


def safety_check(robot_state, action=None, *, arm, **options):
    """Simulator-independent limits check; optional action uses dry_run's core."""
    return _model(arm).safety_check(robot_state, action, **options)
