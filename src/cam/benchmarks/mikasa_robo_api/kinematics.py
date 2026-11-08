"""Existing absolute-pose IK query, kept separate from the native one-step preview.

Used by the MIKASA v2 legacy six-argument API; no simulator task state is read.
"""

import numpy as np
from ..mikasa_robo import to_numpy

JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)


JOINT_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)


LIMIT_TOLERANCE = 0.01


def limits_text():
    return ", ".join(f"J{i + 1} [{lo:+.4f}, {hi:+.4f}]" for i, (lo, hi) in enumerate(zip(JOINT_LOWER, JOINT_UPPER)))


def rpy_matrix(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll), the ``tcp`` sensor's convention."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quat_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_vector(R):
    """Axis-angle vector of a rotation matrix (log map)."""
    angle = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.sin(angle) < 1e-6:  # near pi: use the symmetric part
        values, vectors = np.linalg.eigh((R + R.T) / 2)
        axis = vectors[:, int(np.argmax(values))]
        return axis / np.linalg.norm(axis) * angle
    return axis / (2 * np.sin(angle)) * angle


class ArmKinematics:
    """CPU float64 copy of the arm controller's Panda chain (same URDF and TCP link)."""

    POSITION_TOLERANCE = 0.002  # metres
    ROTATION_TOLERANCE = 0.02  # radians
    PATH_POINTS = 20

    def __init__(self, unwrapped):
        import contextlib
        import io

        import pytorch_kinematics as pk
        import torch

        ctrl = unwrapped.agent.controller.controllers["arm"]
        kin = ctrl.kinematics
        with open(kin.urdf_path, "rb") as stream, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.chain = pk.build_serial_chain_from_urdf(stream.read(), end_link_name=kin.end_link.name).to(
                dtype=torch.float64, device="cpu"
            )
        if [n for n in self.chain.get_joint_parameter_names()] != list(JOINT_NAMES):
            raise ValueError("Kinematic chain joints differ from the Panda arm joints")
        self.torch, self.unwrapped, self.ctrl = torch, unwrapped, ctrl
        self.lower, self.upper = np.array(JOINT_LOWER), np.array(JOINT_UPPER)

    def _base(self):
        raw = to_numpy(self.ctrl.articulation.pose.raw_pose)[0].astype(float)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = quat_matrix(raw[3:7]), raw[:3]
        return T

    def _fk(self, q):
        m = self.chain.forward_kinematics(self.torch.as_tensor(q[None], dtype=self.torch.float64)).get_matrix()
        return m[0].numpy()

    def _jacobian(self, q):
        return self.chain.jacobian(self.torch.as_tensor(q[None], dtype=self.torch.float64))[0].numpy()

    def _error(self, q, target):
        T = self._fk(q)
        return np.concatenate([target[:3, 3] - T[:3, 3], rotation_vector(target[:3, :3] @ T[:3, :3].T)])

    def _done(self, e, position_only):
        return np.linalg.norm(e[:3]) < self.POSITION_TOLERANCE and (
            position_only or np.linalg.norm(e[3:]) < self.ROTATION_TOLERANCE
        )

    def solve(self, target, seed, clamp, iterations, position_only=False):
        """Damped least-squares IK (base frame); returns (q, converged, error6)."""
        q = np.array(seed, dtype=float)
        rows = slice(0, 3) if position_only else slice(0, 6)
        for _ in range(iterations):
            e = self._error(q, target)
            if self._done(e, position_only):
                return q, True, e
            J = self._jacobian(q)[rows]
            e = e[rows]
            # Damping grows near singular configurations (arm close to full extension).
            sigma = np.linalg.svd(J, compute_uv=False)[-1]
            damping = 1e-4 if sigma > 0.05 else 1e-4 + (0.05 - sigma) ** 2
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(e)), e)
            norm = np.linalg.norm(dq)
            q = q + (dq * (0.2 / norm) if norm > 0.2 else dq)
            if clamp:
                q = np.clip(q, self.lower, self.upper)
        e = self._error(q, target)
        return q, bool(self._done(e, position_only)), e

    def dry_run(self, pose):
        q0 = to_numpy(self.ctrl.articulation.get_qpos())[0, :7].astype(float)
        base_inv = np.linalg.inv(self._base())
        world = np.eye(4)
        world[:3, :3], world[:3, 3] = rpy_matrix(*pose[3:]), pose[:3]
        target = base_inv @ world
        bounded, ok, err = self.solve(target, np.clip(q0, self.lower, self.upper), clamp=True, iterations=200)
        free, free_ok, _ = (bounded, True, err) if ok else self.solve(target, q0, clamp=False, iterations=200)
        start = self._fk(q0)
        # Straight TCP path (linear position, geodesic rotation), tracked within limits.
        rotvec = rotation_vector(target[:3, :3] @ start[:3, :3].T)
        q, blocked, blocked_joints = np.clip(q0, self.lower, self.upper), None, []
        for k in range(1, self.PATH_POINTS + 1 if ok else 1):
            s = k / self.PATH_POINTS
            angle = np.linalg.norm(rotvec)
            K = np.zeros((3, 3)) if angle < 1e-12 else np.array(
                [[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]], [-rotvec[1], rotvec[0], 0]]
            ) / angle
            Rs = np.eye(3) + np.sin(s * angle) * K + (1 - np.cos(s * angle)) * K @ K
            waypoint = np.eye(4)
            waypoint[:3, :3], waypoint[:3, 3] = Rs @ start[:3, :3], (1 - s) * start[:3, 3] + s * target[:3, 3]
            q, converged, _ = self.solve(waypoint, q, clamp=True, iterations=60)
            if not converged:
                blocked = round(s, 3)
                near = np.minimum(q - self.lower, self.upper - q)
                blocked_joints = [int(i + 1) for i in np.where(near < LIMIT_TOLERANCE)[0]]
                break
        margin_free = np.minimum(bounded - self.lower, self.upper - bounded)
        nearest = sorted((float(m), int(i + 1)) for i, m in enumerate(margin_free) if m < 0.15)
        nearest_text = ", ".join(f"J{j} {m:.2f} rad" for m, j in nearest) or "none within 0.15 rad"
        singular = bool(np.linalg.svd(self._jacobian(bounded)[:3], compute_uv=False)[-1] < 0.05)
        reach_rpy = None
        if not ok:  # is the position reachable with some other hand orientation?
            q_pos, pos_ok, _ = self.solve(target, np.clip(q0, self.lower, self.upper), clamp=True, iterations=150, position_only=True)
            if pos_ok:
                R = self._base()[:3, :3] @ self._fk(q_pos)[:3, :3]
                reach_rpy = [
                    round(float(np.arctan2(R[2, 1], R[2, 2])), 4),
                    round(float(np.arcsin(np.clip(-R[2, 0], -1, 1))), 4),
                    round(float(np.arctan2(R[1, 0], R[0, 0])), 4),
                ]
        hint = (
            f" The position alone is reachable with hand roll/pitch/yaw {reach_rpy}."
            if reach_rpy
            else " The position is out of reach for any hand orientation; move the target closer to the base."
        )
        violations = [int(i + 1) for i in np.where((free < self.lower - LIMIT_TOLERANCE) | (free > self.upper + LIMIT_TOLERANCE))[0]]
        margin = np.minimum(bounded - self.lower, self.upper - bounded)
        limiting = [int(i + 1) for i in np.where(margin < LIMIT_TOLERANCE)[0]]
        if ok and blocked is None:
            message = "Reachable within joint limits; the straight TCP path stays within limits."
        elif ok:
            message = (
                f"Pose reachable within limits, but the straight TCP path is blocked at {blocked:.0%} "
                f"(joints at limits: {blocked_joints}); approach through an intermediate pose."
            )
        elif free_ok:
            message = (
                f"Unreachable within joint limits: the exact solution needs joints {violations} beyond their limits; "
                f"the closest pose within limits is {np.linalg.norm(err[:3]) * 100:.1f} cm / "
                f"{np.degrees(np.linalg.norm(err[3:])):.1f} deg away with joints {limiting} at limits." + hint
            )
        else:
            message = (
                "Unreachable: beyond the arm's workspace"
                + (" (arm at full extension / singular)" if singular else "")
                + f"; the closest pose within limits is {np.linalg.norm(err[:3]) * 100:.1f} cm / "
                f"{np.degrees(np.linalg.norm(err[3:])):.1f} deg away (joint margins there: {nearest_text})." + hint
            )
        r = lambda v: np.round(np.asarray(v, dtype=float), 5).tolist()
        return {
            "reachable": bool(ok),
            "path_clear": bool(ok and blocked is None),
            "path_blocked_at": blocked,
            "path_blocked_joints": blocked_joints,
            "position_error_m": round(float(np.linalg.norm(err[:3])), 5),
            "rotation_error_rad": round(float(np.linalg.norm(err[3:])), 5),
            "joint_solution": r(bounded),
            "joint_margin": r(margin),
            "joints_at_limit": limiting,
            "nearest_limits": [[j, round(m, 4)] for m, j in nearest],
            "singular": singular,
            "position_reachable_with_rpy": reach_rpy,
            "exact_solution_violates": violations,
            "joints": r(q0),
            "lower": list(JOINT_LOWER),
            "upper": list(JOINT_UPPER),
            "message": message,
        }
