"""Read-only prediction of MIKASA's actual next GPU arm-controller target.

No environment reset/step/evaluate, actor writes, controller.set_action, or
drive-target writes. Only the robot's measured pose/joints, controller and URDF
limits are accessed. Task objects, scoring and outcomes are never inspected.
"""

import numpy as np


def vector(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Expected six finite end-effector values")
    return values


def absolute_command(pose, base_pose, ee_pose_at_base, position_low, position_high, rotation_scale):
    """World XYZ/RPY -> a bounded native command toward that pose (one step).

    Inputs use wxyz quaternions. Rotation uses a base-frame log-map error, then
    the controller's signed normalization scale. Execution and preview both
    subsequently use the native controller's own preprocessing and delta IK.
    """
    from scipy.spatial.transform import Rotation

    pose = vector(pose)
    base, current = np.asarray(base_pose), np.asarray(ee_pose_at_base)
    base_r = Rotation.from_quat(base[[4, 5, 6, 3]])
    current_r = Rotation.from_quat(current[[4, 5, 6, 3]])
    target_p = base_r.inv().apply(pose[:3] - base[:3])
    target_r = base_r.inv() * Rotation.from_euler("xyz", pose[3:])
    offset = target_p - current[:3]
    low, high = np.asarray(position_low), np.asarray(position_high)
    translation = 2 * (offset - low) / (high - low) - 1
    rotation = (target_r * current_r.inv()).as_rotvec() / rotation_scale
    rotation_norm = float(np.linalg.norm(rotation))
    clipped = bool(np.any(np.abs(translation) > 1) or rotation_norm > 1)
    rotation /= max(1.0, rotation_norm)
    return np.concatenate([np.clip(translation, -1, 1), rotation]), clipped


def joint_report(current, target, limits, names):
    """URDF joint-target limits; finite JSON, including solver failure."""
    current, limits = np.asarray(current), np.asarray(limits)
    target = np.asarray(target) if target is not None else None
    valid = target is not None and target.shape == current.shape and np.isfinite(target).all()
    margins = np.minimum(target - limits[:, 0], limits[:, 1] - target) if valid else None
    violations = [names[i] for i in np.flatnonzero(margins < 0)] if valid else []
    return {
        "ik_computed": bool(valid),
        "within_joint_limits": bool(valid and not violations),
        "violating_joints": violations,
        "joint_names": list(names),
        "current_joints_rad": current.tolist(),
        "joint_target_rad": target.tolist() if valid else None,
        "joint_limits_rad": limits.tolist(),
        "joint_margins_rad": margins.tolist() if valid else None,
        "min_joint_margin_rad": float(margins.min()) if valid else None,
        "max_joint_change_rad": float(np.abs(target - current).max()) if valid else None,
    }


def preview(native, values, mode):
    """Preview against the current measured robot; returns only mechanical data."""
    import torch

    if mode not in ("absolute", "delta"):
        raise ValueError("mode must be 'absolute' or 'delta'")
    values = vector(values)
    ctrl = native.agent.controller.controllers["arm"]
    cfg = ctrl.config
    if not (
        native.num_envs == 1 and native.gpu_sim_enabled
        and cfg.use_delta and not cfg.use_target and cfg.normalize_action
        and cfg.frame == "root_translation:root_aligned_body_rotation"
        and ctrl.kinematics.use_gpu_ik
    ):
        raise ValueError("Motion preview requires MIKASA's native single-environment GPU delta-pose controller")

    def array(value):
        return value.detach().cpu().numpy()

    with torch.no_grad():
        q0 = ctrl.articulation.get_qpos().clone()
        current_pose = ctrl.ee_pose_at_base
        if mode == "absolute":
            command, clipped = absolute_command(
                values, array(ctrl.articulation.pose.raw_pose)[0], array(current_pose.raw_pose)[0],
                array(ctrl.action_space_low)[:3], array(ctrl.action_space_high)[:3], cfg.rot_lower,
            )
        else:
            command = np.clip(values, -1, 1)
            clipped = bool(np.any(command != values) or np.linalg.norm(command[3:]) > 1)
        command = torch.as_tensor(command, dtype=torch.float32, device=ctrl.device).reshape(1, 6)
        # These are the exact read-only calculations used by native set_action.
        scaled = ctrl._preprocess_action(command.clone())
        target_pose = ctrl.compute_target_pose(current_pose, scaled)
        target = ctrl.kinematics.compute_ik(
            target_pose, q0, pos_only=False, action=scaled, use_delta_ik_solver=True,
        )
        indices = ctrl.active_joint_indices
        names = [ctrl.articulation.get_active_joints()[int(i)].name for i in indices]
        current = array(q0[:, indices])[0]
        limits = array(ctrl.articulation.get_qlimits()[:, indices])[0]
        report = joint_report(current, array(target)[0] if target is not None else None, limits, names)
        report.update({
            "schema": "cam-mikasa-motion-preview/1",
            "mode": mode,
            "step": int(native.elapsed_steps.item()),
            "command": array(command)[0].tolist(),
            "command_clipped": clipped,
            "scaled_offset": array(scaled)[0].tolist(),
            "scope": "next_arm_joint_target",
            "collision_checked": False,
            "dynamics_checked": False,
            "path_checked": False,
        })
        return report
