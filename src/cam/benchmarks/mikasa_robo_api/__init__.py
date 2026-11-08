"""MIKASA robot queries; only the documented offline utility/model are public."""

PREVIEW_SPEC = {
    "schema": "cam-mikasa-motion-preview/1",
    "modes": ["absolute", "delta"],
    "description": (
        "robot.dry_run(..., mode='absolute' or 'delta') previews the native arm "
        "joint target for ONE next command without moving or consuming a native step. "
        "Absolute: world XYZ metres and roll/pitch/yaw radians. Delta: the six "
        "normalized arm arguments of robot.step. The returned command can be sent "
        "with robot.step(*check['command'], gripper). within_joint_limits checks "
        "the computed arm target only, not collisions, dynamics or the whole path. "
        "Read api_reference/runtime/mikasa_robo/dry_run.md."
        " For development without a simulator, import safety_check from "
        "cam.runtime.mikasa_robo in an offline command script. It checks supplied "
        "joint state and optional action, including position margins and torque "
        "margins when torque or velocity information is supplied. This pure robot-model "
        "calculation is allowed in demo-only mode and uses no simulator attempt. "
        "Read api_reference/runtime/mikasa_robo/safety_check.md."
    ),
}
