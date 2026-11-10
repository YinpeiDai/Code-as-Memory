## Offline MIKASA robot safety check

Available only for MIKASA. In a coding `command` script, use:

```python
from cam.runtime.mikasa_robo import safety_check

state = {"joints": joint_angles_rad, "joint_velocities": joint_velocities_rad_s}
check = safety_check(state, [dx, dy, dz, drx, dry, drz], mode="delta")
print(check["predicted_target"]["margins_rad"])
print(check["estimated_drive_torque"])
```

This uses only supplied robot state and a sanitized Panda geometry/controller
model. It creates no environment, renders no images, makes no RPC, uses no GPU,
and consumes no simulator attempts. It is allowed during demo-only development.
Calls and printed results are retained in the normal coding-command audit.

`robot_state` is a dictionary:

| Field | Shape / units | Use |
|---|---|---|
| `joints` | 7 arm joint angles, radians | Required; J1 through J7 |
| `tcp` | World XYZ metres and roll/pitch/yaw radians | Required for absolute actions; use the same frame as `joints` |
| `joint_velocities` | 7 values, radians/second | Optional; enables a PD-demand estimate with an action or joint target |
| `joint_torques` | 7 values, Nm | Optional supplied torque; compared directly with URDF effort limits |
| `joint_target` | 7 values, radians | Optional explicit target instead of an EEF action |

`action` accepts six arm values, or seven with the gripper ignored. `mode="delta"`
uses the normalized arguments of `robot.step`; `mode="absolute"` uses world
XYZ/RPY, converted to one bounded command toward that pose. Use the returned
`command` to execute the checked EEF command. No action is executed by this API.
Omit `action` for a current-state check. Do not supply both `action` and `joint_target`.

`current` and `predicted_target` report position margins in radians (negative
means outside a joint limit), near-limit joint names and violations. The default
near-limit band is `margin_rad=0.1`. The next target is an estimate of the native
undamped delta-IK calculation, using CPU geometry and SVD; numerical differences
grow near singularities. It is not a full-path IK or physical-motion prediction.

Torque feedback has two separate entries:

- `supplied_torque`: compares supplied torques to URDF effort limits: 87 Nm for
  J1–J4 and 12 Nm for J5–J7. The API cannot confirm how those torques were measured.
- `estimated_drive_torque`: estimates the initial unclipped PD demand,
  `1000*(joint_target-joints) - 100*joint_velocities` Nm, and compares it with the
  simulator's 100 Nm drive-force cap per joint. A nested `urdf_effort_comparison`
  also compares that demand with URDF effort limits. These two limits differ.

Each available torque entry includes `margin_nm = limit - abs(torque)`,
`utilization = abs(torque)/limit`, and joints near or exceeding the limit.
`torque_fraction=0.8` defines the near-limit warning band. Missing torque or
velocity inputs produce `status="unavailable"`, not a safe result. A 100% demand
means the initial estimate reaches the reference limit; actual forces can differ.

Demos expose joint angles, not measured joint torques or velocities. Velocities
may be estimated from consecutive joint samples and their 20 Hz step spacing;
such estimates and the resulting torque demand remain approximations. Never
assume a moving arm has zero velocity. Contacts, payload forces, inertia, gravity
compensation and solver dynamics are not predicted. No aggregate "safe" guarantee,
task-success feedback, or hidden task state is returned.

Generated `act_*` code can call `robot.safety_check(state, action, ...)` with the
same inputs, without imports; this also performs only local calculations. The
existing `robot.dry_run(..., mode=...)` instead queries a live episode and is
unavailable for offline demo-only development. Live queries become possible when
the frozen policy is executed by the host.
