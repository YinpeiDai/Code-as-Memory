## MIKASA-Robo motion dry-run

For simulator-free development, including demo-only mode, use the separate
`safety_check` API documented in `safety_check.md`. The live query below needs
an active episode and cannot be invoked from offline coding commands.

Call inside an `act_*` function on the current episode's `robot`. Both MIKASA
adapters support these explicit modes:

```python
check = robot.dry_run(dx, dy, dz, drx, dry, drz, mode="delta")
check = robot.dry_run(x, y, z, roll, pitch, yaw, mode="absolute")
if check["within_joint_limits"]:
    robot.step(*check["command"], gripper)
```

Delta inputs are the six normalized arm arguments of `robot.step`, with exactly
the same clipping, scaling and native controller. Absolute inputs use world XYZ
in metres and roll/pitch/yaw in radians (`Rz(yaw) Ry(pitch) Rx(roll)`). An absolute
request becomes **one bounded step toward the pose** using base-frame translation
and rotation error. Use the returned `command` to execute that checked step;
check again after moving. Gripper position is supplied separately to `step`.

The result includes `within_joint_limits`, `violating_joints` (Panda joint names),
`joint_target_rad`, `joint_limits_rad`, signed `joint_margins_rad`,
`min_joint_margin_rad`, `max_joint_change_rad`, the measured `current_joints_rad`,
`command`, `command_clipped`, and the observation `step`. Negative margins mean
the native controller would request a joint target outside its limits. If IK
does not produce a finite result, `ik_computed` and `within_joint_limits` are false.

This checks the commanded **arm joint target**, not actual future motion under
contacts or inertia. It does not check collisions, gripper contacts, whole-path
reachability, or task success. A valid target can still produce undesirable motion.
No candidate is executed automatically, rejected, or modified by this query.
Some tasks suppress commands during an initial cue phase. The preview checks
the requested robot command; it does not reveal whether the task will suppress it.

Queries do not advance physics/time, consume native steps, generate frames, or
change controller targets or task state. They are available only inside an
episode; development still uses counted training rollouts through the existing
six CaM tools. Each action routine can make at most `max_action_steps` queries.
Inputs/results are recorded in `dry_runs.jsonl`, also copied into that attempt's
training feedback. Unavailable on other benchmarks.

For compatibility, the v2 adapter's older call without an explicit `mode` retains
its absolute-pose IK query. Use an explicit mode for the next-command check above.
