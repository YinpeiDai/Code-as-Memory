## Offline RMBench robot checks

In a coding `command` script:

```python
from cam.runtime.rmbench import RobotModel, dry_run, safety_check

state = {"joints": frame.left_joints, "joint_source": "commanded"}
# target: [x, y, z, qw, qx, qy, qz], metres and a nonzero quaternion.
preview = dry_run(state, target, arm="left")
checks = safety_check(state, target, arm="left")
assert checks["dry_run"] == preview
```

Both functions use the same `RobotModel` core. `dry_run` searches for an IK
solution within released joint limits. `safety_check` also reports current and
candidate joint margins, plus supplied torque/velocity comparisons when those
inputs are available. Neither executes an action, creates an environment, makes
an RPC, reads a dataset, uses a GPU, or consumes a simulator attempt. Both are
available during demo-only development. Command results use the normal audit.

The shipped model contains only robot joint geometry, limits and fixed
`demo_clean` base/action-frame calibration, with source checksums. It uses the
released per-arm planner transforms, including the aloha frame bias and yaw
correction. It does not expose tasks, object geometry, contacts or scoring.

### Inputs

`arm` is explicitly `"left"` or `"right"`. `robot_state` accepts only:

| Field | Meaning |
|---|---|
| `joints` | Required six arm joint angles in radians, in native joint order |
| `joint_source` | `"commanded"`, `"measured"`, or `"unspecified"` (default); caller-declared provenance |
| `joint_target` | Optional six explicit target joint angles for `safety_check` without an action |
| `joint_velocities` | Optional six supplied joint velocities, rad/s |
| `joint_torques` | Optional six supplied torques, Nm |

RMBench demo and online `left_joints`/`right_joints` contain **commanded targets**,
not measured encoders. The native planner starts from measured joints. This API
uses only what you supply; it cannot recover measured joints or certify the
declared provenance. Contact or tracking error can therefore change the native
planner's answer. Do not assume zero velocity for a moving arm.

An action is exactly one arm's seven-value absolute world target, in the same
convention as `*_ee_pose` and `robot.step`. It is not a 16-value dual-arm action,
XYZ/RPY, a gripper-centre target, or a normalized delta. Quaternions are normalized;
zero/nonfinite quaternions are rejected. Grippers and interactions between arms
are outside this check. Supply `joint_target` or an action, never both.

### Outputs and scope

`dry_run` returns `status="ik_found"` or `"ik_not_found"`, `ik_found`, a
`joint_solution` only on success, the best `candidate`, position/rotation
residuals, current joint margins, and solver statistics. Defaults are 2 mm and
0.02 radians acceptance tolerances, four deterministic attempts and `max_nfev=150`
per attempt. These are numerical solver settings, not task-success thresholds.
Keyword overrides: `position_tolerance_m`, `rotation_tolerance_rad`, `max_attempts`
(1–8), `max_nfev` (1–500). SciPy's evaluation count excludes numerical-Jacobian
evaluations. Search failure does not prove unreachability.

`safety_check` accepts the same IK options when an action is supplied, and
`margin_rad=0.1` for reporting near-limit joints. Its `target` is null when no
target was supplied or IK found no solution. Missing torque/velocity inputs
return `status="unavailable"`. Supplied torques are compared to URDF effort,
not native drive-force caps; they are not estimates of future torque. The native
controller follows a timed trajectory, so we do not estimate PD demand from the
final endpoint alone.

The released URDF uses ±10 rad arm position limits, 100 Nm effort and 1,000 rad/s
velocity limits. These are benchmark configuration values, **not manufacturer
safety limits**. There is no aggregate `safe` result.

All outputs explicitly mark `native_plan_status="not_checked"`,
`collision_checked=False`, `path_checked=False`, `dynamics_checked=False`, and
`simulator_used=False`. This helper does not run CuRobo or check its table/self
collisions, task objects, the other arm, gripper motion, trajectory timing, or
execution. An IK solution is not a native motion plan and does not guarantee that
`robot.step` will succeed. Native planner-status feedback remains disabled.
The separate post-action pose check below uses only public observations.
Older prepared snapshots retain their original feedback behavior.

`RobotModel(arm).forward_kinematics(joints)` returns the modeled world action pose
corresponding to those joints under the planner calibration, not a measured
end-effector observation. The class exposes the same `dry_run`/`safety_check`
methods without needing an `arm` argument per call.

### Inside generated action code

Generated functions cannot import modules; use the supplied `robot`:

```python
check = robot.dry_run(*target, arm="left")
# Uses only robot.now().left_joints, explicitly labelled as commanded.
state = {"joints": robot.now().left_joints, "joint_source": "commanded"}
check = robot.safety_check(state, target, arm="left")
```

`robot.dry_run` also accepts `robot_state=state` to supply the state explicitly,
and `mode="absolute"` (optional). Its query count is bounded by
`max_action_steps`; it consumes no action or physics step and makes no RPC.
Unlike MIKASA's live dry-run, it has no native-controller access and produces no
host `dry_runs.jsonl` RPC record. Returned feedback can be included in normal
policy decision logs. `robot.safety_check` uses only its explicit arguments.

New preparations copy and hash the helper/model/reference into the isolated
runtime. Existing prepared runs retain their original API and results.

### Post-action MotionError

In new preparations, `robot.step` compares both arms' public post-action
`left_ee_pose`/`right_ee_pose` with the submitted world-frame targets. Position
error greater than **0.02 m**, or shortest quaternion rotation error greater
than **0.1 rad**, raises catchable `MotionError` with per-arm errors. Quaternions
are normalized for comparison; `q` and `-q` are equivalent. These fixed,
task-independent tolerances are recorded in the action specification as
`motion_feedback`, schema `cam-rmbench-observed-motion/1`. They are wrapper
tracking tolerances, not native planner settings or task-success thresholds.

This calculation runs inside the policy process using only public sensor data
and the submitted action. It does not query the planner, call `dry_run`, inspect
collisions or determine why a target was missed. A held pose already within
tolerance is accepted even without movement. Gripper execution is not checked.
Missing/invalid pose data produces a "cannot verify motion" error; it never falls
back to simulator internals. `planner_failure_feedback` remains false.

The action is already counted and `robot.now()`/`frames()` already contain the
returned observations before the exception. Either arm or gripper may have
moved; inspect the resulting state before recovery. There is no automatic retry,
extra native action, rollback or scoring change. Terminal `EpisodeEnd` takes
precedence, including success partway through a motion.
