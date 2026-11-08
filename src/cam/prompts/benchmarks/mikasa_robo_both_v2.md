## MIKASA-Robo demo-first development and motion verification

Follow the shared v2 protocol: infer strategy, memory and candidate completion
conditions from all assigned demos, build and check an initial policy, then
validate and snapshot before interacting. You may reread and analyze assigned
demos without limit; offline work consumes no simulator attempts.

Use the entire assigned rollout budget for exploration and verification. For a
50-attempt assignment, use all 50, including every remaining attempt after the
policy appears ready or succeeds. Unchanged-code verification is valid. Failed
and diagnostic attempts count. The host selects the next pinned identity, which
may come from the demos; no seed overrides or extra probes are allowed. Record
assumptions, findings and revisions, and report final changes left unverified.

### Robot checks

Read the declared action interface and
`api_reference/runtime/mikasa_robo/{safety_check,dry_run}.md`.

- **Offline:** import `safety_check` from `cam.runtime.mikasa_robo` in command
  scripts. `safety_check(state, action, mode="delta")` estimates the next joint
  target; omit the action for a state-only check. Absolute actions require the
  corresponding TCP pose and `mode="absolute"`. Generated actions use
  `robot.safety_check(state, action, ...)` without imports.
- **During a counted rollout:** `robot.dry_run(..., mode="delta" or "absolute")`
  checks the next arm target with the active native controller without stepping.
  For an absolute query, execute its returned `command` if proceeding. Recheck
  after moving; read the reference for query limits.

Use supplied joint state only. Velocities estimated from consecutive 20 Hz joint
samples are approximate; missing velocity or torque is unknown, not zero. Keep
URDF effort limits separate from simulator drive caps. The offline model may
differ from live IK. Neither check establishes collision avoidance, future torque
safety, dynamic stability or task success; joint margins alone cannot prove these.

### Motion verification

Commands are incremental pose targets relative to the measured pose. Check units,
rotation signs, scaling, clipping and angle wrapping. A small one-step response
does not justify a large feedback gain. Mock models and `validate` cannot establish
physical stability; document which dynamics your offline checks omit.

Use early rollouts to compare commanded and measured approach/hold motion. Inspect
`actions.npy`, `observations.h5`, video, result and available `dry_runs.jsonl` under
`/feedback/RNNN/`.

- Look for alternating rotation commands, saturation, overshoot and continued
  joint motion during holds. Stable TCP position alone does not mean a settled
  arm; distinguish real rotation from Euler-angle wrapping.
- If motion oscillates, check conventions and test smaller bounded corrections,
  damping or evidence-based deadbands/change limits. Change one factor at a time;
  smoothing can add delay. Base arrival/settling decisions on measured motion.
- Keep approach, transport and retreat paths inside checked joint margins.
  Reachability depends on orientation and joint configuration. Check intermediate
  steps as well as endpoints; shorten commands or revise the path when margins
  shrink or motion overshoots. Avoid large blind movements of the arm or carried
  object. Budget cautious motion and settling within the episode limit.

Each `robot.step` advances one control step; Python sleep does not let the
simulated arm settle.

### Explore uncertain completion conditions

Use demo measurements to form hypotheses, then investigate them through assigned
rollouts. For button tasks, questions may include release, retreat/lift height,
the lift's reference point, timing and settling. These are possible conditions,
not universal requirements. Vary one relevant factor at a time, account for layout
variation, and avoid treating one outcome as proof of an exact hidden threshold.

Distinguish commands issued, observed effects and inferred progress in memory or
action outcomes. Visible contact and policy progress labels do not confirm native
acceptance. Revise from your own allowed observations and training outcomes,
revalidate and snapshot, then use remaining attempts to verify. Benchmark source,
private diagnostics and test feedback remain unavailable.
