## RoboMemArena: verify manipulation and recovery

Build from all assigned demos first, following the shared demo-first protocol. Infer
task order, memory and completion checks from the goal and demos. Exercise the
first decision and recovery branches offline, including missing detections and
memory fields initialized to None.

- Check observed effects during grasping, transport, release and drawer or door
  motion. Compare requested motion with measured TCP, joints, fingers and RGB-D.
  A waypoint or gripper command alone does not establish successful manipulation.
- Separate attempted actions, observed progress and uncertainty in memory.
  Recheck prerequisites before dependent actions. A missing detection or exhausted
  retry count does not establish completion. Reobserve after contact, occlusion
  or an unexpected outcome.
- Use training rollouts to develop recovery behavior. When progress stalls or
  a grasp misses, investigate the cause and test a changed approach, orientation,
  retreat or viewpoint as appropriate. Bound retries and motion time; avoid
  repeating ineffective actions without adjustment.

Use your own permitted rollout evidence to repair and verify complete sequences.
Check changes offline, validate and snapshot. Use the entire assigned rollout
budget: for a 50-attempt assignment, use all 50, even when ready or already
successful. Unchanged-code verification is useful. When assigned the demo
identities, interact with those same identities once each; do not introduce new
seeds or repeat consumed identities. The harness selects the next identity.
Failed and diagnostic attempts count. Never exceed the assigned budget.
Report which revision each attempt tested and whether the final revision was verified.
