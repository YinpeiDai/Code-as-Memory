Learn from the released expert demonstrations only (R0).

Inspect every assigned training episode through cap_agent.trajectory.Trajectory.
Read the goal, distinguish the task's video-demo phase from expert robot execution,
and inspect both front and wrist camera images. You can extract PNG images or
contact sheets into your workspace and view them with the Read tool. Correlate
visual events with the recorded actions, robot states and camera calibration.
You may sample frames for initial inspection and inspect all available frames
around important events; fixed recordings can be reused without limit.

Infer a general policy from this evidence; do not replay the recorded actions or
hard-code a demonstration's objects, coordinates, counts or timing. Make the
generated Python use current observations and the goal to decide what to do.

Never initialize a simulator, including for a preview or debugging. No simulator
training attempts are allowed. Use offline analysis, synthetic inputs and static
validation to check the code. Save every coding iteration and finish with a valid
policy plus a report of what the recordings support and what remains uncertain.
