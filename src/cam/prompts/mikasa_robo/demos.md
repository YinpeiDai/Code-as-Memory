## MIKASA-Robo demo-only guidance

- Measure cue, delay and execution phases from the demos. Retain the observations needed after a cue disappears, and distinguish observed events from inferred progress in memory.
- Commands are incremental targets relative to the measured pose. Compare commands with recorded TCP and joint motion; check units, rotation signs, scaling, clipping and Euler-angle wrapping. A small one-step response does not justify a large feedback gain.
- Budget cue time, movement and settling within the native step limit. Each `robot.step` advances one control step; Python sleep does not advance simulation.
- Use the supplied offline `safety_check` on recorded or synthetic robot states. Missing velocity or torque is unknown, not zero. These checks do not establish collision avoidance, physical stability or task success. No simulator rollouts or live previews are allowed during demo-only development.
