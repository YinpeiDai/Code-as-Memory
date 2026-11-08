## Learn from demonstrations

Inspect every assigned recording: instruction, setup variation, both cameras,
task-video phase and expert execution. Use RGB-D, calibration and robot state
when supplied. Revisit relevant events and test generated perception and parsing
against the recordings; use synthetic inputs to exercise control logic.

Derive the memory representation from the demonstrations: which observations
must persist, when memory updates or resets, and how it controls later actions.
Infer candidate completion conditions from expert behavior and final recorded
states, and cross-check them across every assigned recording. Distinguish
evidence-supported conditions from unresolved hypotheses; successful examples
alone do not uniquely identify the simulator's exact success test. Document
the evidence and uncertainties in the policy notes.

Zero simulator attempts are allowed, including resets, live previews and debugging.
An explicitly supplied pure offline robot-model API may be used on recorded or
synthetic robot states; it must not create, reset or step a simulator.
`rollout` is disabled. Reuse the assigned recordings without limit, then validate
and submit the general program with its checks and remaining uncertainties.
