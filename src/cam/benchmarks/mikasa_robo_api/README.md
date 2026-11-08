# MIKASA-Robo robot queries

`preview.py` and `kinematics.py` are host-only. `preview.py` checks the next native GPU controller
arm target for a normalized delta command or a bounded step toward a world
absolute pose. It reads robot proprioception, controller parameters and joint
limits. It never calls `step`, `evaluate`, `set_action`, pose setters or GPU
apply operations, and never reads task objects or scoring state.

`api.md` documents live queries. The generic
`Robot` RPC client dispatches to the private broker and MIKASA adapter. Requests
and mechanical results are recorded in `dry_runs.jsonl`. `kinematics.py` retains
the older v2 absolute IK query for compatibility; explicit modes select the
native next-command preview.

The separate public offline helper `offline.py` and sanitized `panda_model.json`
are copied into the MIKASA runtime/reference with an initializer and
`safety_check.md`. They contain robot math and mechanical parameters only. The
offline API reads supplied state/actions, never a simulator, task or scorer.
URDF effort limits differ from native PD drive caps; both are named explicitly.
Initial PD-demand estimates require joint velocities and a target. Neither these
estimates nor supplied torque comparisons predict future contact forces.

Unit and sandbox checks: `pytest tests/test_mikasa_dry_run.py`.
For a recorded training rollout, run the host-only regression with the simulator
interpreter and a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src envs/mikasa_robo/.venv/bin/python \
  scripts/check_mikasa_dry_run.py RUN --attempt 1 --output NEW_DIAGNOSTIC_DIR
```

It checks repeated queries leave simulator/controller state unchanged, compares
predicted targets with the actual native controller, and requires recorded RGB,
TCP, gripper and step values to match exactly. Diagnostic output is not learner
feedback. Prepare fresh runs after changing the interface or implementation.
