# MIKASA-Robo · demo_interaction_50

**4,444/4,500 successful held-out episodes (98.76%)**, with complete scored coverage.
This setting contains 90 final policies. Their executable Python bytes are
preserved; schemas and provenance are in the [solution catalog](../../../configs/solutions/mikasa_robo/demo_interaction_50.json).

The primary publication comparison uses the [five-task VLA suite](../../mikasa_robo_vla_suite/README.md).
This full-benchmark release used 50 demos and one interaction per assigned demo
identity, with CPU physics. Current evaluation uses GPU physics for ShellGameTouch.
Full historical development evidence is unavailable for some tasks; current
compatibility checks do not establish that provenance.
[Results and scopes](../../../docs/results/mikasa_robo.md).

## Evaluate

```bash
cam prepare runs --solution solutions/mikasa_robo/demo_interaction_50 --tasks SeqOfColors3-VLA-v0
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

Install the [MIKASA-Robo environment](../../../envs/mikasa_robo/README.md) first.
Omit `--tasks` to prepare all policies. No demonstration download or model call
is needed to evaluate frozen code. A new run never replaces the published score.

## Develop this setting

Shared core + demo-first interaction mode + MIKASA public interface + `prompts/mikasa_robo/both.md`. SeqOfColors3 also gets its explicitly documented cue-order clarification. Historical policies used earlier prompt revisions; do not substitute the current prompt hash for historical evidence.

Use `configs/mikasa_robo/experiments/demo_interaction_50.json`, a downloaded demonstration package,
and the [development commands](../../../docs/benchmarks/mikasa_robo.md).
Each preparation pins its exact prompt; original runs retain their own provenance.
