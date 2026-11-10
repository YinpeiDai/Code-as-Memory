# MIKASA-Robo · demo_interaction_50

**4,444/4,500 successful held-out episodes (98.76%)**, with complete scored coverage.
This setting contains 90 final policies. Their executable Python bytes are
preserved; schemas and provenance are in the [solution catalog](../../../configs/solutions/mikasa_robo/demo_interaction_50.json).

We also developed fresh policies on the five task identities used in prior VLA
publications. Their [three settings](../../mikasa_robo_vla_suite/README.md) and
[results](../../../docs/results/mikasa_robo.md) are part of MIKASA and
are reported separately from this full-90 historical release.

Each task uses 50 demos and one interaction on each of those exact identities, with CPU physics. All 90 policies and 4,500 result records are available. Full development/video evidence covers the initial ten tasks; the other 80 are imported results. Their missing historical source snapshots and reset-seed evidence are not independently verified.

## Evaluate

```bash
cam prepare validate_runs --solution solutions/mikasa_robo/demo_interaction_50 --tasks SeqOfColors3-VLA-v0
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

Install the [MIKASA-Robo environment](../../../envs/mikasa_robo/README.md) first.
Omit `--tasks` to prepare all policies. No demonstration download or model call
is needed to evaluate frozen code. A new run never replaces the published score.

## Develop this setting

Shared core + demo-first interaction mode + MIKASA public interface + `prompts/mikasa_robo/both.md`. SeqOfColors3 also gets its explicitly documented cue-order clarification. Initial-ten and imported policies used different historical prompt revisions; do not substitute the current prompt hash for either.

Use `configs/mikasa_robo/experiments/demo_interaction_50.json`, a downloaded demonstration package,
and the [development commands](../../../docs/benchmarks/mikasa_robo.md).
Each new preparation saves its exact prompt. Historical prompt/configuration
provenance remains attached to the source run; current wording is not a claim
of exact historical reproduction.
