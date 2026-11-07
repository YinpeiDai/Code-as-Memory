# Statistics contract

The evaluation unit is `(run, task, requested interaction budget, test episode)`.
A full regular RoboMME setting has **16 × 50 = 800 episodes**; a one-task run has 50.

| Measure | Definition |
|---|---|
| `successes` | Native terminal successes among valid scored episodes |
| `success_rate` | Successes / expected episodes, only when all are scored; otherwise `null` |
| `observed_success_rate` | Successes / scored episodes so far, explicitly partial |
| `harness_error` | Infrastructure outcome; excluded from the partial denominator and prevents a complete rate |
| Efficiency | Mean native execution control steps on successful episodes, including native penalties; not robot API call count |
| Paired efficiency | Both means on the same successful `(task, episode)` intersection; requires matching benchmark hashes and reports the matched count |
| Behavior | Observed decisions and terminal statuses, without inferred reasoning labels |
| Difficulty | Native test metadata, not training proportions |

Native failure/timeout/error, generated-code error, policy timeout, and decision
limit are scored policy outcomes. Native status is authoritative after a terminal
event, even if the policy then closes its transport. Infrastructure errors must
be diagnosed and cannot be dropped from a claimed complete evaluation.

Each plotted point is one exact completed evaluation. Lines connect observed
budgets; no fitted curves, confidence bands, or invented replicates. Partial
rates are labeled and excluded from final curves. Figure exports follow the same
rules.

Development cost is counted once per run across all retained sessions, not once
per evaluated checkpoint. Claude request IDs deduplicate stream events; final
session usage takes precedence. Claude dollars are **CLI estimates**, not invoices.
Codex token totals are retained; unreported dollars are `null`, not zero.
Known-dollar subtotals and completeness are explicit. GPU/CPU/storage costs are
excluded. Frozen policy execution makes zero inference calls and incurs zero
model API cost.

Historical CAP results differ in data, sensors, prompts, and benchmark revisions.
They require a provenance-aware importer before inclusion in CaM comparisons.
