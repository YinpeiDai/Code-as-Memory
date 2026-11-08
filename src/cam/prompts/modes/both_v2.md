## Demonstrations + interaction v2: demos first, verify and explore

Learn primarily from the assigned demonstrations. Before the first simulator
attempt, inspect every assigned recording, develop an initial general Python
policy offline, and check perception, goal parsing and control using the
recordings and synthetic inputs where useful. Record episode/frame evidence,
assumptions, offline check results and unresolved questions in task_analysis.md.
Call validate and snapshot before starting interaction. Offline checks do not
establish physical success.

Then use rollouts to verify the solution and resolve what the demos leave
uncertain, such as action effects, timing, or what counts as success. Before each
rollout, briefly record what you want to check. Verifying unchanged code on a new
assigned seed is enough; no new hypothesis or code change is required.
A focused exploratory rollout can also help resolve an uncertainty even if it
is unlikely to complete the task. Use only permitted observations and training
feedback.

After each rollout, inspect the result, revisit demos as needed, and record the
findings and any code changes in CHANGELOG.md. Fix issues supported by the
evidence, check the changes offline, validate and snapshot.

When the assignment provides distinct training seeds, use all remaining rollout
attempts to verify and strengthen the solution, even if it already seems ready
for testing. Continue through the assigned seed pool and fix any issues you find.
Successful verification does not require a code change. With a repeating seed
schedule, explain any early stopping.

Only rollout provides simulator access. The harness selects the next assigned
training seed automatically. Every attempt counts, including failures,
initialization errors and diagnostics. Never exceed the budget or request
unassigned seeds or uncounted probes. Finish offline validation and report
attempts used, remaining uncertainties, and whether the final code was verified
by a rollout. The driver saves R_N before attempt N+1 and freezes the final
validated code.
