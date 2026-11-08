## Demonstrations + interaction v1: choose evidence freely

Develop a general Python policy using the assigned demonstrations and counted
training interactions. Decide freely when to inspect demonstrations, write code,
run offline checks, or interact with the simulator. There is no required
demo-first phase or fixed alternation. An early diagnostic interaction may guide
which recordings or events to inspect; demonstrations remain available throughout
development.

Choose the evidence that best resolves the current question. Compare feedback
with relevant demonstration windows and check perception, goal parsing and
control offline when useful. Before each rollout, record its purpose and expected
observations in task_analysis.md. Afterward, inspect feedback and record the
outcome, interpretation and resulting code changes in CHANGELOG.md. Avoid blind
retries; preserve evidence and snapshot each complete iteration.

Only rollout provides simulator access. Training episodes follow the assignment's
pinned schedule automatically. Failures, initialization, diagnostics and early stops all consume
the same total attempt budget. Stop early if justified; finish offline validation
and report actual attempts used and remaining uncertainties. The driver saves
R_N before attempt N+1 and freezes the final validated code.
