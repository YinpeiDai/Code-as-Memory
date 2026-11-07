## Learn from interaction

Expert recordings and expert actions are unavailable. Learn through the counted
`rollout` tool on the assigned training episodes, which cycle automatically.
The first attempt is also an opportunity to observe the task. State what each
attempt will test; inspect its observations, actions and outcome before revising.

All attempts count, including initialization failures, crashes, early stops and
diagnostics. There are no free simulator probes. At the budget limit, finish
offline checks and validation. Checkpoints contain only feedback available before
the next attempt; the final validated code is the submitted policy.
