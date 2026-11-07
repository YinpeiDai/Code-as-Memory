## Learn from demonstrations and interaction

Inspect every assigned demonstration and develop an initial program offline.
Check perception, goal parsing and control using the recordings and synthetic
inputs. Use counted simulator attempts to verify capabilities or resolve stated
uncertainties, revisiting relevant recordings after feedback.

Before each rollout, state its purpose and expected observations. Afterward,
record the outcome and the resulting code change. Assigned training episodes
cycle automatically. Failures, initialization, diagnostics and early stops all
consume the same total budget. Stop early if justified; finish offline validation.
The driver saves R_N before attempt N+1 and freezes the final validated code.
