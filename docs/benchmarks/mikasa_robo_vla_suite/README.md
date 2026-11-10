# MIKASA-Robo VLA comparison suite

`mikasa_robo_vla_suite` selects five standard tasks from the full 90-task
`mikasa_robo` benchmark:

- `ShellGameTouch-VLA-v0`
- `InterceptMedium-VLA-v0`
- `RememberColor3-VLA-v0`
- `RememberColor5-VLA-v0`
- `RememberColor9-VLA-v0`

This is a comparison namespace using the existing MIKASA simulator, environment,
metadata and public interface. It does not duplicate or rename the full benchmark.
The `suite` configuration field is pinned in every run; directory reports keep
the two result tracks separate.

| Setting | Demos/task | Training attempts/task | Held-out episodes/task |
|---|---:|---:|---:|
| `demo_only_25` | 25 | 0 | 50 |
| `demo_only_50` | 50 | 0 | 50 |
| `demo_interaction_50` | 50 | 50, one per assigned demo identity | 50 |

All 15 policies start blank with Claude Code / Opus 5.5 and the current shared
prompts and MIKASA public guidance. The 25-demo arm materializes only the exact
published selected identities. Every arm uses the same held-out identities,
native scoring and limits. ShellGameTouch uses GPU physics/IK; the other four
use CPU physics/IK. All use the existing 256×256 RGB-D interface and GPU rendering.

```bash
.venv/bin/python scripts/develop_suite.py start \
  configs/mikasa_robo_vla_suite/suite.json \
  --dataset /path/to/mikasa_robo/demos --output runs --gpus 0 1
```

The command returns a campaign directory under `runs/mikasa_robo_vla_suite/`.
Three settings share eight coding workers (3/3/2), one training simulator lease
per GPU, and a serialized evaluation queue with five episode workers per GPU.
Frozen policies are evaluated automatically with no model calls. Completed arms
export exact executable bytes to `solutions/mikasa_robo_vla_suite/<setting>/`
and immutable catalogs to `configs/solutions/mikasa_robo_vla_suite/<setting>.json`
inside the campaign. `README.md` and `status.json` track all 15 policies and actual
scored/expected coverage. Failed attempts remain available for review.

```bash
.venv/bin/python scripts/develop_suite.py status CAMPAIGN
```

The task selection supports comparison with the five-task table in
[SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533). Report data budgets,
interaction access, observations, physics/IK, evaluation count and model costs
alongside scores. Matching the task set alone does not establish identical
training or evaluation protocols. The paper's VLA baseline values are attributed
to MemoryVLA++; its GMP reference uses per-task training and tuned horizons.

Keep the [full MIKASA results](../mikasa_robo/README.md) separate. A revised
90-task aggregate incorporating the GPU ShellGameTouch evaluation must identify
the replaced task and its new policy/evaluation provenance, retain the original
aggregate, and disclose the imported 80 tasks' missing historical evidence.
