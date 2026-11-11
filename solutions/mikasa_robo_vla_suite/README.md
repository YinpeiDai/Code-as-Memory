# MIKASA-Robo · five-task publication comparison

These are the five MIKASA task identities used in prior VLA publications:
ShellGameTouch, InterceptMedium and RememberColor3/5/9 (`-VLA-v0` IDs).
They share the full benchmark's simulator and public interface. We ran both
this comparison and the full 90-task benchmark; their policies and counts remain distinct.

| Setting | Policies | Successful/total | Success rate |
|---|---:|---:|---:|
| [25 demos only](demo_only_25/README.md) | 5 | 153/250 | 61.2% |
| [50 demos only](demo_only_50/README.md) | 5 | 175/250 | 70.0% |
| [Up to 250 demos only](demo_only_250/README.md) | 5 | 197/250 | 78.8% |
| [50 demos + 50 interactions](demo_interaction_50/README.md) | 5 | 250/250 | 100% |

All published policy code is byte-identical to its frozen development checkpoint.
Each setting README contains evaluation commands and development provenance.
No demonstrations or model calls are required to evaluate published code.

[Both MIKASA scopes and per-task results](../../docs/results/mikasa_robo.md) ·
[Reproduction](../../docs/reproducing.md#mikasa-five-task-publication-comparison) ·
[Immutable catalogs](../../configs/solutions/mikasa_robo_vla_suite/)
