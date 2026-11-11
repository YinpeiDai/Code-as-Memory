# MIKASA-Robo · up to 250 demos only

Five Robot Memory policies achieve **197/250 successful held-out episodes (78.8%)**.
Development uses successful 256×256 RGB-D replays and **zero training interactions**.

| Task | Demos | Successful/total |
|---|---:|---:|
| `ShellGameTouch-VLA-v0` | 237 | 50/50 |
| `InterceptMedium-VLA-v0` | 56 | 30/50 |
| `RememberColor3-VLA-v0` | 250 | 50/50 |
| `RememberColor5-VLA-v0` | 250 | 49/50 |
| `RememberColor9-VLA-v0` | 246 | 18/50 |

After [installing MIKASA](../../../envs/mikasa_robo/README.md), evaluate all five:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo_vla_suite/demo_only_250 \
  --output runs/mikasa_vla_eval/demo_only_250 --gpus 0 1 --workers-per-gpu 5
```

Evaluation needs no demos or model calls. ShellGameTouch uses GPU physics/IK;
the other four tasks use CPU physics/IK. All use GPU rendering.

These files exactly match the frozen policies from `mikasa_robo_vla_suite_250`.
The [catalog](../../../configs/solutions/mikasa_robo_vla_suite/demo_only_250.json)
pins memory schemas, code hashes, prompts, assigned budgets and original results.
[Development evidence](https://huggingface.co/datasets/Yinpei/CaM_develop_logs/blob/main/archives/mikasa_robo_vla_suite_demo_only_250.tar.zst)
extracts to `public_runs/dev_logs/mikasa_robo_vla_suite/demo_only_250`.
The [RGB-D demos](https://huggingface.co/datasets/Yinpei/mikasa_robo-vla-suite-250-demos)
download separately to `public_runs/demos/mikasa_robo_vla_suite_250_demo`.

[Results and prompt provenance](../../../docs/results/mikasa_robo.md) ·
[Develop new policies](../../../docs/benchmarks/mikasa_robo.md)
