# 🧠 Code-as-Memory: Robots That Remember with Code

**What if robot memory could just be an executable Python program?**

**Code-as-Memory (CaM)** introduces **executable code as explicit, persistent robot memory**. A coding agent distills demonstrations into a Python program that maintains and updates memory through structured variables (e.g., `dict`, `list`), enabling robots to **perceive, remember, reason and act** entirely through Python execution, without relying on any external models during inference.

⚡ **No Training. No Model Calls. Near-Zero Latency.**
- **No model training** during development
- **Zero model/API calls** during inference
- **<5 ms inference latency** through lightweight Python execution

CaM achieves **near-perfect performance across four robot memory benchmarks**, outperforming prior approaches at a fraction of the **computational cost, financial expense, and inference latency**.

![CaM success rates and API costs across RMBench, MIKASA-robo, RoboMME, and RoboMemArena](docs/figures/benchmark_comparison.png)


[More results can be found here →](docs/results.md)


## Rollout Examples

### RMBench

**Task: swap_blocks**

> **Task Goal:** There are three traies on the table, and two blocks are placed in two different traies. You may move only one block at a time, and each tray can hold at most one block. Swap the positions of the two blocks. Finally press the button.

https://github.com/user-attachments/assets/1e420cf9-cc9e-45bd-bab0-020dd7fbaa15

**Task: cover_blocks**

> **Task Goal:** On the table, red, green, and blue blocks are arranged randomly along with three lids. From the current viewpoint, cover the blocks from left to right using the lids, and then uncover them again in the sequence red, green, and blue.

https://github.com/user-attachments/assets/01af08e7-6f9c-444a-af09-63cfa6ce91b8

### MIKASA-Robo

**Task: BatteriesCheckerEasy-3-VLA-v0**

> **Task Goal:** Find all working batteries by inserting each one into the socket, observing the lamp result, and then pressing the button to confirm.

https://github.com/user-attachments/assets/1e8cb1f9-d766-495d-b78b-b522a689a100

**Task: SeqOfColors3-VLA-v0**

> **Task Goal:** Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button.

https://github.com/user-attachments/assets/dd791a43-1ff3-490f-a2fc-8ee6e420bce1

### RoboMME

**Task: PatternLock**

> **Task Goal:** watch the video carefully, then use the stick attached to the robot to retrace the same pattern

https://github.com/user-attachments/assets/2292461f-dd57-4521-8db6-99f427e4d449

**Task: VideoUnmaskSwap**

> **Task Goal:** watch the video carefully, then pick up the container hiding the green cube, finally pick up another container hiding the blue cube

https://github.com/user-attachments/assets/c535af07-06f6-4fa2-8a74-d7a1595f0425

### RoboMemArena

**Task: T04_ButterNonEmptyDrawer**

> **Task Goal:** Open and close each drawer to check it, in this order: top, middle, bottom. Put butter into the drawer that already contains an object.

https://github.com/user-attachments/assets/77102ab7-a06a-40c5-a770-db522e6a6fb2

**Task: T07_PourSauceFrypanTwice**

> **Task Goal:** Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer.

https://github.com/user-attachments/assets/d044e7d0-28ba-464c-9f66-5654e48bafb2

## Try it

[Install the shared harness and a benchmark environment](docs/getting_started.md),
then evaluate its published memory programs. No demonstrations or model credentials
are needed for this step.

```bash
# All 16 RoboMME tasks; five concurrent evaluation jobs per GPU.
python scripts/evaluate_solutions.py solutions/robomme/demo_only_25 \
  --output validate_runs/robomme --gpus 0 1 --workers-per-gpu 5
```

| Benchmark | Published memory programs | Run and reproduce |
|---|---|---|
| RMBench | [50 demonstrations](solutions/rmbench/demo_only_50/) | [Guide](docs/benchmarks/rmbench/README.md) |
| MIKASA-Robo | [50 demonstrations + interaction](solutions/mikasa_robo/demo_interaction_50/) | [Guide](docs/benchmarks/mikasa_robo/README.md) |
| RoboMME | [25 demonstrations](solutions/robomme/demo_only_25/) | [Guide](docs/benchmarks/robomme/README.md) |
| RoboMemArena | [50 demonstrations + interaction](solutions/robomemarena/demo_interaction_50/) | [Guide](docs/benchmarks/robomemarena/README.md) |

## Reproduce development

The demonstrations are successful trajectories collected through replay:
**25 per task for RoboMME and up to 50 per task for the other benchmarks**.
CaM develops memory code using only the assigned trajectories and, when interaction
is enabled, training rollouts from those same demonstration seeds.

[Download the demonstrations and configure your coding client](docs/getting_started.md),
then choose a data path and a directory for development logs:

```bash
python scripts/develop_batch.py start configs/robomme/experiments/demo_only_25.json \
  --dataset /path/to/demos/robomme --output validate_runs/robomme_development \
  --workers 8 --gpus 0 1
```


Apache-2.0. Benchmark code and assets retain their upstream licenses; see [NOTICE](NOTICE).
