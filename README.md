# 🧠 Code-as-Memory (CaM): Robots That Remember with Code

**What if robot memory could just be an executable Python program?**

**Code-as-Memory (CaM)** introduces **executable code as explicit, persistent robot memory**. A coding agent distills demonstrations into a Python program that maintains and updates memory through structured variables (e.g., `dict`, `list`), enabling robots to **perceive, remember, reason and act** entirely through Python execution, without relying on any external models during inference.

⚡ **No Training. No Model Calls. Near-Zero Latency.**
- **No model training** during development
- **Zero model/API calls** during inference
- **<5 ms inference latency** through lightweight Python execution

CaM achieves **near-perfect performance across four robot memory benchmarks**, outperforming prior approaches at a fraction of the **computational cost, financial expense, and inference latency**.

![CaM success rates and API costs across RMBench, MIKASA-robo, RoboMME, and RoboMemArena](docs/figures/benchmark_comparison.png)


[More results can be found here →](docs/results.md)


## Successful Evaluation Examples

### RMBench

**swap_blocks**

> There are three traies on the table, and two blocks are placed in two different traies. You may move only one block at a time, and each tray can hold at most one block. Swap the positions of the two blocks. Finally press the button.

[![Play swap_blocks: recorded memory, positions and end-effector trail](docs/media/rmbench-swap-blocks.jpg)](docs/media/rmbench-swap-blocks.mp4)

**cover_blocks**

> On the table, red, green, and blue blocks are arranged randomly along with three lids. From the current viewpoint, cover the blocks from left to right using the lids, and then uncover them again in the sequence red, green, and blue.

[![Play cover_blocks: recorded memory, positions and end-effector trail](docs/media/rmbench-cover-blocks.jpg)](docs/media/rmbench-cover-blocks.mp4)

### MIKASA-Robo

**BatteriesCheckerEasy-3-VLA-v0**

> Find all working batteries by inserting each one into the socket, observing the lamp result, and then pressing the button to confirm.

[![Play BatteriesCheckerEasy-3-VLA-v0: recorded memory, positions and end-effector trail](docs/media/mikasa-batteries.jpg)](docs/media/mikasa-batteries.mp4)

**SeqOfColors3-VLA-v0**

> Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button.

[![Play SeqOfColors3-VLA-v0: recorded memory, positions and end-effector trail](docs/media/mikasa-color-sequence.jpg)](docs/media/mikasa-color-sequence.mp4)

### RoboMME

**PatternLock**

> watch the video carefully, then use the stick attached to the robot to retrace the same pattern

[![Play PatternLock: recorded memory, positions and end-effector trail](docs/media/robomme-pattern-lock.jpg)](docs/media/robomme-pattern-lock.mp4)

**VideoUnmaskSwap**

> watch the video carefully, then pick up the container hiding the green cube, finally pick up another container hiding the blue cube

[![Play VideoUnmaskSwap: recorded memory, positions and end-effector trail](docs/media/robomme-hidden-objects.jpg)](docs/media/robomme-hidden-objects.mp4)

### RoboMemArena

**T04_ButterNonEmptyDrawer**

> Open and close each drawer to check it, in this order: top, middle, bottom. Put butter into the drawer that already contains an object.

[![Play T04_ButterNonEmptyDrawer: recorded memory, positions and end-effector trail](docs/media/robomemarena-drawers.jpg)](docs/media/robomemarena-drawers.mp4)

**T07_PourSauceFrypanTwice**

> Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer.

[![Play T07_PourSauceFrypanTwice: recorded memory, positions and end-effector trail](docs/media/robomemarena-pour-twice.jpg)](docs/media/robomemarena-pour-twice.mp4)

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
