# CaM · Robot Memory as Code

[Results](#results) · [Try it](#try-it) · [Demonstrations](docs/getting_started.md#demonstration-packages) · [Documentation](docs/README.md)

**CaM learns robot memory as code.**

A coding agent turns demonstrations and optional practice into a Python program
that remembers, decides, and acts.

The learned program carries memory across observations and runs with
**zero model calls during evaluation**.

## Results

| Benchmark | Best setting | Successful / evaluated | Success rate |
|---|---|---:|---:|
| RoboMME | 25 demonstrations | 795 / 800 | **99.38%** |
| RMBench | Up to 50 demonstrations | 1,000 / 1,000 | **100.00%** |
| MIKASA-Robo | 50 demonstrations + 50 interactions | 4,444 / 4,500 | **98.76%** |
| RoboMemArena | Up to 50 demonstrations + matching interactions | 1,271 / 1,300 | **97.77%** |

![CaM and selected published baselines across RoboMME, RMBench, MIKASA-Robo, and RoboMemArena](docs/figures/benchmark_comparison.png)

The comparison uses nine shared RMBench tasks and five shared MIKASA task identities
(CaM: **90.0%**); the table above covers each full suite. Published methods use
different training and evaluation protocols.

[More results, coverage, and provenance →](docs/results.md)

## Learn once, execute as code

RoboMME's 16 memory programs used **$29.44 in reported development API cost**.
Their 800 held-out evaluations made **zero model calls**.

![RoboMME memory-decision timing distribution and development versus evaluation API cost](docs/figures/robomme_efficiency.png)

Timing measures perception and memory decisions on recorded observations, excluding
action-tool execution and simulation. [Measurement details and figure sources](docs/figures/README.md).

## Robot memory in action

Selected successful held-out episodes. Click an animation for the full recording.
Previews adjust playback speed and camera framing; [episode sources and details](docs/media/README.md).

<table>
  <tr><th colspan="2">RoboMME</th></tr>
  <tr>
    <td width="50%"><a href="docs/media/robomme-pattern-lock.mp4"><img src="docs/media/robomme-pattern-lock.gif" width="100%" alt="RoboMME: the robot retraces a previously shown pattern"></a></td>
    <td width="50%"><a href="docs/media/robomme-hidden-objects.mp4"><img src="docs/media/robomme-hidden-objects.gif" width="100%" alt="RoboMME: the robot uncovers remembered objects after their containers move"></a></td>
  </tr>
  <tr>
    <td><b>Remember a pattern</b><br>Watch, remember, and retrace.<br><sub>PatternLock</sub></td>
    <td><b>Track hidden objects</b><br>Find the right objects after they disappear from view.<br><sub>VideoUnmaskSwap</sub></td>
  </tr>
  <tr><th colspan="2">RMBench</th></tr>
  <tr>
    <td><a href="docs/media/rmbench-swap-blocks.mp4"><img src="docs/media/rmbench-swap-blocks.gif" width="100%" alt="RMBench: the robot swaps two blocks through an empty tray"></a></td>
    <td><a href="docs/media/rmbench-cover-blocks.mp4"><img src="docs/media/rmbench-cover-blocks.gif" width="100%" alt="RMBench: the robot covers blocks and uncovers them in color order"></a></td>
  </tr>
  <tr>
    <td><b>Keep track of a swap</b><br>Move one block at a time through an empty tray.<br><sub>swap_blocks</sub></td>
    <td><b>Remember what is covered</b><br>Uncover hidden blocks in the requested color order.<br><sub>cover_blocks</sub></td>
  </tr>
  <tr><th colspan="2">MIKASA-Robo</th></tr>
  <tr>
    <td><a href="docs/media/mikasa-batteries.mp4"><img src="docs/media/mikasa-batteries.gif" width="100%" alt="MIKASA-Robo: the robot tests batteries and remembers the working ones"></a></td>
    <td><a href="docs/media/mikasa-color-sequence.mp4"><img src="docs/media/mikasa-color-sequence.gif" width="100%" alt="MIKASA-Robo: the robot recalls a sequence of colored cubes"></a></td>
  </tr>
  <tr>
    <td><b>Remember which batteries work</b><br>Test, observe, and retain each result.<br><sub>BatteriesCheckerEasy-3</sub></td>
    <td><b>Recall a color sequence</b><br>Use an earlier cue to guide later actions.<br><sub>SeqOfColors3</sub></td>
  </tr>
  <tr><th colspan="2">RoboMemArena</th></tr>
  <tr>
    <td><a href="docs/media/robomemarena-drawers.mp4"><img src="docs/media/robomemarena-drawers.gif" width="100%" alt="RoboMemArena: the robot inspects drawers and returns to the occupied drawer"></a></td>
    <td><a href="docs/media/robomemarena-pour-twice.mp4"><img src="docs/media/robomemarena-pour-twice.gif" width="100%" alt="RoboMemArena: the robot pours sauce twice and puts the bottle away"></a></td>
  </tr>
  <tr>
    <td><b>Remember what is inside</b><br>Inspect the drawers, then return to the right one.<br><sub>T04 · ButterNonEmptyDrawer</sub></td>
    <td><b>Count completed pours</b><br>Pour twice, then put the bottle away.<br><sub>T07 · PourSauceFrypanTwice</sub></td>
  </tr>
</table>

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
| RoboMME | [25 demonstrations](solutions/robomme/demo_only_25/) | [Guide](docs/benchmarks/robomme/README.md) |
| RMBench | [50 demonstrations](solutions/rmbench/demo_only_50/) | [Guide](docs/benchmarks/rmbench/README.md) |
| MIKASA-Robo | [50 demonstrations + interaction](solutions/mikasa_robo/demo_interaction_50/) | [Guide](docs/benchmarks/mikasa_robo/README.md) |
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

**Eight concurrent development jobs; five evaluation jobs per GPU.**
[Batch evaluation, prompts, and reproduction details](docs/reproducing.md) ·
[Five-episode sanity checks](docs/solutions.md#sanity-checks) ·
[Live dashboard and experiment galleries](docs/website.md)

[All documentation](docs/README.md) · [Architecture](docs/architecture.md) ·
[Selected revalidation tasks](docs/validatd_task.md)

Apache-2.0. Benchmark code and assets retain their upstream licenses; see [NOTICE](NOTICE).
