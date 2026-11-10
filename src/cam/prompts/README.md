# Development prompts

`cam prepare` composes and pins the complete prompt in each task's `prompt.md`.
The shared `core.md` and `modes/` apply across benchmarks. Every prompt also includes
its benchmark's public interface guide from `src/cam/interface/<benchmark>/`.

| Benchmark | Additional guidance |
|---|---|
| [RoboMME](robomme/README.md) | Public observation/action API and the shared learning mode |
| [MIKASA-Robo](mikasa_robo/README.md) | Demo-first interaction guidance; one documented cue-order clarification |
| [RMBench](rmbench/README.md) | Public motion/observation API; shared demo-only prompt |
| [RoboMemArena](robomemarena/README.md) | Demo-first interaction guidance with full assigned-budget use |

Use each benchmark's plan in `configs/<benchmark>/experiments/`. New runs use the
current prompt. Historical scores retain their original prompts and hashes;
setting READMEs explain the differences. Evaluation of frozen solutions runs no prompt.
