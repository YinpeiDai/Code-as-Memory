# Reproducible settings and policy metadata

Keep these files with the source: they are required for reproduction.

- `<benchmark>/*_<client>.json`: coding client, learning mode, budgets, and runtime limits.
- `<benchmark>/experiments/`: portable all-task plans; `--dataset`, `--output`, and `--tasks` select local inputs and storage.
- `solutions/<benchmark>/`: published policy memory schemas, checksums, interface identities, source provenance, and evaluation settings.
- `mikasa_robo_vla_suite/`: five-task comparison plans for 25 demos, 50 demos,
  and 50 demos plus 50 interactions. The simulator benchmark remains `mikasa_robo`;
  the explicit `suite` field keeps paths and reported results separate from all 90 tasks.

## MIKASA: two evaluation scopes

`mikasa_robo/experiments/demo_interaction_50.json` selects the full 90 tasks.
`mikasa_robo_vla_suite/` selects the five tasks used in prior VLA publications,
so we report both broad benchmark coverage and comparison on the same task list.
These are two scopes of one benchmark, sharing its environment and demo package.

[Five-task plans, exact budgets and commands](mikasa_robo_vla_suite/README.md) ·
[Reproducing both scopes](../docs/reproducing.md) ·
[MIKASA results](../docs/benchmarks/mikasa_robo/results.md)

Only final Python programs and setting READMEs belong in `solutions/`.
Historical imports and catalog hashes remain unchanged. RMBench's
`*_observed_motion.json` catalogs target the current public interface; original
catalogs remain provenance for historical scores. Superseded experimental plans
are archived locally under ignored `legacy/`.

The default coding plans use Claude Code / Opus 5.5. Configure the referenced
client settings for your own account before development. Both clients share the
same assigned evidence and isolated tool interface.
