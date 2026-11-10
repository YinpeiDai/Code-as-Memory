# Reproducible settings and policy metadata

Keep these files with the source: they are required for reproduction.

- `<benchmark>/*_<client>.json`: coding client, learning mode, budgets, and runtime limits.
- `<benchmark>/experiments/`: portable all-task plans; `--dataset`, `--output`, and `--tasks` select local inputs and storage.
- `solutions/<benchmark>/`: published policy memory schemas, checksums, interface identities, source provenance, and evaluation settings.
- `mikasa_robo_vla_suite/`: five-task comparison plans for 25 demos, 50 demos,
  and 50 demos plus 50 interactions. The simulator benchmark remains `mikasa_robo`;
  the explicit `suite` field keeps paths and reported results separate from all 90 tasks.

Only final Python programs and setting READMEs belong in `solutions/`.
Historical imports and catalog hashes remain unchanged. RMBench's
`*_observed_motion.json` catalogs target the current public interface; original
catalogs remain provenance for historical scores. Superseded experimental plans
are archived locally under ignored `legacy/`.

The default coding plans use Claude Code / Opus 5.5. Configure the referenced
client settings for your own account before development. Both clients share the
same assigned evidence and isolated tool interface.
