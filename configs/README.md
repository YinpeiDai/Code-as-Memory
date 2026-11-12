# Reproducible settings and policy metadata

Keep these files with the source: they are required for reproduction.

- `<benchmark>/demos.json`, `both.json`, `interaction.json` (or versioned modes): run settings, budgets, runtime limits and the selected coding client.
- `<benchmark>/experiments/`: portable all-task plans; `--dataset`, `--output`, and `--tasks` select local inputs and storage.
- `solutions/<benchmark>/`: published policy memory schemas, checksums, interface identities, source provenance, and evaluation settings.
- `mikasa_robo_vla_suite/`: five-task comparison plans for 25 demos, 50 demos,
  and 50 demos plus 50 interactions. The simulator benchmark remains `mikasa_robo`;
  the explicit `suite` field keeps paths and reported results separate from all 90 tasks.
- `mikasa_robo_vla_suite_250/`: the same five tasks with up to 250 successful
  RGB-D demos and zero interaction. Its dataset is
  `public_runs/demos/mikasa_robo_vla_suite_250_demo`; published code and its catalog
  use the `mikasa_robo_vla_suite/demo_only_250` setting.

## MIKASA: two evaluation scopes

`mikasa_robo/experiments/demo_interaction_50.json` selects the full 90 tasks.
`mikasa_robo_vla_suite/` selects the five tasks used in prior VLA publications,
so we report both broad benchmark coverage and comparison on the same task list.
These are two scopes of one benchmark, sharing its environment and demo package.

[Five-task plans, exact budgets and commands](mikasa_robo_vla_suite/README.md) ·
[Reproducing both scopes](../docs/reproducing.md) ·
[MIKASA results](../docs/results/mikasa_robo.md)

Only final Python programs and setting READMEs belong in `solutions/`.
Historical imports remain unchanged. Public coding metadata names the model
without account settings; original launch metadata stays in archived runs. RMBench's
`*_observed_motion.json` catalogs target the current public interface; original
catalogs remain provenance for historical scores.

Run configuration filenames describe the learning mode. Set the `coding` object
inside a configuration to choose your client, model, provider and account settings;
separate copies of every mode for each client are unnecessary. For a custom run,
copy the configuration to a local JSON file and edit that object before preparation.
Resolved client/model settings are still pinned in each run for reproducibility.

Public presets select Opus 5.5 (`claude-opus-5-5`) with `provider: "client"`.
This inherits routing and authentication from the client's process environment;
export the settings required by your account before starting development. If your
endpoint requires a different identifier for Opus 5.5, set `coding.model` in your
local configuration. Client settings files remain disabled to keep prompts and
tools isolated.
