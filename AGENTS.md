# Reproducing CaM experiments

Read [docs/reproducing.md](docs/reproducing.md) before preparing or running an
experiment. It describes the canonical run hierarchy, demonstration validation,
coding-agent filesystem access, and the separation of learning from held-out tests.

- Use the requested coding client/model and learning setting. Prepare through
  `cam prepare`; every new task gets its own timestamped run and generated README.
- Keep model-generated code development inside the CaM tool interface. The host
  orchestrator may inspect the repository and repair infrastructure, but must not
  supply hidden simulator state, benchmark implementation, historical solutions,
  or test feedback to the learner.
- Preserve original datasets, logs, revisions, and scored evaluation outcomes.
  Never edit a prepared source snapshot to bypass its integrity checks. Prepare
  a new timetag after changing the harness, configuration, prompt, or data.
- Finish learning and freeze the policy before evaluating held-out episodes.
  Record actual scored/expected counts and distinguish infrastructure failures.
- Keep the generated run README and reproduction guide accurate when changing
  directory layout, tool exposure, or sandbox mounts.
