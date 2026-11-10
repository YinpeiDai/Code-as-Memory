# Robot Memory solutions

Final Python programs, grouped by benchmark and setting. Open a setting README
for its result, exact evaluation command, and development-prompt provenance.

- [RoboMME](robomme/demo_only_25/README.md)
- [MIKASA-Robo](mikasa_robo/demo_interaction_50/README.md)
- RMBench: [25 demos](rmbench/demo_only_25/README.md), [50 demos](rmbench/demo_only_50/README.md)
- RoboMemArena: [25 demos](robomemarena/demo_only_25/README.md), [50 demos](robomemarena/demo_only_50/README.md), [interaction](robomemarena/demo_interaction_50/README.md)

Required memory declarations and integrity/provenance metadata are in
[configs/solutions](../configs/solutions/). `cam prepare --solution` imports and
freezes unchanged final code without demonstrations, learning, or model calls.
See [running solutions](../docs/solutions.md).
