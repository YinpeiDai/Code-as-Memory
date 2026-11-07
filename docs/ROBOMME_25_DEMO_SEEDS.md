# Historical 25-demo selection

The original regular RoboMME run `regular_demo_v2_restart_20261105_040321` used
**25 demos × 16 tasks = 400 demonstrations**, with zero training simulator
attempts. Its test result was **795/800 (99.375%)**. This is historical evidence,
not a full reproduction with the current CaM runner.

## Sampling rule

Within each task and difficulty, sort training episodes by original episode ID.
Select ranks using `np.rint(np.linspace(0, n - 1, k)).astype(int)`:

| Difficulty | Available | Selected | Zero-based ranks |
|---|---:|---:|---|
| Easy | 50 | 9 | `0, 6, 12, 18, 24, 31, 37, 43, 49` |
| Medium | 25 | 8 | `0, 3, 7, 10, 14, 17, 21, 24` |
| Hard | 25 | 8 | `0, 3, 7, 10, 14, 17, 21, 24` |

Episode ID `N` refers to raw group `episode_N`; it is not the simulator seed.
Reproduction requires the matching recordings, metadata, and benchmark revision.

```bash
cam extract --benchmark robomme --raw-data /path/to/robomme_raw_h5 \
  --output data/robomme_25_rgbd --count 25 --workers 2
```

The current selector matches all 400 historical episode IDs. Exact selected IDs
and seeds are written to the extraction's `manifest.json`; seeds remain host-only.
The current schema and prompt differ from the original runner, so matching the
selection does not imply identical regenerated policies.

## Provenance

- Original manifest: `regular_25shot_rgbd_20261105/fewshot/manifest.json`.
- SHA-256: `b2a83365cbaaac902ccf22ba73b5a2c026b0bbf4da1dc60500e844fbe47acec7`.
- All 400 entries were checked against saved run metadata, pinned official
  metadata, raw seed/difficulty fields, and extracted H5 groups; no selected
  seed overlaps the task's test split.
- [Historical logs, demonstrations, and rollouts](https://huggingface.co/datasets/Yinpei/robomme-CaM-log-rollout)
  retain the original evidence; access may require repository permission.
- [Local reference paths](legacy_reference_paths.json) identify the original runs.

Dataset files and local CSV/JSON selection exports are excluded from this code
repository. Use the extraction manifest for a new run's exact identities.
