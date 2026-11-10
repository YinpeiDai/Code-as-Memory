# RMBench results

Both settings use no training interaction. The larger pool has 49 demonstrations
for `battery_try`, 49 for `blocks_ranking_try`, 48 for `observe_and_pickup`, and
50 for each remaining task. Each policy has 100 scored/expected held-out episodes.
The selected 50-demo `swap_blocks` policy is the fresh rerun; earlier results remain archived.

[25-demo code](../../solutions/rmbench/demo_only_25/README.md) ·
[50-demo code](../../solutions/rmbench/demo_only_50/README.md)

These are historical results under their pinned interfaces. Current evaluation
uses public post-action poses for `MotionError`; the compatibility catalogs preserve
the code and original provenance without relabelling these scores. Current demo-only
development uses the shared guidance without the retired motion supplement.

## Per-task results

Successful / scored episodes; each denominator also equals the expected count.

| Task | 25 demos only | Up to 50 demos only |
|---|---:|---:|
| `battery_try` | 100/100 | 100/100 |
| `blocks_ranking_try` | 100/100 | 100/100 |
| `cover_blocks` | 28/100 | 100/100 |
| `observe_and_pickup` | 99/100 | 100/100 |
| `place_block_mat` | 100/100 | 100/100 |
| `press_button` | 100/100 | 100/100 |
| `put_back_block` | 100/100 | 100/100 |
| `rearrange_blocks` | 100/100 | 100/100 |
| `swap_T` | 100/100 | 100/100 |
| `swap_blocks` | 100/100 | 100/100 |
| **Total** | **927/1,000 (92.70%)** | **1,000/1,000 (100.00%)** |

## Evidence

Counts were checked against the original `result.json` records under
`public_runs/eval/rmbench/` (locally `/data/daiyp/CaM_public/eval/rmbench/`).
Catalogs under `configs/solutions/rmbench/` retain code hashes and source provenance.
New sanity checks and current-interface reruns do not replace these outcomes.

[Run or develop this benchmark](../benchmarks/rmbench.md) · [All benchmark results](../results.md)
