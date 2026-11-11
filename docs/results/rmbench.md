# RMBench results

The [original simulation training protocol](https://arxiv.org/html/2603.01229v1#S5.SS1)
uses **50 expert demonstrations per task**. CaM uses **25 or up to 50 demos/task**,
with no training interaction. The larger pool has 49 demonstrations
for `battery_try`, 49 for `blocks_ranking_try`, 48 for `observe_and_pickup`, and
50 for each remaining task. Each policy is evaluated on all 100 held-out episodes.

[25 demo-only code solutions](../../solutions/rmbench/demo_only_25/README.md) ·
[50 demo-only code solutions](../../solutions/rmbench/demo_only_50/README.md)

## Per-task results

Successful/total held-out episodes.

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
`public_runs/eval/rmbench/`.
Catalogs under `configs/solutions/rmbench/` retain code hashes and source provenance.
New sanity checks and current-interface reruns do not replace these outcomes.

[Run or develop this benchmark](../benchmarks/rmbench.md) · [All benchmark results](../results.md)
