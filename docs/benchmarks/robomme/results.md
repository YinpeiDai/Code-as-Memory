# RoboMME results

The 16 tasks use 25 demonstrations each and no training interaction. These are the
original frozen runs: **795/800 successes (99.38%)**. Every expected episode is scored.
A current-harness rerun is separate evidence; the released H5 bytes remain unchanged.

[Code and provenance](../../../solutions/robomme/demo_only_25/README.md) ·
[Immutable catalog](../../../configs/solutions/robomme/demo_only_25.json)

## Per-task results

Successful / scored episodes; each denominator also equals the expected count.

| Task | 25 demos only |
|---|---:|
| `BinFill` | 50/50 |
| `ButtonUnmask` | 50/50 |
| `ButtonUnmaskSwap` | 50/50 |
| `InsertPeg` | 50/50 |
| `MoveCube` | 50/50 |
| `PatternLock` | 50/50 |
| `PickHighlight` | 50/50 |
| `PickXtimes` | 50/50 |
| `RouteStick` | 50/50 |
| `StopCube` | 47/50 |
| `SwingXtimes` | 50/50 |
| `VideoPlaceButton` | 50/50 |
| `VideoPlaceOrder` | 48/50 |
| `VideoRepick` | 50/50 |
| `VideoUnmask` | 50/50 |
| `VideoUnmaskSwap` | 50/50 |
| **Total** | **795/800 (99.38%)** |

## Evidence

Counts were checked against the original `result.json` records under
`public_runs/eval/robomme/` (locally `/data/daiyp/CaM_public/eval/robomme/`).
Catalogs under `configs/solutions/robomme/` retain code hashes and source provenance.
New sanity checks and current-interface reruns do not replace these outcomes.

[Run or develop this benchmark](README.md) · [All benchmark results](../../results.md)
