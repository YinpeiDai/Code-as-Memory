# RoboMemArena results

The larger assignment uses 50 demonstrations per task except T04, which has 32.
Interaction uses each assigned demonstration identity once (50/32 attempts).
The 25-demo setting uses 25 per task and no interaction. All 50 expected held-out
episodes per policy are scored; the metric is whole-task success (TSR), not PSR.

[25-demo code](../../../solutions/robomemarena/demo_only_25/README.md) ·
[50-demo code](../../../solutions/robomemarena/demo_only_50/README.md) ·
[Interaction code](../../../solutions/robomemarena/demo_interaction_50/README.md)

The 25-demo arm uses revised offline-review instructions, and interaction adds
manipulation/recovery guidance. These comparisons change prompts as well as data
or interaction. The original prompt hashes and intermediate histories remain archived.

## Per-task results

Successful / scored episodes; each denominator also equals the expected count.

| Task | 25 demos only | Up to 50 demos only | Up to 50 demos + interaction |
|---|---:|---:|---:|
| `T01_CookiesTomatoBasket` | 50/50 | 50/50 | 50/50 |
| `T02_ButterPopcornBasket` | 50/50 | 50/50 | 50/50 |
| `T03_CreamPuddingBasket` | 50/50 | 50/50 | 50/50 |
| `T04_ButterNonEmptyDrawer` | 50/50 | 50/50 | 50/50 |
| `T05_ButterEmptyDrawer` | 0/50 | 50/50 | 50/50 |
| `T06_PourSauceCookiesTwice` | 50/50 | 50/50 | 50/50 |
| `T07_PourSauceFrypanTwice` | 50/50 | 50/50 | 50/50 |
| `T08_ChocolateFrypanPourSauceTwice` | 50/50 | 50/50 | 50/50 |
| `T09_ButterFrypanPourSauceTwice` | 45/50 | 50/50 | 50/50 |
| `T10_PourWineMugTwice` | 50/50 | 50/50 | 50/50 |
| `T11_CookiesTopButterMiddle` | 28/50 | 36/50 | 50/50 |
| `T12_CookiesChocolateMiddleDrawer` | 49/50 | 38/50 | 50/50 |
| `T13_CookiesButterMiddleDrawer` | 29/50 | 41/50 | 50/50 |
| `T14_CookiesChocolateDrawers` | 50/50 | 43/50 | 50/50 |
| `T15_ButterFrypanPourMilkTwice` | 50/50 | 50/50 | 50/50 |
| `T16_PourMilkMugTwice` | 50/50 | 50/50 | 50/50 |
| `T17_ButterChocolateMiddleDrawer` | 32/50 | 37/50 | 50/50 |
| `T18_ChocolateButterCabinet` | 48/50 | 48/50 | 48/50 |
| `T19_SauceMilkJuiceCabinet` | 12/50 | 20/50 | 25/50 |
| `T20_CookiesChocolateMicrowave` | 50/50 | 50/50 | 50/50 |
| `T21_ButterChocolateMicrowave` | 50/50 | 50/50 | 50/50 |
| `T22_PourSauceCookiesMicrowave` | 50/50 | 50/50 | 50/50 |
| `T23_CreamPopcornMicrowave` | 5/50 | 41/50 | 50/50 |
| `T24_CookiesPopcornMicrowave` | 50/50 | 50/50 | 50/50 |
| `T25_ButterCreamPlate` | 33/50 | 42/50 | 48/50 |
| `T26_ChocolateCreamPlate` | 39/50 | 33/50 | 50/50 |
| **Total** | **1,070/1,300 (82.31%)** | **1,179/1,300 (90.69%)** | **1,271/1,300 (97.77%)** |

## Evidence

Counts were checked against the original `result.json` records under
`public_runs/eval/robomemarena/` (locally `/data/daiyp/CaM_public/eval/robomemarena/`).
Catalogs under `configs/solutions/robomemarena/` retain code hashes and source provenance.
New sanity checks and current-interface reruns do not replace these outcomes.

[Run or develop this benchmark](README.md) · [All benchmark results](../../results.md)
