# RoboMemArena results

The [original benchmark dataset](https://arxiv.org/html/2605.10921v1#S3.SS3)
provides **100 successful full-task demonstrations per task** (2,600 across
26 tasks). CaM uses **25 or up to 50 demos/task**.
The larger assignment uses 50 demonstrations per task except T04, which has 32.
Interaction uses each assigned demonstration identity once (50/32 attempts).
The 25-demo setting uses 25 per task and no interaction. Every policy is evaluated
on all 50 held-out episodes. **TSR** is the whole-task success rate; **CSR** averages
the native completed-stage percentage, including partial progress in failed episodes.

[25-demo code](../../solutions/robomemarena/demo_only_25/README.md) ·
[50-demo code](../../solutions/robomemarena/demo_only_50/README.md) ·
[Interaction code](../../solutions/robomemarena/demo_interaction_50/README.md)

The 25-demo arm uses revised offline-review instructions, and interaction adds
manipulation/recovery guidance. These comparisons change prompts as well as data
or interaction. Each source run retains its exact prompt.

## Per-task results

Each setting reports **successful/total** held-out episodes and **CSR**.
The total row includes TSR in parentheses. Overall CSR averages the 26 task means;
each task has 50 episodes.

| Task | 25 demos only | CSR | Up to 50 demos only | CSR | Up to 50 demos + interaction | CSR |
|---|---:|---:|---:|---:|---:|---:|
| `T01_CookiesTomatoBasket` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T02_ButterPopcornBasket` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T03_CreamPuddingBasket` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T04_ButterNonEmptyDrawer` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T05_ButterEmptyDrawer` | 0/50 | 87.50% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T06_PourSauceCookiesTwice` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T07_PourSauceFrypanTwice` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T08_ChocolateFrypanPourSauceTwice` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T09_ButterFrypanPourSauceTwice` | 45/50 | 95.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T10_PourWineMugTwice` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T11_CookiesTopButterMiddle` | 28/50 | 82.40% | 36/50 | 86.00% | 50/50 | 100.00% |
| `T12_CookiesChocolateMiddleDrawer` | 49/50 | 99.33% | 38/50 | 92.00% | 50/50 | 100.00% |
| `T13_CookiesButterMiddleDrawer` | 29/50 | 86.00% | 41/50 | 94.00% | 50/50 | 100.00% |
| `T14_CookiesChocolateDrawers` | 50/50 | 100.00% | 43/50 | 94.40% | 50/50 | 100.00% |
| `T15_ButterFrypanPourMilkTwice` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T16_PourMilkMugTwice` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T17_ButterChocolateMiddleDrawer` | 32/50 | 88.00% | 37/50 | 91.33% | 50/50 | 100.00% |
| `T18_ChocolateButterCabinet` | 48/50 | 96.00% | 48/50 | 96.00% | 48/50 | 96.00% |
| `T19_SauceMilkJuiceCabinet` | 12/50 | 42.00% | 20/50 | 51.33% | 25/50 | 60.00% |
| `T20_CookiesChocolateMicrowave` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T21_ButterChocolateMicrowave` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T22_PourSauceCookiesMicrowave` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T23_CreamPopcornMicrowave` | 5/50 | 46.67% | 41/50 | 88.67% | 50/50 | 100.00% |
| `T24_CookiesPopcornMicrowave` | 50/50 | 100.00% | 50/50 | 100.00% | 50/50 | 100.00% |
| `T25_ButterCreamPlate` | 33/50 | 79.00% | 42/50 | 84.00% | 48/50 | 96.00% |
| `T26_ChocolateCreamPlate` | 39/50 | 82.00% | 33/50 | 75.00% | 50/50 | 100.00% |
| **Total** | **1,070/1,300 (82.31%)** | **91.69%** | **1,179/1,300 (90.69%)** | **94.34%** | **1,271/1,300 (97.77%)** | **98.15%** |

## Evidence

Counts and CSR were checked against all 3,900 original `result.json` records under
`public_runs/eval/robomemarena/`. CSR uses the saved `native_metrics.csr` values;
no new simulator runs or changes to episode scores are involved.
Catalogs under `configs/solutions/robomemarena/` retain code hashes and source provenance.
New sanity checks and current-interface reruns do not replace these outcomes.

[Run or develop this benchmark](../benchmarks/robomemarena.md) · [All benchmark results](../results.md)
