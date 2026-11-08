# Validation

Validated in the locked Python 3.11 environment with `dev` and `robomme` extras.
The official benchmark is pinned at `5753639d62c167ba74fa64fcce8b2f4895d08e7b`.

## Automated and native checks

**33 tests passed.** The suite covers balanced sampling, lossless allowed fields,
seed/split validation, extraction integrity, causal checkpoints, attempt accounting,
recovery, organized run paths, and generated access documentation. It also checks
real Bubblewrap isolation, interpreter aliases, MCP process lifetime, command
audits, client tool restrictions, usage accounting, queues, and report denominators.
A transport test streams 600 RGB-D frames (>500 MiB encoded) through the policy
worker without simulator resets. Ruff F/E9 and JavaScript syntax checks pass.

Read-only raw-data checks verified all **1,600 training identities** across 16
tasks against official metadata, without test-seed overlap. First-frame sensor
fields and 25-demo selections were checked; full extraction additionally checks
completion and every selected frame.

Native training smoke checks on BinFill and VideoPlaceOrder saved RGB-D, actions,
and videos; VideoPlaceOrder retained 942 frames. These intentionally terminate at
`decision_limit` and measure harness operation, not learned-policy performance.
Desktop/mobile browser checks passed with fixtures and the completed real run,
including navigation, overflow, media byte ranges, and private-path rejection.

## Completed ButtonUnmaskSwap run

Run: `robomme/ButtonUnmaskSwap/cc-opus5.5/demo_only/20261107T024349_127546Z`.

| Item | Result |
|---|---|
| Learner | Claude Code 2.1.319 / Bedrock `us.anthropic.claude-opus-5-5`, high effort |
| Evidence | 25 RGB-D demos (9/8/8), audited conversion from the supplied CAP extraction |
| Training simulator attempts | 0 |
| Held-out successes / scored / expected | **50 / 50 / 50**, no infrastructure errors |
| Difficulty | Easy 26/26, medium 12/12, hard 12/12 |
| Evaluation model calls | 0 |
| Development | 16.88 minutes; $3.3644 Claude CLI estimate, excluding GPU costs |
| Code history | 51 revision records, 10 distinct source snapshots |
| Policy SHA-256 | `3a10caad2600c82ec719a7b7c3a6ad51bec36ea6a2d751be6e542ae5e1e64dd6` |

The host initiated evaluation after freeze. All 50 episode identities, policy
hashes, RGB-D/action recordings, and video frame counts were verified. The access
audit found no exposed/used WebSearch tool, Internet socket events, or successful
absolute opens under the checked private prefixes; see its
[scope limits](reproducing.md#audit-and-recovery).
Run data, transcripts, policies, and media remain local under `runs/` and are not
included in this code repository. This single-task result does not reproduce the
historical full 800-episode study. Other models and Plus remain unvalidated here.

## RMBench RGB-D demo-only runs (2026-11-08)

All ten tasks used Claude Code / Opus 5.5, zero simulator training rollouts, and
100 held-out episodes per task. The up-to-50 assignment contains 496 regenerated
successful RGB-D demos: 49 for battery_try, 49 for blocks_ranking_try, 48 for
observe_and_pickup, and 50 for each other task. The separate 25-demo assignment
uses deterministic nested subsets (250 total).

| Selection | Successes / scored / expected | Notes |
|---|---|---|
| Original up-to-50 baseline | 900 / 1,000 / 1,000 | swap_blocks 0/100 from a first-decision None initialization error; other nine tasks 100/100 |
| Original matched 25-demo comparison | 927 / 1,000 / 1,000 | cover_blocks 28/100, observe_and_pickup 99/100, other eight tasks 100/100 |
| Updated up-to-50 dashboard selection | 1,000 / 1,000 / 1,000 | Replaces swap_blocks with the user-requested fresh-policy rerun, 100/100 |

The original comparison uses the identical pinned harness and prompts. The new
swap_blocks run starts from a blank policy with the same 50 demos and the revised
shared prompt requiring final offline execution checks; it imports no prior
solution or test feedback. The earlier explicitly authorized inherited-policy
repair also scored 100/100, but is a separate run. These are single runs, and the
updated selection is not a demo-count-only ablation. There are no unresolved
infrastructure errors in these completed evaluations.

Baseline run tag: `20261108T032001_064494Z`; 25-demo tag:
`20261108T055619_235514Z`; fresh replacement:
`20261108T150726_925267Z_swap_blocks_fresh`. The original policies, logs, scores
and comparison report remain on disk. The dashboard removes the original and
repair swap_blocks entries and displays only the fresh 50-demo run alongside
the separate 25-demo experiment. See [reproduction instructions](reproducing.md).

## Repeat checks

```bash
.venv/bin/pytest -q
uvx ruff check --select F,E9 src tests scripts
node --check src/cam/web/app.js
.venv/bin/python scripts/smoke_robomme.py --task VideoPlaceOrder --gpu 0 \
  --raw-data /path/to/robomme_raw_h5 --output local/validation/native_check
uv run --no-sync --with playwright python scripts/check_site.py sites/robomme \
  --screenshots local/validation/browser
```

Browser checks need Playwright Chromium and a generated site. Codex's offline
compatibility test skips without its CLI/catalog; paid Codex learning requires
the gate to pass. Provider authentication and quotas are installation-specific.
