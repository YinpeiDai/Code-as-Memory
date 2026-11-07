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
