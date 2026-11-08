# Historical prompts and the CaM protocol

[Archived prompts](legacy_prompts/manifest.json) retain source paths and hashes.
A historical run's saved prompt remains authoritative: source prompts changed
over time. Archives never initialize new coding sessions.

| Protocol | Evidence and interfaces | Development instructions |
|---|---|---|
| Plus Python core/demo | Released demos; RGB-D/calibration described | Inspect all demos; synthetic/static checks |
| Regular legacy three-demo | One easy/medium/hard; no depth or calibration; older end-effector frame | Generic legacy assignment |
| Regular modern demo-only | Modern RGB-D, native TCP, calibration, expert commands | Per-episode perception and geometry checks; zero attempts |
| CaM both v1 | Manifest-derived evidence and training identities | Free choice of demo analysis, offline checks, and interaction |
| CaM both v2 | Same evidence, tools, seeds, and 25-attempt ceiling as v1 | Demo-derived solution first; resolve uncertainties through rollouts, then use remaining distinct training seeds for verification |

The old extractor used `robot_endeffector_p/q`; modern data provides measured TCP
in `obs/eef_state`. These frames are not interchangeable. CaM reads native TCP
without an undocumented offset.

Three-demo versus 25-demo results change data format, TCP frame, calibration,
depth, example count/diversity, and verification instructions together. They do
not isolate depth or diversity as the cause. Controlled comparisons need fresh
runs holding other factors fixed.

The historical regular modern assignment was hard-coded to demonstration-only
learning. CaM renders all three modes from the validated configuration; `both`
requires an explicit `both_protocol: "v1"` or `"v2"`. The original simplified
CaM `both.md` has been removed. Both new strategies use the same six-tool harness
and differ through prompt instructions. The historical Plus `both_v2` also adds
candidate IDs, mandatory reviews, selection safeguards, and a reserved final
confirmation cycle. Those mechanisms are not part of CaM's prompt comparison;
CaM v2 does not claim exact historical protocol equivalence. See
[version selection and historical compatibility](demo_interaction.md).

At migration, pinned official regular RoboMME Python files matched the local
regular implementation. The official Plus package differed from completed local
Plus experiments in 37 Python files. Historical Plus reproduction requires its
exact snapshot and a validated adapter; Plus remains a protocol draft here.
