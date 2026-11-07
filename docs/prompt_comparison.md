# Historical prompts and the CaM protocol

[Archived prompts](legacy_prompts/manifest.json) retain source paths and hashes.
A historical run's saved prompt remains authoritative: source prompts changed
over time. Archives never initialize new coding sessions.

| Protocol | Evidence and interfaces | Development instructions |
|---|---|---|
| Plus Python core/demo | Released demos; RGB-D/calibration described | Inspect all demos; synthetic/static checks |
| Regular legacy three-demo | One easy/medium/hard; no depth or calibration; older end-effector frame | Generic legacy assignment |
| Regular modern demo-only | Modern RGB-D, native TCP, calibration, expert commands | Per-episode perception and geometry checks; zero attempts |
| Current CaM | Manifest-derived counts/IDs; RGB-D by default | Shared core + mode + benchmark guide; same isolated tools for both clients |

The old extractor used `robot_endeffector_p/q`; modern data provides measured TCP
in `obs/eef_state`. These frames are not interchangeable. CaM reads native TCP
without an undocumented offset.

Three-demo versus 25-demo results change data format, TCP frame, calibration,
depth, example count/diversity, and verification instructions together. They do
not isolate depth or diversity as the cause. Controlled comparisons need fresh
runs holding other factors fixed.

The historical regular modern assignment was hard-coded to demonstration-only
learning. CaM renders all three modes from the validated configuration. Plus's
later `both_v2` adds candidate IDs, reviews, and repeated verification; CaM's
simple demo-first/counted-feedback protocol does not claim equivalence.

At migration, pinned official regular RoboMME Python files matched the local
regular implementation. The official Plus package differed from completed local
Plus experiments in 37 Python files. Historical Plus reproduction requires its
exact snapshot and a validated adapter; Plus remains a protocol draft here.
