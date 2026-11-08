"""Record how reproducible each pinned RMBench test-seed list is (no simulator).

Compares the pinned first run (``outputs/rmbench/seed_search/<task>.jsonl``) with
an independent repeat of the same search (``--repeat`` logs) and, where
published, with the seeds in Mem-0's evaluation logs (``eval_log.txt`` from
``qiuly/Mem-0-m1mix-RMBench``). Writes ``provenance.agreement`` into
``src/cam/benchmarks/metadata/rmbench/test/record_dataset_<task>_metadata.json``;
the pinned records themselves are never changed.

  .venv/bin/python scripts/rmbench_seed_agreement.py \
      --first outputs/rmbench/seed_search --repeat outputs/rmbench/seed_search_repeat \
      --mem0 outputs/rmbench/mem0_logs/eval_results
"""

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "src/cam/benchmarks/metadata/rmbench/test"


def decisions(path):
    return {row["seed"]: row["kept"] for row in map(json.loads, path.read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--mem0", type=Path)
    parser.add_argument("--mem0-source", default="hf://qiuly/Mem-0-m1mix-RMBench/eval_results/<task>/eval_log.txt")
    args = parser.parse_args()
    for path in sorted(METADATA.glob("record_dataset_*_metadata.json")):
        value = json.loads(path.read_text())
        task = value["env_id"]
        pinned = [int(r["seed"]) for r in value["records"]]
        first = decisions(args.first / f"{task}.jsonl")
        if [s for s, kept in sorted(first.items()) if kept][: len(pinned)] != pinned:
            raise SystemExit(f"{task}: pinned seeds are not the first-run log")
        agreement = {}
        repeat_log = args.repeat / f"{task}.jsonl"
        if repeat_log.exists():
            repeat = decisions(repeat_log)
            common = sorted(set(first) & set(repeat))
            flipped = [s for s in common if first[s] != repeat[s]]
            kept = [s for s in sorted(repeat) if repeat[s]][: len(pinned)]
            agreement["repeat_run"] = {
                "log": str(repeat_log.relative_to(ROOT)) if repeat_log.is_relative_to(ROOT) else str(repeat_log),
                "seeds_compared": len(common),
                "decisions_agree": len(common) - len(flipped),
                "flipped_seeds": flipped,
                "repeat_kept": len(kept),
                "identical_list": kept == pinned,
                "shared_seeds": len(set(kept) & set(pinned)),
            }
        mem0 = args.mem0 / task / "eval_log.txt" if args.mem0 else None
        if mem0 and mem0.exists():
            seeds = [int(s) for s in re.findall(r"seed=(\d+)", mem0.read_text())]
            agreement["mem0_published"] = {
                "source": args.mem0_source.replace("<task>", task),
                "count": len(seeds),
                "identical_list": seeds == pinned,
                "shared_seeds": len(set(seeds) & set(pinned)),
                "only_pinned": sorted(set(pinned) - set(seeds)),
                "only_mem0": sorted(set(seeds) - set(pinned)),
            }
        else:
            agreement["mem0_published"] = None  # Mem-0 published no per-seed log for this task
        value["provenance"]["agreement"] = agreement
        path.write_text(json.dumps(value, indent=2) + "\n")
        summary = {k: (v and {x: v[x] for x in ("identical_list", "shared_seeds")}) for k, v in agreement.items()}
        print(task, json.dumps(summary))


if __name__ == "__main__":
    main()
