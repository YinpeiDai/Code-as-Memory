"""One observation-only native simulator check. Not a learned-policy evaluation."""

import argparse
import os
from pathlib import Path

from cam.benchmarks.registry import get_adapter
from cam.config import RunConfig
from cam.core.history import materialize
from cam.benchmarks.robomme_data import scalar
from cam.harness.broker import execute
from cam.harness.prepare import make_runtime, STARTER
from cam.io import write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BinFill")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--raw-data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=False)
adapter = get_adapter("robomme")
record = adapter.records(args.task, "train")[0]
import h5py

with h5py.File(args.raw_data / f"record_dataset_{args.task}.h5") as raw:
    episode = raw[f"episode_{record['episode']}"]
    assert int(scalar(episode["setup/seed"][()])) == int(record["seed"])
    assert scalar(episode["setup/difficulty"][()]) == record["difficulty"]
write(
    root / "metadata" / f"record_dataset_{args.task}_metadata.json",
    {"records": [record]},
)
write(
    root / "diagnostic.json",
    {
        "purpose": "native adapter smoke check, not experimental results",
        "task": args.task,
        "raw_data": str(args.raw_data),
        "training_record": record,
        "api_calls": 0,
    },
)
materialize(
    root / "program",
    {
        "task.json": (
            '{"task":"' + args.task + '","runtime":"python","memory":[]}'
        ).encode(),
        "policy.md": b"Observation-only diagnostic",
        "tools/policy.py": STARTER.encode(),
    },
)
make_runtime(root / "runtime")
result = execute(
    adapter,
    root / "metadata",
    args.task,
    int(record["episode"]),
    "train",
    root / "program",
    root / "runtime",
    root / "rollout",
    RunConfig(benchmark="robomme", mode="demos", max_decisions=1).to_dict(),
)
print(result)
raise SystemExit(1 if not result["scorable"] else 0)
