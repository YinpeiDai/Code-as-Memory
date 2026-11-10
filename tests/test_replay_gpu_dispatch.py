"""Physical GPU allocation is bounded even when more replay jobs are submitted."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replay_gpu_dispatch", ROOT / "scripts/replay_gpu_dispatch.py")
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def config(tmp_path):
    return {"python": sys.executable, "gpus": [0, 1], "workers_per_gpu": 2,
            "min_free_memory_mib": 16384, "locks": str(tmp_path / "locks"),
            "archive": str(tmp_path / "archive")}


@pytest.mark.parametrize('workers_per_gpu', [2, 4])
def test_concurrent_native_children_obey_gpu_slots(tmp_path, monkeypatch, workers_per_gpu):
    cfg = config(tmp_path)
    cfg['workers_per_gpu'] = workers_per_gpu
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="30000\n"))
    code = """import json, os, sys, time
from pathlib import Path
started = time.time()
time.sleep(0.25)
Path(sys.argv[1]).write_text(json.dumps({'gpu': int(os.environ['CUDA_VISIBLE_DEVICES']),
    'started': started, 'finished': time.time()}))
"""
    jobs = []
    for index in range(8):
        job = tmp_path / f"job_{index}.job.json"
        job.write_text("{}")
        output = tmp_path / f"child_{index}.json"
        jobs.append((job, output))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(dispatch.dispatch, cfg, ["-c", code, str(output)], job) for job, output in jobs]
        assert [f.result() for f in futures] == [0] * 8
    rows = [json.loads(output.read_text()) for _, output in jobs]
    assert {r["gpu"] for r in rows} == {0, 1}
    for gpu in [0, 1]:
        events = sorted([(r["started"], 1) for r in rows if r["gpu"] == gpu]
                        + [(r["finished"], -1) for r in rows if r["gpu"] == gpu])
        active = 0
        for _, delta in events:
            active += delta
            assert 0 <= active <= workers_per_gpu
    records = list((tmp_path / "archive").glob("*/dispatch.json"))
    assert len(records) == 8
    assert all(json.loads(p.read_text())["returncode"] == 0 for p in records)
    assert len(list((tmp_path / "archive").glob("*/*.job.json"))) == 8


@pytest.mark.parametrize("change", [{"gpus": [1, 1]}, {"gpus": [-1]}, {"workers_per_gpu": 0}])
def test_invalid_assignments_are_rejected(tmp_path, change):
    cfg = {**config(tmp_path), **change}
    with pytest.raises(ValueError):
        dispatch.validate(cfg)
