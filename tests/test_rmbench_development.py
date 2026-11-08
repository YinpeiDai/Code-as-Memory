"""RMBench's explicitly assigned full pool must preserve replay shortfalls."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assignment(tmp_path):
    spec = json.loads((ROOT / "configs/experiments/rmbench_rgbd_demos50_all10.json").read_text())
    item = spec["benchmarks"]["rmbench"]
    config = json.loads((ROOT / item["config"]).read_text())
    manifest = {"benchmark": "rmbench", "depth": True, "tasks": {
        task: {"episodes": [{"episode": i} for i in range(item["demo_counts"].get(task, 50))]}
        for task in item["tasks"]}}
    item["datasets"] = [str(tmp_path)]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return item, config, manifest


def test_full_pool_explicit_shortfalls_and_demo_only_prompt(tmp_path):
    from cam.benchmarks.registry import get_adapter
    from cam.config import RunConfig
    from cam.prompts import render

    item, config, manifest = assignment(tmp_path)
    result = script("develop_batch").data_assignment("rmbench", item, config)
    assert sum(result["demo_counts"].values()) == 496
    assert config["mode"] == "demos" and config["budget"] == 0 and config["depth"]
    adapter = get_adapter("rmbench")
    assert item["tasks"] == adapter.tasks()
    task = "observe_and_pickup"
    interface = adapter.interface(task, True)
    prompt = render(RunConfig.from_dict(config), task, manifest["tasks"][task]["episodes"], [], adapter, interface)
    assert "Assigned expert demonstrations: 48" in prompt
    assert "Zero simulator attempts are allowed" in prompt
    assert "No depth," not in prompt
    assert "RGB-D" in prompt


@pytest.mark.parametrize("fault", ["undeclared_shortfall", "padded", "unexpected_missing", "unknown_task", "invalid_count"])
def test_assignment_rejects_silent_count_changes(tmp_path, fault):
    item, config, manifest = assignment(tmp_path)
    if fault == "undeclared_shortfall":
        item.pop("demo_counts")
    elif fault == "padded":
        manifest["tasks"]["battery_try"]["episodes"].append({"episode": 99})
    elif fault == "unexpected_missing":
        manifest["tasks"]["cover_blocks"]["episodes"].pop()
    elif fault == "unknown_task":
        item["demo_counts"]["unknown"] = 48
    elif fault == "invalid_count":
        item["demo_counts"]["battery_try"] = 51
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        script("develop_batch").data_assignment("rmbench", item, config)


def test_queue_waits_for_validation_marker(tmp_path):
    producer, pool = tmp_path / "producer", tmp_path / "pool"
    producer.mkdir()
    pool.mkdir()
    (producer / "status.json").write_text(json.dumps({"status": "assembling"}))
    queue = script("queue_rmbench_development")
    assert not queue.pool_ready(producer, pool)
    (pool / "manifest.json").write_text("{}")
    (pool / "manifest.partial.json").write_text("{}")
    assert not queue.pool_ready(producer, pool)
    (pool / "manifest.partial.json").unlink()
    assert queue.pool_ready(producer, pool)
    (producer / "status.json").write_text(json.dumps({"status": "needs_attention"}))
    with pytest.raises(RuntimeError):
        queue.pool_ready(producer, pool)
