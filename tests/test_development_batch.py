"""Launch readiness: explicit demo counts, finalized data and per-task budgets."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("develop_batch", ROOT / "scripts/develop_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def setup_dataset(tmp_path, name="mikasa_robo_v2", count=50):
    config = json.loads((ROOT / f"configs/{name}/both_v2_claude_code.json").read_text())
    path = tmp_path / "dataset"
    path.mkdir()
    item = {"datasets": [str(path)], "tasks": ["Example"], "demo_count": count}
    manifest = {"benchmark": name, "depth": config["depth"], "tasks": {
        "Example": {"episodes": [{"episode": i, "seed": i} for i in range(count)]}
    }}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return config, item, manifest, path


@pytest.mark.parametrize("name,count", [("mikasa_robo", 25), ("mikasa_robo_v2", 50)])
def test_explicit_budgets_and_legacy_default(tmp_path, name, count):
    config, item, _, path = setup_dataset(tmp_path, name, count)
    if count == 25:
        item.pop("demo_count")
    result = batch.data_assignment(name, item, config)
    assert result == {"demo_count": count, "existing": {str(path): ["Example"]}, "missing": []}


@pytest.mark.parametrize("fault", ["partial", "shortfall", "missing_task", "wrong_benchmark", "wrong_depth", "wrong_budget"])
def test_incomplete_or_mismatched_data_cannot_start(tmp_path, fault):
    config, item, manifest, path = setup_dataset(tmp_path)
    if fault == "partial":
        (path / "manifest.json").rename(path / "manifest.partial.json")
    elif fault == "shortfall":
        manifest["tasks"]["Example"]["episodes"].pop()
    elif fault == "missing_task":
        item["tasks"].append("Missing")
    elif fault == "wrong_benchmark":
        manifest["benchmark"] = "mikasa_robo"
    elif fault == "wrong_depth":
        manifest["depth"] = False
    elif fault == "wrong_budget":
        config["budget"] = 25
    if fault != "partial":
        (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        batch.data_assignment("mikasa_robo_v2", item, config)


def test_new_plan_reuses_exact_prior_tasks_and_fifty_fifty():
    old = json.loads((ROOT / "configs/experiments/mikasa_25_newseeds_10tasks.json").read_text())
    new = json.loads((ROOT / "configs/experiments/mikasa_v2_50_newseeds_10tasks.json").read_text())
    item = new["benchmarks"]["mikasa_robo_v2"]
    assert item["tasks"] == old["benchmarks"]["mikasa_robo"]["tasks"]
    assert len(item["tasks"]) == 10 and item["demo_count"] == 50 and item["require_prepared_data"]
    config = json.loads((ROOT / item["config"]).read_text())
    assert config["budget"] == config["training_seed_count"] == 50
    assert config["coding"]["agent"] == "claude_code"
    assert config["coding"]["model"] == "us.anthropic.claude-opus-5-5"


def test_start_rejects_partial_data_before_creating_batch_or_launching(tmp_path, monkeypatch):
    config, item, _, path = setup_dataset(tmp_path)
    (path / "manifest.json").rename(path / "manifest.partial.json")
    item["config"] = "config.json"
    (tmp_path / item["config"]).write_text(json.dumps(config))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"benchmarks": {"mikasa_robo_v2": item}}))
    monkeypatch.setattr(batch, "ROOT", tmp_path)

    def launch_forbidden(*args, **kwargs):
        pytest.fail("Incomplete data must not start a supervisor or coding client")

    monkeypatch.setattr(batch.subprocess, "Popen", launch_forbidden)
    with pytest.raises(ValueError, match="not finalized"):
        batch.start(SimpleNamespace(plan=plan, workers=8, gpus=[0, 1]))
    assert not (tmp_path / "outputs").exists()


def test_cpu_interaction_plan_retains_assignment_and_pins_backend():
    from cam.config import RunConfig

    old = json.loads((ROOT / "configs/experiments/mikasa_v2_50_newseeds_10tasks.json").read_text())
    plan = json.loads((ROOT / "configs/experiments/mikasa_v2_cpu_50_demoseeds_10tasks.json").read_text())
    item = plan["benchmarks"]["mikasa_robo_v2"]
    original = old["benchmarks"]["mikasa_robo_v2"]
    assert {k: v for k, v in item.items() if k != "config"} == {
        k: v for k, v in original.items() if k != "config"
    }
    config = json.loads((ROOT / item["config"]).read_text())
    assert RunConfig.from_dict(config).physics_backend == "cpu"
    assert config["training_seed_policy"] == "demo_cycle"
    assert config["training_seed_count"] == config["budget"] == item["demo_count"] == 50
    previous = json.loads((ROOT / original["config"]).read_text())
    changed = {"physics_backend", "training_seed_policy"}
    assert {k: v for k, v in config.items() if k not in changed} == {
        k: v for k, v in previous.items() if k not in changed
    }


def test_demo_seed_batch_requires_full_demo_assignment(tmp_path):
    config, item, _, _ = setup_dataset(tmp_path)
    config.update(training_seed_policy="demo_cycle", physics_backend="cpu")
    assert batch.data_assignment("mikasa_robo_v2", item, config)["missing"] == []
    config.update(training_seed_count=25, budget=25)
    with pytest.raises(ValueError, match="every assigned demonstration"):
        batch.data_assignment("mikasa_robo_v2", item, config)


def test_cpu_remaining_eighty_completes_task_set_without_duplicates():
    from cam.benchmarks.registry import get_adapter

    ten = json.loads((ROOT / "configs/experiments/mikasa_v2_cpu_50_demoseeds_10tasks.json").read_text())["benchmarks"]["mikasa_robo_v2"]
    remaining = json.loads((ROOT / "configs/experiments/mikasa_v2_cpu_50_demoseeds_remaining80.json").read_text())["benchmarks"]["mikasa_robo_v2"]
    assert len(remaining["tasks"]) == len(set(remaining["tasks"])) == 80
    assert set(remaining["tasks"]).isdisjoint(ten["tasks"])
    assert set(remaining["tasks"]) | set(ten["tasks"]) == set(get_adapter("mikasa_robo_v2").tasks())
    assert remaining["config"] == ten["config"]
    assert remaining["demo_count"] == 50 and remaining["require_prepared_data"]
    assert remaining["datasets"] == ["/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50"]


def test_mikasa_demo_only_followup_has_fifty_demos_and_no_rollouts(tmp_path):
    from cam.benchmarks.registry import get_adapter
    from cam.config import RunConfig
    from cam.prompts import render

    plan = json.loads((ROOT / "configs/experiments/mikasa_v2_demos50_10tasks.json").read_text())
    item = plan["benchmarks"]["mikasa_robo_v2"]
    old = json.loads((ROOT / "configs/experiments/mikasa_25_newseeds_10tasks.json").read_text())
    assert item["tasks"] == old["benchmarks"]["mikasa_robo"]["tasks"]
    config = json.loads((ROOT / item["config"]).read_text())
    assert config["mode"] == "demos" and config["budget"] == 0
    assert item["demo_count"] == 50 and config["depth"]
    _, fake, _, _ = setup_dataset(tmp_path)
    assert batch.data_assignment("mikasa_robo_v2", fake, config)["demo_count"] == 50
    adapter = get_adapter("mikasa_robo_v2")
    task = item["tasks"][0]
    prompt = render(RunConfig.from_dict(config), task, [{"episode": i} for i in range(50)], [], adapter, adapter.interface(task, True))
    for text in ("Assigned expert demonstrations: 50", "Zero simulator attempts are allowed", "Derive the memory representation", "candidate completion conditions", "safety_check", "pure offline"):
        assert text in prompt
