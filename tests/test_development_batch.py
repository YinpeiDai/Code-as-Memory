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


@pytest.mark.parametrize("fault", [None, "unfinished", "early_freeze", "changed_policy", "changed_checkpoint"])
def test_automatic_evaluation_requires_finished_full_budget_and_intact_policy(run_factory, fault):
    from cam.core.budget import Ledger
    from cam.core.history import freeze
    from cam.io import read, write

    root = run_factory("both", 3, both_protocol="v2")
    folder = root / "tasks/BinFill"
    config = read(folder / "state/config.json")
    ledger = Ledger(config["ledger"])
    count = 2 if fault == "early_freeze" else 3
    for episode in config["train_episodes"][:count]:
        attempt = ledger.reserve(config, episode)
        ledger.update(attempt, status="fail", steps=1, step_calls=1)
    final = freeze(config)
    write(folder / "logs/completion.json", {"status": "failed" if fault == "unfinished" else "finished"})
    if fault == "changed_policy":
        (folder / "policy/policy.md").write_text("changed after freeze")
    if fault == "changed_checkpoint":
        (folder / "checkpoints/R003/policy.md").write_text("changed after freeze")
    if fault:
        with pytest.raises(ValueError):
            batch.frozen_evaluation(root, "BinFill")
    else:
        assert batch.frozen_evaluation(root, "BinFill") == {
            "budget": 3, "expected": 3, "program_hash": final["program_hash"]}


@pytest.mark.parametrize("evaluation_failure", [False, True])
def test_evaluation_starts_while_other_tasks_learn_with_four_episode_workers(tmp_path, monkeypatch, evaluation_failure):
    import threading
    import cam.benchmarks.base
    import cam.benchmarks.registry
    import cam.reporting.report
    from cam.io import read, tree_hash, write

    root = tmp_path / "batch"
    harness = root / "inputs/checkout/src/cam"
    harness.mkdir(parents=True)
    names = ("Fast", "Slow")
    runs = {task: tmp_path / "runs" / task for task in names}
    plan = {"workers": 2, "gpus": [0, 1], "python": "python", "runs": str(tmp_path / "runs"),
            "tag": "test", "environment": {}, "harness_hash": tree_hash(harness),
            "evaluation": {"enabled": True, "workers_per_gpu": 2, "task_concurrency": 1},
            "benchmarks": {"mikasa_robo_v2": {"tasks": list(names), "source": "fake", "source_hash": "source",
                "asset_hashes": {}, "existing": {"dataset": list(names)}, "missing": [],
                "config": "config.json", "demo_count": 50}}}
    write(root / "plan.json", plan)
    monkeypatch.setattr(cam.benchmarks.registry, "get_adapter", lambda *args: object())
    monkeypatch.setattr(cam.benchmarks.base, "source_hash", lambda *args: "source")
    monkeypatch.setattr(batch, "boundary_check", lambda *args: None)
    monkeypatch.setattr(batch, "frozen_evaluation", lambda *args: {"budget": 50, "expected": 50, "program_hash": "hash"})
    def report(run):
        failed = evaluation_failure and Path(run).name == "Fast"
        return {"settings": [{"budget": 50, "scored": 2 if failed else 50, "expected": 50,
            "successes": 1 if failed else 30, "success_rate": None if failed else 0.6,
            "infrastructure_errors": int(failed)}]}

    monkeypatch.setattr(cam.reporting.report, "build", report)
    slow_learning, first_evaluation = threading.Event(), threading.Event()
    evaluated = []

    class Process:
        pid = 123

        def __init__(self, argv, **kwargs):
            self.argv, self.log = argv, kwargs["stdout"]

        def wait(self):
            argv = self.argv
            if "prepare" in argv:
                self.log.write(json.dumps({"runs": [{"root": str(runs[t]), "task": t} for t in names]}).encode())
            elif "learn" in argv:
                task = Path(argv[argv.index("learn") + 1]).name
                if task == "Slow":
                    slow_learning.set()
                    assert first_evaluation.wait(5), "Evaluation waited for every learner to finish"
                write(runs[task] / "tasks" / task / "logs/completion.json", {"status": "finished"})
            elif "evaluate" in argv:
                assert argv[argv.index("--budgets") + 1] == "50"
                assert argv[argv.index("--gpus") + 1:argv.index("--workers-per-gpu")] == ["0", "1"]
                assert argv[argv.index("--workers-per-gpu") + 1] == "2"
                task = Path(argv[argv.index("evaluate") + 1]).name
                if task == "Fast":
                    assert slow_learning.wait(5)
                    assert not (runs["Slow"] / "tasks/Slow/logs/completion.json").exists()
                    first_evaluation.set()
                evaluated.append(task)
                if evaluation_failure and task == "Fast":
                    return 1
            return 0

    monkeypatch.setattr(batch.subprocess, "Popen", Process)
    batch.worker(root)
    status = read(root / "status.json")
    assert status["status"] == ("needs_attention" if evaluation_failure else "finished")
    assert evaluated == list(names)
    for row in status["tasks"].values():
        assert row["status"] == "finished"
        if evaluation_failure and row["task"] == "Fast":
            assert row["evaluation_status"] == "failed"
            assert row["evaluation"]["scored"] == 2 and row["evaluation"]["infrastructure_errors"] == 1
        else:
            assert row["evaluation_status"] == "finished" and row["evaluation"]["scored"] == 50
