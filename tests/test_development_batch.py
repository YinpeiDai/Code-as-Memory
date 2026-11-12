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


def batch_protocol(config):
    from cam.config import RunConfig
    return RunConfig.from_dict(config).both_protocol


def setup_dataset(tmp_path, name="mikasa_robo", count=50):
    config = json.loads((ROOT / f"configs/{name}/" / ("both.json" if name in ("mikasa_robo", "robomemarena") else "both_v2.json")).read_text())
    if name == "robomemarena" and count == 25:
        config.update(budget=25, training_seed_count=25, training_seed_policy="disjoint_once")
    path = tmp_path / "dataset"
    path.mkdir()
    item = {"datasets": [str(path)], "tasks": ["Example"], "demo_count": count}
    manifest = {"benchmark": name, "depth": config["depth"], "tasks": {
        "Example": {"episodes": [{"episode": i, "seed": i} for i in range(count)]}
    }}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return config, item, manifest, path


@pytest.mark.parametrize("name,count", [("robomemarena", 25), ("mikasa_robo", 50)])
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
        manifest["benchmark"] = "robomme"
    elif fault == "wrong_depth":
        manifest["depth"] = False
    elif fault == "wrong_budget":
        config["budget"] = 25
    if fault != "partial":
        (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        batch.data_assignment("mikasa_robo", item, config)




def test_start_rejects_partial_data_before_creating_batch_or_launching(tmp_path, monkeypatch):
    config, item, _, path = setup_dataset(tmp_path)
    (path / "manifest.json").rename(path / "manifest.partial.json")
    item["config"] = "config.json"
    (tmp_path / item["config"]).write_text(json.dumps(config))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"benchmarks": {"mikasa_robo": item}}))
    monkeypatch.setattr(batch, "ROOT", tmp_path)

    def launch_forbidden(*args, **kwargs):
        pytest.fail("Incomplete data must not start a supervisor or coding client")

    monkeypatch.setattr(batch.subprocess, "Popen", launch_forbidden)
    with pytest.raises(ValueError, match="not finalized"):
        batch.start(SimpleNamespace(plan=plan, workers=8, gpus=[0, 1]))
    assert not (tmp_path / "outputs").exists()




def test_demo_seed_batch_requires_full_demo_assignment(tmp_path):
    config, item, _, _ = setup_dataset(tmp_path)
    config.update(training_seed_policy="demo_cycle", physics_backend="cpu")
    assert batch.data_assignment("mikasa_robo", item, config)["missing"] == []
    config.update(training_seed_count=25, budget=25)
    with pytest.raises(ValueError, match="every assigned demonstration"):
        batch.data_assignment("mikasa_robo", item, config)


def test_demo_seed_shortfall_pins_exact_attempt_count(tmp_path):
    config, item, _, path = setup_dataset(tmp_path, "robomemarena", 32)
    config.update(budget=50, training_seed_count=50, training_seed_policy="demo_cycle")
    item.update(demo_count=50, demo_counts={"Example": 32}, interaction_counts={"Example": 32})
    assignment = batch.data_assignment("robomemarena", item, config)
    effective = batch.task_configuration(config, assignment, "Example")
    assert effective["budget"] == effective["training_seed_count"] == 32
    assert {k: v for k, v in effective.items() if k not in ("budget", "training_seed_count")} == {
        k: v for k, v in config.items() if k not in ("budget", "training_seed_count")}
    assert config["budget"] == 50
    assert assignment["existing"] == {str(path): ["Example"]}
    for fault in ({"Example": 31}, {"Example": 51}, {"Unknown": 32}, {"Example": True}):
        item["interaction_counts"] = fault
        with pytest.raises(ValueError):
            batch.data_assignment("robomemarena", item, config)
    item["interaction_counts"] = {"Example": 32}
    config["training_seed_policy"] = "disjoint_once"
    with pytest.raises(ValueError, match="demo-seed"):
        batch.data_assignment("robomemarena", item, config)


def test_robomemarena_demo_seed_plan_uses_actual_demos_once_each():
    from cam.interface.registry import get_adapter

    plan = json.loads((ROOT / "configs/robomemarena/experiments/demo_interaction_50.json").read_text())
    item = plan["benchmarks"]["robomemarena"]
    config = json.loads((ROOT / item["config"]).read_text())
    assert item["tasks"] == get_adapter("robomemarena").tasks()
    assert len(item["tasks"]) == 26 and item["demo_count"] == 50
    assert item["interaction_counts"] == item["demo_counts"] == {"T04_ButterNonEmptyDrawer": 32}
    assert config["budget"] == config["training_seed_count"] == 50
    assert config["training_seed_policy"] == "demo_cycle" and batch_protocol(config) == "demo_first"
    assert config["coding"]["model"] == "claude-opus-5-5"






@pytest.mark.parametrize("count", [25, 50])
def test_demo_only_accepts_assigned_count_without_interaction_identities(tmp_path, count):
    _, item, _, path = setup_dataset(tmp_path, "robomemarena", count)
    config = json.loads((ROOT / "configs/robomemarena/demos.json").read_text())
    item["require_prepared_data"] = True
    assert batch.data_assignment("robomemarena", item, config) == {
        "demo_count": count, "existing": {str(path): ["Example"]}, "missing": []}
    item["tasks"].append("Missing")
    with pytest.raises(ValueError, match="Complete the prepared demonstrations"):
        batch.data_assignment("robomemarena", item, config)


@pytest.mark.parametrize("count", [25, 50])
def test_demo_only_plan_covers_every_robomemarena_task(count):
    from cam.interface.registry import get_adapter

    plan = json.loads((ROOT / f"configs/robomemarena/experiments/demo_only_{count}.json").read_text())
    item = plan["benchmarks"]["robomemarena"]
    assert item["tasks"] == get_adapter("robomemarena").tasks()
    assert len(item["tasks"]) == 26 and item["demo_count"] == count
    assert item["require_prepared_data"]
    config = json.loads((ROOT / item["config"]).read_text())
    assert config["mode"] == "demos" and config["budget"] == 0 and config["depth"]


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
@pytest.mark.parametrize("shortfall_config", [False, True])
def test_evaluation_starts_while_other_tasks_learn_with_four_episode_workers(tmp_path, monkeypatch, evaluation_failure, shortfall_config):
    import threading
    import cam.interface.base
    import cam.interface.registry
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
            "benchmarks": {"mikasa_robo": {"tasks": list(names), "source": "fake", "source_hash": "source",
                "asset_hashes": {}, "existing": {"dataset": list(names)}, "missing": [],
                "config": "config.json", "demo_count": 50}}}
    if shortfall_config:
        plan["benchmarks"]["mikasa_robo"]["task_configs"] = {"Slow": "shortfall.json"}
    write(root / "plan.json", plan)
    monkeypatch.setattr(cam.interface.registry, "get_adapter", lambda *args: object())
    monkeypatch.setattr(cam.interface.base, "source_hash", lambda *args: "source")
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
                assigned = argv[argv.index("--tasks") + 1:]
                expected_config = "shortfall.json" if shortfall_config and assigned == ["Slow"] else "config.json"
                assert argv[argv.index("--config") + 1] == expected_config
                if shortfall_config:
                    assert len(assigned) == 1
                self.log.write(json.dumps({"runs": [{"root": str(runs[t]), "task": t} for t in assigned]}).encode())
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
