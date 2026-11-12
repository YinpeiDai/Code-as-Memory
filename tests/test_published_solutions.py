"""Final policies can enter the existing evaluator without demos or learning."""

from pathlib import Path
import shutil
from copy import deepcopy

import pytest

from cam.interface.base import source_hash
from cam.interface.registry import get_adapter
from cam.core.budget import Ledger
from cam.core.history import freeze, select_checkpoint
from cam.harness.jobs import learn_task
from cam.harness.prepare import validate_run
from cam.harness.solutions import load_solutions, prepare_solutions, record_hash
from cam.io import read, sha256, write
from cam.config import COMPARISON_SUITES


@pytest.fixture
def published(run_factory, dataset, tmp_path):
    source = run_factory()
    task = source / "tasks/BinFill"
    freeze(read(task / "state/config.json"))
    data, native = dataset
    adapter = get_adapter("robomme", native)
    solution = tmp_path / "solutions/robomme/demo_only_3"
    target = solution / "BinFill/policy.py"
    target.parent.mkdir(parents=True)
    target.write_bytes((task / "policy/tools/policy.py").read_bytes())
    plan = read(source / "run.json")
    spec = {
        "schema": "cam-solutions/1", "benchmark": "robomme", "experiment": "demo_only_3",
        "benchmark_source_sha256": source_hash(adapter), "evaluation_config": plan["config"],
        "tasks": {"BinFill": {
            "task_spec": read(task / "policy/task.json"), "code_sha256": {"policy.py": sha256(target)},
            "test_records_sha256": record_hash(adapter.records("BinFill", "test")),
            "interface_sha256": record_hash(adapter.interface("BinFill", True)),
        }},
    }
    catalog = tmp_path / "configs/solutions/robomme/demo_only_3.json"
    write(catalog, spec)
    # Published evaluation must work with no historical run or demo H5 remaining.
    shutil.rmtree(source)
    shutil.rmtree(data)
    return solution, native, catalog


def test_prepare_frozen_evaluation_without_data_or_model(published, tmp_path, monkeypatch):
    from cam.harness import prepare

    solution, native, _ = published

    def forbidden(*args, **kwargs):
        raise AssertionError("Published evaluation cannot load demos or select training seeds")

    monkeypatch.setattr(prepare, "validate_dataset", forbidden)
    monkeypatch.setattr(prepare, "training_schedule", forbidden)
    monkeypatch.setattr(prepare, "render", forbidden)
    result = prepare_solutions(tmp_path / "eval", solution, benchmark_source=native, timetag="test")
    root = Path(result["runs"][0]["root"])
    plan = validate_run(root, full_data=True)
    assert plan["evaluation_only"] and not plan["fresh_policy"]
    assert plan["dataset_reference"] is None
    assert plan["raw_data_reference"] is None
    assert plan["demonstrations_per_task"] == {"BinFill": 0}
    assert plan["test_episodes"] == {"BinFill": [0, 1, 2]}
    task = root / "tasks/BinFill"
    assert not (task / "demonstrations").exists()
    assert not list(root.rglob("*.h5"))
    cfg = read(task / "state/config.json")
    ledger = Ledger(cfg["ledger"])
    assert ledger.config()["sealed"] and ledger.rows() == []
    assert cfg["train_episodes"] == []
    checkpoint, frozen = select_checkpoint(task, 0)
    assert frozen["training"]["attempts"] == 0
    assert (checkpoint / "tools/policy.py").read_bytes() == (solution / "BinFill/policy.py").read_bytes()
    assert "no model calls" in (root / "README.md").read_text()
    with pytest.raises(ValueError, match="already frozen"):
        learn_task(root, "BinFill", 0)
    with pytest.raises(ValueError, match="sealed"):
        ledger.reserve(cfg, 0)
    with pytest.raises(FileExistsError):
        prepare_solutions(tmp_path / "eval", solution, benchmark_source=native, timetag="test")
    (checkpoint / "tools/policy.py").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        select_checkpoint(task, 0)


@pytest.mark.parametrize("fault", ["code", "extra", "symlink", "memory", "metadata", "interface", "benchmark"])
def test_reject_changed_published_inputs(published, tmp_path, fault):
    solution, native, catalog = published
    spec = read(catalog)
    item = spec["tasks"]["BinFill"]
    path = solution / "BinFill/policy.py"
    if fault == "code":
        path.write_text("raise RuntimeError('never execute on host')")
    elif fault == "extra":
        (path.parent / "scratch.py").write_text("# intermediate code is not published")
    elif fault == "symlink":
        external = tmp_path / "external.py"
        path.rename(external)
        path.symlink_to(external)
    elif fault == "memory":
        item["task_spec"]["task"] = "OtherTask"
    elif fault == "metadata":
        item["test_records_sha256"] = "wrong"
    elif fault == "interface":
        item["interface_sha256"] = "wrong"
    else:
        spec["benchmark_source_sha256"] = "wrong"
    write(catalog, spec)
    with pytest.raises(ValueError):
        prepare_solutions(tmp_path / "eval", solution, benchmark_source=native)
    assert not (tmp_path / "eval").exists()


def test_shipped_rmbench_final_policies_are_complete_code_only():
    repo = Path(__file__).resolve().parents[1]
    for experiment, count in [("demo_only_25", 250), ("demo_only_50", 496)]:
        solution = repo / "solutions/rmbench" / experiment
        spec, tasks, loaded = load_solutions(solution)
        assert len(tasks) == 10
        assert sum(row["assigned_demos"] for row in spec["tasks"].values()) == count
        assert all(path.suffix == ".py" for path in solution.rglob("*") if path.is_file() and path != solution / "README.md")
        for task in tasks:
            assert loaded["files"][task]["task.json"]
            assert spec["tasks"][task]["training_attempts"] == 0
        if experiment == "demo_only_50":
            assert spec["tasks"]["swap_blocks"]["source_run_tag"] == "20261108T150726_925267Z_swap_blocks_fresh"


def test_shipped_robomme_25_demo_policies_match_the_795_of_800_release():
    repo = Path(__file__).resolve().parents[1]
    solution = repo / "solutions/robomme/demo_only_25"
    spec, tasks, loaded = load_solutions(solution)
    assert len(tasks) == 16
    assert spec["archived_result"] == {
        "expected": 800, "scored": 800, "successes": 795, "success_rate": 795 / 800
    }
    assert sum(row["assigned_demos"] for row in spec["tasks"].values()) == 400
    assert sum(row["archived_result"]["successes"] for row in spec["tasks"].values()) == 795
    assert all(p.suffix == ".py" for p in solution.rglob("*") if p.is_file() and p != solution / "README.md")
    assert spec["evaluation_config"]["depth"] is True
    assert spec["evaluation_config"]["budget"] == 0
    for task in tasks:
        item = spec["tasks"][task]
        assert item["training_attempts"] == item["source_checkpoint"] == 0
        assert item["source_run_tag"] == "regular_demo_v2_restart_20261105_040321"
        assert loaded["files"][task]["task.json"]


def test_cli_solution_path_does_not_require_dataset_or_config(monkeypatch, capsys):
    from cam import cli
    from cam.harness import solutions

    calls = []
    monkeypatch.setattr(solutions, "prepare_solutions", lambda *args: calls.append(args) or {"runs": []})
    assert cli.main(["prepare", "out", "--solution", "solutions/rmbench/demo_only_50", "--tasks", "swap_blocks"]) == 0
    assert calls[0][2] == ["swap_blocks"]
    assert calls[0][-1] is None
    assert '"runs"' in capsys.readouterr().out


def test_rmbench_observed_motion_catalogs_preserve_code_and_historical_provenance():
    repo = Path(__file__).resolve().parents[1]
    adapter = get_adapter("rmbench")
    for experiment in ("demo_only_25", "demo_only_50"):
        solution = repo / "solutions/rmbench" / experiment
        old_path = repo / "configs/solutions/rmbench" / (experiment + ".json")
        old = read(old_path)
        catalog = old_path.with_name(experiment + "_observed_motion.json")
        new, tasks, loaded = load_solutions(solution, catalog=catalog)
        assert new["interface_revision"]["source_catalog_sha256"] == sha256(old_path)
        assert new["experiment"] == experiment + "_observed_motion"
        assert new["evaluation_config"] == old["evaluation_config"]
        assert new["benchmark_source_sha256"] == old["benchmark_source_sha256"]
        for task in tasks:
            item = new["tasks"][task]
            interface = adapter.interface(task, new["evaluation_config"]["depth"])
            assert item["interface_sha256"] == record_hash(interface)
            assert item["source_interface_sha256"] == old["tasks"][task]["interface_sha256"]
            assert {k: v for k, v in item.items() if k not in ("interface_sha256", "source_interface_sha256")} == {
                k: v for k, v in old["tasks"][task].items() if k != "interface_sha256"
            }
            assert loaded["files"][task]["tools/policy.py"] == (solution / task / "policy.py").read_bytes()


@pytest.mark.parametrize("arm,demos,attempts,successes", [
    ("demo_only_25", 25, 0, [10, 44, 50, 48, 1]),
    ("demo_only_50", 50, 0, [43, 41, 45, 31, 15]),
    ("demo_interaction_50", 50, 50, [50] * 5),
])
def test_shipped_mikasa_comparison_policies(arm, demos, attempts, successes):
    repo = Path(__file__).resolve().parents[1]
    name = "mikasa_robo_vla_suite"
    spec, tasks, loaded = load_solutions(repo / "solutions" / name / arm)
    assert tasks == list(COMPARISON_SUITES[name]["tasks"])
    assert spec["suite"] == spec["evaluation_config"]["suite"] == name
    assert spec["benchmark"] == spec["evaluation_config"]["benchmark"] == "mikasa_robo"
    assert spec["evaluation_config"]["budget"] == 0 and loaded["provenance"]["learning_performed"] is False
    for task, successes in zip(tasks, successes):
        row = spec["tasks"][task]
        assert row["assigned_demos"] == demos
        assert row["training_attempts"] == row["source_checkpoint"] == attempts
        assert row["archived_result"] == {"successes": successes, "scored": 50, "expected": 50, "success_rate": successes / 50}
        assert row["physics_backend"] == ("gpu" if task == "ShellGameTouch-VLA-v0" else "cpu")
        assert all(len(row[k]) == 64 for k in ("source_harness_sha256", "source_prompt_sha256", "source_program_hash"))


@pytest.mark.parametrize("fault", ["missing_suite", "missing_config_suite", "extra_task", "wrong_benchmark"])
def test_reject_comparison_catalog_namespace_mismatch(tmp_path, fault):
    repo = Path(__file__).resolve().parents[1]
    solution = repo / "solutions/mikasa_robo_vla_suite/demo_only_25"
    spec = read(repo / "configs/solutions/mikasa_robo_vla_suite/demo_only_25.json")
    if fault == "missing_suite":
        spec.pop("suite")
    elif fault == "missing_config_suite":
        spec["evaluation_config"].pop("suite")
    elif fault == "extra_task":
        spec["tasks"]["TakeItBack-VLA-v0"] = deepcopy(next(iter(spec["tasks"].values())))
    else:
        spec["evaluation_config"]["benchmark"] = "robomme"
    catalog = tmp_path / "catalog.json"
    write(catalog, spec)
    with pytest.raises(ValueError, match="suite"):
        load_solutions(solution, catalog, tasks=["ShellGameTouch-VLA-v0"])
