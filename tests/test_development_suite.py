"""The five-task comparison stays separate from full MIKASA and preserves exports."""
import importlib.util
from pathlib import Path

import pytest

from cam.config import COMPARISON_SUITES, RunConfig
from cam.core.history import freeze
from cam.io import read, sha256, write
from cam.reporting.catalog import index_directory

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.spec_from_file_location("develop_suite", ROOT / "scripts/develop_suite.py")
suite = importlib.util.module_from_spec(module)
module.loader.exec_module(suite)
NAME = "mikasa_robo_vla_suite"


def test_three_arms_have_exact_tasks_model_and_budgets():
    spec = read(ROOT / "configs" / NAME / "suite.json")
    assert spec["tasks"] == list(COMPARISON_SUITES[NAME]["tasks"])
    assert sum(arm["workers"] for arm in spec["arms"].values()) == 8
    assert len(spec["tasks"]) * len(spec["arms"]) == spec["expected_policies"] == 15
    for name, expected in spec["arms"].items():
        plan = read(ROOT / expected["plan"])
        item = plan["benchmarks"]["mikasa_robo"]
        cfg = RunConfig.from_dict(read(ROOT / item["config"]))
        assert cfg.suite == NAME and cfg.benchmark == "mikasa_robo"
        assert item["tasks"] == spec["tasks"]
        assert item["demo_count"] == expected["demo_count"]
        assert cfg.budget == expected["interaction_count"]
        assert cfg.coding.model == "claude-opus-5-5"
        assert cfg.physics_backend == "cpu" and cfg.depth and not cfg.builtin_demo
        assert cfg.training_seed_count == (50 if cfg.mode == "both" else None)
        if cfg.mode == "both":
            assert cfg.both_protocol == "demo_first" and cfg.training_seed_policy == "demo_cycle"
    assert RunConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="suite"):
        RunConfig(benchmark="robomme", mode="demos", suite=NAME)


def test_directory_does_not_merge_suite_with_full_mikasa(tmp_path):
    task = "ShellGameTouch-VLA-v0"
    for label, setting, comparison in [("full", "demo_interaction", None),
                                       ("25", "demo_only_25", NAME), ("50", "demo_only_50", NAME)]:
        run = tmp_path / label / setting / "same_tag"
        write(run / "run.json", {"config": {"benchmark": "mikasa_robo", "suite": comparison},
              "tasks": [task], "test_episodes": {task: [0]}, "physics_backend_by_task": {task: "gpu"}})
        write(run / "tasks" / task / "rollouts/test/R000/ep000/result.json",
              {"episode": 0, "success": True, "status": "success", "scorable": True})
    indexed = index_directory(tmp_path)
    assert len(indexed["groups"]) == 3
    assert {(g["benchmark"], g["setting"]) for g in indexed["groups"]} == {
        ("mikasa_robo", "demo_interaction"), (NAME, "demo_only_25"), (NAME, "demo_only_50")}
    assert all(r["latest"] and r["physics_backend"] == "gpu" for r in indexed["runs"])
    assert all(r["simulator_benchmark"] == "mikasa_robo" for r in indexed["runs"])


def test_export_keeps_exact_frozen_bytes_and_records_partial_coverage(run_factory, tmp_path, monkeypatch):
    # A synthetic simulator-free run exercises the same frozen export path.
    monkeypatch.setitem(COMPARISON_SUITES, NAME, {"benchmark": "robomme", "tasks": ("BinFill",)})
    root = run_factory()
    task = root / "tasks/BinFill"
    frozen = freeze(read(task / "state/config.json"))
    original = (task / "policy/tools/policy.py").read_bytes()
    plan = read(root / "run.json")
    plan["config"]["suite"] = NAME
    plan["physics_backend_by_task"] = {"BinFill": "cpu"}
    write(root / "run.json", plan)
    campaign = tmp_path / "campaign"
    arm = "demo_only_25"
    batch = campaign / "batch"
    write(campaign / "suite.json", {"suite": NAME, "benchmark": "robomme", "tasks": ["BinFill"],
          "arms": {arm: {"interaction_count": 0}}, "expected_policies": 1,
          "evaluation_episodes_per_task": 3, "gpus": [0, 1], "comparison_note": "Test"})
    write(campaign / "batches.json", {arm: {"batch": str(batch)}})
    write(batch / "plan.json", {"benchmarks": {"robomme": {"source_hash": "native"}}})
    state = {"status": "running", "tasks": {"robomme/BinFill": {"run": str(root), "status": "finished",
             "evaluation_status": "running", "evaluation": {"expected": 3, "scored": 3, "successes": 2, "success_rate": 2 / 3}}}}
    write(batch / "status.json", state)
    partial = suite.refresh(campaign)
    assert partial["frozen"] == 1 and partial["scored"] == 0 and partial["expected"] == 3
    suite.export_arm(campaign, arm)
    assert not (campaign / "configs/solutions").exists()
    state["status"] = "finished"
    write(batch / "status.json", state)
    suite.export_arm(campaign, arm)
    exported = campaign / "solutions" / NAME / arm / "BinFill/policy.py"
    assert exported.read_bytes() == original
    catalog = campaign / "configs/solutions" / NAME / f"{arm}.json"
    item = read(catalog)["tasks"]["BinFill"]
    assert item["code_sha256"]["policy.py"] == sha256(exported)
    assert item["source_program_hash"] == frozen["program_hash"]
    before = catalog.read_bytes()
    suite.export_arm(campaign, arm)
    assert catalog.read_bytes() == before
    exported.write_text("# changed")
    with pytest.raises(ValueError, match="checksum"):
        suite.export_arm(campaign, arm)
