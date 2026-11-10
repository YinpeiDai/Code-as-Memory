"""Versioned learning instructions share evidence, tools, and attempt accounting."""

from dataclasses import replace
from pathlib import Path

import pytest

from cam.config import BOTH_PROTOCOLS, CodingConfig, RunConfig, default_evaluation_budgets, learning_setting
from cam.core.budget import Ledger
from cam.harness.prepare import prepare_runs, validate_run
from cam.harness.service import TOOLS
from cam.io import read, write
from cam.prompts import ROOT, render
from cam.providers.clients import TOOL_NAMES
from cam.reporting.report import build
from cam.reporting.site import build_site


@pytest.mark.parametrize("protocol", [None, "", "v0", "legacy", "../both"])
def test_both_requires_an_explicit_supported_version(protocol):
    with pytest.raises(ValueError, match="explicit both_protocol"):
        RunConfig(benchmark="robomme", mode="both", budget=20, both_protocol=protocol)


@pytest.mark.parametrize("mode,budget", [("demos", 0), ("interaction", 20)])
def test_version_cannot_silently_affect_other_modes(mode, budget):
    with pytest.raises(ValueError, match="only valid"):
        RunConfig(benchmark="robomme", mode=mode, budget=budget, both_protocol="v2")


def test_all_versioned_templates_and_client_independent_prompts():
    configs = Path(__file__).resolve().parents[1] / "configs"
    templates = sorted(configs.glob("*/both_*.json"))
    assert templates
    benchmark_clients = set()
    for path in templates:
        config = RunConfig.from_dict(read(path))
        # MIKASA assigns 50 demos and one attempt on each demo identity.
        assert config.budget == (50 if config.benchmark in ("mikasa_robo", "mikasa_robo_v2", "robomemarena") else 25)
        assert path.name == (f"both_{config.coding.agent}.json" if config.benchmark in ("mikasa_robo", "robomemarena")
                             else f"both_{config.both_protocol}_{config.coding.agent}.json")
        assert RunConfig.from_dict(config.to_dict()) == config
        benchmark_clients.add((config.benchmark, config.coding.agent))
    for benchmark, client in benchmark_clients:
        if benchmark in ("mikasa_robo", "robomemarena"):
            continue
        versions = [
            read(configs / benchmark / f"both_{version}_{client}.json")
            for version in ("v1", "v2")
        ]
        assert {k: v for k, v in versions[0].items() if k != "both_protocol"} == {
            k: v for k, v in versions[1].items() if k != "both_protocol"
        }
    assert not (ROOT / "modes/both.md").exists()
    for version in ("v1", "v2"):
        claude = RunConfig(benchmark="robomme", mode="both", budget=20, both_protocol=version)
        codex = replace(claude, coding=CodingConfig(agent="codex", model="gpt-6.1-sol", provider="openai"))
        episodes = [{"episode": 2}, {"episode": 5}, {"episode": 8}]
        assert render(claude, "BinFill", episodes) == render(codex, "BinFill", episodes)
    assert {t["name"] for t in TOOLS} == set(TOOL_NAMES) == {
        "command", "read_image", "usage", "snapshot", "validate", "rollout"
    }


def test_two_versions_pin_distinct_prompts_with_identical_evidence_and_harness(dataset, tmp_path):
    data, source = dataset
    roots, plans, prompts, states = [], [], [], []
    for version in ("v1", "v2"):
        result = prepare_runs(
            tmp_path / "runs", data,
            RunConfig(benchmark="robomme", mode="both", budget=2, both_protocol=version),
            benchmark_source=source, timetag="same_comparison_tag",
        )
        root = Path(result["runs"][0]["root"])
        roots.append(root)
        plan = validate_run(root, full_data=True)
        plans.append(plan)
        task = root / "tasks/BinFill"
        state = read(task / "state/config.json")
        states.append(state)
        prompts.append((task / "prompt.md").read_text())
        assert plan["config"]["both_protocol"] == state["both_protocol"] == version
        assert Ledger(state["ledger"]).config()["both_protocol"] == version
        assert result["setting"] == root.parent.name == f"demo_interaction_{version}"
        assert f"Demo + interaction protocol: {version}" in prompts[-1]
        assert (ROOT / "modes" / f"both_{version}.md").read_text() in prompts[-1]
        assert (task / "workspace/PROTOCOL.md").read_text() == prompts[-1]
        assert f"{version}:" in (root / "README.md").read_text()
        report = build(root)
        assert report["learning_setting"] == f"demo_interaction_{version}"
        assert f"/demo_interaction_{version}/" in report["name"]
        # V2 is a prompt strategy, not a hidden hard gate: both have identical
        # accounting and allow their first counted attempt without new tool calls.
        ledger = Ledger(state["ledger"])
        attempt = ledger.reserve(state, state["train_episodes"][0])
        ledger.update(attempt, steps=0, step_calls=0, status="harness_error")
        assert len(ledger.rows()) == 1

    assert prompts[0] != prompts[1]
    assert "There is no required\ndemo-first phase" in prompts[0]
    assert "Before the first simulator\nattempt, inspect every assigned recording" in prompts[1]
    assert states[0]["train_records"] == states[1]["train_records"]
    for key in ("harness_hash", "runtime_hash", "benchmark_hash", "interfaces", "test_episodes"):
        assert plans[0][key] == plans[1][key]
    h5 = "tasks/BinFill/demonstrations/record_dataset_BinFill.h5"
    assert plans[0]["inputs"][h5] == plans[1]["inputs"][h5]
    assert plans[0]["inputs"]["tasks/BinFill/prompt.md"] != plans[1]["inputs"]["tasks/BinFill/prompt.md"]

    site = tmp_path / "site"
    build_site(roots, [], site)
    runs = read(site / "data.json")["runs"]
    assert {r["learning_setting"] for r in runs} == {"demo_interaction_v1", "demo_interaction_v2"}
    assert len({r["name"] for r in runs}) == 2

    # Once prepared, editing the version in task state invalidates the pin.
    changed = {**states[0], "both_protocol": "v2"}
    write(roots[0] / "tasks/BinFill/state/config.json", changed)
    with pytest.raises(ValueError, match="Pinned input changed"):
        validate_run(roots[0])


def test_historical_unversioned_reports_are_not_relabeled():
    assert learning_setting("both") == "demo_interaction"
    with pytest.raises(ValueError, match="explicit both_protocol"):
        RunConfig.from_dict({"benchmark": "robomme", "mode": "both", "budget": 20})


@pytest.mark.parametrize("mode,budget,version", [
    ("demos", 0, None), ("interaction", 25, None), ("both", 25, "v1"), ("both", 25, "v2"),
])
def test_prompts_never_read_source_derived_task_guides(mode, budget, version):
    class Adapter:
        guide = "robomme/guide.md"

        def task_guide(self, task):
            raise AssertionError("Private success conditions must never reach a learner")

    config = RunConfig(benchmark="robomme", mode=mode, budget=budget, both_protocol=version)
    prompt = render(config, "BinFill", [{"episode": 0}], adapter=Adapter())
    assert "Official success test" not in prompt and "## Task contract" not in prompt
    if version == "v2":
        assert "what counts as success" in prompt and "focused exploratory rollout" in prompt
        assert "Maximum simulator attempts: 25" in prompt


@pytest.mark.parametrize("mode,budget,version", [
    ("demos", 0, None), ("interaction", 25, None), ("both", 25, "v1"), ("both", 25, "v2"),
])
def test_authorized_order_correction_is_scoped_to_exact_benchmark_task(mode, budget, version):
    class Adapter:
        guide = "mikasa_robo/guide.md"

        def task_guide(self, task):
            raise AssertionError("A static correction must not read simulator task guides")

        diagnostic_task_contract = task_guide

    for benchmark, task in [
        ("mikasa_robo", "SeqOfColors3-VLA-v0"),
        ("mikasa_robo_v2", "SeqOfColors3-VLA-v0"),
        ("mikasa_robo", "SeqOfColors5-VLA-v0"),
        ("mikasa_robo_v2", "SeqOfColors5-VLA-v0"),
        ("mikasa_robo_v2", "SeqOfColors3-Long-VLA-v0"),
        ("mikasa_robo_v2", "ChainOfColors3-VLA-v0"),
        ("rmbench", "SeqOfColors3-VLA-v0"),
    ]:
        selected_version = None if benchmark == "mikasa_robo" and mode == "both" else version
        config = RunConfig(benchmark=benchmark, mode=mode, budget=budget, both_protocol=selected_version)
        prompt = render(config, task, [{"episode": 0}], adapter=Adapter())
        corrected = benchmark in ("mikasa_robo", "mikasa_robo_v2") and task == "SeqOfColors3-VLA-v0"
        assert ("## Task instruction clarification" in prompt) == corrected
        if corrected:
            assert "in the order they appeared" in prompt
            assert 'overrides "in any order"' in prompt
        assert (ROOT / "core.md").read_text() in prompt
        assert "Official success test" not in prompt and "## Task contract" not in prompt


def test_preparation_records_and_pins_clarification(run_factory, monkeypatch):
    from cam.prompts import TASK_CLARIFICATIONS, task_clarification

    # Exercise preparation with the small synthetic dataset, without a simulator.
    relative = TASK_CLARIFICATIONS[("mikasa_robo_v2", "SeqOfColors3-VLA-v0")]
    monkeypatch.setitem(TASK_CLARIFICATIONS, ("robomme", "BinFill"), relative)
    root = run_factory()
    entry = task_clarification("robomme", "BinFill")
    plan = validate_run(root)
    assert plan["task_clarifications"] == {"BinFill": entry}
    assert entry["text"] in (root / "tasks/BinFill/prompt.md").read_text()
    assert entry["text"] in (root / "tasks/BinFill/workspace/PROTOCOL.md").read_text()
    assert entry["text"].rstrip() in (root / "README.md").read_text()
    pinned = root / "inputs/harness" / entry["source"]
    assert pinned.read_text() == entry["text"]
    pinned.write_text("Changed correction\n")
    with pytest.raises(ValueError, match="Pinned source changed"):
        validate_run(root)


@pytest.mark.parametrize("used", [0, 7, 25])
def test_v2_hard_cap_early_stopping_and_final_checkpoint(run_factory, used):
    from cam.core.history import freeze, select_checkpoint

    root = run_factory("both", 25, both_protocol="v2")
    task = root / "tasks/BinFill"
    cfg = read(task / "state/config.json")
    ledger = Ledger(cfg["ledger"])
    for index in range(used):
        attempt = ledger.reserve(cfg, cfg["train_episodes"][index % 3])
        ledger.update(attempt, steps=0, step_calls=0, status="harness_error" if index == 0 else "fail")
    if used == 25:
        with pytest.raises(ValueError, match="budget exhausted"):
            ledger.reserve(cfg, cfg["train_episodes"][25 % 3])
    frozen = freeze(cfg)
    assert frozen["training"]["attempts"] == used
    assert select_checkpoint(task, 25)[1]["checkpoint"] == used
    with pytest.raises(ValueError, match="sealed"):
        ledger.reserve(cfg, cfg["train_episodes"][used % 3])
    assert [row["budget"] for row in build(root)["settings"]] == [1, 5, 10, 20, 25]


@pytest.mark.parametrize("mode,limit,expected", [
    ("demos", 0, [0]), ("interaction", 2, [1, 2]),
    ("both", 20, [1, 5, 10, 20]), ("both", 25, [1, 5, 10, 20, 25]),
])
def test_default_evaluation_includes_configured_final_budget(mode, limit, expected):
    assert default_evaluation_budgets({"mode": mode, "budget": limit}) == expected
