"""Scope and pin MIKASA's public motion guidance without simulator access."""

import pytest

from cam.config import RunConfig
from cam.harness.prepare import validate_run
from cam.prompts import BENCHMARK_MODE_GUIDANCE, ROOT, benchmark_mode_guidance, render


@pytest.mark.parametrize("mode,budget,protocol", [
    ("demos", 0, None), ("interaction", 25, None),
    ("both", 25, "v1"), ("both", 25, "v2"),
])
def test_guidance_is_only_for_mikasa_both_v2(mode, budget, protocol):
    class Adapter:
        guide = "mikasa_robo/guide.md"

        def task_guide(self, task):
            raise AssertionError("Private task information must not enter the prompt")

        diagnostic_task_contract = task_guide

    supplement = (ROOT / "mikasa_robo/both.md").read_text()
    for benchmark in ("mikasa_robo_v2", "robomme", "rmbench", "robomemarena"):
        selected_protocol = None if benchmark == "robomemarena" and mode == "both" else protocol
        config = RunConfig(benchmark=benchmark, mode=mode, budget=budget, both_protocol=selected_protocol)
        enabled = benchmark in ("mikasa_robo", "mikasa_robo_v2") and protocol == "v2"
        for task in ("BunchOfColors3-VLA-v0", "TakeItBack-VLA-v0"):
            prompt = render(config, task, [{"episode": 1}], [{"episode": 2}], adapter=Adapter())
            assert prompt.count(supplement) == int(enabled)
            guidance = benchmark_mode_guidance(config)
            assert (guidance is not None and guidance["text"] == supplement) == enabled
            mode_file = f"both_{protocol}" if mode == "both" else mode
            mode_text = (ROOT / "modes" / f"{mode_file}.md").read_text()
            if config.both_protocol == "demo_first":
                mode_text = (ROOT / "modes/both_v2.md").read_text().replace("Demonstrations + interaction v2:", "Demonstrations + interaction:", 1)
            assert mode_text in prompt
            assert (ROOT / "core.md").read_text() in prompt
            assert "Maximum simulator attempts: " + str(budget) in prompt


@pytest.mark.parametrize("benchmark,mode,budget", [
    ("mikasa_robo_v2", "both", 2),
    ("robomemarena", "both", 2),
    ("mikasa_robo", "demos", 0),
])
def test_preparation_pins_guidance_and_records_its_provenance(run_factory, monkeypatch, benchmark, mode, budget):
    # The synthetic RoboMME fixture exercises full preparation without MIKASA
    # data, benchmark implementation, a native simulator, or a coding client.
    protocol = None if mode == "demos" else "demo_first" if benchmark == "robomemarena" else "v2"
    relative = BENCHMARK_MODE_GUIDANCE[(benchmark, mode, protocol)]
    fixture_protocol = "v2" if mode == "both" else None
    monkeypatch.setitem(BENCHMARK_MODE_GUIDANCE, ("robomme", mode, fixture_protocol), relative)
    root = run_factory(mode, budget, both_protocol=fixture_protocol)
    manifest = validate_run(root)
    entry = manifest["benchmark_mode_guidance"]
    task = root / "tasks/BinFill"
    pinned = root / "inputs/harness" / entry["source"]
    assert pinned.read_text() == entry["text"]
    assert entry["text"] in (task / "prompt.md").read_text()
    assert (task / "prompt.md").read_text() == (task / "workspace/PROTOCOL.md").read_text()
    assert f"(inputs/harness/{entry['source']})" in (root / "README.md").read_text()
    pinned.write_text("Changed guidance\n")
    with pytest.raises(ValueError, match="Pinned source changed"):
        validate_run(root)


@pytest.mark.parametrize("benchmark", ["mikasa_robo", "mikasa_robo_v2", "robomme", "rmbench", "robomemarena"])
def test_demo_only_notes_preserve_shared_prompt_and_zero_attempts(benchmark):
    class Adapter:
        guide = "mikasa_robo/guide.md"

        def task_guide(self, task):
            raise AssertionError("Private task information must not enter the prompt")

        diagnostic_task_contract = task_guide

    config = RunConfig(benchmark=benchmark, mode="demos", budget=0)
    supplement = (ROOT / "mikasa_robo/demos.md").read_text()
    interaction = (ROOT / "mikasa_robo/both.md").read_text()
    expected = benchmark in ("mikasa_robo", "mikasa_robo_v2")
    for task in ("ShellGameTouch-VLA-v0", "InterceptMedium-VLA-v0", "RememberColor3-VLA-v0", "RememberColor5-VLA-v0", "RememberColor9-VLA-v0"):
        prompt = render(config, task, [{"episode": i} for i in range(250)], [], adapter=Adapter())
        assert prompt.count(supplement) == int(expected)
        assert interaction not in prompt
        assert (ROOT / "modes/demos.md").read_text() in prompt
        assert "Assigned expert demonstrations: 250;" in prompt
        assert "Maximum simulator attempts: 0 total." in prompt
        assert "`rollout` is disabled" in prompt
