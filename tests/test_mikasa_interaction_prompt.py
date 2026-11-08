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
        guide = "mikasa_robo_v2.md"

        def task_guide(self, task):
            raise AssertionError("Private task information must not enter the prompt")

        diagnostic_task_contract = task_guide

    supplement = (ROOT / "benchmarks/mikasa_robo_both_v2.md").read_text()
    for benchmark in ("mikasa_robo", "mikasa_robo_v2", "robomme", "rmbench", "robomemarena"):
        config = RunConfig(benchmark=benchmark, mode=mode, budget=budget, both_protocol=protocol)
        enabled = benchmark in ("mikasa_robo", "mikasa_robo_v2") and protocol == "v2"
        for task in ("BunchOfColors3-VLA-v0", "TakeItBack-VLA-v0"):
            prompt = render(config, task, [{"episode": 1}], [{"episode": 2}], adapter=Adapter())
            assert prompt.count(supplement) == int(enabled)
            assert (benchmark_mode_guidance(config) is not None) == enabled
            mode_file = f"both_{protocol}" if mode == "both" else mode
            assert (ROOT / "modes" / f"{mode_file}.md").read_text() in prompt
            assert (ROOT / "core.md").read_text() in prompt
            assert "Maximum simulator attempts: " + str(budget) in prompt


def test_preparation_pins_guidance_and_records_its_provenance(run_factory, monkeypatch):
    # The synthetic RoboMME fixture exercises full preparation without MIKASA
    # data, benchmark implementation, a native simulator, or a coding client.
    relative = BENCHMARK_MODE_GUIDANCE[("mikasa_robo_v2", "both", "v2")]
    monkeypatch.setitem(BENCHMARK_MODE_GUIDANCE, ("robomme", "both", "v2"), relative)
    root = run_factory("both", 2, both_protocol="v2")
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
