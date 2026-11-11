"""RoboMemArena guidance stays within its requested benchmark and learning mode."""

import pytest

from cam.config import RunConfig
from cam.prompts import ROOT, benchmark_mode_guidance, render


@pytest.mark.parametrize("mode,budget,protocol", [
    ("demos", 0, None), ("interaction", 25, None),
    ("both", 25, "v1"), ("both", 25, "v2"),
])
def test_guidance_is_only_for_robomemarena_demo_first(mode, budget, protocol):
    class Adapter:
        guide = "robomemarena/guide.md"

        def task_guide(self, task):
            raise AssertionError("Private task information must not enter the prompt")

        diagnostic_task_contract = task_guide

    supplement = (ROOT / "robomemarena/both.md").read_text()
    for benchmark in ("robomemarena", "mikasa_robo", "mikasa_robo_v2", "robomme", "rmbench"):
        selected_protocol = None if benchmark in ("mikasa_robo", "robomemarena") and mode == "both" else protocol
        config = RunConfig(benchmark=benchmark, mode=mode, budget=budget, both_protocol=selected_protocol)
        enabled = benchmark == "robomemarena" and mode == "both"
        for task in ("T01_CookiesTomatoBasket", "T26_ChocolateCreamPlate"):
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
            assert f"Maximum simulator attempts: {budget}" in prompt


def test_default_templates_are_demo_first_without_version_labels():
    from pathlib import Path
    from cam.io import read

    repo = Path(__file__).resolve().parents[1]
    class Adapter:
        guide = 'robomemarena/guide.md'
    cfg = RunConfig.from_dict(read(repo/'configs/robomemarena/both.json'))
    assert cfg.both_protocol == 'demo_first' and cfg.setting == 'demo_interaction'
    assert cfg.budget == cfg.training_seed_count == 50 and cfg.training_seed_policy == 'demo_cycle'
    assert cfg.depth and not cfg.builtin_demo
    demos = [{'episode': i} for i in range(50)]
    prompt = render(cfg, 'T01_CookiesTomatoBasket', demos, demos, adapter=Adapter())
    assert 'v2' not in prompt and 'once' in prompt
    assert benchmark_mode_guidance(cfg)['source'] == 'cam/prompts/robomemarena/both.md'
    with pytest.raises(ValueError, match='demo-first'):
        RunConfig(benchmark='robomemarena', mode='both', budget=50, both_protocol='v1')
