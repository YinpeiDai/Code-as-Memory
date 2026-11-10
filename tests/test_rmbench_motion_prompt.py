"""RMBench uses shared learning instructions and current public interface docs."""

import pytest

from cam.config import RunConfig
from cam.prompts import ROOT, benchmark_mode_guidance, render


@pytest.mark.parametrize("mode,budget,protocol", [
    ("demos", 0, None), ("interaction", 25, None),
    ("both", 25, "v1"), ("both", 25, "v2"),
])
def test_rmbench_uses_shared_prompt_and_current_public_api(mode, budget, protocol):
    class Adapter:
        guide = "rmbench/guide.md"

        def task_guide(self, task):
            raise AssertionError("Private task information must not enter the prompt")

        diagnostic_task_contract = task_guide

    retired_heading = "## RMBench offline motion checks and waypoint design"
    cfg = RunConfig(benchmark="rmbench", mode=mode, budget=budget, both_protocol=protocol)
    for task in ("cover_blocks", "swap_T", "put_back_block"):
        prompt = render(cfg, task, [{"episode": 1}], [], adapter=Adapter())
        assert retired_heading not in prompt
        assert benchmark_mode_guidance(cfg) is None
        assert "compares each public `*_ee_pose` with its requested target" in prompt
        if mode == "demos":
            assert prompt.count("Before finishing, review the complete code") == 1
            assert "`last` is initially `None`" in prompt
        mode_file = f"both_{protocol}" if mode == "both" else mode
        assert (ROOT / "core.md").read_text() in prompt
        assert (ROOT / "modes" / f"{mode_file}.md").read_text() in prompt
