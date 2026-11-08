from pathlib import Path

import pytest

from cam.cli import main
from cam.config import CodingConfig, RunConfig
from cam.harness.prepare import coding_run_name, prepare_runs, validate_run
from cam.io import read, write
from cam.reporting.report import build


@pytest.mark.parametrize(
    "mode,setting,budget,both_protocol",
    [
        ("demos", "demo_only", 0, None),
        ("interaction", "interaction_only", 2, None),
        ("both", "demo_interaction_v1", 2, "v1"),
        ("both", "demo_interaction_v2", 2, "v2"),
    ],
)
def test_organized_run_and_access_readme(
    dataset, tmp_path, mode, setting, budget, both_protocol
):
    dataset, source = dataset
    config = RunConfig(
        benchmark="robomme",
        mode=mode,
        budget=budget,
        both_protocol=both_protocol,
        coding=CodingConfig(model="us.anthropic.claude-opus-5-5"),
    )
    base = tmp_path / "runs"
    result = prepare_runs(
        base, dataset, config, benchmark_source=source, timetag="20261107T000000Z"
    )
    root = Path(result["runs"][0]["root"])
    assert (
        root
        == base
        / "robomme/BinFill/cc-opus5.5"
        / setting
        / "20261107T000000Z"
    )
    assert validate_run(root)["tasks"] == ["BinFill"]
    doc = (root / "README.md").read_text()
    assert (
        "inputs/benchmark/robomme/" in doc
        and "not copied into the coding workspace" in doc
    )
    assert (
        "/demonstrations" in doc
        and "/feedback" in doc
        and "During policy execution" in doc
    )
    assert (base / "README.md").exists()
    assert build(root)["name"] == str(root.relative_to(base))
    with pytest.raises(FileExistsError):
        prepare_runs(
            base, dataset, config, benchmark_source=source, timetag="20261107T000000Z"
        )
    assert (
        read(root / "run.json")["config"]["coding"]["model"]
        == "us.anthropic.claude-opus-5-5"
    )


@pytest.mark.parametrize(
    "agent,provider,model,expected",
    [
        ("claude_code", "bedrock", "us.anthropic.claude-opus-5-5", "cc-opus5.5"),
        ("claude_code", "anthropic", "claude-opus-5-5", "cc-opus5.5"),
        (
            "claude_code",
            "bedrock",
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "cc-sonnet4.5-20250929-v1%3A0",
        ),
        ("claude_code", "anthropic", "claude-opus-4-20250514", "cc-opus4-20250514"),
        ("codex", "openai", "gpt-6.1-sol", "cx-gpt6.1-sol"),
        ("claude_code", "bedrock", "provider/model", "cc-provider%2Fmodel"),
    ],
)
def test_short_model_folder_preserves_variants(agent, provider, model, expected):
    config = CodingConfig(agent=agent, provider=provider, model=model)
    assert coding_run_name(config) == expected
    assert config.model == model


def test_prepare_cli_and_invalid_tag(dataset, tmp_path, capsys):
    dataset, source = dataset
    config = tmp_path / "config.json"
    write(config, RunConfig(benchmark="robomme", mode="demos").to_dict())
    base = tmp_path / "runs"
    assert (
        main(
            [
                "prepare",
                str(base),
                "--dataset",
                str(dataset),
                "--config",
                str(config),
                "--benchmark-source",
                str(source),
                "--tasks",
                "BinFill",
                "--timetag",
                "trial",
            ]
        )
        == 0
    )
    import json

    result = json.loads(capsys.readouterr().out)
    assert Path(result["runs"][0]["root"]).is_dir()
    with pytest.raises(ValueError, match="Timetag"):
        prepare_runs(
            base, dataset, read(config), benchmark_source=source, timetag="../escape"
        )
