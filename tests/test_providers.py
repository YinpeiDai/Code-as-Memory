import json
from pathlib import Path
import shutil

import pytest
from cam.config import CodingConfig
from cam.io import read
from cam.providers.clients import invocation
from cam.providers.usage import summarize_sessions


def test_usage_preserves_unknown_cost_and_deduplicates(tmp_path):
    logs = tmp_path / "claude"
    session = logs / "session_000"
    session.mkdir(parents=True)
    message = {
        "type": "assistant",
        "session_id": "a",
        "message": {
            "id": "m",
            "model": "custom-model",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    }
    final = {
        "type": "result",
        "session_id": "a",
        "total_cost_usd": 0.25,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    (session / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in [message, message, final]) + "\n"
    )
    result = summarize_sessions(logs, "claude_code")
    assert (
        result["reported_cost_usd"] == 0.25
        and result["sessions"][0]["requests_logged"] == 1
    )
    assert result["tokens"]["input_tokens"] == 10
    (session / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 12,
                },
            }
        )
        + "\n"
    )
    result = summarize_sessions(logs, "codex")
    assert result["reported_cost_usd"] is None and not result["cost_complete"]
    assert result["tokens"]["cached_input_tokens"] == 80
    assert result["evaluation_api_cost_usd"] == 0


@pytest.mark.isolation
def test_codex_cli_exposes_only_cam_io_tools(run_factory, tmp_path):
    """Real CLI + real MCP server, fake local inference. No paid model request."""
    if (
        not shutil.which("codex")
        or not (Path.home() / ".codex/models_cache.json").exists()
    ):
        pytest.skip(
            "Local Codex CLI and model catalog are required for the offline compatibility probe"
        )
    root = run_factory()
    session = tmp_path / "mock_session"
    session.mkdir()
    call = invocation(
        CodingConfig(agent="codex", provider="openai", model="gpt-6.1-sol"),
        root,
        "BinFill",
        0,
        session,
    )
    from cam.providers.probe import probe_codex

    report = probe_codex(call, session)
    assert report["passed"] and report["api_calls"] == 0
    calls = list((root / "tasks/BinFill/audit").glob("call_*.json"))
    assert len(calls) == 1 and read(calls[0])["tool"] == "usage"
