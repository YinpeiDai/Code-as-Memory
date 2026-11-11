import json
from pathlib import Path
import shutil

import pytest
from cam.config import CodingConfig
from cam.io import read
from cam.providers.clients import TOOL_NAMES, invocation, unexpected_tool
from cam.providers.usage import summarize_sessions


@pytest.mark.parametrize("status", ["pending", "failed", "connected"])
def test_claude_rejects_empty_tools_even_when_server_claims_connected(status):
    assert "missing" in unexpected_tool("claude_code", {
        "type": "system", "subtype": "init", "tools": [],
        "mcp_servers": [{"name": "cam", "status": status}],
    })


@pytest.mark.parametrize("status", ["pending", "failed", "connected"])
def test_claude_requires_six_tools_and_connected_server(status):
    event = {
        "type": "system", "subtype": "init",
        "tools": ["mcp__cam__" + name for name in TOOL_NAMES],
        "mcp_servers": [{"name": "cam", "status": status}],
    }
    assert (unexpected_tool("claude_code", event) is None) == (status == "connected")
    event["tools"].append("mcp__cam__shell")
    assert "Unexpected" in unexpected_tool("claude_code", event)


def test_real_cam_probe_calls_usage_without_training(run_factory, tmp_path):
    from cam.providers.probe import probe_cam

    root = run_factory()
    session = tmp_path / "claude_session"
    session.mkdir()
    call = invocation(CodingConfig(agent="claude_code"), root, "BinFill", 0, session)
    assert call.environment["MCP_CONNECTION_NONBLOCKING"] == "0"
    report = probe_cam(session / "mcp.json", session / "mcp_probe")
    assert report["passed"] and report["api_calls"] == report["training_attempts"] == 0
    calls = list((root / "tasks/BinFill/audit").glob("call_*.json"))
    assert len(calls) == 1 and read(calls[0])["tool"] == "usage"


def test_failed_cam_probe_prevents_paid_client(run_factory, monkeypatch):
    from cam.harness.jobs import learn_task
    import cam.providers.probe as probe

    root = run_factory(coding=CodingConfig(agent="claude_code"))

    def fail(*args):
        raise RuntimeError("MCP startup failed")

    monkeypatch.setattr(probe, "probe_cam", fail)
    monkeypatch.setattr("cam.harness.jobs.subprocess.check_output", lambda *a, **kw: "test")

    def no_launch(*args, **kwargs):
        pytest.fail("Paid client must not launch after an MCP probe failure")

    monkeypatch.setattr("cam.harness.jobs.subprocess.Popen", no_launch)
    result = learn_task(root, "BinFill", 0)
    assert result["status"] == "failed" and result["error"] == "MCP startup failed"


@pytest.mark.isolation
@pytest.mark.parametrize("scratch_dependency", [True, False])
def test_final_validation_uses_only_publishable_program(run_factory, monkeypatch, scratch_dependency):
    import os
    import sys
    import numpy as np
    from cam.harness import jobs
    from cam.core.budget import Ledger
    from cam.providers.clients import Invocation

    root = run_factory()
    task = root / "tasks/BinFill"
    policy = task / "workspace/tools/policy.py"
    policy.write_text("# Finished policy\n" + policy.read_text())
    if scratch_dependency:
        policy.write_text("DATA = np.load('/workspace/scratch/data.npy')\n" + policy.read_text())
        np.save(task / "workspace/scratch/data.npy", np.zeros(3))
    monkeypatch.setattr(jobs, "invocation", lambda *args: Invocation(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], os.environ.copy(), root))
    monkeypatch.setattr("cam.providers.probe.probe_cam", lambda *args: None)
    result = jobs.learn_task(root, "BinFill", 0)
    assert result["status"] == ("failed" if scratch_dependency else "finished")
    assert (task / "policy").exists() is not scratch_dependency
    assert Ledger(task / "state/ledger.sqlite").config().get("sealed", False) is not scratch_dependency
    if scratch_dependency:
        assert "Final isolated validation failed" in result["error"]


def test_killed_learning_supervisor_stops_client_and_releases_service_lock(run_factory, tmp_path):
    import fcntl
    import os
    import signal
    import subprocess
    import sys
    import time

    root = run_factory()
    task = root / "tasks/BinFill"
    marker = tmp_path / "client.json"
    client = f'''import json, os, subprocess, sys, time
from pathlib import Path
from cam.harness.service import Service
sys.stdin.read()
service = Service(Path({str(root)!r}), "BinFill", 0)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path({str(marker)!r}).write_text(json.dumps([os.getpid(), child.pid]))
time.sleep(60)
'''
    supervisor_code = f'''import os, sys
from pathlib import Path
from cam.harness import jobs
from cam.providers.clients import Invocation
import cam.providers.probe as probe
jobs.invocation = lambda *args: Invocation([sys.executable, "-c", {client!r}], os.environ.copy(), Path({str(root)!r}))
probe.probe_cam = lambda *args: None
jobs.learn_task(Path({str(root)!r}), "BinFill", 0)
'''
    supervisor = subprocess.Popen([sys.executable, "-c", supervisor_code], start_new_session=True)
    client_pids = []
    def alive(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "Fake coding client did not start"
        client_pids = read(marker)
        supervisor.kill()
        supervisor.wait(timeout=5)
        deadline = time.monotonic() + 5
        while any(alive(pid) for pid in client_pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(alive(pid) for pid in client_pids), "Coding processes outlived SIGKILLed supervisor"
        for path in (task / "state/learning.lock", task / "audit/service.lock"):
            with path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait()
        process = task / "logs/session_000/process.json"
        if process.exists():
            try:
                os.killpg(read(process)["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


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
