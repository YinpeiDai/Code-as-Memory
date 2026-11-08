"""Pre-inference MCP checks and offline coding-client compatibility probes."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time
from cam.io import read, write


def probe_cam(config_path, session, timeout=120):
    """Exercise the exact CaM stdio server before paying for Claude inference.

    Only initialization, tool discovery and the read-only usage tool are called;
    no policy, demonstration or simulator is involved.
    """
    from cam.harness.isolation import terminate
    from cam.providers.clients import TOOL_NAMES

    session = Path(session)
    session.mkdir()
    server = read(config_path)["mcpServers"]["cam"]
    env = os.environ.copy()
    env.update(server.get("env", {}))
    started = time.monotonic()
    report = {
        "passed": False,
        "api_calls": 0,
        "simulator_resets": 0,
        "training_attempts": 0,
    }
    proc = None
    try:
        with (session / "stderr.txt").open("wb") as errors:
            proc = subprocess.Popen(
                [server["command"], *server.get("args", [])],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                start_new_session=True,
            )
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)

                def request(ident, method, params):
                    message = {
                        "jsonrpc": "2.0", "id": ident,
                        "method": method, "params": params,
                    }
                    proc.stdin.write((json.dumps(message) + "\n").encode())
                    proc.stdin.flush()
                    remaining = max(0, timeout - (time.monotonic() - started))
                    if not selector.select(remaining):
                        raise TimeoutError("CaM MCP probe timed out during " + method)
                    line = proc.stdout.readline()
                    if not line:
                        raise RuntimeError(
                            "CaM MCP server exited during " + method
                            + "; inspect " + str(session / "stderr.txt")
                        )
                    reply = json.loads(line)
                    if (
                        reply.get("id") != ident
                        or "error" in reply or "result" not in reply
                    ):
                        raise RuntimeError(
                            "Invalid CaM MCP response during " + method + ": " + repr(reply)
                        )
                    return reply["result"]

                report["initialize"] = request(1, "initialize", {
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "cam-host-startup-probe", "version": "1"},
                })
                proc.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
                proc.stdin.flush()
                tools = request(2, "tools/list", {})["tools"]
                report["tools"] = [tool["name"] for tool in tools]
                if sorted(report["tools"]) != sorted(TOOL_NAMES):
                    raise RuntimeError(
                        "CaM MCP probe did not expose exactly the six required tools"
                    )
                usage = request(3, "tools/call", {"name": "usage", "arguments": {}})
                if usage.get("isError") or not usage.get("content"):
                    raise RuntimeError("CaM MCP usage probe failed: " + repr(usage))
                report["usage"] = usage
                proc.stdin.close()
                if proc.wait(timeout=5):
                    raise RuntimeError("CaM MCP server failed after usage probe")
                report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        if proc is not None:
            terminate(proc)
        report["seconds"] = time.monotonic() - started
        write(session / "probe.json", report)
    return report


def probe_codex(call, session):
    """No paid inference. Reject undeclared I/O tools before a real coding session."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"models":[],"data":[]}')

        def do_POST(self):
            import gzip

            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            request = json.loads(body)
            requests.append(request)
            function = {
                "type": "function_call",
                "id": "fc_mock",
                "call_id": "call_mock",
                "namespace": "mcp__cam",
                "name": "usage",
                "arguments": "{}",
                "status": "completed",
            }
            message = {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Offline probe completed.",
                        "annotations": [],
                    }
                ],
            }
            item = function if len(requests) == 1 else message
            response = {
                "id": "resp_mock_" + str(len(requests)),
                "object": "response",
                "status": "completed",
                "output": [item],
                "model": request["model"],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            }
            events = [
                {
                    "type": "response.created",
                    "response": {**response, "status": "in_progress", "output": []},
                },
                {"type": "response.output_item.added", "output_index": 0, "item": item},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": response},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    overrides = {
        "model_provider": "cam_mock",
        "model_providers.cam_mock.name": "Offline test",
        "model_providers.cam_mock.base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "model_providers.cam_mock.wire_api": "responses",
        "model_providers.cam_mock.requires_openai_auth": False,
        "model_providers.cam_mock.supports_websockets": False,
        "features.enable_request_compression": False,
    }
    argv = call.argv[:-1]
    for key, value in overrides.items():
        argv += ["--config", key + "=" + json.dumps(value)]
    argv += ["-"]
    try:
        result = subprocess.run(
            argv,
            input="Run the offline tool probe.",
            text=True,
            capture_output=True,
            cwd=call.cwd,
            env=call.environment,
            timeout=45,
        )
    finally:
        server.shutdown()
        server.server_close()
    if result.returncode:
        raise RuntimeError("Codex compatibility probe failed: " + result.stderr[-4000:])
    if len(requests) != 2:
        raise RuntimeError("Codex did not complete the local MCP probe")
    (session / "observed_requests.json").write_text(json.dumps(requests, indent=2))
    (session / "observed_events.jsonl").write_text(result.stdout)
    tools = requests[0].get("tools", [])
    (session / "observed_tools.json").write_text(json.dumps(tools, indent=2))
    names = [
        name
        for t in tools
        for name in (
            [t["name"] + "__" + x["name"] for x in t["tools"]]
            if t["type"] == "namespace"
            else [t.get("name", t.get("type"))]
        )
    ]
    allowed = {
        "mcp__cam__" + n
        for n in ("command", "read_image", "usage", "snapshot", "validate", "rollout")
    } | {
        "update_plan",
        "request_user_input",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
    }
    if set(names) - allowed:
        raise RuntimeError(
            "Codex exposed unexpected tools: " + ", ".join(set(names) - allowed)
        )
    if "mcp__cam__usage" not in names:
        raise RuntimeError("CaM MCP tools are missing")
    events = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    if not any(e.get("item", {}).get("type") == "mcp_tool_call" for e in events):
        raise RuntimeError("Codex failed the real MCP usage call")
    report = {
        "passed": True,
        "api_calls": 0,
        "simulator_resets": 0,
        "tools": names,
        "local_mock_requests": len(requests),
    }
    write(session / "probe.json", report)
    return report
