"""Explicit CLI tool configuration, independent of task and benchmark prompts."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

from cam.io import read, write

TOOL_NAMES = ["command", "read_image", "usage", "snapshot", "validate", "rollout"]
CODEX_DISABLED = (
    "shell_tool",
    "shell_snapshot",
    "apps",
    "plugins",
    "hooks",
    "memories",
    "multi_agent",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "view_image",
    "image_generation",
    "code_mode_host",
    "code_mode",
    "skill_search",
    "skill_mcp_dependency_install",
    "goals",
    "workspace_dependencies",
    "tool_suggest",
)


@dataclass
class Invocation:
    argv: list
    environment: dict
    cwd: Path


def invocation(config, root, task, gpu, session):
    """The CLI authenticates on the host; all model-accessible I/O is CaM MCP."""
    root, session = Path(root).resolve(), Path(session).resolve()
    cwd = session / "client"
    cwd.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    server = {
        "command": sys.executable,
        "args": [
            str(root / "runner.py"),
            "_service",
            str(root),
            task,
            "--gpu",
            str(gpu),
        ],
        "env": {
            "PYTHONPATH": str(root / "inputs/harness"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    if config.agent == "claude_code":
        write(session / "mcp.json", {"mcpServers": {"cam": server}})
        env["MCP_TOOL_TIMEOUT"] = str(max(2200000, config.timeout * 1000))
        if config.provider == "bedrock":
            env.update(
                CLAUDE_CODE_USE_BEDROCK="1",
                AWS_PROFILE=config.aws_profile,
                AWS_REGION=config.aws_region,
                AWS_DEFAULT_REGION=config.aws_region,
            )
            for name in (
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL",
            ):
                env.pop(name, None)
        else:
            env.pop("CLAUDE_CODE_USE_BEDROCK", None)
        argv = [
            "claude",
            "-p",
            "--model",
            config.model,
            "--effort",
            config.reasoning_effort,
            "--bare",
            "--disable-slash-commands",
            "--setting-sources",
            "",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            str(session / "mcp.json"),
            "--allowedTools",
            "mcp__cam__*",
            "--permission-mode",
            "auto",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
        ]
    else:
        catalog_path = (
            Path(config.model_catalog)
            if config.model_catalog
            else Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
            / "models_cache.json"
        )
        catalog = read(catalog_path)
        candidates = [m for m in catalog["models"] if m["slug"] == config.model]
        if len(candidates) != 1:
            raise ValueError(
                "Codex model must appear once in its catalog; set coding.model_catalog explicitly"
            )
        model = dict(candidates[0])
        # Disable optional model-selected tool surfaces as well as global features.
        model.update(
            apply_patch_tool_type=None,
            experimental_supported_tools=[],
            node_repl_disabled=True,
            include_skills_usage_instructions=False,
            include_plugin_usage_instructions=False,
            include_apps_usage_instructions=False,
            supports_search_tool=False,
            use_responses_lite=False,
            multi_agent_version=None,
            multi_agent_reasoning_effort=None,
        )
        model.pop("model_messages", None)
        model.pop("tool_mode", None)
        model["base_instructions"] = (
            "Complete the assigned Code-as-Memory task using only the provided CaM MCP tools."
        )
        write(session / "model_catalog.json", {"models": [model]})
        argv = [
            "codex",
            "exec",
            "--model",
            config.model,
            "--sandbox",
            "read-only",
            "--json",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--cd",
            str(cwd),
            "--output-last-message",
            str(session / "final.txt"),
        ]
        values = {
            "approval_policy": "never",
            "web_search": "disabled",
            "model_catalog_json": str(session / "model_catalog.json"),
            "project_doc_max_bytes": 0,
            "project_doc_fallback_filenames": [],
            "model_reasoning_effort": config.reasoning_effort,
            "mcp_servers.cam.command": server["command"],
            "mcp_servers.cam.args": server["args"],
            "mcp_servers.cam.env.PYTHONPATH": server["env"]["PYTHONPATH"],
            "mcp_servers.cam.env.PYTHONDONTWRITEBYTECODE": "1",
            "mcp_servers.cam.required": True,
            "mcp_servers.cam.enabled_tools": TOOL_NAMES,
            "mcp_servers.cam.tool_timeout_sec": 2200,
            "mcp_servers.cam.startup_timeout_sec": 60,
            "mcp_servers.cam.default_tools_approval_mode": "approve",
            "mcp_servers.cam.supports_parallel_tool_calls": False,
            "features.skip_host_skill_discovery": True,
        }
        for key, value in values.items():
            argv += ["--config", key + "=" + json.dumps(value)]
        for feature in CODEX_DISABLED:
            argv += ["--disable", feature]
        argv += ["-"]
    return Invocation(argv, env, cwd)


def unexpected_tool(agent, event):
    if (
        agent == "claude_code"
        and event.get("type") == "system"
        and event.get("subtype") == "init"
    ):
        extra = [
            name for name in event.get("tools", []) if not name.startswith("mcp__cam__")
        ]
        if extra:
            return "Unexpected coding tools: " + ", ".join(extra)
    if agent == "codex":
        item = event.get("item", {})
        if item.get("type") in ("command_execution", "web_search", "file_change"):
            return "Unexpected built-in tool invocation: " + item["type"]
        if item.get("type") == "mcp_tool_call" and (
            item.get("server") != "cam" or item.get("tool") not in TOOL_NAMES
        ):
            return "Unexpected MCP tool invocation"
    return None
