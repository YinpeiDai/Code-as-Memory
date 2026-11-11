# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""Report saved provider usage without guessing rates or treating CLI estimates as bills."""

import json
import time

TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

# Standard list prices effective 2026-10-22, USD per million tokens.
# Explicit profiles avoid applying global prices to regional Bedrock inference.
PRICING_SOURCE = "https://assets.claude.com/5633dc848caa431115468f31c7085d695bbfb9cc.pdf"
LIVE_RATES = {
    ("bedrock", "us.anthropic.claude-opus-5-5"): (4.4, 22, 5.5, 8.8, 0.22),
    ("bedrock", "global.anthropic.claude-opus-5-5"): (4, 20, 5, 8, 0.20),
    ("anthropic", "claude-opus-5-5"): (4, 20, 5, 8, 0.20),
}


def estimate_requests(requests, coding):
    """Viewer-only standard-rate estimate; never replace saved reported costs."""
    rates = LIVE_RATES.get((coding.get("provider"), coding.get("model")))
    if rates is None or not requests:
        return None
    total = 0.0
    for request in requests:
        if request.get("model") != "claude-opus-5-5":
            return None  # Mixed or unknown models need their own verified rates.
        usage = request.get("usage", {})
        if not usage:
            return None
        if usage.get("service_tier", "standard") != "standard":
            return None
        creation = usage.get("cache_creation", {})
        long_writes = creation.get("ephemeral_1h_input_tokens", 0)
        short_writes = creation.get("ephemeral_5m_input_tokens", 0)
        short_writes += max(0, usage.get("cache_creation_input_tokens", 0) - short_writes - long_writes)
        tokens = (usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                  short_writes, long_writes, usage.get("cache_read_input_tokens", 0))
        total += sum(count * rate for count, rate in zip(tokens, rates)) / 1_000_000
    return total


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class Usage:
    def __init__(self):
        self.requests = {}
        self.active = {}
        self.results = {}

    def feed(self, record):
        session = record.get("session_id", "unknown")
        channel = (session, record.get("parent_tool_use_id"))
        event = record.get("event", {}) if record.get("type") == "stream_event" else {}
        message = (
            event.get("message", {})
            if event.get("type") == "message_start"
            else record.get("message", {})
            if record.get("type") == "assistant"
            else {}
        )
        if message.get("id"):
            key = (session, message["id"])
            self.active[channel] = key
            row = self.requests.setdefault(
                key,
                {
                    "session_id": session,
                    "message_id": message["id"],
                    "model": message.get("model"),
                    "usage": {},
                    "stream_complete": False,
                    "timestamp": record.get("timestamp"),
                    "request_id": record.get("request_id"),
                    "cost_usd": None,
                },
            )
            row["request_id"] = record.get("request_id") or row["request_id"]
            row["timestamp"] = row["timestamp"] or record.get("timestamp")
            self.merge(row, message.get("usage", {}))
        if (
            event.get("type") in ("message_delta", "message_stop")
            and channel in self.active
        ):
            row = self.requests[self.active[channel]]
            self.merge(row, event.get("usage", {}))
            if event["type"] == "message_stop":
                row["stream_complete"] = True
        if record.get("type") == "result":
            self.results[session] = {
                k: record.get(k)
                for k in (
                    "total_cost_usd",
                    "usage",
                    "modelUsage",
                    "num_turns",
                    "subtype",
                    "is_error",
                )
            }

    @staticmethod
    def merge(row, usage):
        for key in TOKEN_KEYS:
            if key in usage:
                row["usage"][key] = max(row["usage"].get(key, 0), usage[key])
        for key in ("cache_creation", "output_tokens_details"):
            for name, value in usage.get(key, {}).items():
                target = row["usage"].setdefault(key, {})
                target[name] = max(target.get(name, 0), value)

    def summary(self):
        queries = list(self.requests.values())
        results = list(self.results.values())
        sums = {k: sum(r["usage"].get(k, 0) for r in queries) for k in TOKEN_KEYS}
        session_usage = {
            k: sum((r.get("usage") or {}).get(k, 0) for r in results)
            for k in TOKEN_KEYS
        }
        costs = [
            r["total_cost_usd"]
            for r in results
            if isinstance(r.get("total_cost_usd"), (int, float))
        ]
        return {
            "schema": "cam-api-costs/1",
            "updated": time.time(),
            "queries_logged": len(queries),
            "queries_with_complete_stream": sum(r["stream_complete"] for r in queries),
            "sessions_with_final_usage": len(results),
            "reported_cost_usd": sum(costs) if costs else None,
            "cost_basis": "Claude CLI reported estimate; not verified against AWS billing",
            "per_query_dollar_cost_available": False,
            "query_token_sums": sums,
            "final_session_token_totals": session_usage if results else None,
            "query_usage_matches_session_totals": sums == session_usage
            if results
            else None,
            "session_results": results,
            "gpu_cost_included": False,
            "policy_execution_api_cost_usd": 0,
            "note": "Historical assistant records can contain partial output-token counts. Use final session totals. New runs also retain streaming usage events; dollar totals remain the CLI estimate.",
        }

    def save(self, folder):
        write(folder / "api_cost.json", self.summary())
        path = folder / "api_queries.jsonl"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            "".join(json.dumps(row) + "\n" for row in self.requests.values())
        )
        tmp.replace(path)
