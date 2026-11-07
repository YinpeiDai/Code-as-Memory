# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""Execute a generated Python policy, with no inference-time language model.

tools/*.py must define perceive_decide(obs, goal, memory, last). It returns
{"memory": {...}, "act": {"tool": "act_name", "args": {...}}}.
The same observation, act-tool and episode contracts as the original harness apply.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .tools import ACT, PERCEIVE, ToolError, to_json


@dataclass
class CodeDecision:
    memory_in: dict
    memory_written: dict
    act: dict

    def record(self):
        return {
            "memory_in": self.memory_in,
            "memory_written": self.memory_written,
            "act": self.act,
            "cost": 0.0,
            "runtime": "python",
        }


def validate(ws, toolkit):
    toolkit.validate(
        "perceive_decide", {"goal": "", "memory": {}, "last": None}, PERCEIVE
    )
    if not toolkit.names(ACT):
        raise ToolError("A Python policy needs at least one act_* tool")


def decide(ws, toolkit, goal, obs, memory, last):
    before = copy.deepcopy(memory)
    value = toolkit.call(
        "perceive_decide",
        obs,
        {"goal": goal, "memory": copy.deepcopy(memory), "last": last},
        PERCEIVE,
    )
    if not isinstance(value, dict) or set(value) != {"memory", "act"}:
        raise ToolError("perceive_decide must return exactly memory and act")
    fields = {f.name for f in ws.fields}
    if not isinstance(value["memory"], dict) or set(value["memory"]) != fields:
        raise ToolError("perceive_decide must return every declared memory field")
    act = value["act"]
    if not isinstance(act, dict) or set(act) != {"tool", "args"}:
        raise ToolError("act must contain tool and args")
    toolkit.validate(act["tool"], act["args"], ACT)
    return CodeDecision(before, to_json(value["memory"], digits=12), act)
