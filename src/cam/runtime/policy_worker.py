# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""Sandboxed policy runner; initial demo frames arrive as bounded messages."""

import sys
import traceback

from cam.core.workspace import Workspace
from cam.core.code_policy import validate, decide
from cam.runtime.observations import Frame, Observations
from cam.runtime.robot import Robot, EpisodeEnd
from cam.core.tools import ACT, ToolTimeout, outcome
from cam.runtime.wire import send, receive


def initial_observations(reader):
    initial = receive(reader, 2 * 1024**2)
    if initial.get("kind") != "initial" or not 1 <= initial["frame_count"] <= 100000:
        raise ValueError("Invalid initial observation header")
    frames = []
    for index in range(initial["frame_count"]):
        message = receive(reader, 8 * 1024**2)
        if message.get("kind") != "frame" or message.get("index") != index:
            raise ValueError("Missing or out-of-order initial frame")
        frames.append(Frame(message["frame"]))
    return (
        initial["goal"],
        frames,
        initial.get("limits", {}),
        initial.get("action_spec", {}),
    )


def main():
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    try:
        ws = Workspace.load("/workspace")
        toolkit = ws.toolkit()
        validate(ws, toolkit)
        if "--validate" in sys.argv:
            send(writer, {"valid": True, "tools": toolkit.names(ACT)})
            return
        send(writer, {"kind": "ready"})
        goal, frames, limits, action_spec = initial_observations(reader)
        memory = {f.name: None for f in ws.fields}
        last = None
        for _ in range(limits.get("max_decisions", 40)):
            obs = Observations(tuple(frames))
            decision = decide(ws, toolkit, goal, obs, memory, last)
            send(
                writer,
                {
                    "kind": "decision",
                    "row": {
                        "step": obs.current.step,
                        "frames": [obs.frames[0].label, obs.current.label],
                        **decision.record(),
                    },
                },
            )
            memory = decision.memory_written
            robot = Robot(
                obs.current,
                reader,
                writer,
                action_spec=action_spec,
                max_steps=limits.get("max_action_steps", 1500),
            )
            result = outcome(
                toolkit.call, decision.act["tool"], robot, decision.act["args"], ACT
            )
            send(writer, {"kind": "outcome", "output": result})
            last = f"{decision.act['tool']} -> {result}"
            frames = robot.frames() or [robot.now()]
        send(writer, {"kind": "done", "status": "decision_limit"})
    except EpisodeEnd as exc:
        send(writer, {"kind": "done", "status": exc.status})
    except ToolTimeout as exc:
        send(writer, {"kind": "done", "status": "policy_timeout", "error": str(exc)})
        if "--validate" in sys.argv:
            raise SystemExit(1)
    except BaseException:
        send(
            writer,
            {
                "kind": "done",
                "status": "policy_error",
                "error": traceback.format_exc()[-8000:],
            },
        )
        if "--validate" in sys.argv:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
