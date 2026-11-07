# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""Public action API backed by an observation-only RPC connection."""

import numpy as np
from .observations import Frame
from cam.runtime.wire import send, receive


class EpisodeEnd(BaseException):
    def __init__(self, status):
        self.status = status


class MotionError(Exception):
    pass


class Robot:
    """now(), frames(), step(x,y,z,roll,pitch,yaw,gripper); one step per call.

    XYZ are world metres, RPY radians; gripper -1 closes, +1 opens. Gripper
    is ignored for stick tasks. Use gradual waypoints and observed TCP feedback.
    """

    def __init__(self, current, reader, writer, stick=False, max_steps=1500):
        self._current, self._reader, self._writer = current, reader, writer
        self._produced = []
        self._count = 0
        self._stick = stick
        self._max_steps = max_steps

    def now(self):
        return self._produced[-1] if self._produced else self._current

    def frames(self):
        return list(self._produced)

    def step(self, x, y, z, roll, pitch, yaw, gripper):
        self._count += 1
        if self._count > self._max_steps:
            raise MotionError("Action-tool step limit exceeded")
        action = [
            float(v)
            for v in (
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
                0 if self._stick else np.clip(gripper, -1, 1),
            )
        ]
        if not np.isfinite(action).all():
            raise MotionError("Action must contain finite numbers")
        send(self._writer, {"kind": "step", "action": action})
        result = receive(self._reader)
        self._produced.extend(Frame(f) for f in result.get("frames", []))
        if result["status"] in ("success", "fail", "timeout", "error"):
            raise EpisodeEnd(result["status"])
        if result.get("motion_error"):
            raise MotionError(result["motion_error"])
        return self.now()
