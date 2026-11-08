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
    """now(), frames(), step(*action); one native benchmark step per step() call.

    ``robot.action_names`` lists the action components in order. Pass them
    positionally, or all by name, e.g. ``robot.step(x=..., y=..., ...)``. Bounds
    in ``robot.action_low``/``robot.action_high`` are applied; components in
    ``robot.action_ignored`` are sent as zero. The benchmark guide defines units
    and frames. Use gradual targets and observed feedback.

    ``robot.dry_run(..., mode="absolute" or "delta")`` is a MIKASA-only preview
    of the native controller's NEXT arm joint target. Six values specify world
    XYZ/RPY (absolute) or the normalized arm arguments of step (delta). The
    returned ``command`` is the six-value command checked; append a gripper
    argument to execute it. No state/time advance, collision or dynamics check.
    The benchmark's public dry-run reference documents the result and limits.
    The older call without mode retains the v2 absolute IK query where available.
    """

    def __init__(self, current, reader, writer, action_spec=None, max_steps=1500):
        self._current, self._reader, self._writer = current, reader, writer
        self._produced = []
        self._count = 0
        self._dry_runs = 0
        spec = action_spec or {}
        names = spec.get("names") or ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        self._names = tuple(names)
        self._low = tuple(spec.get("low") or [None] * len(names))
        self._high = tuple(spec.get("high") or [None] * len(names))
        self._ignored = tuple(spec.get("ignored") or ())
        self._motion_preview = spec.get("motion_preview")
        self._max_steps = max_steps

    @property
    def action_names(self):
        return self._names

    @property
    def action_low(self):
        return self._low

    @property
    def action_high(self):
        return self._high

    @property
    def action_ignored(self):
        return self._ignored

    def now(self):
        return self._produced[-1] if self._produced else self._current

    def frames(self):
        return list(self._produced)

    def _command(self, values, named):
        if named:
            if values or set(named) != set(self._names):
                raise MotionError(
                    "Pass every action component by name, or all positionally: "
                    + ", ".join(self._names)
                )
            values = [named[name] for name in self._names]
        if len(values) != len(self._names):
            raise MotionError(
                f"step expects {len(self._names)} values: {', '.join(self._names)}"
            )
        try:
            action = [float(v) for v in values]
        except (TypeError, ValueError):
            raise MotionError("Action must contain numbers") from None
        if not np.isfinite(action).all():
            raise MotionError("Action must contain finite numbers")
        for i, name in enumerate(self._names):
            if name in self._ignored:
                action[i] = 0.0
                continue
            if self._low[i] is not None:
                action[i] = max(action[i], self._low[i])
            if self._high[i] is not None:
                action[i] = min(action[i], self._high[i])
        return action

    def dry_run(self, *pose, mode=None, **named):
        if mode is not None and mode not in ("absolute", "delta"):
            raise MotionError("mode must be 'absolute' or 'delta'")
        if mode is not None and not self._motion_preview:
            raise MotionError("Motion dry-run is not available for this benchmark")
        names = (("dx", "dy", "dz", "drx", "dry", "drz") if mode == "delta"
                 else ("x", "y", "z", "roll", "pitch", "yaw"))
        if named:
            if pose or set(named) != set(names):
                raise MotionError("Pass every pose component by name, or all positionally: " + ", ".join(names))
            pose = [named[name] for name in names]
        if len(pose) != 6:
            raise MotionError("dry_run expects 6 values: " + ", ".join(names))
        try:
            pose = [float(v) for v in pose]
        except (TypeError, ValueError):
            raise MotionError("Pose must contain numbers") from None
        if not np.isfinite(pose).all():
            raise MotionError("Pose must contain finite numbers")
        self._dry_runs += 1
        if self._dry_runs > self._max_steps:
            raise MotionError("Action-tool dry_run limit exceeded")
        request = ({"kind": "motion_preview", "mode": mode, "values": pose}
                   if mode is not None else {"kind": "dry_run", "pose": pose})
        send(self._writer, request)
        result = receive(self._reader)
        if result.get("error"):
            raise MotionError(result["error"])
        return result["dry_run"]

    def safety_check(self, robot_state, action=None, **options):
        """MIKASA-only offline check on caller-supplied state; no RPC or step.

        See api_reference/runtime/mikasa_robo/safety_check.md. During offline
        development import safety_check from cam.runtime.mikasa_robo instead.
        """
        if not self._motion_preview:
            raise MotionError("Offline safety_check is only available for MIKASA")
        from cam.runtime.mikasa_robo import safety_check

        return safety_check(robot_state, action, **options)

    def step(self, *values, **named):
        action = self._command(values, named)
        self._count += 1
        if self._count > self._max_steps:
            raise MotionError("Action-tool step limit exceeded")
        send(self._writer, {"kind": "step", "action": action})
        result = receive(self._reader)
        self._produced.extend(Frame(f) for f in result.get("frames", []))
        if result["status"] in ("success", "fail", "timeout", "error"):
            raise EpisodeEnd(result["status"])
        if result.get("motion_error"):
            raise MotionError(result["motion_error"])
        return self.now()
