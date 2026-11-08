# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""One observation frame: phase, step and the benchmark's declared sensors.

Each declared sensor is a read-only NumPy attribute, e.g. ``frame.front_rgb``;
``frame.sensors`` lists the names. The benchmark guide and assignment describe
the sensors, units and frames. No object truth or simulator state is forwarded.
"""

from dataclasses import dataclass
import re
from cam.runtime.wire import is_packed, unpack

NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class Frame:
    def __init__(self, record):
        self.phase, self.step = record["phase"], record["step"]
        names = []
        for name, value in record.items():
            if name in ("phase", "step"):
                continue
            if not NAME.fullmatch(name) or name in ("label", "sensors") or not is_packed(value):
                raise ValueError(f"Invalid observation field: {name!r}")
            setattr(self, name, unpack(value))
            names.append(name)
        self.sensors = tuple(names)

    @property
    def label(self):
        return ("D" if self.phase == "demo" else "E") + str(self.step)


@dataclass(frozen=True)
class Observations:
    frames: tuple

    @property
    def current(self):
        return self.frames[-1]

    def sampled(self, stride):
        return [
            self.frames[i]
            for i in sorted(
                set(range(0, len(self.frames), stride)) | {len(self.frames) - 1}
            )
        ]
