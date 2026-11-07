# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""RGB, optional aligned depth, calibration and robot proprioception.

Depth-enabled studies provide front_depth/wrist_depth: HxW int16 millimetres
along the camera optical axis, with 0 indicating unknown depth. RGB-only
historical studies do not expose these attributes. No object truth is forwarded.
"""

from dataclasses import dataclass
from cam.runtime.wire import FIELDS, DEPTH_FIELDS, unpack


class Frame:
    def __init__(self, record):
        self.phase, self.step = record["phase"], record["step"]
        for name in FIELDS:
            setattr(self, name, unpack(record[name]))
        if any(name in record for name in DEPTH_FIELDS):
            for name in DEPTH_FIELDS:
                value = unpack(record[name])
                if (
                    value.dtype.name != "int16"
                    or value.shape
                    != getattr(self, name.replace("depth", "rgb")).shape[:2]
                ):
                    raise ValueError("Depth must be aligned HxW int16 millimetres")
                setattr(self, name, value)

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
