"""Lossless allowed-observation H5 and paired-camera rollout video."""

# Adapted from the Apache-2.0 CAP broker; see NOTICE.
import numpy as np
import h5py
import imageio.v2 as imageio
from cam.runtime.wire import observation_fields, DEPTH_FIELDS


class Capture:
    def __init__(self, out):
        self.h5 = h5py.File(out / "observations.h5", "w")
        self.n = 0
        self.h5.attrs.update(
            schema="cam-observations/1",
            phase_encoding="0=demo, 1=exec",
            depth_units="millimetres",
        )
        self.fields = None
        self.video = imageio.get_writer(
            out / "episode.mp4",
            fps=20,
            codec="libx264",
            quality=7,
            ffmpeg_params=["-threads", "1"],
            ffmpeg_log_level="error",
        )

    def append(self, frames):
        for frame in frames:
            if not self.n:
                self.fields = observation_fields(
                    any(key in frame for key in DEPTH_FIELDS)
                )
                for key in self.fields:
                    a = frame[key]
                    self.h5.create_dataset(
                        key,
                        shape=(0, *a.shape),
                        maxshape=(None, *a.shape),
                        chunks=(1, *a.shape),
                        dtype=a.dtype,
                        compression="gzip",
                        compression_opts=3,
                    )
                for key in ("step", "phase"):
                    self.h5.create_dataset(
                        key, shape=(0,), maxshape=(None,), dtype="i8"
                    )
            if set(frame) != (set(self.fields) | {"phase", "step"}):
                raise ValueError("Observation fields changed within a rollout")
            for key in self.fields:
                d = self.h5[key]
                d.resize(self.n + 1, axis=0)
                d[self.n] = frame[key]
            for key, value in [
                ("step", frame["step"]),
                ("phase", int(frame["phase"] == "exec")),
            ]:
                d = self.h5[key]
                d.resize(self.n + 1, axis=0)
                d[self.n] = value
            self.video.append_data(np.hstack([frame["front_rgb"], frame["wrist_rgb"]]))
            self.n += 1
        self.h5.flush()

    def close(self):
        self.video.close()
        self.h5.close()
