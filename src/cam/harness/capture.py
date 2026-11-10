"""Lossless allowed-observation H5 and tiled-camera rollout video."""

# Adapted from the Apache-2.0 CAP broker; see NOTICE.
import json

import numpy as np
import h5py
import imageio.v2 as imageio


def tile(frame, names):
    """Side-by-side RGB tiles; shorter tiles are zero-padded, never resampled."""
    images = []
    for name in names:
        image = np.asarray(frame[name])
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        images.append(image.astype(np.uint8))
    height = max(image.shape[0] for image in images)
    padded = [
        np.pad(image, ((0, height - image.shape[0]), (0, 0), (0, 0)))
        for image in images
    ]
    tiled = np.hstack(padded)
    # H.264 needs even dimensions.
    return np.pad(tiled, ((0, tiled.shape[0] % 2), (0, tiled.shape[1] % 2), (0, 0)))


class Capture:
    def __init__(self, out, sensors=None, video=("front_rgb", "wrist_rgb")):
        self.h5 = h5py.File(out / "observations.h5", "w")
        self.n = 0
        self.h5.attrs.update(
            schema="cam-observations/1",
            phase_encoding="0=demo, 1=exec",
            depth_units="millimetres",
        )
        if sensors:
            self.h5.attrs["sensors"] = json.dumps(sensors)
        self.fields = list(sensors) if sensors else None
        self.video_fields = list(video)
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
                if self.fields is None:
                    self.fields = [k for k in frame if k not in ("phase", "step")]
                for key in self.fields:
                    a = np.asarray(frame[key])
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
            self.video.append_data(tile(frame, self.video_fields))
            self.n += 1
        self.h5.flush()

    def close(self):
        self.video.close()
        self.h5.close()


class ResultsCapture:
    """Count observations without writing images, videos or arrays."""

    def __init__(self):
        self.n = 0

    def append(self, frames):
        self.n += len(frames)

    def close(self):
        pass
