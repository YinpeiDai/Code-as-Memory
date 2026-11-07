"""Read assigned CaM expert recordings; this mount is absent at policy execution."""

from pathlib import Path
import h5py
import numpy as np

ROOT = Path("/demonstrations")


def training_episodes(task):
    with h5py.File(ROOT / f"record_dataset_{task}.h5") as source:
        if (
            source.attrs["schema"] != "cam-demonstrations/1"
            or source.attrs["task"] != task
        ):
            raise ValueError("Unassigned or unsupported demonstration data")
        return sorted(int(key.removeprefix("episode_")) for key in source)


class Trajectory:
    def __init__(self, task, episode):
        if episode not in training_episodes(task):
            raise ValueError("Unassigned training episode")
        self._file = h5py.File(ROOT / f"record_dataset_{task}.h5")
        self._group = self._file[f"episode_{episode}"]
        self.goal = self._group.attrs["goal"]
        self.difficulty = self._group.attrs["difficulty"]
        self.action_alignment = self._group.attrs["action_alignment"]
        self.step_semantics = self._group.attrs["step_semantics"]
        self.tcp_frame = "native measured TCP"
        self.T = len(self._group["step"])
        self.is_demo = self._group["is_demo"][:].astype(bool)
        execution = np.flatnonzero(~self.is_demo)
        self.exec_start = int(execution[0]) if len(execution) else self.T
        self.front_K, self.wrist_K = (
            self._group[camera + "_K"][:] for camera in ("front", "wrist")
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._file.close()

    def front_rgb(self, t):
        return self._group["front_rgb"][t]

    def wrist_rgb(self, t):
        return self._group["wrist_rgb"][t]

    def front_depth(self, t):
        return self._group["front_depth"][t] if "front_depth" in self._group else None

    def wrist_depth(self, t):
        return self._group["wrist_depth"][t] if "wrist_depth" in self._group else None

    def tcp(self, t):
        return self._group["tcp"][t]

    def joints(self, t):
        return self._group["joints"][t]

    def fingers(self, t):
        return self._group["fingers"][t]

    def front_extrinsic(self, t):
        return self._group["front_extrinsic"][t]

    def wrist_extrinsic(self, t):
        return self._group["wrist_extrinsic"][t]

    def joint_action(self, t):
        return self._group["joint_action"][t]

    def eef_action(self, t):
        return self._group["eef_action"][t]

    def waypoint_action(self, t):
        return (
            self._group["waypoint_action"][t]
            if self._group["waypoint_action_valid"][t]
            else None
        )

    def step(self, t):
        return int(self._group["step"][t])
