"""Read assigned CaM expert recordings; this mount is absent at policy execution.

``Trajectory(task, episode)`` exposes ``goal``, ``difficulty``, ``T``,
``is_demo``, ``exec_start``, ``action_alignment``, ``step_semantics`` and the
recorded fields. ``traj.fields`` lists per-frame arrays and ``traj.static``
per-episode arrays (e.g. calibration). ``traj.get(name, t)`` reads frame ``t``
(or every frame when ``t`` is None); ``traj.<field>(t)`` is shorthand.
``traj.description`` maps field names to their units and meaning.
"""

import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path("/demonstrations")
SCHEMAS = ("cam-demonstrations/1", "cam-demonstrations/2", "cap-regular-fewshot/2")
EPISODE_KEYS = ("step", "is_demo")


def _open(task):
    source = h5py.File(ROOT / f"record_dataset_{task}.h5", "r")
    if source.attrs["schema"] not in SCHEMAS or source.attrs["task"] != task:
        source.close()
        raise ValueError("Unassigned or unsupported demonstration data")
    return source


def training_episodes(task):
    with _open(task) as source:
        return sorted(int(key.removeprefix("episode_")) for key in source)


class Trajectory:
    def __init__(self, task, episode):
        if episode not in training_episodes(task):
            raise ValueError("Unassigned training episode")
        self._file = _open(task)
        self._group = self._file[f"episode_{episode}"]
        attrs = self._group.attrs
        self.episode = int(episode)
        self.goal = attrs["goal"]
        self.difficulty = attrs.get("difficulty")
        self.action_alignment = attrs.get("action_alignment", self._file.attrs.get("action_alignment"))
        self.step_semantics = attrs.get("step_semantics", self._file.attrs.get("step_semantics"))
        self.tcp_frame = self._file.attrs.get("tcp_frame", "native measured TCP")
        declared = json.loads(self._file.attrs.get("fields", "{}"))
        self.T = len(self._group["step"])
        self.is_demo = self._group["is_demo"][:].astype(bool)
        execution = np.flatnonzero(~self.is_demo)
        self.exec_start = int(execution[0]) if len(execution) else self.T
        if self._file.attrs["schema"] == "cam-demonstrations/2":
            frame_names, static_names = set(declared["frame"]), set(declared["static"])
            if frame_names & static_names or set(self._group) != frame_names | static_names:
                self.close()
                raise ValueError("Recorded fields differ from the declaration")
            self.description = {**declared["frame"], **declared["static"]}
        else:
            # v1 RoboMME calibration is static even when T happens to equal 3.
            static_names = {"front_K", "wrist_K"} & set(self._group)
            frame_names = set(self._group) - static_names
            self.description = declared
        self.static = {
            name: self._group[name][()]
            for name in sorted(static_names)
        }
        self.fields = tuple(sorted(frame_names))
        if "front_K" in self.static:
            self.front_K, self.wrist_K = self.static["front_K"], self.static["wrist_K"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._file.close()

    def get(self, name, t=None):
        if name in self.static:
            return self.static[name]
        if name not in self.fields:
            raise KeyError(f"No recorded field {name!r}; fields: {', '.join(self.fields)}")
        data = self._group[name]
        return data[:] if t is None else data[t]

    def __getattr__(self, name):
        if name.startswith("_") or name not in self.__dict__.get("fields", ()):
            raise AttributeError(name)
        return lambda t=None: self.get(name, t)

    def front_depth(self, t):
        return self._group["front_depth"][t] if "front_depth" in self._group else None

    def wrist_depth(self, t):
        return self._group["wrist_depth"][t] if "wrist_depth" in self._group else None

    def waypoint_action(self, t):
        return (
            self._group["waypoint_action"][t]
            if self._group["waypoint_action_valid"][t]
            else None
        )

    def step(self, t):
        return int(self._group["step"][t])
