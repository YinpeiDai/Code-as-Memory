import h5py
import numpy as np
import pytest

from cam.config import RunConfig
from cam.data.extract import extract
from cam.harness.prepare import prepare
from cam.io import write


@pytest.fixture
def raw(tmp_path):
    source, data = tmp_path / "benchmark/robomme", tmp_path / "raw"
    data.mkdir()
    (source / "__init__.py").parent.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    train = [
        {
            "task": "BinFill",
            "episode": i,
            "seed": 1000 + i,
            "difficulty": ("easy", "medium", "hard")[i // 3],
        }
        for i in range(9)
    ]
    test = [
        {
            "task": "BinFill",
            "episode": i,
            "seed": 2000 + i,
            "difficulty": ("easy", "medium", "hard")[i],
        }
        for i in range(3)
    ]
    for split, records in [("train", train), ("test", test)]:
        write(
            source / "env_metadata" / split / "record_dataset_BinFill_metadata.json",
            {"records": records},
        )
    with h5py.File(data / "record_dataset_BinFill.h5", "w") as h5:
        for r in train:
            group = h5.create_group(f"episode_{r['episode']}")
            for k, v in {
                "seed": r["seed"],
                "difficulty": r["difficulty"],
                "task_goal": "Move the red cube",
                "front_camera_intrinsic": np.eye(3),
                "wrist_camera_intrinsic": np.eye(3),
            }.items():
                group.create_dataset("setup/" + k, data=v)
            for t in range(3):
                g = group.create_group(f"timestep_{t}")
                for c in ("front", "wrist"):
                    g.create_dataset(
                        "obs/" + c + "_rgb",
                        data=np.full((16, 16, 3), 25 + t, dtype=np.uint8),
                    )
                    g.create_dataset(
                        "obs/" + c + "_depth",
                        data=np.full((16, 16, 1), 1234, dtype=np.int16),
                    )
                    g.create_dataset(
                        "obs/" + c + "_camera_extrinsic",
                        data=np.c_[np.eye(3), np.zeros(3)],
                    )
                for k, v in {
                    "eef_state": np.array([0.4, 0.2, 0.3, 0, 1.5, 0]),
                    "joint_state": np.zeros(7),
                    "gripper_state": np.zeros(2),
                    "oracle_segmentation": np.ones((16, 16)),
                }.items():
                    g.create_dataset("obs/" + k, data=v)
                for k, v in {
                    "joint_action": np.zeros(8),
                    "eef_action": np.zeros(7),
                    "waypoint_action": np.full(7, np.nan)
                    if t == 0
                    else np.arange(7, dtype=float) / 7,
                }.items():
                    g.create_dataset("action/" + k, data=v)
                g.create_dataset("info/is_video_demo", data=t == 0)
                g.create_dataset("info/is_completed", data=t == 2)
                g.create_dataset("info/hidden_subgoal", data="privileged")
    return data, source


@pytest.fixture
def dataset(raw, tmp_path):
    data, source = raw
    target = tmp_path / "extracted"
    extract(data, target, count=3, benchmark_source=source, workers=1, videos=False)
    return target, source


@pytest.fixture
def run_factory(dataset, tmp_path):
    target, source = dataset

    def make(mode="demos", budget=0, **kwargs):
        root = tmp_path / (mode + "_run")
        prepare(
            root,
            target,
            RunConfig(benchmark="robomme", mode=mode, budget=budget, **kwargs),
            benchmark_source=source,
        )
        return root

    return make
