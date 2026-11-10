"""Export only public robot geometry/calibration, without importing RMBench.

Host maintenance utility, never mounted in the learner runtime. Requires PyYAML.
The resulting JSON has no task, scene, camera, scoring or episode information.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def export_model(assets, source):
    assets, source = Path(assets), Path(source)
    config = yaml.safe_load((assets / "config.yml").read_text())
    urdf = assets / config["urdf_path"]
    joints = {j.attrib["name"]: j for j in ET.parse(urdf).getroot().findall("joint")}
    model = {
        "schema": "cam-rmbench-robot-model/1",
        "robot": "aloha-agilex", "setting": "demo_clean",
        "world_root_pose_wxyz": config["robot_pose"][0],
        "gripper_bias_m": config["gripper_bias"],
        "action_offset_m": 0.12,
        "delta_matrix": config["delta_matrix"],
        "limit_reference": "Released RMBench URDF; not manufacturer safety limits",
        "arms": {}, "provenance": {},
    }
    paths = {"config.yml": assets / "config.yml", config["urdf_path"].removeprefix("./"): urdf}
    for index, arm in enumerate(("left", "right")):
        cfg = yaml.safe_load((assets / f"curobo_{arm}.yml").read_text())
        paths[f"curobo_{arm}_tmp.yml"] = assets / f"curobo_{arm}_tmp.yml"
        chain = []
        parent = cfg["robot_cfg"]["kinematics"]["base_link"]
        for name in config["arm_joints_name"][index]:
            j = joints[name]
            assert j.attrib["type"] == "revolute" and j.find("parent").attrib["link"] == parent
            parent = j.find("child").attrib["link"]
            origin = j.find("origin")
            chain.append({
                "name": name, "type": "revolute", "child_link": parent,
                "xyz": list(map(float, origin.attrib["xyz"].split())),
                "rpy": list(map(float, origin.attrib["rpy"].split())),
                "axis": list(map(float, j.find("axis").attrib["xyz"].split())),
                "limits": {k: float(v) for k, v in j.find("limit").attrib.items()},
            })
        kin = cfg["robot_cfg"]["kinematics"]
        assert len(chain) == 6 and parent == kin["ee_link"]
        # These are the released planner.py aloha-agilex calibration rotations.
        model["arms"][arm] = {
            "base_link": kin["base_link"], "ee_link": kin["ee_link"],
            "frame_bias_m": cfg["planner"]["frame_bias"],
            "planner_yaw_rad": (-0.02, -0.01)[index], "chain": chain,
        }
    for name in ("envs/robot/robot.py", "envs/robot/planner.py"):
        paths[name] = source / name
    model["provenance"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    }
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export_model(args.assets, args.source), indent=2) + "\n")
