"""RMBench dataset schema and validation for prepared RGB/RGB-D demonstrations.

Use the published RGB-D demonstrations directly with cam prepare. Historical
collection/conversion sources live outside the package in legacy/rmbench/.
Validation here is opt-in; normal preparation reads the release manifest only.
"""

from pathlib import Path

from cam.data import demonstrations
from cam.data.demonstrations import declaration
from cam.io import read
from cam.interface.rmbench import adapter as rmbench
from cam.interface.rmbench.adapter import ARMS, CAMERAS, HEIGHT, WIDTH

RGBD_PROTOCOL = "rmbench-native-expert-rgbd/1"


def fields():
    frame = {}
    for name, (_, group) in CAMERAS.items():
        frame[name + "_rgb"] = {
            "dtype": "uint8",
            "shape": [HEIGHT, WIDTH, 3],
            "description": f"{name} camera RGB decoded from the released JPEG ({group})",
        }
    for name, (_, group) in CAMERAS.items():
        frame[name + "_K"] = {
            "dtype": "float32",
            "shape": [3, 3],
            "description": f"{name} camera 3x3 OpenCV intrinsics in pixels ({group}/intrinsic_matrix)",
        }
        frame[name + "_extrinsic"] = {
            "dtype": "float32",
            "shape": [3, 4],
            "description": f"{name} camera 3x4 OpenCV world-to-camera extrinsics in metres, "
            f"inv(cam2world_gl @ diag(1,-1,-1,1))[:3] of {group}/extrinsics_matrix",
        }
    for arm in ARMS:
        frame[arm + "_ee_pose"] = {
            "dtype": "float32",
            "shape": [7],
            "description": f"{arm} end-effector pose, world x,y,z metres and quaternion w,x,y,z "
            f"(state/{arm}_ee_poses; the frame commanded by ee actions)",
        }
        frame[arm + "_joints"] = {
            "dtype": "float32",
            "shape": [6],
            "description": f"{arm} arm commanded joint targets in radians (state/{arm}_arm_joint_states)",
        }
        frame[arm + "_gripper"] = {
            "dtype": "float32",
            "shape": [],
            "description": f"{arm} commanded gripper opening, 0 closed to 1 open (state/{arm}_ee_joint_states)",
        }
    return declaration(frame)



def rgbd_fields():
    declared = fields()
    for name in rmbench.CAMERAS:
        declared["frame"][name + "_rgb"]["description"] = name + " RGB rendered during native expert regeneration"
        declared["frame"][name + "_depth"] = {
            "dtype": "int16", "shape": [rmbench.HEIGHT, rmbench.WIDTH],
            "description": name + " aligned optical-axis depth, millimetres; 0 invalid/out of range",
        }
    return declared



def validate_dataset(root, verify_hashes=True):
    """Shared v2 validation plus RMBench identity checks against pinned metadata."""
    if read(Path(root) / "manifest.json").get("depth"):
        return validate_rgbd_dataset(root, verify_hashes)
    manifest = demonstrations.validate_dataset(root, verify_hashes)
    if manifest["benchmark"] != "rmbench" or manifest["depth"] or manifest["fields"] != fields():
        raise ValueError("Not an RGB-only RMBench extraction with the declared fields")
    metadata = rmbench.METADATA
    for task, item in manifest["tasks"].items():
        train = read(metadata / "train" / f"record_dataset_{task}_metadata.json")["records"]
        seeds = {int(r["episode"]): int(r["seed"]) for r in train}
        test_path = metadata / "test" / f"record_dataset_{task}_metadata.json"
        test = {int(r["seed"]) for r in read(test_path)["records"]} if test_path.exists() else set()
        for row in item["episodes"]:
            if seeds.get(int(row["episode"])) != int(row["seed"]) or int(row["seed"]) in test:
                raise ValueError(f"{task}: episode {row['episode']} identity or test overlap")
            if row["demo_frames"] != 0:
                raise ValueError("RMBench has no task-provided video frames")
    return manifest



def validate_rgbd_dataset(root, verify_hashes=True):
    manifest = demonstrations.validate_dataset(root, verify_hashes)
    if manifest["benchmark"] != "rmbench" or not manifest["depth"] or manifest["fields"] != rgbd_fields():
        raise ValueError("Not a regenerated RMBench RGB-D dataset")
    if manifest.get("replay", {}).get("protocol") != RGBD_PROTOCOL:
        raise ValueError("Missing regeneration provenance")
    adapter = rmbench.RMBench()
    for task, item in manifest["tasks"].items():
        train = {int(r["episode"]): int(r["seed"]) for r in adapter.records(task, "train")}
        test = {int(r["seed"]) for r in adapter.records(task, "test")}
        accepted = {int(r["episode"]): r for r in item["replay"]["identities"] if r["success"] and r["status"] == "success"}
        for r in item["episodes"]:
            ep, seed = int(r["episode"]), int(r["seed"])
            if train.get(ep) != seed or seed in test or ep not in accepted or int(accepted[ep]["seed"]) != seed:
                raise ValueError("Unverified training identity or success")
            if r["demo_frames"] != 0:
                raise ValueError("RMBench has no reset video")
    return manifest



def extract(raw_data, output, benchmark="rmbench", count=25, tasks=None, depth=False,
            videos=True, workers=2, benchmark_source=None, resume=False):
    """Retired acquisition entry point; published data needs no regeneration."""
    raise NotImplementedError(
        "RMBench collection/conversion is archived under legacy/rmbench/. "
        "Use the published RGB-D dataset with cam prepare --dataset DATA "
        "--config configs/rmbench/demos.json; no replay or dataset "
        "revalidation is required. See docs/benchmarks/rmbench.md."
    )
