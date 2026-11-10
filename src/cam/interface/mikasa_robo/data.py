"""Read and validate the released MIKASA RGB-D demonstrations; no replay dependency."""
from cam.interface.registry import get_adapter
from cam.data import demonstrations

BENCHMARK = "mikasa_robo"
IMAGE = (256, 256, 3)

def spec(dtype, shape, description):
    return {"dtype": dtype, "shape": list(shape), "description": description}


DECLARED = demonstrations.declaration(
    {
        "front_rgb": spec("uint8", IMAGE, "front (MIKASA base_camera) RGB 256x256"),
        "wrist_rgb": spec("uint8", IMAGE, "wrist (MIKASA hand_camera) RGB 256x256"),
        "front_depth": spec("int16", IMAGE[:2], "front depth, int16 optical-axis millimetres, 0 unknown"),
        "wrist_depth": spec("int16", IMAGE[:2], "wrist depth, int16 optical-axis millimetres, 0 unknown"),
        "front_extrinsic": spec("float32", (3, 4), "front 3x4 world-to-camera [R t], OpenCV axes, metres"),
        "wrist_extrinsic": spec("float32", (3, 4), "wrist 3x4 world-to-camera [R t], OpenCV axes, metres"),
        "tcp": spec("float32", (6,), "measured TCP XYZ metres (world) and roll/pitch/yaw radians"),
        "joints": spec("float32", (7,), "Panda arm joint positions J1..J7, radians"),
        "fingers": spec("float32", (2,), "Panda finger joint positions, metres"),
        "action": spec("float32", (7,), "recorded expert command in robot.step units (normalized pd_ee_delta_pose)"),
    },
    {
        "front_K": spec("float32", (3, 3), "front 3x3 intrinsics of the 256x256 image (OpenCV)"),
        "wrist_K": spec("float32", (3, 3), "wrist 3x3 intrinsics of the 256x256 image (OpenCV)"),
    },
)


def validate_dataset(root, verify_hashes=True):
    manifest = demonstrations.validate_dataset(root, verify_hashes)
    if manifest["benchmark"] not in (BENCHMARK, "mikasa_robo_v2") or manifest["fields"] != DECLARED or not manifest["depth"]:
        raise ValueError("Expected the released MIKASA 256x256 RGB-D dataset")
    adapter = get_adapter(BENCHMARK)
    for task, item in manifest["tasks"].items():
        train = {int(r["episode"]): int(r["seed"]) for r in adapter.records(task, "train")}
        for row in item["episodes"]:
            if train.get(int(row["episode"])) != int(row["seed"]):
                raise ValueError(f"Demonstration identity is not a pinned training episode: {task}/{row['episode']}")
    return manifest


def extract(*args, **kwargs):
    raise ValueError("Use the released MIKASA RGB-D demonstrations. Replay generation is local legacy tooling, excluded from the public package.")
