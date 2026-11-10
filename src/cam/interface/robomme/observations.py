"""Allowlisted RoboMME sensor conversion; privileged simulator fields stay private."""

# Adapted from the regular CAP broker; see NOTICE.
import numpy as np
from cam.runtime.wire import DEPTH_FIELDS

KEYS = {
    "front_rgb": "front_rgb_list",
    "wrist_rgb": "wrist_rgb_list",
    "front_extrinsic": "front_camera_extrinsic_list",
    "wrist_extrinsic": "wrist_camera_extrinsic_list",
    "tcp": "eef_state_list",
    "joints": "joint_state_list",
    "fingers": "gripper_state_list",
}


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def observations(
    obs, info, first_step, reset=False, builtin_demo=True, include_depth=False
):
    """Rebuild every response; never forward raw obs/info dictionaries."""
    n = len(obs["front_rgb_list"])
    result = []
    for i in range(n):
        phase = "demo" if reset and i < n - 1 else "exec"
        if phase == "demo" and not builtin_demo:
            continue
        step = i if phase == "demo" else 0 if reset else first_step + i
        row = {"phase": phase, "step": step}
        for name, key in KEYS.items():
            v = array(obs[key][i])
            if name == "tcp":
                v = v.reshape(-1)[:6]
            elif name.endswith("extrinsic"):
                v = v.reshape(-1)[:12].reshape(3, 4)
            elif not name.endswith("rgb"):
                v = v.reshape(-1)
            if name.endswith("rgb"):
                v = v.astype(np.uint8)
            row[name] = v
        for cam in ("front", "wrist"):
            row[cam + "_K"] = array(info[cam + "_camera_intrinsic"]).reshape(3, 3)
        if include_depth:
            for name in DEPTH_FIELDS:
                value = array(obs[name + "_list"][i])
                if value.ndim == 3 and value.shape[-1] == 1:
                    value = value[..., 0]
                if (
                    value.dtype != np.int16
                    or value.shape != row[name.replace("depth", "rgb")].shape[:2]
                ):
                    raise ValueError("Depth must be aligned HxW int16 millimetres")
                row[name] = value
        result.append(row)
    return result
