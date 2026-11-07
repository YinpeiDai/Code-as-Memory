"""Local readiness checks without simulator reset, login or model inference."""

import importlib.util
import os
import platform
import shutil
import subprocess
import sys


def doctor(simulator=False):
    tools = {
        name: shutil.which(name)
        for name in ("bwrap", "strace", "ffmpeg", "claude", "codex")
    }
    modules = ["numpy", "scipy", "PIL", "cv2", "h5py", "imageio"]
    if simulator:
        modules += ["torch", "sapien", "mani_skill"]
    imports = {name: importlib.util.find_spec(name) is not None for name in modules}
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "tools": tools,
        "modules": imports,
        "api_calls": 0,
        "simulator_resets": 0,
    }
    if tools["bwrap"]:
        probe = subprocess.run(
            [
                "bwrap",
                "--unshare-all",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "--",
                "/usr/bin/true",
            ],
            capture_output=True,
            timeout=10,
        )
        result["bubblewrap_works"] = probe.returncode == 0
        result["bubblewrap_error"] = probe.stderr.decode(errors="replace")
    result["ready"] = (
        sys.version_info[:2] == (3, 11)
        and all(imports.values())
        and all(tools[name] for name in ("bwrap", "strace", "ffmpeg"))
        and result.get("bubblewrap_works", False)
    )
    return result
