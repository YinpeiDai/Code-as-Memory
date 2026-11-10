"""Local readiness checks without simulator reset, login or model inference."""

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import json


def doctor(simulator=False, benchmark=None):
    tools = {
        name: shutil.which(name)
        for name in ("bwrap", "strace", "ffmpeg", "claude", "codex")
    }
    modules = ["numpy", "scipy", "PIL", "cv2", "h5py", "imageio"]
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
    if simulator:
        from cam.interface.registry import get_adapter, simulator_python
        checks = {}
        for name in ([benchmark] if benchmark else ["robomme", "mikasa_robo", "rmbench", "robomemarena"]):
            try:
                python = simulator_python(get_adapter(name))
                required = {"robomme": ["torch", "sapien", "mani_skill", "robomme"],
                            "mikasa_robo": ["torch", "sapien", "mani_skill"],
                            "rmbench": ["torch", "sapien", "curobo"],
                            "robomemarena": ["torch", "robosuite", "mujoco"]}[name]
                code = "import importlib.util,json;print(json.dumps({m:importlib.util.find_spec(m) is not None for m in " + repr(required) + "}))"
                check = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=30)
                found = json.loads(check.stdout) if check.returncode == 0 else {}
                checks[name] = {"python": str(python), "modules": found, "ready": bool(found) and all(found.values())}
                if check.returncode:
                    checks[name]["error"] = check.stderr[-2000:]
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                checks[name] = {"ready": False, "error": str(error)}
        result["simulators"] = checks
        result["ready"] = result["ready"] and all(check["ready"] for check in checks.values())
    return result
