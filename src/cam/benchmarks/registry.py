"""Implemented adapters and explicitly pending benchmarks.

``environment`` is the benchmark's dedicated uv environment, relative to the CaM
checkout (``envs/<name>/.venv``); ``CAM_<NAME>_PYTHON`` overrides its interpreter.
Benchmarks without an environment run their simulator in the CaM environment.
"""

import importlib
import os
from pathlib import Path

BENCHMARKS = {
    "robomme": {
        "status": "implemented",
        "url": "https://github.com/RoboMME/robomme_benchmark",
        "adapter": "cam.benchmarks.robomme:RoboMME",
        "data": "cam.benchmarks.robomme_data",
    },
    "robomme_plus": {
        "status": "protocol_draft",
        "url": "https://github.com/RoboMME/robomme_benchmark",
    },
    "mikasa_robo": {
        "status": "implemented",
        "url": "https://github.com/CognitiveAISystems/MIKASA-Robo",
        "adapter": "cam.benchmarks.mikasa_robo:MikasaRobo",
        "data": "cam.benchmarks.mikasa_robo_data",
        "environment": "envs/mikasa_robo/.venv",
    },
    "mikasa_robo_v2": {
        "status": "implemented",
        "url": "https://github.com/CognitiveAISystems/MIKASA-Robo",
        "adapter": "cam.benchmarks.mikasa_robo_v2:MikasaRoboV2",
        "data": "cam.benchmarks.mikasa_robo_v2_data",
        "environment": "envs/mikasa_robo/.venv",
    },
    "rmbench": {
        "status": "planned",
        "url": "https://github.com/robotwin-Platform/rmbench",
    },
    "robomemarena": {
        "status": "planned",
        "url": "https://github.com/OpenHelix-Team/RoboMemArena",
    },
    "remembench": {
        "status": "planned",
        "url": "https://github.com/ShahRutav/ReMemBench",
    },
    "activearena": {
        "status": "planned",
        "url": "https://github.com/leeibo/ActiveArena",
    },
}


def _load(reference):
    module, name = reference.split(":")
    return getattr(importlib.import_module(module), name)


def get_adapter(name, source=None):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    entry = BENCHMARKS[name]
    if "adapter" not in entry:
        raise NotImplementedError(
            f"{name}: {entry['status']}; no runnable adapter claimed"
        )
    return _load(entry["adapter"])(source)


def data_protocol(name):
    entry = BENCHMARKS.get(name, {})
    if "data" not in entry:
        raise NotImplementedError(f"Raw data protocol not implemented for {name}")
    return importlib.import_module(entry["data"])


def checkout():
    """The CaM source checkout containing this package, if any."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/cam").is_dir():
            return parent
    return None


def simulator_python(adapter):
    """Interpreter for a benchmark's simulator broker, or None for the CaM environment."""
    entry = BENCHMARKS.get(adapter.name, {})
    override = os.environ.get(f"CAM_{adapter.name.upper()}_PYTHON")
    if override:
        return Path(override)
    if "environment" not in entry:
        return None
    root = checkout()
    if root is None:
        raise FileNotFoundError(
            f"Set CAM_{adapter.name.upper()}_PYTHON to the {adapter.name} environment interpreter"
        )
    python = root / entry["environment"] / "bin/python"
    if not python.exists():
        raise FileNotFoundError(
            f"{adapter.name} environment missing: {python}; see envs/{adapter.name}/README.md"
        )
    return python
