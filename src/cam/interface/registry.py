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
        "adapter": "cam.interface.robomme.adapter:RoboMME",
        "data": "cam.interface.robomme.data",
        "environment": "envs/robomme/.venv",
    },
    "robomme_plus": {
        "status": "protocol_draft",
        "url": "https://github.com/RoboMME/robomme_benchmark",
    },
    "mikasa_robo": {
        "status": "implemented",
        "url": "https://github.com/CognitiveAISystems/MIKASA-Robo",
        "adapter": "cam.interface.mikasa_robo.adapter:MikasaRobo",
        "data": "cam.interface.mikasa_robo.data",
        "environment": "envs/mikasa_robo/.venv",
    },
    "mikasa_robo_v2": {
        "status": "compatibility_alias",
        "alias_for": "mikasa_robo",
        "url": "https://github.com/CognitiveAISystems/MIKASA-Robo",
        "adapter": "cam.interface.mikasa_robo.adapter:MikasaRobo",
        "data": "cam.interface.mikasa_robo.data",
        "environment": "envs/mikasa_robo/.venv",
    },
    "rmbench": {
        "status": "implemented",
        "url": "https://github.com/robotwin-Platform/rmbench",
        "adapter": "cam.interface.rmbench.adapter:RMBench",
        "data": "cam.interface.rmbench.data",
        "environment": "envs/rmbench/.venv",
    },
    "robomemarena": {
        "status": "implemented",
        "url": "https://github.com/OpenHelix-Team/RoboMemArena",
        "adapter": "cam.interface.robomemarena.adapter:RoboMemArena",
        "data": "cam.interface.robomemarena.data",
        "environment": "envs/robomemarena/.venv",
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
    if name == "mikasa_robo_legacy":
        # Local historical tooling is deliberately excluded from distributions.
        root = checkout()
        if root is None:
            raise ValueError("The RGB-only MIKASA adapter requires a local legacy checkout")
        path = root / "legacy/mikasa_robo/adapter.py"
        if not path.is_file():
            raise ValueError("The RGB-only MIKASA adapter is local legacy tooling")
        import importlib.util
        spec = importlib.util.spec_from_file_location("cam_legacy_mikasa", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.MikasaRoboLegacy(source)
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    entry = BENCHMARKS[name]
    if "adapter" not in entry:
        raise NotImplementedError(
            f"{name}: {entry['status']}; no runnable adapter claimed"
        )
    adapter = _load(entry["adapter"])(source)
    if entry.get("alias_for"):
        adapter.name = name  # Preserve historical configuration identity without a second implementation.
    return adapter


def data_protocol(name):
    entry = BENCHMARKS.get(name, {})
    if "data" not in entry:
        raise NotImplementedError(f"Raw data protocol not implemented for {name}")
    return importlib.import_module(entry["data"])


def benchmark_matches(expected, actual):
    """Only the former RGB-D name aliases the default; RGB-only data is distinct."""
    return expected == actual or {expected, actual} == {"mikasa_robo", "mikasa_robo_v2"}


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
