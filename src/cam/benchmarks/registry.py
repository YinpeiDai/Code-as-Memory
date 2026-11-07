"""Implemented adapters and explicitly pending benchmarks."""

from .robomme import RoboMME

BENCHMARKS = {
    "robomme": {
        "status": "implemented",
        "url": "https://github.com/RoboMME/robomme_benchmark",
    },
    "robomme_plus": {
        "status": "protocol_draft",
        "url": "https://github.com/RoboMME/robomme_benchmark",
    },
    "mikasa_robo": {
        "status": "planned",
        "url": "https://github.com/CognitiveAISystems/MIKASA-Robo",
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


def get_adapter(name, source=None):
    if name == "robomme":
        return RoboMME(source)
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    raise NotImplementedError(
        f"{name}: {BENCHMARKS[name]['status']}; no runnable adapter claimed"
    )


def data_protocol(name):
    if name == "robomme":
        from . import robomme_data

        return robomme_data
    raise NotImplementedError(f"Raw data protocol not implemented for {name}")
