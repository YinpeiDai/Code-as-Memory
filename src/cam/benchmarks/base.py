"""The simulator boundary. Model drivers and statistics do not import a simulator."""

from pathlib import Path
from typing import Protocol
from dataclasses import dataclass


@dataclass
class EpisodeStart:
    goal: str
    frames: list[dict]
    context: dict


@dataclass
class StepResult:
    frames: list[dict]
    status: str
    steps: int
    motion_error: str | None = None
    private_error: dict | None = None


class BenchmarkAdapter(Protocol):
    name: str
    package: str
    source: Path

    def tasks(self) -> list[str]: ...
    def records(self, task: str, split: str) -> list[dict]: ...
    def make_environment(
        self, metadata: Path, task: str, episode: int, split: str, depth: bool
    ): ...
    def observations(self, obs, info, first_step: int, **options) -> list[dict]: ...
    def action_spec(self, task: str) -> dict: ...
    def begin(self, env, *, builtin_demo: bool, depth: bool) -> EpisodeStart: ...
    def advance(self, env, action, context: dict, *, depth: bool) -> StepResult: ...
    def select_training(
        self, extracted_records: list[dict], count: int | None
    ) -> list[dict]: ...
