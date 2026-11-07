"""Explicit run settings shared by every benchmark and learning mode."""

from dataclasses import asdict, dataclass, field

MODES = ("demos", "interaction", "both")


@dataclass(frozen=True)
class CodingConfig:
    agent: str = "claude_code"
    model: str = "us.anthropic.claude-opus-5-5"
    provider: str = "bedrock"
    reasoning_effort: str = "high"
    aws_profile: str = "sagemaker"
    aws_region: str = "us-east-1"
    timeout: int = 21600
    model_catalog: str | None = None

    def __post_init__(self):
        if self.agent not in ("claude_code", "codex"):
            raise ValueError("Supported coding agents: claude_code, codex")
        allowed = (
            ("bedrock", "anthropic") if self.agent == "claude_code" else ("openai",)
        )
        if self.provider not in allowed or not self.model or self.timeout < 1:
            raise ValueError("Invalid coding model/provider configuration")


@dataclass(frozen=True)
class RunConfig:
    benchmark: str
    mode: str
    budget: int = 0
    training_seed_count: int | None = 3
    depth: bool = True
    builtin_demo: bool = True
    perceptual_variations: bool = False
    max_decisions: int = 40
    max_action_steps: int = 1500
    policy_timeout: int = 1800
    coding: CodingConfig = field(default_factory=CodingConfig)

    def __post_init__(self):
        if self.mode not in MODES or self.budget < 0:
            raise ValueError(
                "Choose demos, interaction or both and a nonnegative budget"
            )
        if self.training_seed_count is not None and self.training_seed_count < 1:
            raise ValueError(
                "training_seed_count must be positive or null for all extracted seeds"
            )
        if self.mode == "demos" and self.budget != 0:
            raise ValueError("Demo-only development permits zero simulator attempts")
        if self.mode != "demos" and self.budget == 0:
            raise ValueError("Interaction modes require a positive attempt budget")
        if min(self.max_decisions, self.max_action_steps, self.policy_timeout) < 1:
            raise ValueError("Execution limits must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["coding"] = CodingConfig(**value.get("coding", {}))
        return cls(**value)
