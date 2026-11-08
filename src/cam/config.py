"""Explicit run settings shared by every benchmark and learning mode."""

from dataclasses import asdict, dataclass, field

MODES = ("demos", "interaction", "both")
BOTH_PROTOCOLS = {
    "v1": "Free use of demonstrations and interaction",
    "v2": "Demonstrations first; interaction verifies and explores uncertainties",
}


def default_evaluation_budgets(config):
    """Causal checkpoints, including the actual configured development ceiling."""
    if config["mode"] == "demos":
        return [0]
    limit = int(config["budget"])
    return sorted({n for n in (1, 5, 10, 20, limit) if 0 < n <= limit})


def learning_setting(mode, both_protocol=None):
    """Name the setting; missing versions identify historical unversioned reports."""
    setting = {
        "demos": "demo_only",
        "interaction": "interaction_only",
        "both": "demo_interaction",
    }[mode]
    if mode == "both" and both_protocol is not None:
        if both_protocol not in BOTH_PROTOCOLS:
            raise ValueError("both_protocol must be v1 or v2")
        setting += "_" + both_protocol
    return setting


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
    training_seed_policy: str = "demo_cycle"
    depth: bool = True
    builtin_demo: bool = True
    perceptual_variations: bool = False
    max_decisions: int = 40
    max_action_steps: int = 1500
    policy_timeout: int = 1800
    coding: CodingConfig = field(default_factory=CodingConfig)
    both_protocol: str | None = None
    physics_backend: str | None = None

    def __post_init__(self):
        if self.physics_backend is not None and (
            self.benchmark != "mikasa_robo_v2" or self.physics_backend not in ("cpu", "gpu")
        ):
            raise ValueError("physics_backend is a MIKASA-Robo v2 override: cpu or gpu")
        if self.mode not in MODES or self.budget < 0:
            raise ValueError(
                "Choose demos, interaction or both and a nonnegative budget"
            )
        if self.mode == "both" and self.both_protocol not in BOTH_PROTOCOLS:
            raise ValueError("mode='both' requires an explicit both_protocol: v1 or v2")
        if self.mode != "both" and self.both_protocol is not None:
            raise ValueError("both_protocol is only valid for mode='both'")
        if self.training_seed_count is not None and self.training_seed_count < 1:
            raise ValueError(
                "training_seed_count must be positive or null for all extracted seeds"
            )
        if self.training_seed_policy not in ("demo_cycle", "disjoint_once"):
            raise ValueError("training_seed_policy must be demo_cycle or disjoint_once")
        if self.training_seed_policy == "disjoint_once" and (
            self.mode == "demos" or self.training_seed_count is None
            or self.training_seed_count < self.budget
        ):
            raise ValueError("disjoint_once requires interaction and at least one distinct training seed per attempt")
        if self.mode == "demos" and self.budget != 0:
            raise ValueError("Demo-only development permits zero simulator attempts")
        if self.mode != "demos" and self.budget == 0:
            raise ValueError("Interaction modes require a positive attempt budget")
        if min(self.max_decisions, self.max_action_steps, self.policy_timeout) < 1:
            raise ValueError("Execution limits must be positive")

    def to_dict(self):
        return asdict(self)

    @property
    def setting(self):
        return learning_setting(self.mode, self.both_protocol)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["coding"] = CodingConfig(**value.get("coding", {}))
        return cls(**value)
