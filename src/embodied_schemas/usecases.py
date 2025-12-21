"""Use case template schemas for embodied AI applications.

Defines constraint templates for different deployment scenarios:
drones, quadrupeds, bipeds, AMRs, and edge AI systems.
"""

from enum import Enum
from pydantic import BaseModel, Field


class ConstraintCriticality(str, Enum):
    """How critical a constraint is to meet."""

    SAFETY_CRITICAL = "safety_critical"  # Must meet, safety implications
    HARD = "hard"  # Must meet for functionality
    MEDIUM = "medium"  # Should meet, degraded experience if not
    SOFT = "soft"  # Nice to have


class PlatformType(str, Enum):
    """Types of embodied AI platforms."""

    DRONE = "drone"
    QUADRUPED = "quadruped"
    BIPED = "biped"
    AMR = "amr"
    MANIPULATOR = "manipulator"
    EDGE = "edge"
    VEHICLE = "vehicle"


class Operator(str, Enum):
    """Comparison operators for success criteria."""

    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    EQ = "eq"  # Equal


class Constraint(BaseModel):
    """A single constraint specification."""

    # Value constraints (use the appropriate one)
    max_value: float | None = Field(None, description="Maximum allowed value")
    min_value: float | None = Field(None, description="Minimum required value")
    target_value: float | None = Field(None, description="Target value")

    # Classification
    tier: str | None = Field(
        None, description="Tier classification: real_time, low_power, etc."
    )
    criticality: ConstraintCriticality = Field(
        ConstraintCriticality.MEDIUM, description="How critical this constraint is"
    )

    # Context
    unit: str | None = Field(None, description="Unit of measurement: ms, watts, mb, etc.")
    notes: str | None = Field(None, description="Additional context")


class SuccessCriterion(BaseModel):
    """A measurable success criterion for validation."""

    metric: str = Field(..., description="Metric name: latency_ms, power_watts, recall, etc.")
    target: float = Field(..., description="Target value")
    operator: Operator = Field(..., description="Comparison operator")
    criticality: ConstraintCriticality = Field(
        ConstraintCriticality.HARD, description="How critical this criterion is"
    )


class PerceptionRequirement(BaseModel):
    """Perception capability requirements."""

    tasks: list[str] = Field(
        default_factory=list,
        description="Required perception tasks: object_detection, depth, etc.",
    )
    target_classes: list[str] = Field(
        default_factory=list,
        description="Target object classes: person, vehicle, obstacle, etc.",
    )
    detection_range_m: list[float] | None = Field(
        None, description="Required detection range [min, max] in meters"
    )
    field_of_view_deg: float | None = Field(
        None, description="Minimum horizontal field of view in degrees"
    )
    frame_rate_fps: float | None = Field(
        None, description="Minimum frame rate for perception"
    )


class PlatformSpec(BaseModel):
    """Platform specification for the use case."""

    type: PlatformType = Field(..., description="Platform type")
    size_class: str | None = Field(
        None, description="Size class: micro, small, medium, large"
    )
    indoor_outdoor: str = Field(
        "both", description="Operating environment: indoor, outdoor, both"
    )
    autonomous_level: int | None = Field(
        None, description="Autonomy level (1-5, SAE-style)"
    )


class RecommendedConfig(BaseModel):
    """A recommended hardware/model/sensor configuration."""

    id: str = Field(..., description="ID of the recommended item")
    variant: str | None = Field(None, description="Specific variant if applicable")
    notes: str | None = Field(None, description="Why this is recommended")
    priority: int = Field(1, description="Priority ranking (1 = highest)")


class UseCaseEntry(BaseModel):
    """Complete use case template entry."""

    # Identity
    id: str = Field(..., description="Unique identifier, e.g., drone_obstacle_avoidance")
    name: str = Field(..., description="Human-readable name")
    category: str = Field(..., description="Category: drone, quadruped, biped, amr, edge")
    description: str = Field(..., description="Detailed description of the use case")

    # Platform
    platform: PlatformSpec = Field(..., description="Platform specification")

    # Perception requirements
    perception: PerceptionRequirement = Field(
        ..., description="Perception capability requirements"
    )

    # Constraints (keyed by dimension: latency, power, memory, accuracy, cost, weight)
    constraints: dict[str, Constraint] = Field(
        default_factory=dict,
        description="Constraints keyed by dimension name",
    )

    # Preferences (soft optimization goals)
    preferences: list[str] = Field(
        default_factory=list,
        description="Optimization preferences: minimize_power, maximize_accuracy, etc.",
    )

    # Implied constraints (derived from platform/use case)
    implied: list[str] = Field(
        default_factory=list,
        description="Implied constraints: real_time, battery_powered, outdoor_rated, etc.",
    )

    # Recommendations
    recommended_hardware: list[RecommendedConfig] = Field(
        default_factory=list, description="Recommended hardware configurations"
    )
    recommended_models: list[RecommendedConfig] = Field(
        default_factory=list, description="Recommended model configurations"
    )
    recommended_sensors: list[RecommendedConfig] = Field(
        default_factory=list, description="Recommended sensor configurations"
    )

    # Success criteria (for validation)
    success_criteria: list[SuccessCriterion] = Field(
        default_factory=list, description="Measurable success criteria"
    )

    # Metadata
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")
    maintainer: str | None = Field(None, description="Maintainer contact")
    references: list[str] = Field(
        default_factory=list, description="Reference documents or URLs"
    )

    model_config = {"extra": "forbid"}
