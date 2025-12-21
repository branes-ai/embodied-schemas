"""Constraint ontology and tier definitions.

Defines classification tiers for latency, power, and other constraints
used in embodied AI codesign.
"""

from enum import Enum
from pydantic import BaseModel, Field


class LatencyTier(str, Enum):
    """Latency requirement tiers."""

    ULTRA_REAL_TIME = "ultra_real_time"  # < 10ms, 100+ FPS
    REAL_TIME = "real_time"  # < 33.3ms, 30+ FPS
    INTERACTIVE = "interactive"  # < 100ms, 10+ FPS
    NEAR_REAL_TIME = "near_real_time"  # < 500ms
    BATCH = "batch"  # < 1000ms+


class PowerClass(str, Enum):
    """Power consumption classes."""

    ULTRA_LOW_POWER = "ultra_low_power"  # < 2W
    LOW_POWER = "low_power"  # < 10W
    MEDIUM_POWER = "medium_power"  # < 30W
    HIGH_POWER = "high_power"  # < 100W
    DATACENTER = "datacenter"  # 100W+


class MemoryClass(str, Enum):
    """Memory capacity classes."""

    TINY = "tiny"  # < 256MB
    SMALL = "small"  # < 1GB
    MEDIUM = "medium"  # < 4GB
    LARGE = "large"  # < 16GB
    VERY_LARGE = "very_large"  # 16GB+


class AccuracyClass(str, Enum):
    """Accuracy requirement classes."""

    BEST_EFFORT = "best_effort"  # No strict requirement
    STANDARD = "standard"  # Typical accuracy
    HIGH = "high"  # High accuracy required
    SAFETY_CRITICAL = "safety_critical"  # Must meet strict thresholds


class LatencyTierSpec(BaseModel):
    """Specification for a latency tier."""

    tier: LatencyTier = Field(..., description="Tier identifier")
    max_ms: float = Field(..., description="Maximum latency in ms")
    fps_equivalent: float = Field(..., description="Equivalent FPS")
    description: str = Field(..., description="Human-readable description")
    typical_use_cases: list[str] = Field(
        default_factory=list, description="Typical use cases"
    )


class PowerClassSpec(BaseModel):
    """Specification for a power class."""

    power_class: PowerClass = Field(..., description="Power class identifier")
    max_watts: float = Field(..., description="Maximum power in watts")
    typical_source: str = Field(..., description="Typical power source")
    examples: list[str] = Field(default_factory=list, description="Example hardware")
    suitable_platforms: list[str] = Field(
        default_factory=list, description="Suitable platform types"
    )


class ImplicationRule(BaseModel):
    """A constraint implication rule.

    e.g., "drone" implies "battery_powered" which implies "low_power"
    """

    condition: str = Field(..., description="The triggering condition")
    implies: list[str] = Field(
        default_factory=list, description="Constraints that are implied"
    )
    prefers: list[str] = Field(
        default_factory=list, description="Constraints that are preferred"
    )
    avoids: list[str] = Field(
        default_factory=list, description="Constraints to avoid"
    )
    not_allowed: list[str] = Field(
        default_factory=list, description="Constraints that are not allowed"
    )


# Pre-defined tier specifications
LATENCY_TIERS: dict[LatencyTier, LatencyTierSpec] = {
    LatencyTier.ULTRA_REAL_TIME: LatencyTierSpec(
        tier=LatencyTier.ULTRA_REAL_TIME,
        max_ms=10.0,
        fps_equivalent=100.0,
        description="Sub-10ms for high-speed control loops",
        typical_use_cases=["high_speed_tracking", "industrial_inspection", "drone_racing"],
    ),
    LatencyTier.REAL_TIME: LatencyTierSpec(
        tier=LatencyTier.REAL_TIME,
        max_ms=33.3,
        fps_equivalent=30.0,
        description="30+ FPS, standard real-time perception",
        typical_use_cases=[
            "drone_obstacle_avoidance",
            "quadruped_locomotion",
            "amr_navigation",
        ],
    ),
    LatencyTier.INTERACTIVE: LatencyTierSpec(
        tier=LatencyTier.INTERACTIVE,
        max_ms=100.0,
        fps_equivalent=10.0,
        description="Responsive but not strictly real-time",
        typical_use_cases=["human_robot_interaction", "gesture_recognition"],
    ),
    LatencyTier.NEAR_REAL_TIME: LatencyTierSpec(
        tier=LatencyTier.NEAR_REAL_TIME,
        max_ms=500.0,
        fps_equivalent=2.0,
        description="Near real-time processing",
        typical_use_cases=["video_analytics", "smart_camera"],
    ),
    LatencyTier.BATCH: LatencyTierSpec(
        tier=LatencyTier.BATCH,
        max_ms=1000.0,
        fps_equivalent=1.0,
        description="Offline or batch processing",
        typical_use_cases=["surveillance_archival", "quality_inspection_offline"],
    ),
}


POWER_CLASSES: dict[PowerClass, PowerClassSpec] = {
    PowerClass.ULTRA_LOW_POWER: PowerClassSpec(
        power_class=PowerClass.ULTRA_LOW_POWER,
        max_watts=2.0,
        typical_source="coin_cell",
        examples=["coral_edge_tpu", "ncs2"],
        suitable_platforms=["iot", "wearable", "always_on_sensor"],
    ),
    PowerClass.LOW_POWER: PowerClassSpec(
        power_class=PowerClass.LOW_POWER,
        max_watts=10.0,
        typical_source="small_battery",
        examples=["jetson_nano_5w", "raspberry_pi"],
        suitable_platforms=["drone", "small_robot", "edge_camera"],
    ),
    PowerClass.MEDIUM_POWER: PowerClassSpec(
        power_class=PowerClass.MEDIUM_POWER,
        max_watts=30.0,
        typical_source="large_battery",
        examples=["jetson_orin_nano", "hailo_8"],
        suitable_platforms=["quadruped", "amr", "large_drone"],
    ),
    PowerClass.HIGH_POWER: PowerClassSpec(
        power_class=PowerClass.HIGH_POWER,
        max_watts=100.0,
        typical_source="ac_power",
        examples=["jetson_agx_orin", "rtx_4090"],
        suitable_platforms=["biped", "vehicle", "edge_server"],
    ),
    PowerClass.DATACENTER: PowerClassSpec(
        power_class=PowerClass.DATACENTER,
        max_watts=500.0,
        typical_source="rack_power",
        examples=["a100", "h100", "mi250x"],
        suitable_platforms=["cloud", "datacenter"],
    ),
}


# Platform-to-constraint implications
PLATFORM_IMPLICATIONS: dict[str, ImplicationRule] = {
    "drone": ImplicationRule(
        condition="drone",
        implies=["battery_powered", "weight_sensitive", "real_time"],
        prefers=["low_power", "compact"],
        avoids=["high_power", "heavy"],
    ),
    "quadruped": ImplicationRule(
        condition="quadruped",
        implies=["battery_powered", "vibration_resistant", "real_time"],
        prefers=["medium_power", "rugged"],
        avoids=["datacenter"],
    ),
    "biped": ImplicationRule(
        condition="biped",
        implies=["high_compute", "real_time"],
        prefers=["high_power"],
        avoids=["ultra_low_power"],
    ),
    "amr": ImplicationRule(
        condition="amr",
        implies=["safety_critical", "real_time"],
        prefers=["medium_power"],
        avoids=[],
    ),
    "edge": ImplicationRule(
        condition="edge",
        implies=["always_on", "remote_deployment"],
        prefers=["low_power", "compact"],
        avoids=["datacenter", "high_power"],
    ),
}


def get_latency_tier(latency_ms: float) -> LatencyTier:
    """Get the latency tier for a given latency value."""
    for tier, spec in sorted(LATENCY_TIERS.items(), key=lambda x: x[1].max_ms):
        if latency_ms <= spec.max_ms:
            return tier
    return LatencyTier.BATCH


def get_power_class(power_watts: float) -> PowerClass:
    """Get the power class for a given power value."""
    for power_class, spec in sorted(POWER_CLASSES.items(), key=lambda x: x[1].max_watts):
        if power_watts <= spec.max_watts:
            return power_class
    return PowerClass.DATACENTER


def get_platform_implications(platform: str) -> ImplicationRule | None:
    """Get constraint implications for a platform type."""
    return PLATFORM_IMPLICATIONS.get(platform.lower())
