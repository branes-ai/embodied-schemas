"""Mission, battery, and capability tier schemas for embodied AI systems.

Defines data structures for mission profiles, battery configurations, and
capability tier classifications used in embodied AI codesign.
"""

from enum import Enum
from pydantic import BaseModel, Field, field_validator


class CapabilityTierName(str, Enum):
    """Capability tier identifiers for embodied AI systems."""

    WEARABLE_AI = "wearable_ai"
    MICRO_AUTONOMY = "micro_autonomy"
    INDUSTRIAL_EDGE = "industrial_edge"
    EMBODIED_AI = "embodied_ai"
    AUTOMOTIVE_AI = "automotive_ai"


class BatteryChemistry(str, Enum):
    """Common battery chemistries for embodied AI systems."""

    LIPO = "lipo"  # Lithium Polymer - high energy density, drone-grade
    LI_ION = "li_ion"  # Lithium Ion - general purpose
    LIFEPO4 = "lifepo4"  # Lithium Iron Phosphate - safety, longevity
    LI_ION_HV = "li_ion_hv"  # High-voltage Li-ion - automotive grade
    NIMH = "nimh"  # Nickel-Metal Hydride - legacy, low cost


class SubsystemType(str, Enum):
    """Subsystem types for power allocation."""

    PERCEPTION = "perception"
    CONTROL = "control"
    MOVEMENT = "movement"
    OVERHEAD = "overhead"


class CoolingType(str, Enum):
    """Cooling methods for thermal management."""

    PASSIVE = "passive"
    ACTIVE_FAN = "active_fan"
    LIQUID = "liquid"


class ThermalConstraints(BaseModel):
    """Thermal constraints for a capability tier."""

    max_ambient_temp_c: float = Field(40.0, description="Maximum ambient temperature in Celsius")
    max_junction_temp_c: float = Field(85.0, description="Maximum junction temperature in Celsius")
    typical_cooling: CoolingType = Field(CoolingType.PASSIVE, description="Typical cooling method")
    sustained_power_derating: float = Field(
        0.8, description="Fraction of peak power sustainable (0.0-1.0)"
    )


class FormFactorConstraints(BaseModel):
    """Physical form factor constraints for a capability tier."""

    max_compute_weight_kg: float | None = Field(
        None, description="Maximum compute module weight in kg"
    )
    max_compute_volume_cm3: float | None = Field(
        None, description="Maximum compute module volume in cm3"
    )
    max_total_system_weight_kg: float | None = Field(
        None, description="Maximum total system weight in kg"
    )
    typical_form_factors: list[str] = Field(
        default_factory=list, description="Typical form factors for this tier"
    )


class PowerAllocation(BaseModel):
    """Power allocation ratios across subsystems.

    Ratios must sum to 1.0 (validated).
    """

    perception: float = Field(0.35, description="Fraction allocated to perception (0.0-1.0)")
    control: float = Field(0.20, description="Fraction allocated to control (0.0-1.0)")
    movement: float = Field(0.35, description="Fraction allocated to movement (0.0-1.0)")
    overhead: float = Field(0.10, description="Fraction allocated to overhead (0.0-1.0)")

    @field_validator("perception", "control", "movement", "overhead")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        """Validate that ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {v}")
        return v

    def validate_sum(self) -> bool:
        """Check that ratios sum to 1.0 (within tolerance)."""
        total = self.perception + self.control + self.movement + self.overhead
        return abs(total - 1.0) < 0.001


class DutyCycle(BaseModel):
    """Duty cycle specification for a subsystem or mission phase."""

    active_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of time in active state (0.0-1.0)"
    )
    peak_ratio: float = Field(
        0.2, ge=0.0, le=1.0, description="Fraction of active time at peak power (0.0-1.0)"
    )
    description: str = Field("", description="Description of the duty cycle pattern")

    @property
    def effective_ratio(self) -> float:
        """Effective power ratio considering peak vs typical.

        Simplified model: peak power is 1.5x typical.
        """
        peak_contribution = self.active_ratio * self.peak_ratio * 1.5
        typical_contribution = self.active_ratio * (1.0 - self.peak_ratio) * 1.0
        return peak_contribution + typical_contribution


class OperatingTemperatureRange(BaseModel):
    """Operating temperature range specification."""

    min_c: float = Field(..., description="Minimum operating temperature in Celsius")
    max_c: float = Field(..., description="Maximum operating temperature in Celsius")


# =============================================================================
# Entry Models (YAML-backed)
# =============================================================================


class CapabilityTierEntry(BaseModel):
    """Capability tier definition for embodied AI systems.

    Capability tiers represent classes of systems with similar power envelopes,
    application domains, and operational constraints.
    """

    id: str = Field(..., description="Unique identifier, e.g., wearable_ai")
    name: CapabilityTierName = Field(..., description="Tier name enum value")
    display_name: str = Field(..., description="Human-readable name")
    power_min_w: float = Field(..., description="Minimum power envelope in watts")
    power_max_w: float = Field(..., description="Maximum power envelope in watts")
    description: str = Field(..., description="Detailed description of the tier")

    # Typical characteristics
    typical_applications: list[str] = Field(
        default_factory=list, description="Typical application domains"
    )
    example_platforms: list[str] = Field(
        default_factory=list, description="Example hardware platforms"
    )

    # Constraints
    thermal: ThermalConstraints = Field(
        default_factory=ThermalConstraints, description="Thermal constraints"
    )
    form_factor: FormFactorConstraints = Field(
        default_factory=FormFactorConstraints, description="Form factor constraints"
    )

    # Mission characteristics
    typical_mission_hours: list[float] = Field(
        [1.0, 8.0], description="Typical mission duration range [min, max] in hours"
    )
    typical_battery_wh_per_kg: float = Field(
        150.0, description="Typical battery energy density in Wh/kg"
    )

    # Power allocation
    typical_allocation: PowerAllocation | None = Field(
        None, description="Typical power allocation across subsystems"
    )

    # Metadata
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}

    @property
    def power_range_str(self) -> str:
        """Format power range as string."""
        if self.power_max_w < 1.0:
            return f"{self.power_min_w * 1000:.0f}-{self.power_max_w * 1000:.0f}mW"
        elif self.power_min_w < 1.0:
            return f"{self.power_min_w:.1f}-{self.power_max_w:.0f}W"
        return f"{self.power_min_w:.0f}-{self.power_max_w:.0f}W"

    @property
    def typical_power_w(self) -> float:
        """Typical operating power (geometric mean of range)."""
        return (self.power_min_w * self.power_max_w) ** 0.5

    def contains_power(self, power_w: float) -> bool:
        """Check if a power level falls within this tier."""
        return self.power_min_w <= power_w <= self.power_max_w


class MissionProfileEntry(BaseModel):
    """Mission profile for an embodied AI application.

    Captures the operational characteristics including duration, duty cycles,
    and power requirements for different mission phases.
    """

    id: str = Field(..., description="Unique identifier, e.g., drone_inspection")
    name: str = Field(..., description="Profile name")
    display_name: str = Field(..., description="Human-readable name")
    tier: CapabilityTierName = Field(..., description="Associated capability tier")
    description: str = Field(..., description="Detailed description")
    typical_duration_hours: float = Field(..., description="Expected mission duration in hours")

    # Duty cycles
    perception_duty: DutyCycle = Field(..., description="Perception subsystem duty cycle")
    control_duty: DutyCycle = Field(..., description="Control subsystem duty cycle")
    movement_duty: DutyCycle = Field(..., description="Movement subsystem duty cycle")

    # Power requirements (optional, for reference)
    peak_perception_power_w: float | None = Field(
        None, description="Peak perception power requirement in watts"
    )
    cruise_perception_power_w: float | None = Field(
        None, description="Cruising perception power in watts"
    )
    peak_movement_power_w: float | None = Field(
        None, description="Peak movement power (sprinting, climbing) in watts"
    )
    cruise_movement_power_w: float | None = Field(
        None, description="Cruising movement power in watts"
    )

    # Environment and constraints
    environment: str = Field("indoor", description="Operating environment")
    constraints: list[str] = Field(
        default_factory=list, description="Additional operational constraints"
    )

    # Metadata
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}

    def estimate_average_power_multiplier(
        self, allocation: PowerAllocation | None = None
    ) -> float:
        """Estimate average power as multiplier of rated power.

        Returns a value < 1.0 representing the fraction of rated power
        expected during typical operation based on duty cycles.

        Args:
            allocation: Optional power allocation. Uses default if not provided.
        """
        if allocation is None:
            allocation = PowerAllocation()

        perception_contrib = allocation.perception * self.perception_duty.effective_ratio
        control_contrib = allocation.control * self.control_duty.effective_ratio
        movement_contrib = allocation.movement * self.movement_duty.effective_ratio
        overhead_contrib = allocation.overhead * 1.0  # Overhead always on

        return perception_contrib + control_contrib + movement_contrib + overhead_contrib


class BatteryEntry(BaseModel):
    """Battery configuration for an embodied AI system."""

    id: str = Field(..., description="Unique identifier, e.g., drone_small")
    name: str = Field(..., description="Configuration name")
    chemistry: BatteryChemistry = Field(..., description="Battery chemistry type")
    capacity_wh: float = Field(..., description="Total capacity in watt-hours")
    voltage_nominal: float = Field(..., description="Nominal voltage")
    weight_kg: float = Field(..., description="Total battery weight in kg")
    volume_cm3: float = Field(..., description="Total volume in cubic centimeters")

    # Discharge characteristics
    max_discharge_rate_c: float = Field(
        1.0, description="Maximum continuous discharge rate (C-rate)"
    )
    peak_discharge_rate_c: float = Field(
        2.0, description="Peak discharge rate (short bursts)"
    )

    # Longevity
    cycle_life: int = Field(500, description="Expected cycle life at 80% DoD")

    # Operating conditions
    operating_temp: OperatingTemperatureRange = Field(
        ..., description="Operating temperature range"
    )

    # Classification
    typical_tier: CapabilityTierName | None = Field(
        None, description="Typical capability tier for this battery"
    )

    # Metadata
    description: str = Field("", description="Description of use case")
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}

    @property
    def energy_density_wh_per_kg(self) -> float:
        """Gravimetric energy density in Wh/kg."""
        return self.capacity_wh / self.weight_kg if self.weight_kg > 0 else 0.0

    @property
    def energy_density_wh_per_l(self) -> float:
        """Volumetric energy density in Wh/L."""
        volume_l = self.volume_cm3 / 1000.0
        return self.capacity_wh / volume_l if volume_l > 0 else 0.0

    @property
    def max_continuous_power_w(self) -> float:
        """Maximum continuous discharge power in watts."""
        return self.capacity_wh * self.max_discharge_rate_c

    @property
    def peak_power_w(self) -> float:
        """Peak discharge power in watts."""
        return self.capacity_wh * self.peak_discharge_rate_c

    def estimate_runtime_hours(
        self, average_power_w: float, safety_margin: float = 0.9
    ) -> float:
        """Estimate runtime at a given average power draw.

        Args:
            average_power_w: Average power consumption in watts
            safety_margin: Usable capacity fraction (default 0.9 = 90%)

        Returns:
            Estimated runtime in hours
        """
        if average_power_w <= 0:
            return float("inf")

        usable_capacity = self.capacity_wh * safety_margin
        return usable_capacity / average_power_w

    def can_support_power(self, power_w: float, continuous: bool = True) -> bool:
        """Check if battery can support a given power level."""
        if continuous:
            return power_w <= self.max_continuous_power_w
        return power_w <= self.peak_power_w
