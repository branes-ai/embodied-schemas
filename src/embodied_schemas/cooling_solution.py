"""Cooling-solution schemas: thermal removal capability, peer of ProcessNode.

A complete compute product is the blend of three orthogonal pieces of data:
ProcessNode (silicon fabrication), CoolingSolution (thermal removal), and
SKU configuration (architectural choices). Cooling does not belong inside
ProcessNode -- the same node ships in fanless edge modules and liquid-cooled
datacenter cards. The blend is called a ComputeSolution.

The thermal-hotspot validator consumes ``max_power_density_w_per_mm2`` to
flag blocks whose per-area power exceeds what the cooling solution can
remove. The EM validator consumes ``junction_c_max`` to look up the
corresponding ``em_j_max`` figure on the ProcessNode.

Naming note: ``CoolingMechanism`` here is intentionally finer-grained than
``CoolingType`` in mission.py. The mission.py enum (PASSIVE / ACTIVE_FAN /
LIQUID) is used for capability-tier and mission-profile constraints --
a coarse three-bucket classification. ``CoolingMechanism`` distinguishes
e.g., ``passive_heatsink_small`` from ``vapor_chamber`` because the
thermal-hotspot validator needs that resolution.
"""

from enum import Enum

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from embodied_schemas.process_node import DataConfidence
from embodied_schemas.serialization import omit_if_default
from embodied_schemas.swapc2 import ValueBasis


class CoolingMechanism(str, Enum):
    """Class of thermal-removal mechanism.

    Ordered roughly by increasing capability:

    PASSIVE_FANLESS:        no fins, ambient convection only (badge / drone).
    PASSIVE_HEATSINK_SMALL: small fins, no fan (edge module).
    PASSIVE_HEATSINK_LARGE: large fins, no fan (industrial fanless).
    ACTIVE_FAN:             heatsink + fan (PCIe card, NUC, laptop).
    VAPOR_CHAMBER:          spread-plate vapor chamber + fan (high-end consumer).
    LIQUID_COOLED:          AIO / custom loop (workstation, gaming flagship).
    DATACENTER_DTC:         direct-to-chip cold-plate (hyperscaler accel).
    IMMERSION:              two-phase immersion bath (max-density datacenter).
    """

    PASSIVE_FANLESS = "passive_fanless"
    PASSIVE_HEATSINK_SMALL = "passive_heatsink_small"
    PASSIVE_HEATSINK_LARGE = "passive_heatsink_large"
    ACTIVE_FAN = "active_fan"
    VAPOR_CHAMBER = "vapor_chamber"
    LIQUID_COOLED = "liquid_cooled"
    DATACENTER_DTC = "datacenter_direct_to_chip"
    IMMERSION = "immersion"


class CoolingSolutionEntry(BaseModel):
    """One cooling-solution catalog entry.

    Authored once, referenced by SKUs per thermal profile (e.g., a 15W
    profile points to ``passive_heatsink_large_15w``, a 50W profile points
    to ``active_fan_50w``). The thermal-hotspot validator does not read
    cooling-type strings -- it reads ``max_power_density_w_per_mm2``.
    """

    # Identity
    id: str = Field(
        ...,
        description=(
            "Unique id, e.g., 'passive_heatsink_large_30w', 'active_fan_100w'."
        ),
    )
    name: str = Field(..., description="Human-readable name")
    cooling_mechanism: CoolingMechanism = Field(
        ..., description="Class of thermal-removal mechanism"
    )

    # Thermal envelope
    max_power_density_w_per_mm2: float = Field(
        ...,
        gt=0,
        description=(
            "Per-area power-removal ceiling. The thermal-hotspot validator "
            "flags any silicon_bin block whose dynamic + leakage W / area "
            "exceeds this value. Function of fin geometry, airflow, ambient."
        ),
    )
    max_total_w: float = Field(
        ...,
        gt=0,
        description="Whole-package thermal envelope -- sum of all blocks must fit",
    )
    ambient_c_max: float = Field(
        ...,
        description="Maximum ambient operating temperature (C) for this rating",
    )
    surface_c_max: float | None = Field(
        None,
        description=(
            "Maximum temperature at the solution's reference surface (C), for "
            "entries rated at a surface rather than in air -- e.g. a SMARC / COM "
            "heat spreader, which vendors rate at the spreader plate. "
            "ambient_c_max must not exceed it"
        ),
    )
    junction_c_max: float = Field(
        ...,
        description=(
            "Maximum junction temperature (C). Feeds the EM validator's "
            "lookup of em_j_max_by_temp_c on the ProcessNode."
        ),
    )

    # Form factor / mechanical
    form_factor_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form constraints, e.g., 'height<=10mm', 'fanless', "
            "'requires_airflow_>=200lfm', 'requires_cold_plate'."
        ),
    )
    weight_g: float | None = Field(
        None,
        ge=0,
        description="Solution weight in grams (mechanical budget). With "
        "mass_g_per_w set, the fixed part of the mass",
    )
    cost_usd: float | None = Field(
        None,
        ge=0,
        description="Approximate BOM cost contribution in USD (variable unit "
        "cost, RFC 0001 D9). With cost_usd_per_w set, the fixed part",
    )

    # SWaP-C² (RFC 0001 R1, phase S1). All optional; see ``swapc2.py``.
    dimensions_mm: tuple[float, float, float] | None = Field(
        None,
        description="Stated L x W x H in mm. Mounted on the product's top face: "
        "the envelope takes the larger footprint and adds H",
    )
    volume_cm3: float | None = Field(
        None, ge=0, description="Volume in cm^3; with volume_cm3_per_w, the fixed part"
    )
    parasitic_power_w: float | None = Field(
        None,
        ge=0,
        description="Electrical load of fans / pumps, at the vehicle input rail",
    )
    mass_g_per_w: float | None = Field(
        None, ge=0, description="Mass added per W removed (sizing model)"
    )
    volume_cm3_per_w: float | None = Field(
        None, ge=0, description="Volume added per W removed (sizing model)"
    )
    cost_usd_per_w: float | None = Field(
        None, ge=0, description="Unit cost added per W removed (sizing model)"
    )
    sizing_source: str | None = Field(
        None,
        description="Estimator id + version and parameter sources, for entries whose "
        "SWaP-C² fields scripts/swapc2_estimators.py writes (cooling_sizing_v1)",
    )
    basis: ValueBasis = Field(
        ValueBasis.ESTIMATED,
        description="Basis of this entry's size / mass / power / cost figures. "
        "Mechanism-class entries are estimates; a vendor heatsink part is 'datasheet'",
    )

    # Provenance
    source: str = Field(..., description="Citation: vendor datasheet, thermal-design ref")
    confidence: DataConfidence = Field(..., description="Provenance confidence")
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")
    notes: str = Field("", description="Additional notes")

    model_config = {"extra": "forbid"}

    # v14 additive fields are left out of dumps while unset (``serialization``).
    @model_serializer(mode="wrap")
    def _omit_unset_additions(self, handler: SerializerFunctionWrapHandler):
        return omit_if_default(self, handler, (
            "surface_c_max",
            "dimensions_mm",
            "volume_cm3",
            "parasitic_power_w",
            "mass_g_per_w",
            "volume_cm3_per_w",
            "cost_usd_per_w",
            "sizing_source",
            "basis",
        ))

    @model_validator(mode="after")
    def _ambient_below_surface(self) -> "CoolingSolutionEntry":
        """Heat cannot flow from a surface into hotter air."""
        if self.surface_c_max is not None and self.ambient_c_max > self.surface_c_max:
            raise ValueError(
                f"ambient_c_max {self.ambient_c_max} exceeds surface_c_max {self.surface_c_max}"
            )
        return self

    @property
    def is_active(self) -> bool:
        """Whether the mechanism draws electrical power (fan, pump)."""
        return self.cooling_mechanism in _ACTIVE_MECHANISMS

    @staticmethod
    def _sized(base: float | None, per_w: float | None, watts: float) -> float | None:
        if base is None and per_w is None:
            return None
        return (base or 0.0) + (per_w or 0.0) * watts

    def mass_g_at(self, watts: float) -> float | None:
        """Mass in g when removing ``watts``; None if the entry states none."""
        return self._sized(self.weight_g, self.mass_g_per_w, watts)

    def cost_usd_at(self, watts: float) -> float | None:
        """Unit cost in USD when removing ``watts``; None if the entry states none."""
        return self._sized(self.cost_usd, self.cost_usd_per_w, watts)

    def volume_cm3_at(self, watts: float) -> float | None:
        """Volume in cm^3 when removing ``watts``. A fanless entry with no
        stated size occupies nothing; otherwise None if the entry states none."""
        sized = self._sized(self.volume_cm3, self.volume_cm3_per_w, watts)
        if sized is not None:
            return sized
        if self.dimensions_mm is not None:
            length, width, height = self.dimensions_mm
            return length * width * height / 1000.0
        if self.cooling_mechanism == CoolingMechanism.PASSIVE_FANLESS:
            return 0.0
        return None


_ACTIVE_MECHANISMS = {
    CoolingMechanism.ACTIVE_FAN,
    CoolingMechanism.VAPOR_CHAMBER,
    CoolingMechanism.LIQUID_COOLED,
    CoolingMechanism.DATACENTER_DTC,
}
