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

from pydantic import BaseModel, Field

from embodied_schemas.process_node import DataConfidence


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
        None, description="Solution weight in grams (mechanical budget)"
    )
    cost_usd: float | None = Field(
        None, description="Approximate BOM cost contribution in USD"
    )

    # Provenance
    source: str = Field(..., description="Citation: vendor datasheet, thermal-design ref")
    confidence: DataConfidence = Field(..., description="Provenance confidence")
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")
    notes: str = Field("", description="Additional notes")

    model_config = {"extra": "forbid"}
