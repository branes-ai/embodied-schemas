"""Unified ComputeProduct schema.

v1 (KPU-only) shipped in PRs #15-#18. v2 adds GPUBlock as the second
discriminated-union member; see ``gpu_block.py`` and
``graphs/docs/designs/gpu-compute-product-schema-extension.md``.

Implements the v1 scope from the assessment at
``graphs/docs/assessments/kpu-as-generic-compute-product.md``: a unified
spine for compute products with per-die structure (process_node_id +
silicon_bin + die area + clocks per die) and a discriminated ``blocks``
union for category-specific architectural detail.

v1 scope (now v1.0, additive): KPU monolithic only.

  - ``BlockKind.KPU`` discriminator value defined.
  - One ``Die`` per ComputeProduct (monolithic).
  - Inter-die ``Interconnect`` list reserved for future use (empty in v1).
  - ``ThermalProfile`` / ``Power`` / ``Market`` are vendor-neutral
    duplicates of the existing KPU-prefixed types so the adapter (PR #3)
    is mechanical.
  - ``LifecycleStatus`` enum replaces today's ``is_discontinued: bool``.

v2 scope (additive): GPU block kind.

  - ``BlockKind.GPU`` discriminator value added.
  - ``GPUBlock`` and supporting GPU-shaped sub-types live in
    ``gpu_block.py`` (separate module to keep this file focused on
    the spine + discriminator wiring).
  - ``AnyBlock`` union now accepts ``KPUBlock | GPUBlock``.
  - Existing KPU YAMLs validate identically -- v2 only adds new types,
    it does not modify or rename anything in v1.

Deferred to v3+:

  - CPU / NPU / DSP / Memory / IO / Bridge / ISP / VideoCodec /
    AudioCodec block kinds (GPU shipped in v2)
  - Per-die ``voltage_rails`` / ``clock_domains`` (multi-rail DVFS)
  - Per-die thermal coupling and 3D stacking metadata
  - ``Switch`` first-class entity
  - ``CoherenceDomain`` overlay
  - ``contains: list[ProductRef]`` for board / system level products
  - ``harvested_from: ProductRef`` for harvested-SKU lineage
  - ``Cluster`` / ``Quadrant`` hierarchy from
    ``graphs/docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md``
  - Software / SDK / Telemetry / Sensor I/O / Certifications / Security
    / Form-factor / Operating environment / Mission profile -- all
    additive when needed

This module is additive in PR #1: it does not modify ``KPUEntry`` or any
other existing schema. The migration path is parallel + adapter:

  - Existing ``data/kpus/<vendor>/`` YAMLs continue to load via
    ``load_kpus()`` returning ``KPUEntry`` instances
  - New ``data/compute_products/<vendor>/`` YAMLs load as
    ``ComputeProduct`` instances (loader added in PR #2)
  - ``graphs.hardware`` adapter (PR #3) wraps any remaining ``KPUEntry``
    to look like a ``ComputeProduct`` so consumers see one interface
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# Reuse existing KPU sub-types verbatim for ``KPUBlock`` content. The
# adapter in PR #3 maps ``KPUEntry.kpu_architecture`` to
# ``KPUBlock(...)`` field-by-field with no transformation.
from embodied_schemas.kpu import (
    KPUClocks,
    KPUMemorySubsystem,
    KPUNoCSpec,
    KPUSiliconBin,
    KPUTheoreticalPerformance,
    KPUThermalProfile,
    KPUTileSpec,
)
from embodied_schemas.process_node import DataConfidence


# ---------------------------------------------------------------------------
# Top-level enums
# ---------------------------------------------------------------------------

class ProductKind(str, Enum):
    """High-level product category. v1 only ships ``CHIP``; ``MCM`` /
    ``CHIPLET`` / ``BOARD`` / ``SYSTEM`` are reserved for v2+ when chiplet,
    multi-die, board-level, and rack-level products land."""

    CHIP = "chip"           # monolithic single-die product
    MCM = "mcm"             # multi-chip module (deferred to v2)
    CHIPLET = "chiplet"     # chiplet package (deferred to v2)
    BOARD = "board"         # board-level product (deferred to v2)
    SYSTEM = "system"       # rack / chassis level (deferred to v2)


class PackagingKind(str, Enum):
    """Physical packaging classification. Independent of ProductKind --
    a CHIP product has a packaging kind too (BGA, LGA, SXM, etc.)."""

    MONOLITHIC = "monolithic"
    MCM = "mcm"
    CHIPLET = "chiplet"
    BOARD = "board"
    SYSTEM = "system"


class LifecycleStatus(str, Enum):
    """Product lifecycle state. Replaces today's ``is_discontinued: bool``
    with a richer enum that captures the full procurement-relevant
    lifecycle. v1 uses the canonical values; v2 may extend with
    intermediate states."""

    ENGINEERING_SAMPLE = "engineering_sample"
    PILOT = "pilot"
    PRODUCTION = "production"
    MATURE = "mature"
    NRND = "nrnd"           # Not Recommended for New Designs
    LTB = "ltb"             # Last-Time Buy window open
    EOL = "eol"             # End of Life


class DieRole(str, Enum):
    """What this die is for. v1 always uses ``COMPUTE`` (KPU monolithic);
    ``MEMORY`` / ``IO`` / ``BRIDGE`` / ``MIXED`` are reserved for v2+ when
    chiplet / HBM / V-Cache / IOD products land."""

    COMPUTE = "compute"
    MEMORY = "memory"
    IO = "io"
    BRIDGE = "bridge"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Block discriminated union
# ---------------------------------------------------------------------------

class BlockKind(str, Enum):
    """Discriminator for ``Block`` subclasses. v1 ships ``KPU``; v2
    adds ``GPU``; v3 adds ``CPU``; v4 adds ``NPU``; v5 adds ``CGRA``;
    v6 adds ``DPU``. Future block kinds (``DSP``, ``TPU``, ``MEMORY``,
    ``IO``, ``BRIDGE``, ``ISP``, ``VIDEO_CODEC``, ``AUDIO_CODEC``,
    ``RADAR_DSP``, ``LIDAR_PREPROC``) come in subsequent PRs as their
    catalogs are added."""

    KPU = "kpu"
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    CGRA = "cgra"
    DPU = "dpu"


class KPUBlock(BaseModel):
    """KPU compute block. Carries the architectural description that today
    lives in ``KPUEntry.kpu_architecture``: heterogeneous tile mix,
    on-chip mesh NoC, and tile-local memory subsystem.

    The ``noc`` field stays here in v1 (matches today's
    ``KPUArchitecture``). v2 may extract intra-die NoCs into the parent
    ``Die.interconnects`` list and reserve ``Die.interconnects`` for all
    interconnect levels uniformly. Hold off until at least one consumer
    needs the unified view.
    """

    kind: Literal[BlockKind.KPU] = Field(
        BlockKind.KPU,
        description="Discriminator -- always BlockKind.KPU for this class",
    )
    total_tiles: int = Field(
        ..., gt=0, description="Total tiles across all tile classes"
    )
    multi_precision_alu: list[str] = Field(
        default_factory=list,
        description="Precisions supported chip-wide, e.g., ['int4','int8','bf16','fp32']",
    )
    tiles: list[KPUTileSpec] = Field(
        ..., description="Per-tile-class specifications (heterogeneous tile mix)"
    )
    noc: KPUNoCSpec = Field(..., description="Intra-die NoC topology")
    memory: KPUMemorySubsystem = Field(
        ..., description="Tile-local memory hierarchy (L1 per PE, L2/L3 per tile)"
    )

    model_config = {"extra": "forbid"}


# Imported here (after KPUBlock is defined) to keep the discriminator
# union local. ``GPUBlock`` lives in ``gpu_block.py``, ``CPUBlock``
# in ``cpu_block.py``, ``NPUBlock`` in ``npu_block.py``, ``CGRABlock``
# in ``cgra_block.py``, ``DPUBlock`` in ``dpu_block.py`` -- each
# block kind's supporting types form a self-contained module.
from embodied_schemas.gpu_block import GPUBlock  # noqa: E402
from embodied_schemas.cpu_block import CPUBlock  # noqa: E402
from embodied_schemas.npu_block import NPUBlock  # noqa: E402
from embodied_schemas.cgra_block import CGRABlock  # noqa: E402
from embodied_schemas.dpu_block import DPUBlock  # noqa: E402

# Discriminated union for ``Die.blocks``. v1 had one element (KPUBlock);
# v2 added GPUBlock; v3 added CPUBlock; v4 added NPUBlock; v5 added
# CGRABlock; v6 adds DPUBlock. Future PRs extend this with
# ``DSPBlock``, ``TPUBlock``, ``MemoryBlock``, etc. and Pydantic
# dispatches by the ``kind`` discriminator.
AnyBlock = Annotated[
    Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock, CGRABlock, DPUBlock],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Interconnect (basic v1 shape)
# ---------------------------------------------------------------------------

class InterconnectLevel(str, Enum):
    """Scope of an Interconnect. v1 reserves ``NOC_INTRA_DIE`` though the
    KPU monolithic case keeps the NoC inside ``KPUBlock``; future levels
    (``DIE_TO_DIE``, ``PACKAGE_TO_PACKAGE``, ``NODE_TO_NODE``,
    ``RACK_TO_RACK``, ``STORAGE``) come with chiplet / system products."""

    NOC_INTRA_DIE = "noc_intra_die"
    DIE_TO_DIE = "die_to_die"
    PACKAGE_TO_PACKAGE = "package_to_package"
    NODE_TO_NODE = "node_to_node"
    RACK_TO_RACK = "rack_to_rack"
    STORAGE = "storage"


class TopologyKind(str, Enum):
    """Network topology of an Interconnect. v1 typically uses
    ``POINT_TO_POINT`` or ``MESH_2D`` only; richer topologies populate as
    needed."""

    POINT_TO_POINT = "point_to_point"
    RING = "ring"
    MESH_2D = "mesh_2d"
    MESH_3D = "mesh_3d"
    TORUS_2D = "torus_2d"
    TORUS_3D = "torus_3d"
    FAT_TREE = "fat_tree"
    DRAGONFLY = "dragonfly"
    HYPERCUBE = "hypercube"
    ALL_TO_ALL = "all_to_all"
    HUB_AND_SPOKE = "hub_and_spoke"
    SWITCHED = "switched"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"


class Interconnect(BaseModel):
    """Inter-die or intra-die link. v1 KPU monolithic does not populate
    this list (NoC stays in ``KPUBlock``); the field is present so the
    schema is extension-ready for chiplet products in v2."""

    interconnect_id: str = Field(..., description="Unique id within the die / product")
    level: InterconnectLevel
    topology: TopologyKind
    per_link_bandwidth_gbps: float = Field(..., gt=0)
    per_link_latency_ns: float | None = Field(None, ge=0)
    per_link_energy_pj_per_byte: float | None = Field(None, ge=0)
    num_links: int = Field(..., gt=0)
    coherent: bool = Field(False, description="Cache-coherent or DMA-only")
    notes: str = Field("")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Die
# ---------------------------------------------------------------------------

class Die(BaseModel):
    """One die within a ComputeProduct. v1 monolithic KPU has exactly one
    die; chiplet products in v2 will have multiple dies with potentially
    different ``process_node_id`` per die.

    ``silicon_bin`` is per-die (the chiplet caveat in the assessment doc)
    -- each die has its own area / transistor decomposition. v1 reuses
    the existing ``KPUSiliconBin`` type; the per-block ``count_ref``
    strings (``tile.<type>``, ``l2_total_kib``, ``noc``, ``memory``) are
    KPU-flavored and assume a ``KPUBlock`` in this die. Future block
    kinds will need their own ``count_ref`` conventions.
    """

    die_id: str = Field(
        ...,
        description=(
            "Unique id within the product, e.g., 'kpu_compute', 'ccd0', "
            "'iod', 'hbm3_stack_0'. v1 KPU monolithic uses 'kpu_compute' "
            "by convention."
        ),
    )
    die_role: DieRole = Field(
        ..., description="What this die is for. v1 KPU always uses COMPUTE."
    )
    process_node_id: str = Field(
        ...,
        description=(
            "References data/process-nodes/<foundry>/<node>.yaml. Per-die "
            "(chiplet caveat); each die can be on a different node."
        ),
    )
    die_size_mm2: float = Field(..., gt=0, description="Die area in mm^2")
    transistors_billion: float = Field(
        ..., gt=0, description="Transistor count in billions"
    )
    silicon_bin: KPUSiliconBin = Field(
        ..., description="Per-block transistor decomposition"
    )
    clocks: KPUClocks = Field(
        ..., description="Per-die clock domain (base + boost)"
    )
    blocks: list[AnyBlock] = Field(
        ...,
        min_length=1,
        description=(
            "Architectural blocks on this die. v1 has exactly one "
            "KPUBlock per die; future block kinds via the discriminated "
            "union."
        ),
    )
    interconnects: list[Interconnect] = Field(
        default_factory=list,
        description=(
            "Inter-die and on-die interconnects for this die. v1 KPU "
            "monolithic has empty list (intra-die NoC stays in KPUBlock); "
            "v2 chiplet products populate with die-to-die links."
        ),
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Packaging, Power, Market
# ---------------------------------------------------------------------------

class Packaging(BaseModel):
    """Physical packaging metadata. v1 KPU always uses
    ``num_dies=1`` and ``MONOLITHIC``; chiplet products in v2 set
    ``num_dies > 1`` and choose the appropriate kind."""

    kind: PackagingKind = Field(...)
    num_dies: int = Field(1, gt=0)
    package_type: str | None = Field(
        None,
        description=(
            "Foundry-specific packaging tag, e.g., 'cowos', 'foveros', "
            "'emib', 'pop', 'flip_chip_bga'. Free-form in v1."
        ),
    )

    model_config = {"extra": "forbid"}


class Power(BaseModel):
    """Chip-level power envelope. Same shape as today's ``KPUPowerSpec``
    but vendor-neutral; the v1 adapter copies fields one-to-one.

    v1 keeps ``thermal_profiles`` as a chip-wide list. v2 may add per-die
    TDP allocation (per the chiplet caveat) and per-rail Vdd / per-domain
    clock per profile (per the voltage/clock domain caveat).
    """

    tdp_watts: float = Field(
        ..., gt=0, description="Default-profile TDP (DERIVED by the generator)"
    )
    max_power_watts: float = Field(..., gt=0)
    min_power_watts: float = Field(..., gt=0)
    idle_power_watts: float | None = Field(None, ge=0)
    default_thermal_profile: str = Field(...)
    thermal_profiles: list[KPUThermalProfile] = Field(..., min_length=1)

    model_config = {"extra": "forbid"}


class Market(BaseModel):
    """Market positioning. Vendor-neutral version of today's
    ``KPUMarket``. v1 keeps the existing field set; lifecycle moves to
    the top-level ``ComputeProduct.lifecycle`` enum."""

    launch_date: str | None = Field(None)
    launch_msrp_usd: float | None = Field(None)
    target_market: str = Field(
        ..., description="edge / embodied / datacenter (free-form in v1)"
    )
    product_family: str = Field(...)
    model_tier: str = Field(
        ...,
        description="entry / mid / high / enthusiast / datacenter",
    )
    is_available: bool = Field(False)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Top-level ComputeProduct
# ---------------------------------------------------------------------------

class ComputeProduct(BaseModel):
    """Unified compute product schema, v1 (KPU-only).

    Lives under ``data/compute_products/<vendor>/<id>.yaml``. References
    a ``ProcessNodeEntry`` per die via ``Die.process_node_id`` and a
    ``CoolingSolutionEntry`` per thermal profile via
    ``Power.thermal_profiles[].cooling_solution_id``.

    The shape is a strict superset (with one extra nesting level) of
    today's ``KPUEntry``: every existing field maps cleanly into either
    the spine, the ``dies[0]`` ``Die``, or the ``dies[0].blocks[0]``
    ``KPUBlock``. The PR #3 adapter performs this mapping mechanically.

    This v1 covers KPU monolithic. v2+ extends to chiplet products
    (multiple dies with mixed process nodes), GPU / CPU / NPU / DSP
    blocks, system-level composition, and the additional metadata axes
    (software, telemetry, sensors, certifications, security,
    form-factor, mission profile) catalogued in the assessment doc.
    """

    # Identity
    id: str = Field(
        ...,
        description=(
            "Unique id, following the naming convention "
            "``kpu_t<count>_<rows>x<cols>_<mem><ch>_<value>nm_<foundry>_<library>`` "
            "for KPU products."
        ),
    )
    name: str = Field(..., description="Human-readable name")
    vendor: str = Field(..., description="Vendor, e.g., 'stillwater'")

    # Product classification
    kind: ProductKind = Field(
        ProductKind.CHIP,
        description="Product category. v1 KPU always uses CHIP.",
    )
    packaging: Packaging = Field(...)

    # Lifecycle (replaces is_discontinued bool)
    lifecycle: LifecycleStatus = Field(
        LifecycleStatus.PRODUCTION,
        description="Procurement-relevant lifecycle state",
    )

    # Per-die structure (at least one die required)
    dies: list[Die] = Field(
        ...,
        min_length=1,
        description=(
            "Per-die structure. v1 KPU monolithic always has exactly "
            "one Die. v2 chiplet products will have multiple, "
            "potentially on different process nodes."
        ),
    )

    # Roll-ups (chip-level)
    performance: KPUTheoreticalPerformance = Field(
        ...,
        description=(
            "Roll-up peak performance across dies / blocks (DERIVED by "
            "the generator). v1 reuses ``KPUTheoreticalPerformance`` "
            "since only KPUBlock exists."
        ),
    )
    power: Power = Field(...)
    market: Market = Field(...)

    # Confidence / provenance
    confidence: DataConfidence = Field(
        DataConfidence.THEORETICAL,
        description="Spec-data provenance confidence",
    )
    notes: str = Field("")
    datasheet_url: str | None = Field(None)
    last_updated: str = Field(..., description="YYYY-MM-DD")

    model_config = {"extra": "forbid"}
