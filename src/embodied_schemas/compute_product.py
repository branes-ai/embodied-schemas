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

v14 scope (additive): SWaP-C², levels of integration, HardwareEntry
absorption. RFC 0001 rev 2-4, phase S1.

  - ``ProductKind.MODULE`` and ``contains: list[ProductRef]`` (D6): a
    module / board / system references the products it is built from.
  - ``swapc2: SWaPC2Spec`` (R1): size, mass, input power and variable unit
    cost, each with provenance. ``swapc2.resolve_swapc2`` adds the cooling
    bound by a thermal profile (D7).
  - ``environmental`` / ``interfaces`` / ``software`` / ``memory`` and the
    packaging form-factor fields (D8): the ``HardwareEntry`` sections that
    have no other home, reused verbatim from ``hardware.py``.
  - ``performance_by_aggregation`` / ``aggregate_peak`` (D4): peak ops/s
    as sum, min and max over compute blocks.

Deferred to v3+ (items not yet shipped):

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

from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

# Reuse existing KPU sub-types verbatim for ``KPUBlock`` content.
# ``KPUBlock`` shares its architectural field set with ``KPUArchitecture``
# through ``KPUArchitectureBase``; convert with ``KPUBlock.from_architecture``
# / ``KPUBlock.to_architecture``.
from embodied_schemas.kpu import (
    KPUArchitecture,
    KPUArchitectureBase,
    KPUClocks,
    KPUSiliconBin,
    KPUTheoreticalPerformance,
    KPUThermalProfile,
    check_performance_rollup,
    check_profile_domain_references,
    derive_kpu_performance,
)
from embodied_schemas.hardware import (
    EnvironmentalSpec,
    FormFactor,
    InterfaceSpec,
    SoftwareSpec,
)
from embodied_schemas.process_node import DataConfidence
from embodied_schemas.serialization import omit_if_default
from embodied_schemas.swapc2 import SWaPC2Spec


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
    MODULE = "module"       # SoM / COM (SMARC, Jetson), M.2 card, OAM / SXM (v14)
    BOARD = "board"         # board-level product
    SYSTEM = "system"       # rack / chassis level

    @property
    def is_silicon(self) -> bool:
        """Chip-level kinds, which must describe at least one die."""
        return self in (ProductKind.CHIP, ProductKind.MCM, ProductKind.CHIPLET)


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
    v6 adds ``DPU``; v7 adds ``TPU``; v9 adds ``DSP`` (closes the
    last category compute fabric gap); **v13 adds ``IO``** (first
    non-compute block kind: memory controllers + PCIe + inter-socket
    coherence on AMD-style chiplet IODs). Future block kinds
    (``MEMORY``, ``BRIDGE``, ``ISP``, ``VIDEO_CODEC``, ``AUDIO_CODEC``,
    ``RADAR_DSP``, ``LIDAR_PREPROC``) come in subsequent PRs as their
    catalogs are added."""

    KPU = "kpu"
    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    CGRA = "cgra"
    DPU = "dpu"
    TPU = "tpu"
    DSP = "dsp"
    IO = "io"


# Serialized key order of a KPUBlock. Pydantic orders inherited fields
# before subclass fields, which would put ``kind`` last; the catalog YAMLs
# (and the graphs generator's YAML output) have always led with ``kind``.
_KPU_BLOCK_DUMP_ORDER = (
    "kind", "total_tiles", "multi_precision_alu", "tiles", "noc", "memory",
)


class KPUBlock(KPUArchitectureBase):
    """KPU compute block. Carries the architectural description that
    ``KPUEntry.kpu_architecture`` carries: heterogeneous tile mix, on-chip
    mesh NoC, and tile-local memory subsystem.

    The architectural fields are defined once on
    ``kpu.KPUArchitectureBase`` (shared with ``KPUArchitecture``); this
    class adds only the ``kind`` discriminator. Convert between the two
    with ``KPUBlock.from_architecture(arch)`` and ``block.to_architecture()``
    rather than copying fields by hand.

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

    # Deliberately no return annotation: Pydantic builds the serialization-mode
    # JSON schema from a wrap serializer's return type, and ``-> Any`` (or
    # ``-> dict``) would replace KPUBlock's schema with ``{}`` / a bare object.
    # Unannotated keeps the field-derived schema (pinned by
    # tests/test_kpu_architecture_block_dedup.py).
    @model_serializer(mode="wrap")
    def _serialize_in_catalog_order(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if not isinstance(data, dict):
            return data
        ordered = {k: data[k] for k in _KPU_BLOCK_DUMP_ORDER if k in data}
        ordered.update((k, v) for k, v in data.items() if k not in ordered)
        return ordered

    @classmethod
    def from_architecture(cls, arch: KPUArchitecture) -> KPUBlock:
        """Build the ``Die.blocks`` form of an architect-facing topology."""
        return cls(**arch.architecture_fields())

    def to_architecture(self) -> KPUArchitecture:
        """The architect-facing ``KPUArchitecture`` form of this block."""
        return KPUArchitecture(**self.architecture_fields())


# Imported here (after KPUBlock is defined) to keep the discriminator
# union local. ``GPUBlock`` lives in ``gpu_block.py``, ``CPUBlock``
# in ``cpu_block.py``, ``NPUBlock`` in ``npu_block.py``, ``CGRABlock``
# in ``cgra_block.py``, ``DPUBlock`` in ``dpu_block.py``, ``TPUBlock``
# in ``tpu_block.py``, ``DSPBlock`` in ``dsp_block.py`` -- each block
# kind's supporting types form a self-contained module.
from embodied_schemas.gpu_block import GPUBlock  # noqa: E402
from embodied_schemas.cpu_block import CPUBlock  # noqa: E402
from embodied_schemas.npu_block import NPUBlock  # noqa: E402
from embodied_schemas.cgra_block import CGRABlock  # noqa: E402
from embodied_schemas.dpu_block import DPUBlock  # noqa: E402
from embodied_schemas.tpu_block import TPUBlock  # noqa: E402
from embodied_schemas.dsp_block import DSPBlock  # noqa: E402
from embodied_schemas.io_block import IOBlock  # noqa: E402

# Discriminated union for ``Die.blocks``. v1 had one element (KPUBlock);
# v2 added GPUBlock; v3 added CPUBlock; v4 added NPUBlock; v5 added
# CGRABlock; v6 added DPUBlock; v7 added TPUBlock; v9 added DSPBlock
# (closed the last category compute fabric gap); **v13 adds IOBlock**
# (first non-compute block kind; non-compute IOD silicon). Future PRs
# extend this with ``MemoryBlock``, etc. and Pydantic dispatches by
# the ``kind`` discriminator.
AnyBlock = Annotated[
    Union[KPUBlock, GPUBlock, CPUBlock, NPUBlock, CGRABlock, DPUBlock, TPUBlock, DSPBlock, IOBlock],
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

    # v14 (D8): mechanical form factor, from ``HardwareEntry.physical``.
    form_factor: FormFactor | None = Field(None, description="Mechanical form factor")
    mounting: str | None = Field(None, description="Mounting, e.g. 'SMARC 314-pin MXM edge'")
    vita_standard: str | None = Field(None, description="VITA standard, e.g. 'VITA 90'")
    sosa_profile: str | None = Field(None, description="SOSA slot / module profile")
    conduction_cooled: bool | None = Field(None, description="Conduction-cooled variant")
    conformal_coated: bool | None = Field(None, description="Conformal coating applied")

    model_config = {"extra": "forbid"}

    # v14 additive fields are left out of dumps while unset (``serialization``).
    @model_serializer(mode="wrap")
    def _omit_unset_additions(self, handler: SerializerFunctionWrapHandler):
        return omit_if_default(self, handler, (
            "form_factor",
            "mounting",
            "vita_standard",
            "sosa_profile",
            "conduction_cooled",
            "conformal_coated",
        ))


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

    @model_validator(mode="after")
    def _validate_default_profile_name(self) -> Power:
        """``default_thermal_profile`` must name a profile (as ``KPUPowerSpec``)."""
        names = [p.name for p in self.thermal_profiles]
        if self.default_thermal_profile not in names:
            raise ValueError(
                f"default_thermal_profile={self.default_thermal_profile!r} "
                f"is not in thermal_profiles (available: {names})"
            )
        return self

    @property
    def default_profile(self) -> KPUThermalProfile:
        """The profile named by ``default_thermal_profile``."""
        return next(p for p in self.thermal_profiles if p.name == self.default_thermal_profile)


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
    suitable_for: list[str] | None = Field(
        None, description="Use-case ids this product suits (from HardwareEntry, D8)"
    )
    target_applications: list[str] | None = Field(
        None, description="Free-form application tags (from HardwareEntry, D8)"
    )

    model_config = {"extra": "forbid"}

    # v14 additive fields are left out of dumps while unset (``serialization``).
    @model_serializer(mode="wrap")
    def _omit_unset_additions(self, handler: SerializerFunctionWrapHandler):
        return omit_if_default(self, handler, ("suitable_for", "target_applications"))


class ProductRef(BaseModel):
    """A product contained in a module / board / system (RFC D6). ``slot``
    and ``role`` carry what ``SystemConfiguration.SlotAssignment`` did."""

    id: str = Field(..., description="ComputeProduct id of the contained product")
    count: int = Field(1, gt=0)
    slot: int | None = Field(None, ge=1, description="Backplane / carrier slot (1-based)")
    role: str | None = Field(None, description="e.g. 'compute', 'switch', 'io'")
    notes: str = ""

    model_config = {"extra": "forbid"}


class EnabledUnits(BaseModel):
    """How many units of one kind a SKU enables -- its floorsweep (RFC 0001
    D6). SKUs of one family share the silicon and differ by the units enabled
    and the memory configuration."""

    unit: str = Field(
        ...,
        pattern=r"^[a-z0-9_]+$",
        description="Unit kind: gpu_sm, cuda_core, tensor_core, cpu_core, dla, pva, ...",
    )
    enabled: int = Field(..., gt=0, description="Units enabled on this SKU")
    physical: int | None = Field(
        None, gt=0, description="Units on the die; None when the vendor does not publish it"
    )
    source: str | None = Field(
        None, description="Source-DB observation key(s) behind `enabled`"
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _physical_covers_enabled(self) -> EnabledUnits:
        if self.physical is not None and self.physical < self.enabled:
            raise ValueError(
                f"{self.unit}: physical {self.physical} < enabled {self.enabled}"
            )
        return self


class SKUSpec(BaseModel):
    """A product SKU within its family (`market.product_family`): the vendor's
    SKU name and part number, and the floorsweep that sets it apart."""

    name: str = Field(..., description="Vendor SKU name, e.g. 'Jetson AGX Orin 64GB'")
    part_number: str | None = Field(None, description="Vendor part number")
    floorsweep: list[EnabledUnits] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _units_unique(self) -> SKUSpec:
        units = [u.unit for u in self.floorsweep]
        if len(units) != len(set(units)):
            raise ValueError(f"SKU {self.name!r}: floorsweep repeats a unit kind")
        return self

    def enabled(self, unit: str) -> int | None:
        """Units of kind ``unit`` enabled on this SKU, or None if not stated."""
        return next((u.enabled for u in self.floorsweep if u.unit == unit), None)


class MemorySummary(BaseModel):
    """Product-level memory as sold (from ``HardwareEntry.capabilities``, D8).
    Per-block memory hierarchies stay on the blocks."""

    memory_gb: float = Field(..., gt=0, description="Total attached memory in GB")
    memory_type: str | None = Field(None, description="e.g. 'LPDDR5', 'HBM3'")
    memory_bandwidth_gbps: float | None = Field(None, gt=0, description="Peak GB/s")

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

    # Per-die structure. Chip-level kinds need at least one die; a module /
    # board / system needs dies or contains (v14).
    dies: list[Die] = Field(
        default_factory=list,
        description=(
            "Per-die structure. v1 KPU monolithic always has exactly "
            "one Die. v2 chiplet products will have multiple, "
            "potentially on different process nodes. A module / board / "
            "system lists only silicon it adds itself; usually none."
        ),
    )
    contains: list[ProductRef] = Field(
        default_factory=list,
        description="Products this module / board / system is built from (D6)",
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

    # v14: SWaP-C² (R1) and the HardwareEntry sections (D8). All optional.
    swapc2: SWaPC2Spec | None = Field(
        None, description="Size, weight, input power and unit cost (RFC 0001 R1)"
    )
    memory: MemorySummary | None = Field(None, description="Memory as sold (D8)")
    sku: SKUSpec | None = Field(
        None, description="SKU name, part number and floorsweep within the product family (D6)"
    )
    environmental: EnvironmentalSpec | None = Field(None, description="Environmental specs (D8)")
    interfaces: InterfaceSpec | None = Field(None, description="I/O interfaces (D8)")
    software: SoftwareSpec | None = Field(None, description="Software ecosystem (D8)")

    # Confidence / provenance
    confidence: DataConfidence = Field(
        DataConfidence.THEORETICAL,
        description="Spec-data provenance confidence",
    )
    notes: str = Field("")
    datasheet_url: str | None = Field(None)
    product_url: str | None = Field(None)
    last_updated: str = Field(..., description="YYYY-MM-DD")

    model_config = {"extra": "forbid"}

    # v14 additive fields are left out of dumps while unset (``serialization``).
    @model_serializer(mode="wrap")
    def _omit_unset_additions(self, handler: SerializerFunctionWrapHandler):
        return omit_if_default(self, handler, (
            "contains",
            "swapc2",
            "memory",
            "environmental",
            "interfaces",
            "software",
            "product_url",
            "sku",
        ))

    @model_validator(mode="after")
    def _check_structure(self) -> ComputeProduct:
        """Chip-level kinds describe their dies. A module / board / system is
        built from dies of its own, contained products, or both (D8)."""
        if self.kind.is_silicon and not self.dies:
            raise ValueError(f"a {self.kind.value} product needs at least one die")
        if not self.kind.is_silicon and not (self.dies or self.contains):
            raise ValueError(f"a {self.kind.value} product needs dies or contains")
        if any(ref.id == self.id for ref in self.contains):
            raise ValueError(f"product {self.id!r} contains itself")
        return self

    @model_validator(mode="after")
    def _check_floorsweep(self) -> ComputeProduct:
        """A stated floorsweep must match the GPU blocks it describes: enabled
        SMs = num_sms, CUDA / Tensor cores = num_sms x the per-SM count."""
        if self.sku is None:
            return self
        gpus = [b for d in self.dies for b in d.blocks if isinstance(b, GPUBlock)]
        if not gpus:
            return self
        sms = sum(g.num_sms for g in gpus)
        expected = {
            "gpu_sm": sms,
            "cuda_core": sum(g.num_sms * g.cuda_cores_per_sm for g in gpus),
            "tensor_core": sum(g.num_sms * g.tensor_cores_per_sm for g in gpus),
        }
        for unit, value in expected.items():
            stated = self.sku.enabled(unit)
            if stated is not None and stated != value:
                raise ValueError(
                    f"{self.id}: floorsweep {unit} = {stated}, but the GPU blocks give {value}"
                )
        return self

    @property
    def performance_by_aggregation(self) -> PeakAggregation:
        """D4: peak ops/s as sum, min and max over this product's own compute
        blocks. Contained products are not expanded; use ``aggregate_peak``
        with a product catalog for that."""
        return aggregate_peak(self)

    @model_validator(mode="after")
    def _check_profile_references(self) -> ComputeProduct:
        """Thermal-profile domain / tile-class references resolve against the
        product's KPU blocks. Power-domain ids are product-wide, because the
        chip-level thermal profiles refer to them by id alone."""
        domains: dict = {}
        tile_class_ids: set[str] = set()
        shared_ids: set[str] = set()
        for die in self.dies:
            for block in die.blocks:
                if not isinstance(block, KPUBlock):
                    continue
                ids = {t.tile_class_id for t in block.tiles}
                shared_ids |= ids & tile_class_ids
                tile_class_ids |= ids
                for d in block.power_domains or []:
                    if d.domain_id in domains:
                        raise ValueError(
                            f"power domain_id {d.domain_id!r} is defined by two KPU "
                            f"blocks; domain ids must be unique across the product"
                        )
                    domains[d.domain_id] = d
        if shared_ids and any(p.tdp_scenario is not None for p in self.power.thermal_profiles):
            raise ValueError(
                f"tile_class_id {sorted(shared_ids)} appears in more than one KPU block, "
                f"so a tdp_scenario keyed by tile_class_id is ambiguous"
            )
        check_profile_domain_references(self.power.thermal_profiles, domains, tile_class_ids)
        return self

    @model_validator(mode="after")
    def _check_performance_rollup(self) -> ComputeProduct:
        """A declared B5 performance roll-up must match the product's KPU tiles
        at the default thermal profile's clock (the generator's convention).
        Products without a KPU block are not checked against tiles."""
        tiles = [
            t
            for die in self.dies
            for block in die.blocks
            if isinstance(block, KPUBlock)
            for t in block.tiles
        ]
        if tiles:
            check_performance_rollup(self.performance, tiles, self.power.default_profile.clock_mhz)
        return self


# ---------------------------------------------------------------------------
# D4: peak aggregation over compute blocks
# ---------------------------------------------------------------------------

class PeakAggregation(BaseModel):
    """Peak ops/s of a product's compute blocks, in three forms (RFC D4).

    - ``sum``: every block busy at once. An upper bound that assumes perfect
      partitioning and ignores shared memory bandwidth and power limits.
    - ``max``: the best single block.
    - ``min``: the weakest block.

    Each is keyed by precision and covers only the blocks that support that
    precision (a zero peak means unsupported). ``block_count`` says how many
    block instances contributed to each precision.
    """

    sum: dict[str, float] = Field(default_factory=dict)
    min: dict[str, float] = Field(default_factory=dict)
    max: dict[str, float] = Field(default_factory=dict)
    block_count: dict[str, int] = Field(default_factory=dict)
    blocks: list[str] = Field(
        default_factory=list, description="Contributing blocks, 'product/die/kind[i]'"
    )
    blocks_without_peak: list[str] = Field(
        default_factory=list, description="Compute blocks that state no peak; left out"
    )
    headline_fallback: list[str] = Field(
        default_factory=list,
        description="Products whose single compute block took the product's "
        "``performance`` headline as its peak",
    )
    not_expanded: list[str] = Field(
        default_factory=list, description="Contained product ids that could not be expanded"
    )

    model_config = {"extra": "forbid"}


def headline_ops(perf: KPUTheoreticalPerformance) -> dict[str, float]:
    """A ``performance`` headline as ops/s by precision, unsupported (zero)
    precisions dropped."""
    if perf.peak_ops_per_sec_by_precision:
        ops = dict(perf.peak_ops_per_sec_by_precision)
    else:
        ops = {
            "int8": perf.int8_tops * 1e12,
            "bf16": perf.bf16_tflops * 1e12,
            "fp32": perf.fp32_tflops * 1e12,
        }
        if perf.int4_tops is not None:
            ops["int4"] = perf.int4_tops * 1e12
    return {k: v for k, v in ops.items() if v > 0}


def block_peak_ops(block: AnyBlock, clock_mhz: float) -> dict[str, float] | None:
    """Peak ops/s by precision of one compute block, or None if it states none.

    A KPU block's peak is derived from its tiles at ``clock_mhz`` (the B5
    roll-up). Other kinds use their optional ``theoretical_performance``.
    """
    if isinstance(block, KPUBlock):
        return headline_ops(derive_kpu_performance(block.tiles, clock_mhz))
    perf = getattr(block, "theoretical_performance", None)
    if perf is None:
        return None
    return {k: v for k, v in perf.peak_ops_per_sec_by_precision.items() if v > 0}


def aggregate_peak(
    product: ComputeProduct,
    products: Mapping[str, ComputeProduct] | None = None,
) -> PeakAggregation:
    """D4 aggregation over the product's own compute blocks and, given a
    product catalog, those of every contained product (expanded by count).

    A product with exactly one compute block, no contents and no per-block
    peak uses its ``performance`` headline for that block. IO blocks are
    not compute and are skipped.
    """
    result = PeakAggregation()
    instances: list[tuple[dict[str, float], int]] = []
    _collect_peaks(product, products, 1, product.id, (), result, instances)
    for ops, mult in instances:
        for precision, value in ops.items():
            result.sum[precision] = result.sum.get(precision, 0.0) + value * mult
            result.min[precision] = min(result.min.get(precision, value), value)
            result.max[precision] = max(result.max.get(precision, value), value)
            result.block_count[precision] = result.block_count.get(precision, 0) + mult
    return result


def _collect_peaks(
    product: ComputeProduct,
    products: Mapping[str, ComputeProduct] | None,
    mult: int,
    prefix: str,
    seen: tuple[str, ...],
    result: PeakAggregation,
    instances: list[tuple[dict[str, float], int]],
) -> None:
    if product.id in seen:
        raise ValueError(f"contains cycle: {' > '.join(seen + (product.id,))}")
    seen = seen + (product.id,)
    clock = product.power.default_profile.clock_mhz
    compute = [
        (f"{prefix}/{die.die_id}/{getattr(b.kind, 'value', b.kind)}[{i}]", b)
        for die in product.dies
        for i, b in enumerate(die.blocks)
        if not isinstance(b, IOBlock)
    ]
    peaks = [(label, block_peak_ops(b, clock)) for label, b in compute]
    if len(peaks) == 1 and peaks[0][1] is None and not product.contains:
        peaks = [(peaks[0][0], headline_ops(product.performance))]
        result.headline_fallback.append(product.id)
    for label, ops in peaks:
        if ops is None:
            result.blocks_without_peak.append(label)
        else:
            result.blocks.append(label if mult == 1 else f"{label} x{mult}")
            instances.append((ops, mult))
    for ref in product.contains:
        child = products.get(ref.id) if products is not None else None
        if child is None:
            result.not_expanded.append(ref.id)
            continue
        _collect_peaks(
            child, products, mult * ref.count, f"{prefix}>{ref.id}", seen, result, instances
        )


def check_contains_references(products: Mapping[str, ComputeProduct]) -> list[str]:
    """Catalog-level ``contains`` checks: every reference resolves, and there
    are no cycles. Returns error messages; empty means clean."""
    errors: list[str] = []
    for pid, product in products.items():
        for ref in product.contains:
            if ref.id not in products:
                errors.append(f"{pid}: contains unknown product {ref.id!r}")

    def visit(pid: str, path: tuple[str, ...]) -> None:
        if pid in path:
            errors.append(f"contains cycle: {' > '.join(path + (pid,))}")
            return
        for ref in products[pid].contains if pid in products else []:
            visit(ref.id, path + (pid,))

    for pid in products:
        visit(pid, ())
    return sorted(set(errors))
