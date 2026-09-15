"""KPU SKU schemas: Knowledge Processing Unit catalog entries.

KPUs are general parallel execution engines (SURE/SARE-capable), distinct
from NPUs. They live under ``data/kpus/<vendor>/`` as a peer of
``data/gpus/``, ``data/cpus/``, and ``data/npus/``.

A KPUEntry references two other catalog entries by id:

- ``process_node_id`` -> a ProcessNodeEntry (silicon fabrication data).
- ``thermal_profiles[].cooling_solution_id`` -> a CoolingSolutionEntry.

The entry's ``silicon_bin`` block decomposes the chip into per-class blocks
(PE arrays per tile class, SRAM banks, NoC routers, memory PHYs, IO ring)
so the validator framework can do per-library area / power / EM math.

Roll-up numbers (``die.transistors_billion``, ``die.die_size_mm2``,
``power.tdp_watts``, ``performance.*``) are hand-authored for now. The
generator (Phase 3) will derive them from ``silicon_bin`` + the referenced
ProcessNode and the validator framework will check consistency.
"""

import math
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.datapath import PEDatapath
from embodied_schemas.overlay import (
    FabricInterconnect,
    FabricOverlayKind,
    NoCOverlay,
    NoCOverlayKind,
    OverlayScope,
)
from embodied_schemas.gpu import Foundry, MemoryType
from embodied_schemas.process_node import CircuitClass, DataConfidence


# ---------------------------------------------------------------------------
# Tile fabric
# ---------------------------------------------------------------------------

class KPUTileScheduleClass(str, Enum):
    """Scheduling discipline for a KPU tile -- determines dataflow pattern."""

    OUTPUT_STATIONARY = "output_stationary"
    WEIGHT_STATIONARY = "weight_stationary"
    INPUT_STATIONARY = "input_stationary"
    NO_LOCAL_REUSE = "no_local_reuse"


class KPUTileKind(str, Enum):
    """What a KPU tile *is* (graphs#268).

    Every tile class occupies sites of the KPU compute/memory checkerboard.

    - ``pe_fabric``: a programmable domain-flow PE array with a declared
      per-PE datapath. All catalog SKUs today.
    - ``systolic``: a fixed-schedule GEMM / conv array (Phase B3).
    - ``fixed_function``: an encapsulated compute segment such as an ISP or
      a VIO pipeline (Phase B3).
    - ``scalar``, ``io_bridge``: reserved; no schema yet.
    """

    PE_FABRIC = "pe_fabric"
    SYSTOLIC = "systolic"
    FIXED_FUNCTION = "fixed_function"
    SCALAR = "scalar"
    IO_BRIDGE = "io_bridge"


class TileFootprint(BaseModel):
    """How many checkerboard compute sites a tile occupies (rows x cols).

    ``absorbs_memory_cells``: the memory cells paired with the sites inside
    the footprint become tile-local (e.g. a VIO tile's on-chip state) instead
    of shared L3.
    """

    rows: int = Field(1, ge=1)
    cols: int = Field(1, ge=1)
    absorbs_memory_cells: bool = False

    model_config = {"extra": "forbid"}

    @property
    def sites(self) -> int:
        return self.rows * self.cols


class LocalMemoryLevel(str, Enum):
    """Kind of a tile-class-local memory."""

    L1 = "l1"
    L2 = "l2"
    LINE_BUFFER = "line_buffer"
    WEIGHT_BUFFER = "weight_buffer"
    ACCUMULATOR = "accumulator"


class LocalMemoryScope(str, Enum):
    """Whether a ``LocalMemory.kib`` figure is per PE or per tile."""

    PE = "pe"      # kib is per PE
    TILE = "tile"  # kib is per tile


class LocalMemory(BaseModel):
    """One tile-class-local memory (per PE or per tile)."""

    level: LocalMemoryLevel
    kib: float = Field(..., gt=0)
    per: LocalMemoryScope = LocalMemoryScope.TILE
    circuit_class: CircuitClass = CircuitClass.SRAM_HD

    model_config = {"extra": "forbid"}


class TilePlacementAffinity(str, Enum):
    """Where on the die a tile class prefers to be placed."""

    ANY = "any"
    IO_EDGE = "io_edge"          # e.g. an ISP next to the MIPI PHYs
    MEMORY_EDGE = "memory_edge"  # next to a DRAM controller
    CENTER = "center"


class TilePlacement(BaseModel):
    """Floorplan hints for a tile class (used by the Phase D placer)."""

    affinity: TilePlacementAffinity = TilePlacementAffinity.ANY
    adjacent_to: list[str] = Field(
        default_factory=list,
        description="tile_class_ids this class should be placed next to "
        "(e.g. a stream-link partner)",
    )

    model_config = {"extra": "forbid"}


_TILE_CLASS_ID_RE = r"^[a-z0-9_]+$"


def tile_class_slug(tile_type: str) -> str:
    """Default ``tile_class_id`` for a tile label: 'INT8-primary' -> 'int8_primary'."""
    return re.sub(r"[^a-z0-9]+", "_", tile_type.lower()).strip("_")


class KPUTileSpec(BaseModel):
    """One specialized tile class within a heterogeneous KPU.

    A KPU is built from a small number of tile classes (typically
    INT8-primary, BF16-primary, Matrix), each with its own PE array,
    standard-cell library, and ops/clock profile. The ``num_tiles`` for
    each class is the architectural mix.

    Every catalog tile is a ``pe_fabric`` tile. The Phase B1 fields
    (graphs#268) are optional and backward compatible: ``tile_kind`` and
    ``tile_class_id`` are filled in when absent. When a ``datapath`` is
    declared, the ops it implies must equal ``ops_per_tile_per_clock``.
    """

    tile_kind: Literal[KPUTileKind.PE_FABRIC] = Field(
        KPUTileKind.PE_FABRIC,
        description="Tile kind; pe_fabric is the only kind this class describes",
    )
    tile_type: str = Field(
        ..., description="Human-readable label, e.g., 'INT8-primary', 'Matrix'"
    )
    tile_class_id: str = Field(
        ...,
        pattern=_TILE_CLASS_ID_RE,
        description="Stable id for references (power domains, placement, "
        "silicon). Defaults to a slug of tile_type: 'INT8-primary' -> 'int8_primary'",
    )
    num_tiles: int = Field(..., gt=0, description="Number of tiles of this class")
    pe_array_rows: int = Field(..., gt=0, description="PE array rows per tile")
    pe_array_cols: int = Field(..., gt=0, description="PE array columns per tile")
    pe_circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library used for the PE datapath"
    )
    ops_per_tile_per_clock: dict[str, float] = Field(
        ...,
        description=(
            "Ops per tile per clock by precision name. Keys are precision "
            "strings ('int8', 'int4', 'bf16', 'fp32'). For uniform PE-array "
            "tiles this equals (rows * cols * ops_per_pe). Tensor-core-style "
            "tiles can have higher op density via systolic accumulation."
        ),
    )
    schedule_class: KPUTileScheduleClass = Field(
        KPUTileScheduleClass.OUTPUT_STATIONARY,
        description="Dataflow scheduling class",
    )
    pipeline_fill_cycles: int = Field(
        0, ge=0, description="Cycles to fill the tile pipeline"
    )
    pipeline_drain_cycles: int = Field(
        0, ge=0, description="Cycles to drain the tile pipeline"
    )
    notes: str = Field("", description="Additional notes")

    # --- Phase B1 (graphs#268): optional, backward compatible -------------
    datapath: PEDatapath | None = Field(
        None,
        description="Per-PE datapath. When set, rows * cols * its ops per PE "
        "must equal ops_per_tile_per_clock for every precision",
    )
    footprint: TileFootprint | None = Field(
        None, description="Checkerboard sites occupied; None = 1x1"
    )
    local_memory: list[LocalMemory] | None = Field(
        None,
        description="Tile-class-local memories. None = the chip-level "
        "KPUMemorySubsystem L1/L2 figures apply",
    )
    power_domain_id: str | None = Field(
        None, description="Power domain this tile class belongs to (Phase B4)"
    )
    placement: TilePlacement | None = Field(None, description="Floorplan hints")
    # --- Phase B2 (graphs#268) --------------------------------------------
    interconnect: FabricInterconnect | None = Field(
        None,
        description="PE-to-PE links and overlays inside the tile; None = "
        "nearest-neighbor mesh with no declared overlays",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _default_tile_class_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "tile_class_id" not in data and "tile_type" in data:
            data = {**data, "tile_class_id": tile_class_slug(str(data["tile_type"]))}
        return data

    @model_validator(mode="after")
    def _check_datapath_and_memory(self) -> "KPUTileSpec":
        if self.local_memory:
            levels = [m.level for m in self.local_memory]
            dup = sorted({lv.value for lv in levels if levels.count(lv) > 1})
            if dup:
                raise ValueError(f"tile {self.tile_type!r}: duplicate local_memory levels {dup}")

        dp = self.datapath
        if dp is None:
            return self
        if dp.circuit_class is not None and dp.circuit_class != self.pe_circuit_class:
            raise ValueError(
                f"tile {self.tile_type!r}: datapath {dp.datapath_id!r} is built in "
                f"{dp.circuit_class.value} but pe_circuit_class is "
                f"{self.pe_circuit_class.value}"
            )
        derived = {
            prec: ops * self.pes_per_tile
            for prec, ops in dp.legacy_precision_ops_per_pe().items()
        }
        declared = self.ops_per_tile_per_clock
        missing = sorted(set(declared) - set(derived))
        extra = sorted(set(derived) - set(declared))
        mismatched = sorted(
            p for p in set(declared) & set(derived)
            if not math.isclose(declared[p], derived[p], rel_tol=1e-9)
        )
        if missing or extra or mismatched:
            detail = []
            if missing:
                detail.append(f"declared but not produced by the datapath: {missing}")
            if extra:
                detail.append(f"produced by the datapath but not declared: {extra}")
            for p in mismatched:
                detail.append(
                    f"{p}: declared {declared[p]:g}, datapath gives "
                    f"{self.pes_per_tile} PEs x {derived[p] / self.pes_per_tile:g} = {derived[p]:g}"
                )
            raise ValueError(
                f"tile {self.tile_type!r}: datapath {dp.datapath_id!r} disagrees "
                f"with ops_per_tile_per_clock ({'; '.join(detail)})"
            )
        return self

    @model_validator(mode="after")
    def _check_interconnect(self) -> "KPUTileSpec":
        """Overlays must fit the PE array they are laid over."""
        if self.interconnect is None:
            return self
        rows, cols = self.pe_array_rows, self.pe_array_cols
        for ov in self.interconnect.overlays:
            where = f"tile {self.tile_type!r} overlay {ov.overlay_id!r}"
            # Links of a per-row overlay run along a row (length = cols), and
            # vice versa. A per-tile overlay spans both axes.
            axes = {
                OverlayScope.ROW: [cols],
                OverlayScope.COL: [rows],
                OverlayScope.TILE: [rows, cols],
            }[ov.instances_per]
            if ov.span is not None and ov.span > max(axes) - 1:
                raise ValueError(
                    f"{where}: span {ov.span} does not fit a {rows}x{cols} PE array "
                    f"(max {max(axes) - 1} along its axis)"
                )
            if ov.kind == FabricOverlayKind.TRANSPOSE and rows != cols:
                raise ValueError(f"{where}: transpose needs a square PE array, got {rows}x{cols}")
            if ov.kind == FabricOverlayKind.BUTTERFLY:
                bad = [n for n in axes if n & (n - 1)]
                if bad:
                    raise ValueError(
                        f"{where}: butterfly needs a power-of-two axis, got {bad} "
                        f"for a {rows}x{cols} array"
                    )
        return self

    @property
    def pes_per_tile(self) -> int:
        return self.pe_array_rows * self.pe_array_cols

    @property
    def total_pes(self) -> int:
        return self.num_tiles * self.pes_per_tile

    @property
    def sites_per_tile(self) -> int:
        """Checkerboard compute sites one tile of this class occupies."""
        return self.footprint.sites if self.footprint is not None else 1

    def ops_per_pe_per_clock(self) -> dict[str, float] | None:
        """``"<op>:<format>"`` ops per PE per clock from the datapath, or None
        when no datapath is declared."""
        return self.datapath.ops_per_pe_per_clock() if self.datapath is not None else None


# ---------------------------------------------------------------------------
# NoC + memory subsystem
# ---------------------------------------------------------------------------

class KPUNoCSpec(BaseModel):
    """KPU on-chip network."""

    topology: str = Field(..., description="Topology, e.g., 'mesh_2d', 'torus_2d'")
    mesh_rows: int = Field(..., gt=0)
    mesh_cols: int = Field(..., gt=0)
    flit_bytes: int = Field(..., gt=0, description="NoC flit width in bytes")
    router_circuit_class: CircuitClass = Field(
        CircuitClass.HP_LOGIC,
        description="Library used for NoC router logic; typically HP for low latency",
    )
    bisection_bandwidth_gbps: float | None = Field(
        None, ge=0, description="Bisection bandwidth across the mesh"
    )
    # --- Phase B2 (graphs#268) --------------------------------------------
    overlays: list[NoCOverlay] | None = Field(
        None,
        description="Tile-level overlays on the mesh: express channels, stream "
        "links between tile classes, multicast trees. Endpoints are "
        "tile_class_ids, checked by KPUArchitectureBase",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_overlays(self) -> "KPUNoCSpec":
        if not self.overlays:
            return self
        ids = [o.overlay_id for o in self.overlays]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate NoC overlay_id {dup}")
        extent = max(self.mesh_rows, self.mesh_cols)
        for o in self.overlays:
            if o.kind == NoCOverlayKind.EXPRESS_CHANNEL and o.span is not None and o.span >= extent:
                raise ValueError(
                    f"NoC overlay {o.overlay_id!r}: span {o.span} does not fit a "
                    f"{self.mesh_rows}x{self.mesh_cols} mesh"
                )
        return self

    @property
    def num_routers(self) -> int:
        return self.mesh_rows * self.mesh_cols


class KPUMemorySubsystem(BaseModel):
    """KPU memory hierarchy: per-PE L1, per-tile L2, distributed L3, off-chip DRAM."""

    memory_type: MemoryType = Field(..., description="DRAM technology")
    memory_size_gb: float = Field(..., gt=0, description="Off-chip DRAM capacity")
    memory_bus_bits: int = Field(..., gt=0, description="DRAM bus width in bits")
    memory_bandwidth_gbps: float = Field(..., gt=0, description="Peak DRAM bandwidth")
    memory_controllers: int = Field(..., gt=0, description="Number of memory controllers")
    l3_kib_per_tile: int = Field(
        ..., gt=0, description="Distributed L3 SRAM per tile (KiB)"
    )
    l2_kib_per_tile: int = Field(
        0, ge=0, description="Per-tile L2 SRAM (KiB); 0 if absent"
    )
    l1_kib_per_pe: int = Field(
        0, ge=0, description="Per-PE L1 SRAM (KiB); 0 if absent"
    )

    model_config = {"extra": "forbid"}


class KPUArchitectureBase(BaseModel):
    """The KPU architectural field set, defined once.

    Two schema types carry exactly these fields:

    - ``KPUArchitecture`` -- the architect-facing topology used by
      ``KPUEntry.kpu_architecture`` and the graphs ``KPUSKUInputSpec``.
    - ``compute_product.KPUBlock`` -- the same topology as a ``Die.blocks``
      member, plus the ``kind`` discriminator.

    Both subclass this base, so a new architectural field is added in one
    place and both types (and the ``KPUBlock.from_architecture`` /
    ``KPUBlock.to_architecture`` converters) pick it up. Neither type is a
    subclass of the other: a ``KPUBlock`` is not an instance of
    ``KPUArchitecture`` and vice versa, so crossing between them is always
    an explicit conversion.

    Not used directly in any catalog document.
    """

    total_tiles: int = Field(
        ..., gt=0, description="Total tiles across all tile classes"
    )
    tiles: list[KPUTileSpec] = Field(
        ..., description="Per-tile-class specifications (heterogeneous tile mix)"
    )
    noc: KPUNoCSpec = Field(..., description="Intra-die NoC topology")
    memory: KPUMemorySubsystem = Field(
        ..., description="Tile-local memory hierarchy (L1 per PE, L2/L3 per tile)"
    )
    multi_precision_alu: list[str] = Field(
        default_factory=list,
        description="Precisions supported chip-wide, e.g., ['int4','int8','bf16','fp32']",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_tile_class_references(self) -> "KPUArchitectureBase":
        ids = [t.tile_class_id for t in self.tiles]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate tile_class_id {dup}; each tile class needs a unique id")
        known = set(ids)
        for tile in self.tiles:
            if tile.placement is None:
                continue
            for other in tile.placement.adjacent_to:
                if other == tile.tile_class_id:
                    raise ValueError(f"tile class {other!r} lists itself in placement.adjacent_to")
                if other not in known:
                    raise ValueError(
                        f"tile class {tile.tile_class_id!r}: placement.adjacent_to "
                        f"references unknown tile_class_id {other!r} (known: {sorted(known)})"
                    )
        for ov in self.noc.overlays or []:
            unknown = [e for e in ov.endpoints if e not in known]
            if unknown:
                raise ValueError(
                    f"NoC overlay {ov.overlay_id!r} references unknown tile_class_id "
                    f"{unknown} (known: {sorted(known)})"
                )
        return self

    def architecture_fields(self) -> dict:
        """The shared architectural fields as a shallow dict.

        Sub-models (tiles, noc, memory) are passed by reference, matching
        the field-by-field copies this replaces. Used by the
        ``KPUBlock`` <-> ``KPUArchitecture`` converters.
        """
        return {name: getattr(self, name) for name in KPUArchitectureBase.model_fields}


class KPUArchitecture(KPUArchitectureBase):
    """The architectural topology of a KPU SKU.

    Captures every architectural knob the generator needs to derive
    rolled-up performance, area, and power. Together with the referenced
    ProcessNode this completely determines the spec.

    Fields are defined on ``KPUArchitectureBase`` (shared with
    ``compute_product.KPUBlock``).
    """


# ---------------------------------------------------------------------------
# Silicon bin (per-block transistor decomposition)
# ---------------------------------------------------------------------------

class TransistorSourceKind(str, Enum):
    """How a silicon_bin block's transistor count is derived."""

    FIXED = "fixed"                # absolute Mtx
    PER_PE = "per_pe"              # Mtx per PE in named tile class
    PER_KIB = "per_kib"            # Mtx per KiB of SRAM
    PER_ROUTER = "per_router"      # Mtx per NoC router
    PER_CONTROLLER = "per_controller"  # Mtx per memory controller


class TransistorSource(BaseModel):
    """How a silicon_bin block's transistor count is computed.

    The generator expands these into absolute transistor counts using
    the SKU's kpu_architecture. Examples:

    - ``kind=FIXED, mtx=25.0`` -> 25 M transistors fixed.
    - ``kind=PER_PE, per_unit_mtx=0.012, count_ref="tile.INT8-primary"``
      -> 0.012 Mtx times the total PE count of the INT8-primary tile class.
    - ``kind=PER_KIB, per_unit_mtx=0.06, count_ref="l3_total_kib"`` ->
      0.06 Mtx times total L3 SRAM in KiB.
    - ``kind=PER_ROUTER, per_unit_mtx=1.2, count_ref="noc"`` -> 1.2 Mtx
      times NoC router count.
    - ``kind=PER_CONTROLLER, per_unit_mtx=8.0, count_ref="memory"`` ->
      8 Mtx per memory controller.
    """

    kind: TransistorSourceKind = Field(...)
    mtx: float | None = Field(
        None, ge=0, description="Fixed Mtx (kind=FIXED)"
    )
    per_unit_mtx: float | None = Field(
        None, ge=0, description="Mtx per unit (kind=PER_*)"
    )
    count_ref: str | None = Field(
        None,
        description=(
            "Reference for unit count. Forms: 'tile.<tile_type>' (PE_PER), "
            "'l1_total_kib' / 'l2_total_kib' / 'l3_total_kib' (PER_KIB), "
            "'noc' (PER_ROUTER), 'memory' (PER_CONTROLLER)."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "TransistorSource":
        """Ensure the right field combination is set for each kind.

        Catches malformed silicon_bin entries at load time rather than
        deferring to the silicon_math resolver, which would surface
        them as runtime errors during area / power computation.
        """
        if self.kind == TransistorSourceKind.FIXED:
            if self.mtx is None:
                raise ValueError(
                    "kind=FIXED requires 'mtx' to be set"
                )
            if self.per_unit_mtx is not None or self.count_ref is not None:
                raise ValueError(
                    "kind=FIXED must not set 'per_unit_mtx' or 'count_ref'; "
                    "those are for kind=PER_*"
                )
        else:
            # PER_PE, PER_KIB, PER_ROUTER, PER_CONTROLLER all require both.
            if self.per_unit_mtx is None:
                raise ValueError(
                    f"kind={self.kind.value} requires 'per_unit_mtx' to be set"
                )
            if not self.count_ref:
                raise ValueError(
                    f"kind={self.kind.value} requires 'count_ref' to be set"
                )
            if self.mtx is not None:
                raise ValueError(
                    f"kind={self.kind.value} must not set 'mtx'; "
                    f"that is only for kind=FIXED"
                )
        return self


class SiliconBinBlock(BaseModel):
    """One block in the silicon-area decomposition.

    The validator framework iterates over these to compute per-block area,
    power density, EM J, etc. Every block declares a ``circuit_class`` so
    density / energy / leakage are looked up from the right ProcessNode
    library.
    """

    name: str = Field(..., description="Human-readable block label")
    circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library for area / power lookup"
    )
    transistor_source: TransistorSource = Field(
        ..., description="How this block's transistor count is derived"
    )
    notes: str = Field("", description="Additional notes")

    model_config = {"extra": "forbid"}


class KPUSiliconBin(BaseModel):
    """Per-block silicon-area decomposition of a KPU SKU."""

    blocks: list[SiliconBinBlock] = Field(...)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Roll-up blocks (generator-derived; hand-authored for now)
# ---------------------------------------------------------------------------

class KPUDieSpec(BaseModel):
    """Roll-up die-level numbers.

    Field shape matches the GPU/NPU ``die`` block so the existing
    physical_spec_loader picks up the same fields without changes.
    """

    architecture: str = Field("KPU Tile")
    foundry: Foundry = Field(...)
    process_nm: int = Field(..., gt=0)
    process_name: str = Field(...)
    transistors_billion: float = Field(..., gt=0)
    die_size_mm2: float = Field(..., gt=0)
    is_chiplet: bool = Field(False)
    num_dies: int = Field(1, gt=0)

    model_config = {"extra": "forbid"}


class KPUClocks(BaseModel):
    base_clock_mhz: float = Field(..., gt=0)
    boost_clock_mhz: float = Field(..., gt=0)

    model_config = {"extra": "forbid"}


class KPUTheoreticalPerformance(BaseModel):
    """Roll-up peak ops/s by precision (TOPS / TFLOPS)."""

    int8_tops: float = Field(..., ge=0)
    bf16_tflops: float = Field(..., ge=0)
    fp32_tflops: float = Field(..., ge=0)
    int4_tops: float | None = Field(None, ge=0)

    model_config = {"extra": "forbid"}


class KPUThermalProfile(BaseModel):
    """One operating point: clock, TDP, and the cooling solution it assumes."""

    name: str = Field(..., description="Profile label, e.g., '15W', '30W'")
    tdp_watts: float = Field(..., gt=0)
    clock_mhz: float = Field(..., gt=0)
    cooling_solution_id: str = Field(
        ..., description="References data/cooling-solutions/<id>.yaml"
    )

    # Optional per-precision calibration data. Both maps key on lowercase
    # precision names ('int4', 'int8', 'bf16', 'fp32', etc.) and carry
    # values in [0, 1]. When absent, downstream consumers (the graphs
    # KPU YAML loader, validator framework) fall back to flat
    # placeholders (~0.70 efficiency, ~0.95 utilization) -- those are
    # safe defaults; populate when measured calibration data exists.
    efficiency_factor_by_precision: dict[str, float] | None = Field(
        None,
        description=(
            "Per-precision combined efficiency factor (measured / sustained). "
            "Range [0, 1]; e.g., 0.68 means 68%% of sustained throughput "
            "is achieved on real workloads. Profile-specific because DVFS "
            "throttling, memory contention, and thermal margins shift "
            "with TDP."
        ),
    )
    tile_utilization_by_precision: dict[str, float] | None = Field(
        None,
        description=(
            "Per-precision fraction of tiles actively scheduled for the "
            "precision's primary workload. Range [0, 1]. Lower than 1.0 "
            "when some tile classes are idle during precision-specific "
            "execution (e.g., FP32 only uses BF16-primary tiles)."
        ),
    )
    activity_factor: float | None = Field(
        None, gt=0,
        description=(
            "Multiplier on the WorkloadAssumption's compute_duty_cycle "
            "for THIS profile only. Lets the architect tune a single "
            "profile (typically the lower-power ones) without changing "
            "the chip-wide workload model. Default None = use the "
            "workload duty cycle as-is. Example: profile '15W' with "
            "activity_factor=0.5 dissipates half the dynamic power of "
            "the same clock at activity_factor=1.0."
        ),
    )
    vdd_v: float | None = Field(
        None, gt=0,
        description=(
            "Core supply voltage for THIS operating point in volts. "
            "Default None = use ProcessNode.nominal_vdd_v. Dynamic power "
            "scales by (vdd_v / nominal_vdd_v)^2 -- so lower-power "
            "profiles drop both clock AND voltage (Orin-style DVFS), "
            "and the (V^2 * f) product is what spreads TDP across "
            "profiles. Typical range at 16nm FinFET: 0.55-0.95 V."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "KPUThermalProfile":
        """Ensure efficiency / utilization values are in [0, 1] when set."""
        for attr_name, label in (
            ("efficiency_factor_by_precision", "efficiency_factor"),
            ("tile_utilization_by_precision", "tile_utilization"),
        ):
            mapping = getattr(self, attr_name)
            if mapping is None:
                continue
            for precision, value in mapping.items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{attr_name}[{precision!r}] = {value} is outside "
                        f"[0, 1]; {label} is a unit fraction."
                    )
        return self


class KPUPowerSpec(BaseModel):
    """Power envelope rolled up across thermal profiles."""

    tdp_watts: float = Field(
        ..., gt=0, description="Default thermal-profile TDP"
    )
    max_power_watts: float = Field(..., gt=0)
    min_power_watts: float = Field(..., gt=0)
    idle_power_watts: float | None = Field(None, ge=0)
    default_thermal_profile: str = Field(
        ..., description="Name of the default profile in thermal_profiles"
    )
    thermal_profiles: list[KPUThermalProfile] = Field(..., min_length=1)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_default_profile_name(self) -> "KPUPowerSpec":
        """Ensure ``default_thermal_profile`` names a real profile.

        Catches typos at YAML load time -- otherwise downstream
        consumers (the SKU generator, validator framework, KPU YAML
        loader) hit a confusing KeyError / Optional unwrap later.
        """
        names = [p.name for p in self.thermal_profiles]
        if self.default_thermal_profile not in names:
            raise ValueError(
                f"default_thermal_profile={self.default_thermal_profile!r} "
                f"is not in thermal_profiles "
                f"(available: {names})"
            )
        return self


class KPUMarket(BaseModel):
    launch_date: str | None = Field(None)
    launch_msrp_usd: float | None = Field(None)
    target_market: str = Field(..., description="edge / embodied / datacenter")
    product_family: str = Field("Stillwater KPU")
    model_tier: str = Field(..., description="entry / mid / high / datacenter")
    is_available: bool = Field(False)
    is_discontinued: bool = Field(False)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

class KPUEntry(BaseModel):
    """Complete KPU SKU catalog entry.

    Lives under ``data/kpus/<vendor>/<id>.yaml``. References a ProcessNode
    by ``process_node_id`` and one CoolingSolution per thermal profile by
    ``cooling_solution_id``. The ComputeSolution = ProcessNode +
    CoolingSolution + KPUEntry.
    """

    id: str = Field(..., description="Unique id, e.g., 'stillwater_kpu_t256'")
    name: str = Field(..., description="Human-readable name")
    vendor: str = Field(..., description="Vendor, e.g., 'stillwater'")

    process_node_id: str = Field(
        ...,
        description=(
            "References data/process-nodes/<foundry>/<node>.yaml. The "
            "validator and generator look up density / energy / leakage "
            "from this ProcessNodeEntry."
        ),
    )

    die: KPUDieSpec = Field(..., description="Roll-up die-level spec")
    kpu_architecture: KPUArchitecture = Field(
        ..., description="Architectural topology (tiles, NoC, memory)"
    )
    silicon_bin: KPUSiliconBin = Field(
        ..., description="Per-block transistor decomposition"
    )
    clocks: KPUClocks = Field(...)
    performance: KPUTheoreticalPerformance = Field(
        ..., description="Roll-up peak performance"
    )
    power: KPUPowerSpec = Field(...)
    market: KPUMarket = Field(...)

    notes: str = Field("", description="Additional notes")
    datasheet_url: str | None = Field(None)
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}
