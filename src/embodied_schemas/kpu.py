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
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    SerializerFunctionWrapHandler,
    Tag,
    model_serializer,
    model_validator,
)

from embodied_schemas.datapath import FunctionalUnit, OpKind, PEDatapath
from embodied_schemas.function_core import _FUNCTION_ID_RE, FunctionCore
from embodied_schemas.local_memory import (  # noqa: F401  (re-exported)
    LocalMemory,
    LocalMemoryLevel,
    LocalMemoryScope,
    duplicate_levels,
)
from embodied_schemas.overlay import (
    FabricInterconnect,
    FabricOverlayKind,
    NoCOverlay,
    NoCOverlayKind,
    OverlayScope,
)
from embodied_schemas.power_domain import (
    DomainOperatingPoint,
    PowerDomain,
    PowerDomainKind,
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

    - ``pe_fabric`` (``KPUTileSpec``): a programmable domain-flow PE array
      with a declared per-PE datapath. All catalog SKUs today.
    - ``systolic`` (``SystolicTile``): a fixed-schedule GEMM / conv array.
    - ``fixed_function`` (``FixedFunctionTile``): an encapsulated compute
      segment such as an ISP or a VIO pipeline (a ``FunctionCore``).
    - ``scalar``, ``io_bridge``: reserved; no schema yet, so a tile with one
      of these kinds is rejected.
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


# Serialized key order of every tile kind: identity first, kind-specific
# fields next, the shared optional fields last. This keeps a pe_fabric tile's
# key order exactly as it was before the base class existed.
_TILE_DUMP_HEAD = ("tile_kind", "tile_type", "tile_class_id", "num_tiles")
_TILE_DUMP_TAIL = (
    "notes", "datapath", "footprint", "local_memory", "power_domain_id",
    "placement", "interconnect", "tile_class_ref",
)


class KPUTileBase(BaseModel):
    """Fields every KPU tile kind shares (graphs#268 Phase B3).

    Every tile class occupies ``num_tiles * sites_per_tile`` compute sites of
    the KPU checkerboard, whatever it is inside. ``tile_class_id`` is the
    stable id that power domains, placement hints and NoC overlays refer to.
    Subclasses pin ``tile_kind`` to a Literal and add their own fields.
    """

    tile_kind: KPUTileKind
    tile_type: str = Field(
        ..., description="Human-readable label, e.g., 'INT8-primary', 'Matrix', 'ISP'"
    )
    tile_class_id: str = Field(
        ...,
        pattern=_TILE_CLASS_ID_RE,
        description="Stable id for references (power domains, placement, "
        "silicon). Defaults to a slug of tile_type: 'INT8-primary' -> 'int8_primary'",
    )
    num_tiles: int = Field(..., gt=0, description="Number of tiles of this class")
    notes: str = Field("", description="Additional notes")
    footprint: TileFootprint | None = Field(
        None, description="Checkerboard sites occupied; None = 1x1"
    )
    local_memory: list[LocalMemory] | None = Field(
        None,
        description="Tile-class-local memories. For pe_fabric tiles None means "
        "the chip-level KPUMemorySubsystem L1/L2 figures apply",
    )
    power_domain_id: str | None = Field(
        None, description="Power domain this tile class belongs to (Phase B4)"
    )
    placement: TilePlacement | None = Field(None, description="Floorplan hints")
    # --- Phase B6 (graphs#268) --------------------------------------------
    tile_class_ref: str | None = Field(
        None,
        pattern=_TILE_CLASS_ID_RE,
        description="Tile-class library entry this tile was resolved from "
        "(informational: the tile is self-contained, so nothing needs the library)",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _default_tile_class_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "tile_class_id" not in data and "tile_type" in data:
            data = {**data, "tile_class_id": tile_class_slug(str(data["tile_type"]))}
        return data

    @model_validator(mode="after")
    def _check_local_memory(self) -> "KPUTileBase":
        dup = duplicate_levels(self.local_memory)
        if dup:
            raise ValueError(f"tile {self.tile_type!r}: duplicate local_memory levels {dup}")
        return self

    # Deliberately no return annotation: an annotation would replace the
    # serialization-mode JSON schema (see compute_product.KPUBlock).
    @model_serializer(mode="wrap")
    def _serialize_in_tile_order(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if not isinstance(data, dict):
            return data
        head = [k for k in _TILE_DUMP_HEAD if k in data]
        tail = [k for k in _TILE_DUMP_TAIL if k in data]
        middle = [k for k in data if k not in head and k not in tail]
        return {k: data[k] for k in head + middle + tail}

    @property
    def sites_per_tile(self) -> int:
        """Checkerboard compute sites one tile of this class occupies."""
        return self.footprint.sites if self.footprint is not None else 1

    @property
    def total_sites(self) -> int:
        return self.num_tiles * self.sites_per_tile


class KPUTileSpec(KPUTileBase):
    """A ``pe_fabric`` tile: a programmable domain-flow PE array.

    A KPU is built from a small number of tile classes (typically
    INT8-primary, BF16-primary, Matrix), each with its own PE array,
    standard-cell library, and ops/clock profile. The ``num_tiles`` for
    each class is the architectural mix.

    Every catalog tile is a ``pe_fabric`` tile. The optional fields (B1, B2)
    are backward compatible: ``tile_kind`` and ``tile_class_id`` are filled
    in when absent. When a ``datapath`` is declared, the ops it implies must
    equal ``ops_per_tile_per_clock``. Also available as ``PEFabricTile``.
    """

    tile_kind: Literal[KPUTileKind.PE_FABRIC] = Field(
        KPUTileKind.PE_FABRIC,
        description="Tile kind; pe_fabric is the only kind this class describes",
    )
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
    # --- Phase B1 (graphs#268) --------------------------------------------
    datapath: PEDatapath | None = Field(
        None,
        description="Per-PE datapath. When set, rows * cols * its ops per PE "
        "must equal ops_per_tile_per_clock for every precision",
    )
    # --- Phase B2 (graphs#268) --------------------------------------------
    interconnect: FabricInterconnect | None = Field(
        None,
        description="PE-to-PE links and overlays inside the tile; None = "
        "nearest-neighbor mesh with no declared overlays",
    )

    @model_validator(mode="after")
    def _check_datapath(self) -> "KPUTileSpec":
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
            # vice versa. A per-tile overlay spans both axes, so its span must
            # fit the shorter one.
            axes = {
                OverlayScope.ROW: [cols],
                OverlayScope.COL: [rows],
                OverlayScope.TILE: [rows, cols],
            }[ov.instances_per]
            max_span = min(axes) - 1
            if ov.span is not None and ov.span > max_span:
                raise ValueError(
                    f"{where}: span {ov.span} does not fit a {rows}x{cols} PE array "
                    f"(max {max_span} along its axis)"
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

    def ops_per_pe_per_clock(self) -> dict[str, float] | None:
        """``"<op>:<format>"`` ops per PE per clock from the datapath, or None
        when no datapath is declared."""
        return self.datapath.ops_per_pe_per_clock() if self.datapath is not None else None


PEFabricTile = KPUTileSpec
"""Alias: the ``pe_fabric`` tile kind."""


class SystolicKernel(str, Enum):
    """Kernels a fixed-schedule systolic array executes."""

    GEMM = "gemm"
    CONV2D = "conv2d"
    DEPTHWISE_CONV2D = "depthwise_conv2d"
    ATTENTION = "attention"


class SystolicTile(KPUTileBase):
    """A ``systolic`` tile: a fixed-schedule GEMM / conv array.

    It is not programmable beyond its kernels. Every cell carries the same
    MAC / FMA unit (``mac``, with one mode per number format), which is why it
    is denser and cheaper per op than a pe_fabric tile. Ops per clock are
    derived: ``array_rows * array_cols`` x the unit's ops per clock in each
    mode.
    """

    tile_kind: Literal[KPUTileKind.SYSTOLIC] = KPUTileKind.SYSTOLIC
    array_rows: int = Field(..., gt=0)
    array_cols: int = Field(..., gt=0)
    dataflow: KPUTileScheduleClass = Field(
        KPUTileScheduleClass.WEIGHT_STATIONARY, description="Stationarity of the array"
    )
    circuit_class: CircuitClass = Field(..., description="Library the array is built in")
    mac: FunctionalUnit = Field(..., description="The per-cell MAC / FMA unit")
    supported_kernels: list[SystolicKernel] = Field(
        default_factory=lambda: [SystolicKernel.GEMM, SystolicKernel.CONV2D], min_length=1
    )

    @model_validator(mode="after")
    def _check_array(self) -> "SystolicTile":
        if self.dataflow == KPUTileScheduleClass.NO_LOCAL_REUSE:
            raise ValueError(f"systolic tile {self.tile_type!r}: dataflow cannot be no_local_reuse")
        if self.mac.op not in (OpKind.MAC, OpKind.FMA):
            raise ValueError(
                f"systolic tile {self.tile_type!r}: the cell unit must be a mac or fma, "
                f"got {self.mac.op.value}"
            )
        return self

    @property
    def pes_per_tile(self) -> int:
        return self.array_rows * self.array_cols

    @property
    def total_pes(self) -> int:
        return self.num_tiles * self.pes_per_tile

    @property
    def ops_per_tile_per_clock(self) -> dict[str, float]:
        """Peak ops per tile per clock, by number format (derived)."""
        opi = self.mac.resolved_ops_per_invocation
        return {
            m.operand_format: self.pes_per_tile * m.lanes * opi / m.issue_interval_cycles
            for m in self.mac.modes
        }

    @property
    def pipeline_fill_cycles(self) -> int:
        """Catalog convention: fill / drain of an R x C array = max(R, C)."""
        return max(self.array_rows, self.array_cols)

    @property
    def pipeline_drain_cycles(self) -> int:
        return max(self.array_rows, self.array_cols)


class FixedFunctionTile(KPUTileBase):
    """A ``fixed_function`` tile: one encapsulated compute segment.

    The core (``FunctionCore``) carries the function contract, throughput in
    work units, energy per unit at a reference node, numeric formats, its own
    local memory and silicon. A fixed-function tile has no programmable ops:
    ``ops_per_tile_per_clock`` is empty, so a fixed-function tile never adds
    to programmable TOPS.
    """

    tile_kind: Literal[KPUTileKind.FIXED_FUNCTION] = KPUTileKind.FIXED_FUNCTION
    core: FunctionCore

    @model_validator(mode="after")
    def _memory_lives_in_the_core(self) -> "FixedFunctionTile":
        if self.local_memory is not None:
            raise ValueError(
                f"fixed-function tile {self.tile_type!r}: declare memory in "
                f"core.local_memory, not the tile's local_memory"
            )
        return self

    @property
    def ops_per_tile_per_clock(self) -> dict[str, float]:
        return {}

    @property
    def function_id(self) -> str:
        return self.core.function_id

    @property
    def units_per_clock(self) -> float:
        """Work units per clock of one tile (``core.throughput.unit``)."""
        return self.core.units_per_clock


def _tile_kind_of(value: Any) -> str:
    """Union tag of a tile: its ``tile_kind``, defaulting to pe_fabric (every
    catalog YAML predates the field)."""
    kind = value.get("tile_kind", KPUTileKind.PE_FABRIC) if isinstance(value, dict) else (
        getattr(value, "tile_kind", KPUTileKind.PE_FABRIC)
    )
    return kind.value if isinstance(kind, Enum) else str(kind)


AnyKPUTile = Annotated[
    Union[
        Annotated[KPUTileSpec, Tag(KPUTileKind.PE_FABRIC.value)],
        Annotated[SystolicTile, Tag(KPUTileKind.SYSTOLIC.value)],
        Annotated[FixedFunctionTile, Tag(KPUTileKind.FIXED_FUNCTION.value)],
    ],
    Discriminator(_tile_kind_of),
]
"""Any KPU tile kind, discriminated on ``tile_kind`` (missing = pe_fabric)."""


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
    # The KPU hierarchy is a streaming pipeline, not a cache hierarchy.
    # Each level has a distinct job, and the level a figure belongs to
    # decides what it is multiplied by -- which is how l1 came to be
    # declared per PE and imply 835% of a die in SRAM (graphs#268 F1).
    l3_kib_per_tile: int = Field(
        ...,
        gt=0,
        description=(
            "Distributed L3 SRAM per tile (KiB): the block linear-algebra "
            "reuse scratchpad. Allocated to its own memory tile."
        ),
    )
    l2_kib_per_tile: int = Field(
        0,
        ge=0,
        description=(
            "Per-tile L2 SRAM (KiB); 0 if absent. Reformats L3 blocks so "
            "that streaming preparation is easy at fabric clock rates. Its "
            "placement is a design choice: it can sit in the L3 memory tile "
            "or in the compute tile, depending on the concurrent bank "
            "layout needed to feed L1."
        ),
    )
    l1_kib_per_tile: int = Field(
        0,
        ge=0,
        description=(
            "Per-compute-tile L1 SRAM (KiB); 0 if absent. Tightly coupled "
            "to the fabric edges: it turns row and column fetches out of L2 "
            "into streams of operands pushed into the edges of the fabric. "
            "Per TILE, never per PE -- a PE is an ALU with datapath "
            "registers and a small token CAM, and holds no SRAM."
        ),
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Checkerboard (graphs#268 Phase B4)
# ---------------------------------------------------------------------------

SPARE_SITE = "."
"""``placement_map`` glyph for a spare (empty / whitespace) compute site."""


class CheckerboardPlacement(str, Enum):
    """How tile classes are assigned to compute sites."""

    AUTO = "auto"          # the floorplan placer decides (graphs Phase D)
    EXPLICIT = "explicit"  # ``placement_map`` fixes every site


class SiteGrid(BaseModel):
    """Dimensions of the compute-site grid (rows x cols)."""

    rows: int = Field(..., gt=0)
    cols: int = Field(..., gt=0)

    model_config = {"extra": "forbid"}

    @property
    def sites(self) -> int:
        return self.rows * self.cols


class CheckerboardMemoryCell(BaseModel):
    """The memory cell paired 1:1 with every compute site.

    ``l3_kib`` restates ``KPUMemorySubsystem.l3_kib_per_tile`` and must match
    it.
    """

    l3_kib: int = Field(..., gt=0)
    circuit_class: CircuitClass = CircuitClass.SRAM_HD

    model_config = {"extra": "forbid"}


class CheckerboardSpec(BaseModel):
    """The KPU compute/memory checkerboard, made explicit.

    Every compute site is paired with one memory cell and one NoC router.
    A tile class occupies ``num_tiles * footprint.rows * footprint.cols``
    sites; the remaining sites are ``spare_sites``. The site-accounting
    invariant (checked by ``KPUArchitectureBase``)::

        sum(tile.num_tiles * tile.sites_per_tile) + spare_sites
            == compute_sites.rows * compute_sites.cols

    replaces the implicit ``noc.mesh_rows * noc.mesh_cols == total_tiles``
    that legacy SKUs rely on.

    With ``placement: explicit``, ``placement_map`` gives the tile class of
    every site (``"."`` = spare). A footprint is an axis-aligned
    ``rows x cols`` rectangle of sites, never rotated.
    """

    compute_sites: SiteGrid
    memory_cell: CheckerboardMemoryCell | None = Field(
        None, description="None = memory.l3_kib_per_tile in the default SRAM library"
    )
    placement: CheckerboardPlacement = CheckerboardPlacement.AUTO
    placement_map: list[list[str]] | None = Field(
        None,
        description="explicit placement only: one row per site row, one "
        "tile_class_id (or '.' for a spare site) per site",
    )
    spare_sites: int = Field(0, ge=0, description="Compute sites no tile occupies")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_map_shape(self) -> "CheckerboardSpec":
        grid, pmap = self.compute_sites, self.placement_map
        if self.placement == CheckerboardPlacement.EXPLICIT and pmap is None:
            raise ValueError("checkerboard: placement 'explicit' needs a placement_map")
        if self.placement == CheckerboardPlacement.AUTO and pmap is not None:
            raise ValueError("checkerboard: placement_map requires placement 'explicit'")
        if self.spare_sites > grid.sites:
            raise ValueError(
                f"checkerboard: spare_sites {self.spare_sites} exceeds the "
                f"{grid.rows}x{grid.cols} grid"
            )
        if pmap is None:
            return self
        if len(pmap) != grid.rows or any(len(row) != grid.cols for row in pmap):
            shape = f"{len(pmap)} rows of lengths {sorted({len(r) for r in pmap})}"
            raise ValueError(
                f"checkerboard: placement_map must be {grid.rows}x{grid.cols}, got {shape}"
            )
        glyphs = {c for row in pmap for c in row if c != SPARE_SITE}
        bad = sorted(c for c in glyphs if not re.fullmatch(_TILE_CLASS_ID_RE, c))
        if bad:
            raise ValueError(
                f"checkerboard: placement_map entries {bad} are neither a tile_class_id nor '.'"
            )
        spares = sum(row.count(SPARE_SITE) for row in pmap)
        if spares != self.spare_sites:
            raise ValueError(
                f"checkerboard: placement_map has {spares} spare sites but "
                f"spare_sites is {self.spare_sites}"
            )
        return self


def _check_placement_map(pmap: list[list[str]], tiles: list) -> None:
    """Every tile class's sites in ``pmap`` must decompose into exactly
    ``num_tiles`` footprint rectangles.

    Scanning row-major, the first unclaimed site of a class must be the
    top-left corner of one of its footprints (any footprint covering it
    would otherwise cover an earlier, already-claimed site), so the greedy
    claim below is exact for a single fixed-orientation rectangle per class.
    """
    by_id = {t.tile_class_id: t for t in tiles}
    unknown = sorted({c for row in pmap for c in row if c != SPARE_SITE} - set(by_id))
    if unknown:
        raise ValueError(
            f"checkerboard: placement_map references unknown tile_class_id "
            f"{unknown} (known: {sorted(by_id)})"
        )
    rows, cols = len(pmap), len(pmap[0])
    claimed = [[False] * cols for _ in range(rows)]
    placed = {cid: 0 for cid in by_id}
    for r in range(rows):
        for c in range(cols):
            cid = pmap[r][c]
            if cid == SPARE_SITE or claimed[r][c]:
                continue
            fp = by_id[cid].footprint
            fr, fc = (fp.rows, fp.cols) if fp is not None else (1, 1)
            block = [(r + i, c + j) for i in range(fr) for j in range(fc)]
            if any(y >= rows or x >= cols or pmap[y][x] != cid or claimed[y][x] for y, x in block):
                raise ValueError(
                    f"checkerboard: tile class {cid!r} at site ({r}, {c}) does not "
                    f"form a whole {fr}x{fc} footprint"
                )
            for y, x in block:
                claimed[y][x] = True
            placed[cid] += 1
    wrong = {cid: n for cid, n in placed.items() if n != by_id[cid].num_tiles}
    if wrong:
        detail = ", ".join(
            f"{cid}: {n} placed, num_tiles {by_id[cid].num_tiles}"
            for cid, n in sorted(wrong.items())
        )
        raise ValueError(f"checkerboard: placement_map disagrees with num_tiles ({detail})")


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
    tiles: list[AnyKPUTile] = Field(
        ...,
        description="Per-tile-class specifications (heterogeneous tile mix): "
        "pe_fabric, systolic and fixed_function tiles",
    )
    noc: KPUNoCSpec = Field(..., description="Intra-die NoC topology")
    memory: KPUMemorySubsystem = Field(
        ..., description="Tile-local memory hierarchy (L1 per PE, L2/L3 per tile)"
    )
    multi_precision_alu: list[str] = Field(
        default_factory=list,
        description="Precisions supported chip-wide, e.g., ['int4','int8','bf16','fp32']",
    )
    # --- Phase B4 (graphs#268) --------------------------------------------
    checkerboard: CheckerboardSpec | None = Field(
        None,
        description="Explicit compute-site grid; None = the legacy implicit "
        "one-tile-per-site mesh (noc.mesh_rows x noc.mesh_cols)",
    )
    power_domains: list[PowerDomain] | None = Field(
        None,
        description="Power domains (cluster / tile_class / uncore); None = one "
        "chip-wide domain at the thermal profile's clock and Vdd",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_tile_class_references(self) -> "KPUArchitectureBase":
        ids = [t.tile_class_id for t in self.tiles]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate tile_class_id {dup}; each tile class needs a unique id")
        count = sum(t.num_tiles for t in self.tiles)
        if self.total_tiles != count:
            raise ValueError(
                f"total_tiles {self.total_tiles} != {count}, the sum of num_tiles over tile classes"
            )
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

    @model_validator(mode="after")
    def _check_checkerboard(self) -> "KPUArchitectureBase":
        cb = self.checkerboard
        if cb is None:
            return self
        grid = cb.compute_sites
        used = sum(t.total_sites for t in self.tiles)
        if used + cb.spare_sites != grid.sites:
            detail = " + ".join(
                f"{t.tile_class_id} {t.num_tiles}x{t.sites_per_tile}" for t in self.tiles
            )
            raise ValueError(
                f"checkerboard site accounting: tiles use {used} sites ({detail}) + "
                f"{cb.spare_sites} spare != {grid.rows}x{grid.cols} = {grid.sites} compute sites"
            )
        if (self.noc.mesh_rows, self.noc.mesh_cols) != (grid.rows, grid.cols):
            raise ValueError(
                f"checkerboard: noc mesh {self.noc.mesh_rows}x{self.noc.mesh_cols} must "
                f"match the {grid.rows}x{grid.cols} compute-site grid (one router per site)"
            )
        if cb.memory_cell is not None and cb.memory_cell.l3_kib != self.memory.l3_kib_per_tile:
            raise ValueError(
                f"checkerboard: memory_cell.l3_kib {cb.memory_cell.l3_kib} != "
                f"memory.l3_kib_per_tile {self.memory.l3_kib_per_tile}"
            )
        for t in self.tiles:
            fp = t.footprint
            if fp is not None and (fp.rows > grid.rows or fp.cols > grid.cols):
                raise ValueError(
                    f"tile class {t.tile_class_id!r}: {fp.rows}x{fp.cols} footprint does "
                    f"not fit the {grid.rows}x{grid.cols} compute-site grid"
                )
        if cb.placement_map is not None:
            _check_placement_map(cb.placement_map, self.tiles)
        return self

    @model_validator(mode="after")
    def _check_power_domains(self) -> "KPUArchitectureBase":
        domains = self.power_domains or []
        ids = [d.domain_id for d in domains]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate power domain_id {dup}")
        by_id = {d.domain_id: d for d in domains}
        known = {t.tile_class_id for t in self.tiles}
        owner: dict[str, str] = {}  # tile_class_id -> its tile_class domain
        cluster_sites: dict[tuple[int, int], str] = {}
        for d in domains:
            if d.kind == PowerDomainKind.UNCORE:
                continue
            unknown = [m for m in d.members if m not in known]
            if unknown:
                raise ValueError(
                    f"power domain {d.domain_id!r} references unknown tile_class_id "
                    f"{unknown} (known: {sorted(known)})"
                )
            if d.kind == PowerDomainKind.TILE_CLASS:
                for m in d.members:
                    if m in owner:
                        raise ValueError(
                            f"tile class {m!r} is in two tile_class power domains: "
                            f"{owner[m]!r} and {d.domain_id!r}"
                        )
                    owner[m] = d.domain_id
                continue
            # CLUSTER: the site ranges need the explicit grid.
            if self.checkerboard is None:
                raise ValueError(
                    f"power domain {d.domain_id!r}: a cluster domain's site_ranges "
                    f"need a checkerboard"
                )
            grid = self.checkerboard.compute_sites
            for rng in d.site_ranges:
                if not rng.fits(grid.rows, grid.cols):
                    raise ValueError(
                        f"power domain {d.domain_id!r}: site range rows "
                        f"{rng.row_min}..{rng.row_max}, cols {rng.col_min}..{rng.col_max} "
                        f"is outside the {grid.rows}x{grid.cols} compute-site grid"
                    )
            for site in sorted(d.sites()):
                if site in cluster_sites:
                    raise ValueError(
                        f"cluster power domains {cluster_sites[site]!r} and "
                        f"{d.domain_id!r} overlap at site {site}"
                    )
                cluster_sites[site] = d.domain_id
        for t in self.tiles:
            ref = t.power_domain_id
            if ref is None:
                continue
            if ref not in by_id:
                raise ValueError(
                    f"tile class {t.tile_class_id!r}: power_domain_id {ref!r} is not a "
                    f"defined power domain (defined: {sorted(by_id)})"
                )
            dom = by_id[ref]
            if dom.kind == PowerDomainKind.UNCORE:
                raise ValueError(
                    f"tile class {t.tile_class_id!r}: power_domain_id {ref!r} is an "
                    f"uncore domain"
                )
            if dom.kind == PowerDomainKind.TILE_CLASS and t.tile_class_id not in dom.members:
                raise ValueError(
                    f"tile class {t.tile_class_id!r}: power_domain_id {ref!r} does not "
                    f"list it in members"
                )
            if t.tile_class_id in owner and owner[t.tile_class_id] != ref:
                raise ValueError(
                    f"tile class {t.tile_class_id!r}: power_domain_id {ref!r} disagrees "
                    f"with tile_class domain {owner[t.tile_class_id]!r}, which lists it"
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


PROGRAMMABLE_TILE_KINDS = frozenset({KPUTileKind.PE_FABRIC, KPUTileKind.SYSTOLIC})
"""Tile kinds whose ops count toward peak (programmable) ops/s and TOPS."""

# (precision, legacy field, decimals): the legacy roll-up fields and the
# rounding the graphs SKU generator applies to them (T-ops/s).
_LEGACY_PERF_FIELDS = (
    ("int8", "int8_tops", 1),
    ("bf16", "bf16_tflops", 1),
    ("fp32", "fp32_tflops", 2),
    ("int4", "int4_tops", 1),
)


def _check_rate_map(name: str, rates: dict[str, float]) -> None:
    for key, value in rates.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name}[{key!r}] = {value} must be finite and >= 0")


class KPUTheoreticalPerformance(BaseModel):
    """Roll-up peak ops/s by precision (TOPS / TFLOPS).

    The legacy fields (``int8_tops`` ...) are T-ops/s of the programmable
    tile kinds (pe_fabric + systolic) at the default thermal profile's
    clock. The optional B5 fields (graphs#268) break the same roll-up down:

    - ``peak_ops_per_sec_by_precision``: ops/s by precision name, same name
      and meaning as ``compute_block_common.TheoreticalPerformance``.
    - ``by_tile_kind``: the same, per programmable tile kind.
    - ``fixed_function_throughput``: work units/s per ``function_id``.
      Fixed-function tiles have no programmable ops, so they never enter the
      ops/s fields or the legacy TOPS.

    When ``peak_ops_per_sec_by_precision`` is set, the legacy fields must
    equal it to their stated precision. ``derive_kpu_performance`` computes
    every field from the tiles; ``KPUEntry`` and ``ComputeProduct`` check a
    declared B5 roll-up against their architecture.
    """

    int8_tops: float = Field(..., ge=0)
    bf16_tflops: float = Field(..., ge=0)
    fp32_tflops: float = Field(..., ge=0)
    int4_tops: float | None = Field(None, ge=0)
    # --- Phase B5 (graphs#268) --------------------------------------------
    peak_ops_per_sec_by_precision: dict[str, float] | None = Field(
        None, description="Programmable peak ops/s by precision (pe_fabric + systolic)"
    )
    by_tile_kind: dict[KPUTileKind, dict[str, float]] | None = Field(
        None, description="Peak ops/s by precision, per programmable tile kind"
    )
    fixed_function_throughput: dict[str, float] | None = Field(
        None, description="Work units/s per fixed-function function_id"
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_rollup(self) -> "KPUTheoreticalPerformance":
        peak, kinds = self.peak_ops_per_sec_by_precision, self.by_tile_kind
        if peak is not None:
            _check_rate_map("peak_ops_per_sec_by_precision", peak)
        if self.fixed_function_throughput is not None:
            _check_rate_map("fixed_function_throughput", self.fixed_function_throughput)
            ff_ids = self.fixed_function_throughput
            bad = [k for k in ff_ids if not re.fullmatch(_FUNCTION_ID_RE, k)]
            if bad:
                raise ValueError(f"fixed_function_throughput keys {bad} are not function_ids")
        if kinds is not None:
            other = sorted(k.value for k in kinds if k not in PROGRAMMABLE_TILE_KINDS)
            if other:
                raise ValueError(
                    f"by_tile_kind covers programmable kinds only (pe_fabric, systolic), "
                    f"got {other}"
                )
            for kind, rates in kinds.items():
                _check_rate_map(f"by_tile_kind[{kind.value!r}]", rates)
            if peak is not None:
                summed: dict[str, float] = {}
                for rates in kinds.values():
                    for prec, value in rates.items():
                        summed[prec] = summed.get(prec, 0.0) + value
                if not _rates_close(summed, peak):
                    raise ValueError(
                        "peak_ops_per_sec_by_precision must be the sum of by_tile_kind "
                        f"over kinds: {peak} vs {summed}"
                    )
        if peak is not None:
            for prec, field, decimals in _LEGACY_PERF_FIELDS:
                expected = peak.get(prec, 0.0) / 1e12
                value = getattr(self, field)
                if value is None:
                    if expected > 0:
                        raise ValueError(
                            f"{field} is None but peak_ops_per_sec_by_precision gives "
                            f"{expected:g} T-ops/s"
                        )
                    continue
                if abs(value - expected) > 0.5 * 10**-decimals + 1e-9:
                    raise ValueError(
                        f"{field} = {value} disagrees with peak_ops_per_sec_by_precision"
                        f"[{prec!r}] = {expected:g} T-ops/s"
                    )
        return self

    @property
    def declares_rollup(self) -> bool:
        """Whether any B5 roll-up field is set."""
        return any(
            v is not None
            for v in (
                self.peak_ops_per_sec_by_precision,
                self.by_tile_kind,
                self.fixed_function_throughput,
            )
        )


def _rates_close(a: dict[str, float], b: dict[str, float]) -> bool:
    """Same keys and values within rel 1e-9 (keys whose value is 0 may be absent)."""
    keys = {k for k, v in a.items() if v} | {k for k, v in b.items() if v}
    return all(math.isclose(a.get(k, 0.0), b.get(k, 0.0), rel_tol=1e-9) for k in keys)


def derive_kpu_performance(tiles: list, clock_mhz: float) -> KPUTheoreticalPerformance:
    """The performance roll-up of ``tiles`` (any tile kinds) at ``clock_mhz``.

    - Programmable kinds (pe_fabric, systolic) contribute
      ``num_tiles * ops_per_tile_per_clock * clock`` to the ops/s fields.
    - Fixed-function tiles contribute ``num_tiles * units_per_clock *
      clock`` to ``fixed_function_throughput``; classes sharing a
      ``function_id`` add up.
    - The legacy fields use the graphs generator's rounding (int8 / bf16 /
      int4 to 0.1, fp32 to 0.01 T-ops/s; int4 None when zero). The catalog
      is generated at the default thermal profile's clock.
    """
    hz = clock_mhz * 1e6
    by_kind: dict[KPUTileKind, dict[str, float]] = {}
    ff: dict[str, float] = {}
    for t in tiles:
        if t.tile_kind == KPUTileKind.FIXED_FUNCTION:
            ff[t.function_id] = ff.get(t.function_id, 0.0) + t.num_tiles * t.units_per_clock * hz
            continue
        rates = by_kind.setdefault(t.tile_kind, {})
        for prec, ops in t.ops_per_tile_per_clock.items():
            rates[prec] = rates.get(prec, 0.0) + t.num_tiles * ops * hz
    peak: dict[str, float] = {}
    for rates in by_kind.values():
        for prec, value in rates.items():
            peak[prec] = peak.get(prec, 0.0) + value
    legacy: dict[str, float | None] = {}
    for prec, field, decimals in _LEGACY_PERF_FIELDS:
        tops = peak.get(prec, 0.0) / 1e12
        legacy[field] = round(tops, decimals)
    if not peak.get("int4"):
        legacy["int4_tops"] = None
    return KPUTheoreticalPerformance(
        **legacy,
        peak_ops_per_sec_by_precision=peak,
        by_tile_kind=by_kind,
        fixed_function_throughput=ff or None,
    )


def check_performance_rollup(
    perf: KPUTheoreticalPerformance, tiles: list, clock_mhz: float
) -> None:
    """A declared B5 roll-up must match the tiles at ``clock_mhz``.

    No-op for a legacy performance block (no B5 field set). Called by
    ``KPUEntry`` and ``compute_product.ComputeProduct`` with the default
    thermal profile's clock.
    """
    if not perf.declares_rollup:
        return
    derived = derive_kpu_performance(tiles, clock_mhz)
    if perf.peak_ops_per_sec_by_precision is not None and not _rates_close(
        perf.peak_ops_per_sec_by_precision, derived.peak_ops_per_sec_by_precision
    ):
        raise ValueError(
            f"performance.peak_ops_per_sec_by_precision {perf.peak_ops_per_sec_by_precision} "
            f"!= {derived.peak_ops_per_sec_by_precision} derived from the tiles at "
            f"{clock_mhz:g} MHz"
        )
    if perf.by_tile_kind is not None:
        kinds = set(perf.by_tile_kind) | set(derived.by_tile_kind)
        for kind in sorted(kinds, key=lambda k: k.value):
            declared = perf.by_tile_kind.get(kind, {})
            expected = derived.by_tile_kind.get(kind, {})
            if not _rates_close(declared, expected):
                raise ValueError(
                    f"performance.by_tile_kind[{kind.value!r}] {declared} != {expected} "
                    f"derived from the tiles at {clock_mhz:g} MHz"
                )
    if perf.fixed_function_throughput is not None and not _rates_close(
        perf.fixed_function_throughput, derived.fixed_function_throughput or {}
    ):
        raise ValueError(
            f"performance.fixed_function_throughput {perf.fixed_function_throughput} != "
            f"{derived.fixed_function_throughput or {}} derived from the tiles at "
            f"{clock_mhz:g} MHz"
        )


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
    # --- Phase B4 (graphs#268) --------------------------------------------
    domain_operating_points: dict[str, DomainOperatingPoint] | None = Field(
        None,
        description=(
            "Per power domain operating point, keyed by domain_id. A domain "
            "not listed runs at this profile's clock / Vdd / activity. None "
            "= every domain does (the legacy single-domain behavior)."
        ),
    )
    tdp_scenario: dict[str, float] | None = Field(
        None,
        description=(
            "Concurrency assumption the TDP is derived from: the activity "
            "in [0, 1] of every tile class, keyed by tile_class_id (all "
            "classes, so the scenario is explicit). None = the legacy TDP "
            "formula (every class active at the workload duty cycle)."
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
        for cls_id, activity in (self.tdp_scenario or {}).items():
            if not 0.0 <= activity <= 1.0:
                raise ValueError(
                    f"profile {self.name!r}: tdp_scenario[{cls_id!r}] = {activity} "
                    f"is outside [0, 1]"
                )
        return self


def check_profile_domain_references(
    profiles: list[KPUThermalProfile],
    domains: dict[str, PowerDomain],
    tile_class_ids: set[str],
) -> None:
    """Check thermal-profile references against the architecture.

    - Every ``domain_operating_points`` key must be a defined power domain,
      and only a gateable domain may be gated.
    - A ``tdp_scenario`` must list exactly the tile classes.

    Called where the profiles and the architecture meet: ``KPUEntry`` and
    ``compute_product.ComputeProduct``.
    """
    for p in profiles:
        for did, op in (p.domain_operating_points or {}).items():
            if did not in domains:
                raise ValueError(
                    f"profile {p.name!r}: domain_operating_points key {did!r} is not a "
                    f"defined power domain (defined: {sorted(domains)})"
                )
            if op.gated and not domains[did].gateable:
                raise ValueError(
                    f"profile {p.name!r}: power domain {did!r} is gated but not gateable"
                )
        if p.tdp_scenario is not None:
            unknown = sorted(set(p.tdp_scenario) - tile_class_ids)
            missing = sorted(tile_class_ids - set(p.tdp_scenario))
            if unknown or missing:
                raise ValueError(
                    f"profile {p.name!r}: tdp_scenario must list every tile class "
                    f"exactly (unknown: {unknown}, missing: {missing})"
                )


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

    @property
    def default_profile(self) -> KPUThermalProfile:
        """The profile named by ``default_thermal_profile``."""
        return next(p for p in self.thermal_profiles if p.name == self.default_thermal_profile)


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

    @model_validator(mode="after")
    def _check_profile_references(self) -> "KPUEntry":
        arch = self.kpu_architecture
        check_profile_domain_references(
            self.power.thermal_profiles,
            {d.domain_id: d for d in arch.power_domains or []},
            {t.tile_class_id for t in arch.tiles},
        )
        return self

    @model_validator(mode="after")
    def _check_performance_rollup(self) -> "KPUEntry":
        check_performance_rollup(
            self.performance, self.kpu_architecture.tiles, self.power.default_profile.clock_mhz
        )
        return self
