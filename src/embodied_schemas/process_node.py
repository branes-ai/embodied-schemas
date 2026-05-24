"""Process-node schemas: per-foundry, per-library silicon fabrication data.

Transistor density, energy per op, leakage, and electromigration limits all
vary per process node AND per circuit type / standard-cell library. A modern
PDK ships separate Mtx/mm^2 figures for HP-logic, balanced-logic, LP-logic,
SRAM-HD, SRAM-HC, dual-port SRAM, analog, and IO. A chip's effective density
is a weighted average over how much die area is each library type.

This module models that fact. ProcessNodeEntry is the catalog entry; a SKU
references it by id and decomposes its silicon into blocks tagged with
CircuitClass values.

The schema is shape-stable across confidence levels. Public-estimate entries
(THEORETICAL) and PDK-derived entries (CALIBRATED) populate the same fields
and differ only by the ``confidence`` and ``source`` metadata. When a PDK
arrives, the import tool emits a CALIBRATED ProcessNodeEntry YAML and every
consumer (generator, validator, resource models) sharpens automatically with
no code changes.

Design note: ``Confidence`` (HIGH/MEDIUM/LOW) in benchmarks.py is for
*measurement-result* confidence. ``DataConfidence`` (CALIBRATED / INTERPOLATED
/ THEORETICAL / UNKNOWN) lives here and is for *spec-data provenance*. They
are intentionally different vocabularies.
"""

from enum import Enum

from pydantic import BaseModel, Field

from embodied_schemas.gpu import Foundry


class DataConfidence(str, Enum):
    """Provenance confidence for spec data (distinct from benchmark Confidence).

    CALIBRATED: derived from a PDK or first-party measurement.
    INTERPOLATED: derived from neighbor-node data with a documented model.
    THEORETICAL: hand-authored from public estimates (Wikichip, ISSCC, etc.).
    UNKNOWN: best guess; should not gate any production decision.
    """

    CALIBRATED = "calibrated"
    INTERPOLATED = "interpolated"
    THEORETICAL = "theoretical"
    UNKNOWN = "unknown"


class TransistorTopology(str, Enum):
    """Underlying transistor structure of a process node.

    BULK_PLANAR: legacy bulk planar CMOS (>=28nm typical).
    FINFET: 3D fin transistors; mainstream from 22/16nm down to 5/4nm.
    FD_SOI: fully-depleted silicon-on-insulator; body-bias capable; lower
        peak density but wider DVFS range and ultra-low leakage modes.
    GAA: gate-all-around / nanosheet; emerging at 3nm and below.
    """

    BULK_PLANAR = "bulk_planar"
    FINFET = "finfet"
    FD_SOI = "fd_soi"
    GAA = "gaa"


class CircuitClass(str, Enum):
    """Library / circuit-type classification for silicon area decomposition.

    Each PDK ships separate density / energy / leakage figures per class.
    A SKU's silicon_bin tags every block with one of these so the validator
    and generator can do per-library math instead of one-size-fits-all
    estimates.

    Logic libraries (denser to sparser):
      LP_LOGIC:        low-power logic; small, slow, low leakage.
      ULL_LOGIC:       ultra-low-leakage logic; for retention / always-on.
      BALANCED_LOGIC:  mainstream library; default for compute datapaths.
      HP_LOGIC:        high-performance; larger cells, faster, higher leakage.

    SRAM libraries:
      SRAM_HD:  high-density single-port SRAM (caches, scratchpads).
      SRAM_HC:  high-current single-port (faster access, larger).
      SRAM_HP:  high-performance dual-port (register files, NoC buffers).

    Other:
      ANALOG:  analog blocks (memory PHYs, PLLs, SerDes).
      IO:      pad ring, level shifters.
      MIXED:   weighted average for blocks not decomposed further.
    """

    HP_LOGIC = "hp_logic"
    BALANCED_LOGIC = "balanced_logic"
    LP_LOGIC = "lp_logic"
    ULL_LOGIC = "ull_logic"
    SRAM_HD = "sram_hd"
    SRAM_HC = "sram_hc"
    SRAM_HP = "sram_hp"
    ANALOG = "analog"
    IO = "io"
    MIXED = "mixed"


class LibraryDensity(BaseModel):
    """Per-library transistor density at one process node.

    Values are PDK-provided or public-estimate. Each entry carries its own
    confidence and source so a partially-CALIBRATED node (some libraries
    measured, some estimated) reports honestly.
    """

    circuit_class: CircuitClass = Field(..., description="Library classification")
    mtx_per_mm2: float = Field(
        ..., gt=0, description="Transistor density in millions per mm^2"
    )
    library_name: str | None = Field(
        None,
        description="Foundry's library name, e.g., '9T-HD', 'HPC+', 'ULL-LVT'",
    )
    confidence: DataConfidence = Field(
        ..., description="Provenance confidence for this density figure"
    )
    source: str = Field(
        ...,
        description="Citation: 'TSMC ISSCC 2018', 'PDK rev 2024Q1', 'Wikichip'",
    )
    notes: str | None = Field(None, description="Additional caveats or context")

    model_config = {"extra": "forbid"}


class ProcessNodeEntry(BaseModel):
    """One process-node catalog entry.

    Holds everything a SKU author needs to do area / energy / leakage / EM
    math at a chosen process node, decomposed by standard-cell library.
    Shape stable across confidence levels: PDK-derived entries populate the
    same fields as public-estimate entries.

    Lookup:
      - density_for(class) -> LibraryDensity
      - supports(class) -> bool
      - composite_density(area_by_class) -> float (area-weighted)
    """

    # Identity
    id: str = Field(
        ...,
        description="Unique id: '{foundry}_{node}'. Examples: 'tsmc_n16', 'gf_12fdx'.",
    )
    foundry: Foundry = Field(..., description="Foundry that fabricates this node")
    node_name: str = Field(
        ..., description="Foundry's marketing name, e.g., 'N16', '12FDX', '8LPP'"
    )
    node_nm: int = Field(
        ..., gt=0, description="Marketing-rounded process node in nanometers"
    )

    # Topology
    transistor_topology: TransistorTopology = Field(
        ..., description="Underlying transistor structure (FinFET, FD-SOI, GAA, etc.)"
    )
    body_bias_supported: bool = Field(
        False,
        description=(
            "Whether the node supports back-gate body bias (FD-SOI feature). "
            "Body bias adds Vt range -- enables wider DVFS swings and ULL modes."
        ),
    )
    back_bias_range_mv: tuple[int, int] | None = Field(
        None,
        description="Back-bias voltage range (forward, reverse) in mV (FD-SOI only)",
    )
    vt_options: list[str] = Field(
        default_factory=list,
        description="Threshold-voltage options offered, e.g., ['LVT','SVT','HVT','RVT']",
    )
    nominal_vdd_v: float = Field(
        ..., gt=0, description="Nominal core supply voltage in volts"
    )

    # Per-library data
    densities: dict[CircuitClass, LibraryDensity] = Field(
        ...,
        description=(
            "Map of CircuitClass to its LibraryDensity. Only populate the "
            "classes the foundry actually offers at this node."
        ),
    )
    leakage_w_per_mm2: dict[CircuitClass, float] = Field(
        default_factory=dict,
        description="Per-library leakage power density (W/mm^2) at nominal Vdd / Tj",
    )
    leakage_vdd_exponent: float | None = Field(
        None,
        ge=0,
        description=(
            "Leakage power scales as (Vdd / nominal_vdd_v) ** leakage_vdd_exponent "
            "when the operating Vdd departs from nominal. leakage_w_per_mm2 is the "
            "value AT nominal Vdd; consumers (e.g. the KPU power model) apply this "
            "exponent to derive leakage at lower-power DVFS profiles. Sub-threshold "
            "and gate leakage both fall steeply with Vdd; the effective power "
            "exponent for FinFET is typically ~3-5 (HVT-heavy designs ~6-8). "
            "Default None = leakage held flat across Vdd (legacy behavior)."
        ),
    )
    energy_per_op_pj: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Energy per operation in pJ, keyed by '<circuit_class>:<precision>'. "
            "Examples: 'balanced_logic:int8', 'hp_logic:bf16'."
        ),
    )
    sram_access_pj_per_byte: dict[CircuitClass, float] = Field(
        default_factory=dict,
        description=(
            "Per-byte SRAM access energy by library (sram_hd, sram_hc, sram_hp). "
            "Representative figure for ~32-256 KiB caches at this node. "
            "Used by the KPU power model to derive L2/L3 access power from "
            "sustained byte-rate. L1 access energy is rolled into "
            "energy_per_op_pj for the PE library."
        ),
    )
    dram_io_pj_per_byte: float | None = Field(
        None, ge=0,
        description=(
            "PHY-side DRAM I/O energy in pJ per byte transferred at the chip "
            "package boundary. Excludes the DRAM die's internal energy. "
            "Typical ranges: LPDDR5 ~5-10 pJ/byte, HBM3 ~3-5, DDR5 ~6-10."
        ),
    )
    noc_pj_per_flit_per_hop: dict[CircuitClass, float] = Field(
        default_factory=dict,
        description=(
            "Per-flit-per-hop NoC traversal energy by router library. Typical "
            "16-byte mesh router at N16: ~1.0 pJ/flit/hop on balanced_logic, "
            "~1.5 on hp_logic. Used by the KPU power model to derive on-chip "
            "communication power from estimated cross-tile traffic."
        ),
    )

    # Reliability
    em_j_max_by_temp_c: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Electromigration current-density ceiling (A/cm^2) keyed by junction "
            "temperature in C. Used by the EM validator."
        ),
    )
    routing_metal_width_um: dict[CircuitClass, float] = Field(
        default_factory=dict,
        description="Typical local-routing metal width (um) per library, for EM math",
    )

    # Geometry hints (advisory; consumed by the future floorplanner)
    m0_pitch_nm: int | None = Field(None, description="M0 metal pitch in nm")
    m1_pitch_nm: int | None = Field(None, description="M1 metal pitch in nm")

    # Cooling compatibility (advisory)
    cooling_compatible: list[str] = Field(
        default_factory=list,
        description="CoolingSolution ids commonly paired with this node (advisory)",
    )

    # Provenance
    source: str = Field(
        ..., description="Top-level citation for the entry as a whole"
    )
    confidence: DataConfidence = Field(
        ..., description="Roll-up confidence; weakest of any library entry"
    )
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")
    notes: str = Field("", description="Additional notes")

    model_config = {"extra": "forbid"}

    # ---------------------------- Lookups ----------------------------

    def density_for(self, circuit_class: CircuitClass) -> LibraryDensity:
        """Return the LibraryDensity for ``circuit_class``.

        Raises KeyError if the node doesn't offer that library -- callers
        should check ``supports()`` first or catch the error and surface it
        as a Finding from the block_library_validity validator.
        """
        return self.densities[circuit_class]

    def supports(self, circuit_class: CircuitClass) -> bool:
        """True if this node has a library entry for ``circuit_class``."""
        return circuit_class in self.densities

    def composite_density(
        self, area_mm2_by_class: dict[CircuitClass, float]
    ) -> float:
        """Area-weighted composite density across the supplied block areas.

        Used by the composite_density_envelope validator: a real chip's
        density should fall inside the envelope spanned by its constituent
        libraries' densities.
        """
        total_area = sum(area_mm2_by_class.values())
        if total_area <= 0:
            return 0.0
        total_mtx = sum(
            area * self.densities[c].mtx_per_mm2
            for c, area in area_mm2_by_class.items()
            if c in self.densities
        )
        return total_mtx / total_area

    def density_envelope(self) -> tuple[float, float]:
        """(min, max) density across all libraries this node supports.

        Useful for the composite_density_envelope validator and for
        inspection CLIs that report the node's range.
        """
        if not self.densities:
            return (0.0, 0.0)
        values = [d.mtx_per_mm2 for d in self.densities.values()]
        return (min(values), max(values))
