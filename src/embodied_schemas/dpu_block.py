"""DPU compute block for ``ComputeProduct`` (v6 schema, additive).

PR 2 of the DPU mini-sprint scoped at ``graphs#200``. Adds the sixth
member of the ``Block`` discriminated union after KPU (v1), GPU (v2),
CPU (v3), NPU (v4), and CGRA (v5). Modeled directly off the field set
audited in ``graphs/docs/designs/dpu-compute-product-schema-extension.md``.

Design choice: same per-architecture-types rule the prior sprints
established -- ship DPU-specific sub-types (``DPUComputeFabric``,
``DPUMemorySubsystem``, ``DPUOnDieFabric``, ``DPUThermalProfile``,
``DPUTheoreticalPerformance``) rather than generalize. With 6
architectures and 3+ shared primitives each, the v7 sprint should
finally land the vendor-neutral ``compute_block_common`` unification.

**Fourth cross-block-kind type reuse**: ``DPUOnDieFabric.confidence``
reuses ``DataConfidence`` from ``process_node`` (same as NPU/CGRA);
``DPUMemorySubsystem.external_dram_type`` reuses ``MemoryType`` from
``gpu`` (same as NPU; CGRA used ``host_dram_type`` with the same
underlying enum but different name); ``DPUComputeFabric.circuit_class``
reuses ``CircuitClass`` from ``process_node`` (same as GPU/CPU/NPU/CGRA).

Xilinx Vitis AI on Versal VE2302 (B4096 config) reference SKU
specifics that shaped the design:

  - **FPGA-based reconfigurable architecture** -- first FPGA in the
    catalog. The AIE tiles are hardened (not LUT-based), but the
    surrounding interconnect uses FPGA programmable routing. The
    ``fpga_fabric_overhead_factor`` field on ``DPUComputeFabric``
    models the ~20-30% energy penalty vs equivalent ASIC.
  - **AIE tile hierarchy** -- 64 AIE tiles in B4096 config (64 MACs/
    tile = 4096 total). Modeled as ``num_aie_tiles`` + ``macs_per_tile``
    on ``DPUBlock``. Future SKUs may use heterogeneous tile types
    (AMD Versal Premium with AIE + DSP58 mix); v7 unification.
  - **Static FPGA reconfiguration** -- the DPU bitstream is baked
    into the FPGA at deployment time. Cost is deployment-only
    (~1-3 seconds), not runtime. Modeled as
    ``DPUBlock.is_statically_reconfigurable: bool = True`` +
    ``bitstream_load_time_ms: Optional[float]``. Different semantics
    from CGRA's runtime ``reconfig_overhead_cycles``.
  - **Multi-precision INT-dominant + native FP16 + emulated FP32** --
    Vitis AI B4096: 128 INT8 ops/tile/clock (native) + 32 FP16 ops/
    tile/clock (native AIE-ML) + emulated FP32. Distinguishes from
    INT-only NPUs and FP-emulation-only CGRAs.
  - **Chip-attached DDR4** (NOT host-bus like CGRA's Plasticine).
    Versal VE2302 has on-die DDR4 controllers. Modeled via
    ``has_external_dram`` (NPU-style) rather than ``has_host_dram``
    (CGRA-style); v7 unification can resolve the naming.
  - **Single 20W thermal profile, no DVFS** -- Versal VE2302 ships
    with active-fan cooling. ``DPUThermalProfile`` uses scalar
    ``clock_mhz`` + ``dvfs_enabled`` flag (defaulting False).
  - **AIE-ML v1/v2/HD variants** -- discriminated via
    ``DPUFabricKind`` enum so future Versal Premium / datacenter
    SKUs land as pure YAML additions.

Future-deferred (v7+):

  - Other DPU architectures (AMD Versal AI Core / Premium with
    AIE-ML v2, Intel/Altera AgileX AI). Pure additive; defer to
    YAML PRs once this schema lands.
  - FPGA partial reconfiguration scheduling models (region-level
    dynamic remap). Runtime concern.
  - Versal ARM Cortex-A72 control complex as a sibling CPU block on
    the same die. Would extend the ``Die.blocks`` list to carry
    DPU + CPU blocks per die. Defer to v7.
  - Vendor-neutral compute_block_common unification. After 6
    architectures with consistent reuse patterns, this is the right
    next move.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.gpu import MemoryType
from embodied_schemas.process_node import CircuitClass, DataConfidence


# ---------------------------------------------------------------------------
# Helper enums
# ---------------------------------------------------------------------------

class DPUFabricKind(str, Enum):
    """The DPU AI-engine generation / fabric architecture.

    Xilinx Versal VE2302 ships AIE-ML v1 (this sprint's reference).
    AMD Versal AI Core / Premium ships AIE-ML v2 (different MAC width
    + precision support). AIE-HD targets datacenter (much larger
    arrays). SOFT_LUT_DPU is the historical "DPU implemented entirely
    in FPGA soft logic" variant; mostly obsolete but retained for
    completeness.
    """

    AIE_ML_V1 = "aie_ml_v1"
    AIE_ML_V2 = "aie_ml_v2"
    AIE_HD = "aie_hd"
    SOFT_LUT_DPU = "soft_lut_dpu"


class DPUNoCTopology(str, Enum):
    """On-die fabric topology for DPUs. AIE tile arrays use a
    streaming-mesh interconnect (AIE_MESH) distinct from packet-based
    NoCs. Smaller / hard-DPU SKUs may use simple crossbars."""

    AIE_MESH = "aie_mesh"
    CROSSBAR = "crossbar"


# ---------------------------------------------------------------------------
# Compute fabric (single AIE-ML fabric on most DPUs)
# ---------------------------------------------------------------------------

class DPUComputeFabric(BaseModel):
    """One compute fabric on a DPU. Vitis AI B4096 ships a single
    AIE-ML v1 fabric (64 tiles, 128 INT8 ops/tile/clock). Heterogeneous-
    tile DPUs (Versal AI Core Premium with mixed AIE/DSP58 tiles)
    would carry multiple entries; defer to v7+.

    Mirrors ``CGRAComputeFabric`` field-by-field where possible. The
    shape differences are (1) the discriminator (``fabric_kind`` enum
    is DPU-specific) and (2) the ``fpga_fabric_overhead_factor``
    multiplier that models the FPGA-vs-ASIC energy penalty.
    """

    fabric_kind: DPUFabricKind = Field(...)
    circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library used for this fabric"
    )
    ops_per_unit_per_clock: dict[str, int] = Field(
        ...,
        description=(
            "Ops per AIE tile per clock keyed on precision name. "
            "Vitis AI B4096: {'int8': 128, 'fp16': 32}. The chip-wide "
            "INT8 peak = num_aie_tiles * ops_per_unit_per_clock['int8'] "
            "* clock_hz."
        ),
    )
    energy_per_op_int8_pj: float = Field(
        ..., gt=0,
        description=(
            "Energy per INT8 op in picojoules at the fabric's nominal "
            "operating point. DPUs are INT8-dominant for DNN workloads; "
            "the FPGA fabric overhead is applied separately via "
            "fpga_fabric_overhead_factor. Vitis AI VE2302 16nm AIE-ML "
            "v1: ~0.4 pJ per INT8 op (pre-overhead)."
        ),
    )
    energy_scaling: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Multiplier on ``energy_per_op_int8_pj`` for each precision. "
            "Unlike NPUs (INT-only), DPUs include native FP16 + emulated "
            "FP32 (Vitis AI: {'fp16': 2.5, 'fp32': 5.0}). INT8 baseline "
            "is 1.0 implicitly (omit from this dict)."
        ),
    )
    fpga_fabric_overhead_factor: float = Field(
        1.0, ge=1.0,
        description=(
            "Multiplier applied to fabric energy for FPGA-fabric "
            "overhead vs equivalent ASIC implementation. Models the "
            "~20-30% penalty from LUT-based interconnect + programmable "
            "routing. Default 1.0 for fully-hardened DPU variants; "
            "Vitis AI ships at ~1.2-1.3 for the published AIE+FPGA mix."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_int_precision_required(self) -> "DPUComputeFabric":
        """DPUs must ship at least one of INT4 / INT8 -- the dominant
        DNN inference precisions. Catches typo'd YAMLs that only declare
        FP precisions (which would be wrong for any DNN-class DPU).

        Also rejects non-positive ops values; zero or negative
        ops/unit/clock is not a meaningful capacity number."""
        non_positive = [
            precision for precision, value in self.ops_per_unit_per_clock.items()
            if value <= 0
        ]
        if non_positive:
            raise ValueError(
                f"DPUComputeFabric.ops_per_unit_per_clock values must be "
                f"positive; got non-positive entries for: {sorted(non_positive)}"
            )
        precisions = {k.lower() for k in self.ops_per_unit_per_clock}
        if not ({"int4", "int8"} & precisions):
            raise ValueError(
                "DPUComputeFabric.ops_per_unit_per_clock must include "
                "at least one of {'int4', 'int8'} (DPUs are DNN "
                f"accelerators dominated by integer quantization); got: "
                f"{sorted(precisions)}"
            )
        return self


# ---------------------------------------------------------------------------
# Memory subsystem (scratchpad-dominant + shared L2 + chip-attached DDR)
# ---------------------------------------------------------------------------

class DPUMemorySubsystem(BaseModel):
    """DPU memory hierarchy: per-tile scratchpad + shared L2 + chip-
    attached DRAM. Vitis AI VE2302 has 64 KiB scratchpad per AIE tile
    + 4 MiB shared L2 + 8 GiB DDR4 (on-die controller + on-package
    PHY; NOT host-bus like CGRA's Plasticine).

    Uses NPU-style ``has_external_dram`` naming because the DRAM is
    chip-attached. CGRA's Plasticine uses ``has_host_dram`` because
    its DDR4 is reached via host PCIe. The two architectures map to
    the same downstream concept ("non-SRAM memory tier") but the
    architectural plumbing differs. v7 unification will collapse
    these.
    """

    # AIE tile fabric bandwidth (the streaming-mesh between tiles)
    on_chip_bandwidth_gbps: float = Field(..., gt=0)

    # Per-AIE-tile scratchpad. Always private per-tile; compiler-managed.
    scratchpad_kib_per_tile: int = Field(..., gt=0)

    # Inter-tile shared L2 SRAM (the "LLC" of DPU-land).
    shared_sram_kib: int = Field(..., ge=0)
    shared_sram_layout: Literal["shared", "partitioned"] = Field("shared")

    # External DRAM. Gated by has_external_dram bool. Vitis AI VE2302:
    # True with DDR4 / 8 GiB. Future SoC-bound DPUs may also populate.
    has_external_dram: bool = Field(False)
    external_dram_type: MemoryType | None = Field(default=None)
    external_dram_size_gb: float | None = Field(default=None, ge=0)
    external_dram_bandwidth_gbps: float | None = Field(default=None, ge=0)

    # Energy per byte for on-chip scratchpad / L2 access.
    # ~5 pJ/B for Vitis AI 16nm; cheaper than DDR4 access.
    scratchpad_access_energy_pj_per_byte: float = Field(..., gt=0)

    # Energy per byte for external DRAM access. Only meaningful when
    # has_external_dram=True. ~15 pJ/B for Vitis AI DDR4.
    external_dram_access_energy_pj_per_byte: float = Field(0.0, ge=0)

    # Cache coherence. DPU default is "none" since compiler-routed
    # AIE dataflow has no host-coherent cache.
    coherence_protocol: str = Field(
        "none",
        description="Common: 'none' (DPU default), 'pcie' (host DMA)",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_external_dram_consistency(self) -> "DPUMemorySubsystem":
        """When ``has_external_dram=True`` the external_dram_* fields
        must all be populated; when False they must all be None / 0.
        Mirrors NPU's external_dram validator."""
        if self.has_external_dram:
            missing = []
            if self.external_dram_type is None:
                missing.append("external_dram_type")
            if self.external_dram_size_gb is None or self.external_dram_size_gb <= 0:
                missing.append("external_dram_size_gb")
            if (self.external_dram_bandwidth_gbps is None
                    or self.external_dram_bandwidth_gbps <= 0):
                missing.append("external_dram_bandwidth_gbps")
            if missing:
                raise ValueError(
                    f"has_external_dram=True requires all of "
                    f"external_dram_type, external_dram_size_gb, "
                    f"external_dram_bandwidth_gbps to be populated; "
                    f"missing/zero: {missing}"
                )
        else:
            extras = []
            if self.external_dram_type is not None:
                extras.append("external_dram_type")
            if self.external_dram_size_gb is not None and self.external_dram_size_gb > 0:
                extras.append("external_dram_size_gb")
            if (self.external_dram_bandwidth_gbps is not None
                    and self.external_dram_bandwidth_gbps > 0):
                extras.append("external_dram_bandwidth_gbps")
            if extras:
                raise ValueError(
                    f"has_external_dram=False requires external_dram_* "
                    f"fields to be None / 0; got populated: {extras}"
                )
        return self


# ---------------------------------------------------------------------------
# On-die fabric (AIE tile streaming mesh, often low-confidence)
# ---------------------------------------------------------------------------

class DPUOnDieFabric(BaseModel):
    """DPU on-die interconnect between AIE tiles. Vitis AI VE2302 uses
    an 8x8 AIE_MESH (estimated; Xilinx doesn't publish per-hop
    streaming-fabric details). Confidence defaults to THEORETICAL.

    FOURTH cross-block-kind type reuse: ``confidence`` reuses
    ``DataConfidence`` from ``process_node`` (same as NPU/CGRA).
    """

    topology: DPUNoCTopology = Field(...)
    bisection_bandwidth_gbps: float = Field(..., gt=0)
    unit_count: int = Field(
        ..., gt=0,
        description="Number of fabric endpoints (= num_aie_tiles typically)",
    )
    flit_size_bytes: int = Field(..., gt=0)

    # Mesh-specific (optional; only populated when topology=AIE_MESH)
    mesh_rows: int | None = Field(default=None, gt=0)
    mesh_cols: int | None = Field(default=None, gt=0)

    hop_latency_ns: float = Field(..., ge=0)
    pj_per_flit_per_hop: float = Field(..., ge=0)
    routing_distance_factor: float = Field(1.0, gt=0)

    confidence: DataConfidence = Field(
        DataConfidence.THEORETICAL,
        description=(
            "Provenance of NoC numbers. Xilinx doesn't publish per-hop "
            "AIE streaming-fabric details so THEORETICAL dominates."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_mesh_dims(self) -> "DPUOnDieFabric":
        """When topology=AIE_MESH, both mesh_rows and mesh_cols should
        be set (and their product should match unit_count); when
        topology is CROSSBAR, both should be None."""
        is_mesh = self.topology == DPUNoCTopology.AIE_MESH
        if is_mesh:
            if self.mesh_rows is None or self.mesh_cols is None:
                raise ValueError(
                    f"topology=AIE_MESH requires both mesh_rows and "
                    f"mesh_cols to be set; got rows={self.mesh_rows}, "
                    f"cols={self.mesh_cols}"
                )
            if self.mesh_rows * self.mesh_cols != self.unit_count:
                raise ValueError(
                    f"topology=AIE_MESH: mesh_rows * mesh_cols "
                    f"({self.mesh_rows} * {self.mesh_cols} = "
                    f"{self.mesh_rows * self.mesh_cols}) must equal "
                    f"unit_count ({self.unit_count})"
                )
        else:
            if self.mesh_rows is not None or self.mesh_cols is not None:
                raise ValueError(
                    f"topology={self.topology.value} requires mesh_rows "
                    f"and mesh_cols to be None; got rows={self.mesh_rows}, "
                    f"cols={self.mesh_cols}"
                )
        return self


# ---------------------------------------------------------------------------
# Thermal profile (single operating point on most edge DPUs)
# ---------------------------------------------------------------------------

class DPUThermalProfile(BaseModel):
    """One DPU operating point. Edge-class DPUs typically ship a single
    profile (Vitis AI VE2302: 20W active-fan); datacenter Versal SKUs
    may have DVFS. The schema uses a scalar ``clock_mhz`` +
    ``dvfs_enabled`` flag instead of the GPU/CPU ``ClockDomain``
    (base/boost/sustained) so the common case stays clean. Mirrors
    ``NPUThermalProfile`` / ``CGRAThermalProfile`` field-by-field."""

    name: str = Field(...)
    tdp_watts: float = Field(..., gt=0)
    cooling_solution_id: str = Field(...)

    clock_mhz: float = Field(..., gt=0, description="Operating frequency")
    dvfs_enabled: bool = Field(
        False,
        description=(
            "False is the edge-DPU default (single fixed operating point). "
            "True only for the rare DPU with multiple thermal profiles."
        ),
    )

    # Per-precision empirical numbers, same shape as GPU/CPU/KPU/NPU/CGRA
    efficiency_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    instruction_efficiency_by_precision: dict[str, float] = Field(default_factory=dict)
    memory_bottleneck_factor_by_precision: dict[str, float] = Field(default_factory=dict)

    vdd_v: float | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "DPUThermalProfile":
        for attr_name, label in (
            ("efficiency_factor_by_precision", "efficiency_factor"),
            ("instruction_efficiency_by_precision", "instruction_efficiency"),
            ("memory_bottleneck_factor_by_precision", "memory_bottleneck_factor"),
        ):
            mapping = getattr(self, attr_name)
            for precision, value in mapping.items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{attr_name}[{precision!r}] = {value} is outside "
                        f"[0, 1]; {label} is a unit fraction."
                    )
        return self


# ---------------------------------------------------------------------------
# Theoretical performance roll-up
# ---------------------------------------------------------------------------

class DPUTheoreticalPerformance(BaseModel):
    """Roll-up peak ops/sec per precision for the DPU. Same shape as
    GPU/CPU/NPU/CGRA. DPU's distinctive pattern is **native FP16 +
    emulated FP32** (similar to CGRA but FP16 is hardware-native in
    AIE-ML, not emulated like Plasticine)."""

    peak_ops_per_sec_by_precision: dict[str, float] = Field(...)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_positive(self) -> "DPUTheoreticalPerformance":
        for prec, value in self.peak_ops_per_sec_by_precision.items():
            if value < 0:
                raise ValueError(
                    f"peak_ops_per_sec_by_precision[{prec!r}] = {value} "
                    f"must be >= 0"
                )
        return self


# ---------------------------------------------------------------------------
# DPUBlock (the discriminated union member)
# ---------------------------------------------------------------------------

class DPUBlock(BaseModel):
    """DPU compute block. Carries the DPU-specific architectural
    description: AIE tile hierarchy, FPGA reconfiguration model,
    scratchpad + L2 + DDR memory subsystem, AIE streaming-mesh on-die
    fabric, and DPU-only scheduler attributes (multi-model concurrency,
    pair-quantized tile allocation).

    The discriminator value ``BlockKind.DPU`` is wired in
    ``compute_product.py``. Imports are arranged so this module does
    not import from ``compute_product`` (compute_product imports from
    here). Same pattern as GPUBlock / CPUBlock / NPUBlock / CGRABlock.
    """

    kind: Literal["dpu"] = Field(
        "dpu",
        description="Discriminator -- always 'dpu' for DPUBlock",
    )

    # AIE tile hierarchy
    num_aie_tiles: int = Field(
        ..., gt=0,
        description=(
            "Number of AI Engine tiles. Vitis AI B4096 config: 64. "
            "AMD Versal AI Core / Premium: more tiles + heterogeneous "
            "tile types (future YAML). AIE-HD datacenter SKUs: 100s."
        ),
    )
    macs_per_tile: int = Field(
        ..., gt=0,
        description=(
            "Multiply-Accumulate units per AIE tile. Vitis AI B4096: "
            "64 MACs/tile (4096 total). AIE-ML v2 uses different MAC "
            "widths; v7 unification can capture this."
        ),
    )
    simd_lanes_per_tile: int = Field(
        1, gt=0,
        description=(
            "SIMD vector lane count per AIE tile. AIE-ML tiles have "
            "wide vector datapaths (e.g. 8-wide); 1 is the default "
            "for non-SIMD compatibility (mirrors NPU lanes_per_unit)."
        ),
    )

    # FPGA reconfiguration model -- the defining DPU characteristic
    is_statically_reconfigurable: bool = Field(
        True,
        description=(
            "True for FPGA-based DPUs (the bitstream is loaded at "
            "deployment, not runtime). False for future hard-DPU "
            "SKUs that ship as fixed-function silicon."
        ),
    )
    bitstream_load_time_ms: float | None = Field(
        default=None, ge=0,
        description=(
            "Typical FPGA bitstream load time in milliseconds. "
            "Vitis AI VE2302: ~1000-3000 ms (seconds-range). "
            "Deployment-only cost; not a per-inference latency. "
            "Optional because the legacy doesn't carry it."
        ),
    )

    # Single compute fabric in the common case (Vitis AI); could grow
    # to multiple fabrics for a DPU with heterogeneous AIE tile types.
    compute_fabrics: list[DPUComputeFabric] = Field(..., min_length=1)

    # Precisions supported chip-wide -- union of compute_fabrics[*].ops_per_unit_per_clock
    multi_precision_alu: list[str] = Field(default_factory=list)

    memory: DPUMemorySubsystem = Field(...)
    noc: DPUOnDieFabric = Field(...)

    # DPU-only scheduler / mapper attributes
    min_occupancy: float = Field(
        0.3, ge=0.0, le=1.0,
        description=(
            "Lower default (0.3) similar to CGRA -- FPGA fabric overhead "
            "means tile utilization varies more than fixed-function "
            "dataflow."
        ),
    )
    max_concurrent_models: int = Field(
        4, gt=0,
        description=(
            "Maximum compiled models the DPU can run simultaneously "
            "by partitioning AIE tiles. Vitis AI default: 4. Higher "
            "than NPU/CGRA (1 each) -- DPU is a multi-model platform."
        ),
    )
    wave_quantization: int = Field(
        2, gt=0,
        description=(
            "AIE tiles are allocated in pairs per Xilinx documentation. "
            "Default 2 reflects this; future AIE-HD SKUs may use larger "
            "quantization."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_noc_unit_count_matches(self) -> "DPUBlock":
        """noc.unit_count should equal num_aie_tiles. Catches YAMLs
        where the NoC was authored against a different tile count
        than the block declared. Mirrors NPU/CGRA validators."""
        if self.noc.unit_count != self.num_aie_tiles:
            raise ValueError(
                f"noc.unit_count ({self.noc.unit_count}) must equal "
                f"num_aie_tiles ({self.num_aie_tiles})"
            )
        return self
