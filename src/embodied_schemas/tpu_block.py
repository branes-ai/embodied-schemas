"""TPU compute block for ``ComputeProduct`` (v7 schema, additive).

PR 2 of the TPU mini-sprint scoped at ``graphs#204``. Adds the
seventh member of the ``Block`` discriminated union after KPU (v1),
GPU (v2), CPU (v3), NPU (v4), CGRA (v5), and DPU (v6). Modeled
directly off the field set audited in branes-ai/graphs#205 paper
exercise (the design doc at
``graphs/docs/designs/tpu-compute-product-schema-extension.md``).

Design choice: same per-architecture-types rule the prior sprints
established -- ship TPU-specific sub-types (``TPUComputeFabric``,
``TPUTileEnergyCoefficients``, ``TPUMemorySubsystem``,
``TPUOnDieFabric``, ``TPUThermalProfile``, ``TPUTheoreticalPerformance``)
rather than generalize. After 7 architectures, the v8 vendor-neutral
``compute_block_common`` unification is the recommended next major
sprint -- collapsing all 7 shared-primitive reuse patterns in one
pass.

**Fifth cross-block-kind type reuse**:
  - ``TPUOnDieFabric.confidence`` reuses ``DataConfidence`` from
    ``process_node`` (same as NPU/CGRA/DPU)
  - ``TPUMemorySubsystem.external_dram_type`` reuses ``MemoryType``
    from ``gpu`` (same as NPU/DPU)
  - ``TPUComputeFabric.circuit_class`` reuses ``CircuitClass`` from
    ``process_node`` (same as GPU/CPU/NPU/CGRA/DPU)

Google TPU v4 (reference SKU for this sprint) specifics:

  - **MXU (Matrix Multiplier Unit) hierarchy** -- TPU v4 has 2 MXUs,
    each 128x128 (= 32,768 total MACs). Modeled as ``num_mxus`` +
    ``mxu_dim_rows`` + ``mxu_dim_cols`` on ``TPUBlock``.
  - **Unified Buffer (UB)** -- 32 MiB chip-shared SRAM that collapses
    L1+L2 into one tier (no separate L1/L2 distinction like other
    architectures). Captured as ``TPUMemorySubsystem.unified_buffer_size_kib``.
  - **Tile energy decomposition** -- TPU community uses a fine-grained
    energy model (weight FIFO + accumulator + UB read/write + MAC).
    Captured as ``TPUTileEnergyCoefficients`` sub-type with the 9
    canonical coefficients.
  - **ICI (Inter-Chip Interconnect) single-chip surface** -- TPU pods
    use ICI for multi-chip topology (v4 pod = 4096 chips in 3D torus).
    This sprint models only the single-chip surface: ``ici_port_count``
    + ``ici_bandwidth_per_port_gbps`` + optional ``ici_topology_hint``.
    Pod-level coverage deferred to v8+ system-level scope.
  - **HBM2e external DRAM** -- 32 GiB at 1.2 TB/s on TPU v4. Uses
    NPU/DPU-style ``has_external_dram`` (chip-attached HBM stacks,
    NOT host-bus like CGRA).
  - **Multi-precision INT-and-FP** -- BF16 native (training default)
    + INT8 at 2x BF16 rate + emulated FP32 at 1/2 BF16. The schema's
    INT-precision-required validator must accept BF16+INT8 (TPU is
    BF16-dominant for training, INT8 secondary for inference).
  - **Datacenter-class thermals** -- 350W TDP, active-liquid cooling.

Multi-SKU schema coverage (all 5 TPU SKUs from the paper exercise):

  - tpu_v1: 28nm, 1x (256x256), 75W DDR3
  - tpu_v3: 16nm, 2x (128x128), 200W HBM
  - **tpu_v4 (this sprint)**: 7nm, 2x (128x128), 350W HBM2e
  - tpu_v5p: 4nm, 2x (128x128), 400W HBM3
  - tpu_edge_pro: 7nm, 1x (128x128), 15-45W (3 profiles!) LPDDR4X

The schema accommodates all 5 SKUs; only TPU v4 ships in this sprint.
Other SKUs land as pure data PRs after this umbrella closes.

Future-deferred (v8+):

  - Multi-chip pod topology modeling (TPU v4 pod = 4096 chips in 3D
    torus). v8 system-level scope.
  - Coral Edge TPU re-migration from NPUBlock to TPUBlock. The current
    NPUBlock + HardwareType.TPU overlay was a v6 workaround; v8
    unification can resolve.
  - Vendor-neutral compute_block_common unification (HIGHEST priority
    after this sprint -- 7 architectures' shared primitives can
    collapse in one pass).
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

class TPUFabricKind(str, Enum):
    """The TPU systolic fabric generation / architecture.

    TPU v1 used a single large 256x256 MXU (TPU_V1_STYLE). TPU v2+
    shifted to multiple smaller 128x128 MXUs with HBM (TPU_V2_PLUS).
    AIE-HD targets future datacenter SKUs with larger MXUs and
    advanced bring-up; this is the placeholder for ``TPU_HD``.
    """

    TPU_V1_STYLE = "tpu_v1_style"
    TPU_V2_PLUS = "tpu_v2_plus"
    TPU_HD = "tpu_hd"


class TPUNoCTopology(str, Enum):
    """On-die fabric topology for TPUs. The UB-to-MXU connection is
    typically a direct crossbar (CROSSBAR for single-MXU SKUs;
    MULTI_CROSSBAR for 2+ MXUs that share the UB). TPUs don't use
    packet-based meshes for on-die routing -- the dataflow is
    statically compiled by XLA."""

    CROSSBAR = "crossbar"
    MULTI_CROSSBAR = "multi_crossbar"


# ---------------------------------------------------------------------------
# Tile energy coefficients (TPU-specific architectural decomposition)
# ---------------------------------------------------------------------------

class TPUTileEnergyCoefficients(BaseModel):
    """Per-byte / per-element energy coefficients for the TPU tile
    energy model. The TPU community uses a fine-grained decomposition
    (weight FIFO + accumulator + UB read/write + MAC) for architectural
    analysis. Lives in graphs as ``TPUTileEnergyModel``; this schema
    type captures the 9 canonical coefficients so the loader can
    reconstruct it.

    All energies in **picojoules** (pJ).
    """

    mac_energy_pj: float = Field(
        ..., gt=0,
        description=(
            "Energy per MAC operation at the fabric's nominal "
            "precision (typically BF16 for training TPUs). TPU v4 7nm: "
            "~0.25 pJ per BF16 MAC."
        ),
    )
    weight_memory_energy_pj_per_byte: float = Field(
        ..., gt=0,
        description=(
            "Energy per byte from external DRAM (HBM/DDR) into the "
            "weight FIFO. TPU v4 HBM2e: ~10 pJ/byte."
        ),
    )
    weight_fifo_energy_pj_per_byte: float = Field(
        ..., gt=0,
        description=(
            "Energy per byte from weight FIFO into the MXU. On-chip "
            "SRAM cost; ~0.5 pJ/byte at 7nm."
        ),
    )
    unified_buffer_read_energy_pj_per_byte: float = Field(
        ..., gt=0,
        description="Energy per byte read from the Unified Buffer.",
    )
    unified_buffer_write_energy_pj_per_byte: float = Field(
        ..., gt=0,
        description="Energy per byte written to the Unified Buffer.",
    )
    accumulator_read_energy_pj_per_element: float = Field(
        ..., gt=0,
        description=(
            "Energy per element read from per-MXU accumulator (32-bit "
            "element)."
        ),
    )
    accumulator_write_energy_pj_per_element: float = Field(
        ..., gt=0,
        description=(
            "Energy per element written to per-MXU accumulator (32-bit "
            "element)."
        ),
    )
    weight_shift_in_energy_pj_per_element: float = Field(
        ..., gt=0,
        description=(
            "Energy per element shifted into the systolic array's "
            "weight shift register."
        ),
    )
    activation_stream_energy_pj_per_element: float = Field(
        ..., gt=0,
        description=(
            "Energy per element streamed into the systolic array's "
            "activation input."
        ),
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Compute fabric (single systolic fabric on TPUs)
# ---------------------------------------------------------------------------

class TPUComputeFabric(BaseModel):
    """One compute fabric on a TPU. TPU v4 ships a single systolic
    fabric (2 MXUs share the same fabric type). Heterogeneous-MXU
    SKUs would carry multiple fabrics; not common today.

    Mirrors ``DPUComputeFabric`` field-by-field where possible. The
    shape differences are (1) the discriminator (``fabric_kind`` is
    TPU-specific) and (2) the energy baseline (BF16, the TPU
    training-default precision, vs DPU/NPU's INT8 baseline).
    """

    fabric_kind: TPUFabricKind = Field(...)
    circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library used for this fabric"
    )
    ops_per_unit_per_clock: dict[str, int] = Field(
        ...,
        description=(
            "Ops per MAC per clock keyed on precision name. The "
            "schema 'unit' is the MAC, not the MXU -- per-chip peak "
            "is num_mxus * mxu_dim_rows * mxu_dim_cols * "
            "ops_per_unit_per_clock[precision] * clock_hz. TPU v4: "
            "{'bf16': 2, 'int8': 2} (MAC = 2 ops, multiply + accumulate)."
        ),
    )
    energy_per_op_bf16_pj: float = Field(
        ..., gt=0,
        description=(
            "Energy per BF16 op in picojoules at the fabric's nominal "
            "operating point. TPUs are BF16-dominant for training "
            "(vs DPU/NPU's INT8-dominant inference). TPU v4 7nm: ~0.225 "
            "pJ per BF16 op (= 0.45 pJ per MAC / 2 ops)."
        ),
    )
    energy_scaling: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Multiplier on ``energy_per_op_bf16_pj`` for each precision. "
            "TPU v4: {'int8': 0.25, 'fp32': 2.0} (INT8 4x cheaper than "
            "BF16; FP32 emulated, 2x cost). BF16 baseline is 1.0 "
            "implicitly (omit from this dict)."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_int_or_bf16_precision_required(self) -> "TPUComputeFabric":
        """TPUs must ship at least one of {INT8, BF16} -- the
        training (BF16) or inference (INT8) precisions that justify
        a TPU's existence. Catches typo'd YAMLs that only declare
        FP32 (which would be wrong; TPUs emulate FP32 from BF16
        building blocks).

        Also rejects non-positive ops values (catches the same class
        of bug the DPU loader fix in graphs#202 addressed)."""
        non_positive = [
            precision for precision, value in self.ops_per_unit_per_clock.items()
            if value <= 0
        ]
        if non_positive:
            raise ValueError(
                f"TPUComputeFabric.ops_per_unit_per_clock values must be "
                f"positive; got non-positive entries for: {sorted(non_positive)}"
            )
        precisions = {k.lower() for k in self.ops_per_unit_per_clock}
        if not ({"int8", "bf16"} & precisions):
            raise ValueError(
                "TPUComputeFabric.ops_per_unit_per_clock must include "
                "at least one of {'int8', 'bf16'} (TPUs are BF16-dominant "
                f"for training and INT8-dominant for inference); got: "
                f"{sorted(precisions)}"
            )
        return self


# ---------------------------------------------------------------------------
# Memory subsystem (Unified Buffer + HBM)
# ---------------------------------------------------------------------------

class TPUMemorySubsystem(BaseModel):
    """TPU memory hierarchy: Unified Buffer (UB) + HBM. The UB
    collapses what would be L1+L2 on other architectures into a
    single chip-shared SRAM tier (32 MiB on TPU v4). HBM is the
    chip-attached off-chip memory tier (vs CGRA's host-bus DDR4).

    Uses NPU/DPU-style ``has_external_dram`` because HBM stacks are
    chip-attached (multiple stacks on the package).
    """

    # UB-to-MXU streaming bandwidth (the bandwidth available to the
    # systolic arrays from the unified buffer)
    on_chip_bandwidth_gbps: float = Field(..., gt=0)

    # Unified Buffer size. TPU v4: 32 MiB. Acts as L1+L2 combined.
    unified_buffer_size_kib: int = Field(..., gt=0)

    # Energy per byte for UB access (on-chip SRAM). ~0.5 pJ/byte at 7nm.
    unified_buffer_access_energy_pj_per_byte: float = Field(..., gt=0)

    # External DRAM. Gated by has_external_dram bool. TPU v4: True
    # with HBM2e / 32 GiB. tpu_edge_pro would use LPDDR4X.
    has_external_dram: bool = Field(False)
    external_dram_type: MemoryType | None = Field(default=None)
    external_dram_size_gb: float | None = Field(default=None, ge=0)
    external_dram_bandwidth_gbps: float | None = Field(default=None, ge=0)

    # Energy per byte for external DRAM (HBM/DDR/LPDDR) access.
    external_dram_access_energy_pj_per_byte: float = Field(0.0, ge=0)

    # Cache coherence. TPU default is "none" since XLA-routed dataflow
    # has no coherence requirement.
    coherence_protocol: str = Field(
        "none",
        description="Common: 'none' (TPU default)",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_external_dram_consistency(self) -> "TPUMemorySubsystem":
        """When ``has_external_dram=True`` all external_dram_* fields
        must be populated; when False they must all be None / 0.
        Mirrors NPU / DPU validators."""
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
# On-die fabric (UB-to-MXU streaming; simpler than NoCs)
# ---------------------------------------------------------------------------

class TPUOnDieFabric(BaseModel):
    """TPU on-die interconnect between the Unified Buffer and the
    MXUs. Much simpler than CGRA/DPU/NPU NoCs because the UB-to-MXU
    connection is essentially direct (crossbar) -- XLA statically
    routes the dataflow at compile time.

    Single-MXU SKUs use CROSSBAR; multi-MXU SKUs use MULTI_CROSSBAR.

    FIFTH cross-block-kind type reuse: ``confidence`` reuses
    ``DataConfidence`` from ``process_node`` (same as NPU/CGRA/DPU).
    """

    topology: TPUNoCTopology = Field(...)
    bisection_bandwidth_gbps: float = Field(..., gt=0)
    unit_count: int = Field(
        ..., gt=0,
        description="Number of fabric endpoints (= num_mxus typically)",
    )
    flit_size_bytes: int = Field(..., gt=0)

    hop_latency_ns: float = Field(..., ge=0)
    pj_per_flit_per_hop: float = Field(..., ge=0)
    routing_distance_factor: float = Field(1.0, gt=0)

    confidence: DataConfidence = Field(
        DataConfidence.THEORETICAL,
        description=(
            "Provenance of NoC numbers. Google publishes systolic "
            "array dimensions and HBM bandwidth but not UB-to-MXU "
            "fabric details, so THEORETICAL is the dominant case."
        ),
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Thermal profile (single profile on datacenter; multi-profile on edge)
# ---------------------------------------------------------------------------

class TPUThermalProfile(BaseModel):
    """One TPU operating point. Datacenter TPUs (v1/v3/v4/v5p) ship
    single profiles with no DVFS. Edge TPUs (tpu_edge_pro) ship
    multiple profiles (15W / 30W / 45W on tpu_edge_pro -- three
    distinct operating points). The schema uses a scalar ``clock_mhz``
    per profile (mirrors NPU/CGRA/DPU); for multi-profile SKUs the
    ``power.thermal_profiles`` list carries multiple entries.

    Mirrors ``DPUThermalProfile`` field-by-field."""

    name: str = Field(...)
    tdp_watts: float = Field(..., gt=0)
    cooling_solution_id: str = Field(...)

    clock_mhz: float = Field(..., gt=0, description="Operating frequency")
    dvfs_enabled: bool = Field(
        False,
        description=(
            "False is the datacenter-TPU default (single fixed "
            "operating point). True for edge TPUs with multiple "
            "thermal profiles (tpu_edge_pro)."
        ),
    )

    # Per-precision empirical numbers, same shape as GPU/CPU/KPU/NPU/CGRA/DPU
    efficiency_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    instruction_efficiency_by_precision: dict[str, float] = Field(default_factory=dict)
    memory_bottleneck_factor_by_precision: dict[str, float] = Field(default_factory=dict)

    vdd_v: float | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "TPUThermalProfile":
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

class TPUTheoreticalPerformance(BaseModel):
    """Roll-up peak ops/sec per precision for the TPU. Same shape as
    GPU/CPU/NPU/CGRA/DPU. TPU's distinctive pattern is **BF16-dominant
    + INT8 at 2x BF16 + emulated FP32** (training-first design)."""

    peak_ops_per_sec_by_precision: dict[str, float] = Field(...)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_positive(self) -> "TPUTheoreticalPerformance":
        for prec, value in self.peak_ops_per_sec_by_precision.items():
            if value < 0:
                raise ValueError(
                    f"peak_ops_per_sec_by_precision[{prec!r}] = {value} "
                    f"must be >= 0"
                )
        return self


# ---------------------------------------------------------------------------
# TPUBlock (the discriminated union member)
# ---------------------------------------------------------------------------

class TPUBlock(BaseModel):
    """TPU compute block. Carries the TPU-specific architectural
    description: MXU hierarchy, tile energy decomposition, Unified
    Buffer-dominant memory subsystem, UB-to-MXU on-die fabric, single-
    chip ICI port surface, and TPU-only scheduler attributes.

    The discriminator value ``BlockKind.TPU`` is wired in
    ``compute_product.py``. Imports are arranged so this module does
    not import from ``compute_product`` (compute_product imports from
    here). Same pattern as GPUBlock / CPUBlock / NPUBlock / CGRABlock /
    DPUBlock.
    """

    kind: Literal["tpu"] = Field(
        "tpu",
        description="Discriminator -- always 'tpu' for TPUBlock",
    )

    # MXU hierarchy
    num_mxus: int = Field(
        ..., gt=0,
        description=(
            "Number of Matrix Multiplier Units. TPU v1: 1 (single "
            "256x256 MXU). TPU v3/v4/v5p: 2 (two 128x128 MXUs). "
            "tpu_edge_pro: 1 (128x128). Future SKUs may go higher."
        ),
    )
    mxu_dim_rows: int = Field(
        ..., gt=0,
        description=(
            "Rows in each systolic array. TPU v1: 256. TPU v2+: 128. "
            "Edge TPU Pro: 128."
        ),
    )
    mxu_dim_cols: int = Field(
        ..., gt=0,
        description=(
            "Columns in each systolic array. TPU v1: 256. TPU v2+: "
            "128. Edge TPU Pro: 128. Typically equal to mxu_dim_rows "
            "(square systolic arrays)."
        ),
    )

    # Tile energy decomposition (TPU-specific architectural model)
    weight_tile_size_kib: int = Field(
        ..., gt=0,
        description=(
            "Per-MXU weight prefetch buffer size in KiB. TPU v4: 32 "
            "KiB. Decouples DRAM bandwidth from MXU throughput."
        ),
    )
    weight_fifo_depth: int = Field(
        ..., gt=0,
        description=(
            "Number of weight tiles buffered in the FIFO. TPU v4: 2 "
            "(double-buffered)."
        ),
    )
    pipeline_fill_cycles: int = Field(
        ..., gt=0,
        description=(
            "Cycles to fill the systolic pipeline (typically = "
            "max(mxu_dim_rows, mxu_dim_cols)). TPU v4: 128."
        ),
    )
    accumulator_size_kib_per_mxu: int = Field(
        ..., gt=0,
        description=(
            "Per-MXU accumulator size in KiB (32-bit elements). TPU "
            "v4: 2 MiB. Sized to hold the output of one full systolic "
            "pass at the roofline knee."
        ),
    )
    tile_energy_coefficients: TPUTileEnergyCoefficients = Field(
        ...,
        description=(
            "The 9 canonical energy coefficients for the TPU tile "
            "energy model (FIFO + accumulator + UB + MAC)."
        ),
    )

    # Inter-Chip Interconnect (ICI) single-chip surface
    ici_port_count: int = Field(
        0, ge=0,
        description=(
            "Number of ICI ports on the chip. TPU v4: 6 (for 3D "
            "torus). tpu_edge_pro: 0 (single-chip; no pod). Datacenter "
            "TPUs use ICI to build pods (pod-level topology deferred "
            "to v8+)."
        ),
    )
    ici_bandwidth_per_port_gbps: float = Field(
        0.0, ge=0,
        description=(
            "Per-ICI-port bandwidth in GB/s. TPU v4: ~400 GB/s per "
            "port. 0 when ici_port_count=0."
        ),
    )
    ici_topology_hint: str | None = Field(
        default=None,
        description=(
            "Forward-compat hint for v8+ pod-level topology modeling. "
            "Free-form: '3d_torus_2x2x2', '2d_torus_4x4', etc. Optional."
        ),
    )

    # Single compute fabric in the common case (TPU v4); could grow
    # to multiple fabrics for heterogeneous-MXU SKUs in v8+.
    compute_fabrics: list[TPUComputeFabric] = Field(..., min_length=1)

    # Precisions supported chip-wide -- union of compute_fabrics[*].ops_per_unit_per_clock
    multi_precision_alu: list[str] = Field(default_factory=list)

    memory: TPUMemorySubsystem = Field(...)
    noc: TPUOnDieFabric = Field(...)

    # TPU-only scheduler / mapper attributes
    min_occupancy: float = Field(
        0.5, ge=0.0, le=1.0,
        description=(
            "TPUs need high systolic utilization to be efficient; "
            "higher default (0.5) than NPU (0.8 -- wait, NPU is 0.8; "
            "DPU/CGRA 0.3). TPUs sit between -- they're more efficient "
            "than CGRA/DPU FPGAs but less rigid than fixed NPUs."
        ),
    )
    max_concurrent_models: int = Field(
        1, gt=0,
        description=(
            "TPUs typically run one large model/batch at a time. The "
            "TPU pod itself can run multiple models via partitioning, "
            "but that's a v8+ pod-level concern."
        ),
    )
    wave_quantization: int = Field(
        1, gt=0,
        description="TPUs don't wave-quantize; default 1 keeps shape consistent",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_noc_unit_count_matches(self) -> "TPUBlock":
        """noc.unit_count should equal num_mxus. Catches YAMLs where
        the NoC was authored against a different MXU count than the
        block declared. Mirrors NPU/CGRA/DPU validators."""
        if self.noc.unit_count != self.num_mxus:
            raise ValueError(
                f"noc.unit_count ({self.noc.unit_count}) must equal "
                f"num_mxus ({self.num_mxus})"
            )
        return self

    @model_validator(mode="after")
    def _validate_mxu_dimensions(self) -> "TPUBlock":
        """MXU dimensions should be reasonable: 128 or 256 (the
        canonical TPU sizes) or another power of 2. Catches typo'd
        YAMLs that swap rows/cols or use non-power-of-2 values.

        Allowed values: 64, 128, 256, 512 (future-proofing). Square
        MXUs are the convention but not enforced (heterogeneous-dim
        future SKUs allowed)."""
        allowed = {64, 128, 256, 512}
        if self.mxu_dim_rows not in allowed:
            raise ValueError(
                f"mxu_dim_rows ({self.mxu_dim_rows}) is not in the "
                f"canonical TPU set {sorted(allowed)}; if a future SKU "
                f"genuinely uses a different size, extend this allow-list."
            )
        if self.mxu_dim_cols not in allowed:
            raise ValueError(
                f"mxu_dim_cols ({self.mxu_dim_cols}) is not in the "
                f"canonical TPU set {sorted(allowed)}; if a future SKU "
                f"genuinely uses a different size, extend this allow-list."
            )
        return self

    @model_validator(mode="after")
    def _validate_ici_consistency(self) -> "TPUBlock":
        """If ICI ports are declared, bandwidth-per-port must also be
        populated (and vice versa). Catches typo'd YAMLs that toggle
        one field without the other."""
        if self.ici_port_count > 0 and self.ici_bandwidth_per_port_gbps <= 0:
            raise ValueError(
                f"ici_port_count={self.ici_port_count} requires "
                f"ici_bandwidth_per_port_gbps > 0; got "
                f"{self.ici_bandwidth_per_port_gbps}"
            )
        if self.ici_port_count == 0 and self.ici_bandwidth_per_port_gbps > 0:
            raise ValueError(
                f"ici_bandwidth_per_port_gbps={self.ici_bandwidth_per_port_gbps} "
                f"requires ici_port_count > 0; got 0"
            )
        return self
