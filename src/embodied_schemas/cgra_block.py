"""CGRA compute block for ``ComputeProduct`` (v5 schema, additive).

PR 2 of the CGRA mini-sprint scoped at ``graphs#196``. Adds the fifth
member of the ``Block`` discriminated union after KPU (v1), GPU (v2),
CPU (v3), and NPU (v4). Modeled directly off the field set audited
in ``graphs/docs/designs/cgra-compute-product-schema-extension.md``.

Design choice: same per-architecture-types rule the prior sprints
established -- ship CGRA-specific sub-types (``CGRAComputeFabric``,
``CGRAMemorySubsystem``, ``CGRAOnDieFabric``, ``CGRAThermalProfile``,
``CGRATheoreticalPerformance``) rather than generalize. With 5
architectures it's now time to think seriously about v6 unification
(the design doc's "6th sprint" milestone).

**Third cross-block-kind type reuse**: ``CGRAOnDieFabric.confidence``
reuses ``DataConfidence`` from ``process_node`` (same as NPU);
``CGRAMemorySubsystem.host_dram_type`` reuses ``MemoryType`` from
``gpu`` (same as NPU's ``external_dram_type``); ``CGRAComputeFabric.
circuit_class`` reuses ``CircuitClass`` from ``process_node`` (same
as GPU/CPU/NPU). These three data points across NPU (v4) and CGRA
(v5) justify carving out a vendor-neutral ``compute_block_common``
module when the v6 unification sprint lands.

Stanford Plasticine v2 reference SKU specifics that shaped the design:

  - **Reconfiguration overhead** is the defining CGRA characteristic.
    The fabric reconfigures per application; Plasticine v2 reports
    ~1000 cycles for full-fabric remap (Achilles heel vs fixed-
    function NPUs). Modeled as ``CGRABlock.reconfig_overhead_cycles``
    (scalar; partial-reconfig forward-compat flag at
    ``CGRABlock.supports_partial_reconfig``).
  - **PCU (Pattern Compute Unit) hierarchy** -- 32 PCUs * 8 MACs at
    1 GHz = ~10 TOPS INT8 theoretical. PCUs are reconfigurable per-
    application but fixed during a single program execution.
  - **PMU (Pattern Memory Unit)** -- per-PCU scratchpad (~64 KB on
    Plasticine v2). Acts as L1 in NPU-land terms but compiler-managed.
    Modeled as ``CGRAMemorySubsystem.pmu_kib_per_pcu``.
  - **Host-DRAM bus path** -- like Coral Edge TPU (graphs#192), the
    chip has no chip-attached external DRAM controllers; main memory
    is reached through the host bus (DDR4 4 GB on Plasticine v2).
    Modeled separately from NPU's ``has_external_dram`` because the
    architectural distinction is real (bus-mediated vs chip-attached);
    v6 unification can resolve the naming difference.
  - **Multi-precision INT-dominant with FP emulation** -- Plasticine
    runs INT8 natively (10 TOPS) and emulates FP16 (1/4 INT8) and
    FP32 (1/8 INT8). The ``energy_scaling`` dict must include the
    FP entries so downstream energy estimates penalize FP workloads.
  - **Single thermal profile, no DVFS** -- Plasticine v2 is a single
    15W operating point. ``CGRAThermalProfile`` uses scalar
    ``clock_mhz`` + ``dvfs_enabled`` flag (defaulting False).

Future-deferred (v6+):

  - Other CGRA architectures (Wave Computing DPU, Cerebras WSE,
    SambaNova RDU, Tenstorrent Grayskull). Each may exercise extension
    fields not in Plasticine (e.g. wafer-scale routing, TORUS_2D NoC,
    heterogeneous PCU types). Pure additive; defer to YAML PRs.
  - Partial reconfiguration scheduling models (region-level dynamic
    remap). Runtime concern, not static SKU description.
  - DPU (Xilinx Vitis AI on AMD FPGA) -- structurally distinct (FPGA
    reconfig is much finer-grained than CGRA PCU reconfig); separate
    sprint with its own schema.
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

class CGRAFabricKind(str, Enum):
    """The dataflow scheduling discipline a CGRA fabric implements.

    Plasticine's PCU mesh is PCU_SPATIAL_DATAFLOW. Systolic-PCU CGRAs
    (rare; some Wave Computing variants) are SYSTOLIC_PCU. Future
    heterogeneous CGRAs with mixed PCU types are HETEROGENEOUS_PCU.
    """

    PCU_SPATIAL_DATAFLOW = "pcu_spatial_dataflow"
    SYSTOLIC_PCU = "systolic_pcu"
    HETEROGENEOUS_PCU = "heterogeneous_pcu"


class CGRANoCTopology(str, Enum):
    """On-die fabric topology for CGRAs. Plasticine uses MESH_2D;
    SambaNova uses TORUS_2D; small academic CGRAs use CROSSBAR."""

    MESH_2D = "mesh_2d"
    TORUS_2D = "torus_2d"
    CROSSBAR = "crossbar"


# ---------------------------------------------------------------------------
# Compute fabric (single PCU fabric on most CGRAs)
# ---------------------------------------------------------------------------

class CGRAComputeFabric(BaseModel):
    """One compute fabric on a CGRA. Plasticine v2 ships a single
    PCU fabric (32 PCUs, 320 INT8 ops/PCU/clock). Multi-fabric CGRAs
    (heterogeneous PCU types) would carry multiple entries; not
    common today.

    Mirrors ``NPUComputeFabric`` field-by-field where possible. The
    shape difference is in the discriminator (``fabric_kind`` instead
    of ``dataflow_kind``) and the energy baseline (``energy_per_op_int8_pj``
    -- same as NPU because CGRAs are also INT8-dominant for DNN
    workloads, but unlike NPU they can do FP via emulation).
    """

    fabric_kind: CGRAFabricKind = Field(...)
    circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library used for this fabric"
    )
    ops_per_unit_per_clock: dict[str, int] = Field(
        ...,
        description=(
            "Ops per PCU per clock keyed on precision name. "
            "Plasticine v2: {'int8': 320, 'fp16': 80}. The chip-wide "
            "INT8 peak = num_pcus * ops_per_unit_per_clock['int8'] * "
            "clock_hz."
        ),
    )
    energy_per_op_int8_pj: float = Field(
        ..., gt=0,
        description=(
            "Energy per INT8 op in picojoules at the fabric's nominal "
            "operating point. CGRAs are INT8-dominant for DNN workloads; "
            "FP is supported via emulation. Plasticine v2 28nm balanced: "
            "~0.6 pJ per INT8 op."
        ),
    )
    energy_scaling: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Multiplier on ``energy_per_op_int8_pj`` for each precision. "
            "Unlike NPUs which are INT-only, CGRAs commonly include FP "
            "scaling entries (Plasticine: {'fp16': 3.3, 'fp32': 6.7} "
            "for INT8-relative cost of emulated FP). INT8 baseline is "
            "1.0 implicitly (omit from this dict)."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_int_precision_required(self) -> "CGRAComputeFabric":
        """CGRAs must ship at least one of INT4 / INT8 -- the dominant
        DNN inference precisions. Catches typo'd YAMLs that only declare
        FP precisions (which would be wrong for any DNN-class CGRA)."""
        precisions = {k.lower() for k in self.ops_per_unit_per_clock}
        if not ({"int4", "int8"} & precisions):
            raise ValueError(
                "CGRAComputeFabric.ops_per_unit_per_clock must include "
                "at least one of {'int4', 'int8'} (CGRAs are DNN "
                f"accelerators dominated by integer quantization); got: "
                f"{sorted(precisions)}"
            )
        return self


# ---------------------------------------------------------------------------
# Memory subsystem (PMU-dominant + small shared L2 + host DRAM)
# ---------------------------------------------------------------------------

class CGRAMemorySubsystem(BaseModel):
    """CGRA memory hierarchy: PMU-dominant + small shared L2 + host
    memory. Plasticine v2 has 64 KB PMU per PCU + 2 MB shared L2 +
    4 GB host DDR4 (accessed via host bus, not chip-attached).

    The host-DRAM gating mirrors NPU's external-DRAM gating but with
    different naming (``has_host_dram`` vs ``has_external_dram``) to
    signal the bus-mediated nature. v6 unification can resolve the
    naming; for now, the loader-side overlay pattern from Coral
    (graphs#192) handles the peak_bandwidth bottleneck-tier selection.
    """

    # On-chip mesh fabric bandwidth -- replaces the GPU/CPU "DRAM
    # bandwidth" field because most CGRAs are PMU/L2-resident steady-state
    on_chip_bandwidth_gbps: float = Field(..., gt=0)

    # Per-PCU PMU scratchpad ("Pattern Memory Unit" in Plasticine
    # literature). Always private per-PCU; the compiler statically
    # allocates it.
    pmu_kib_per_pcu: int = Field(..., gt=0)

    # Inter-PCU shared L2 SRAM (the "LLC" of CGRA-land). Acts above
    # the per-PCU PMU partitions.
    shared_sram_kib: int = Field(..., ge=0)
    shared_sram_layout: Literal["shared", "partitioned"] = Field("shared")

    # Host DRAM. Gated by has_host_dram bool; when False all the
    # host_dram_* fields must be None / 0 (validator enforces).
    # Plasticine v2: True with DDR4 / 4 GB. Future CGRAs with chip-
    # attached external DRAM (Cerebras WSE) would prefer adding a
    # parallel has_chip_dram path in v6 rather than reusing has_host_dram.
    has_host_dram: bool = Field(False)
    host_dram_type: MemoryType | None = Field(default=None)
    host_dram_size_gb: float | None = Field(default=None, ge=0)
    host_dram_bandwidth_gbps: float | None = Field(default=None, ge=0)

    # Energy per byte for the dominant on-chip memory tier (PMU + L2).
    # ~12 pJ/B for Plasticine 28nm; cheaper than host DRAM access.
    pmu_access_energy_pj_per_byte: float = Field(..., gt=0)

    # Energy per byte for host DRAM access (via host bus). Only
    # meaningful when has_host_dram=True. ~20 pJ/B for Plasticine
    # DDR4 path on 28nm.
    host_dram_access_energy_pj_per_byte: float = Field(0.0, ge=0)

    # Cache coherence. CGRA default is "none" since compiler-routed
    # spatial dataflow has no host-coherent cache. Free-form string
    # for future host-coherent CGRAs (rare).
    coherence_protocol: str = Field(
        "none",
        description="Common: 'none' (CGRA default), 'pcie' (host DMA)",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_host_dram_consistency(self) -> "CGRAMemorySubsystem":
        """When ``has_host_dram=True`` the host_dram_* fields must all
        be populated; when False they must all be None / 0. Catches
        typo'd YAMLs that toggled one field without the other. Mirrors
        the external_dram validator on ``NPUMemorySubsystem``."""
        if self.has_host_dram:
            missing = []
            if self.host_dram_type is None:
                missing.append("host_dram_type")
            if self.host_dram_size_gb is None or self.host_dram_size_gb <= 0:
                missing.append("host_dram_size_gb")
            if (self.host_dram_bandwidth_gbps is None
                    or self.host_dram_bandwidth_gbps <= 0):
                missing.append("host_dram_bandwidth_gbps")
            if missing:
                raise ValueError(
                    f"has_host_dram=True requires all of host_dram_type, "
                    f"host_dram_size_gb, host_dram_bandwidth_gbps to be "
                    f"populated; missing/zero: {missing}"
                )
        else:
            extras = []
            if self.host_dram_type is not None:
                extras.append("host_dram_type")
            if self.host_dram_size_gb is not None and self.host_dram_size_gb > 0:
                extras.append("host_dram_size_gb")
            if (self.host_dram_bandwidth_gbps is not None
                    and self.host_dram_bandwidth_gbps > 0):
                extras.append("host_dram_bandwidth_gbps")
            if extras:
                raise ValueError(
                    f"has_host_dram=False requires host_dram_* fields to "
                    f"be None / 0; got populated: {extras}"
                )
        return self


# ---------------------------------------------------------------------------
# On-die fabric (PCU mesh, often low-confidence for research SKUs)
# ---------------------------------------------------------------------------

class CGRAOnDieFabric(BaseModel):
    """CGRA on-die interconnect between PCUs + PMUs + memory
    controllers. Most CGRAs use 2D meshes (Plasticine v2: 4x8 mesh of
    32 PCUs, estimated). Academic / research CGRAs typically don't
    publish NoC details, so the ``confidence`` field defaults to
    THEORETICAL.

    THIRD cross-block-kind type reuse: ``confidence`` reuses
    ``DataConfidence`` from ``process_node`` (same as NPU). Field
    names align with NPU/GPU/CPU so v6 unification is mechanical.
    """

    topology: CGRANoCTopology = Field(...)
    bisection_bandwidth_gbps: float = Field(..., gt=0)
    unit_count: int = Field(
        ..., gt=0,
        description="Number of fabric endpoints (= num_pcus typically)",
    )
    flit_size_bytes: int = Field(..., gt=0)

    # Mesh-specific (optional; only populated when topology=MESH_2D
    # or TORUS_2D)
    mesh_rows: int | None = Field(default=None, gt=0)
    mesh_cols: int | None = Field(default=None, gt=0)

    hop_latency_ns: float = Field(..., ge=0)
    pj_per_flit_per_hop: float = Field(..., ge=0)
    routing_distance_factor: float = Field(1.0, gt=0)

    confidence: DataConfidence = Field(
        DataConfidence.THEORETICAL,
        description=(
            "Provenance of NoC numbers. Research CGRAs rarely publish "
            "fabric details so THEORETICAL is the dominant case."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_mesh_dims(self) -> "CGRAOnDieFabric":
        """When topology in (MESH_2D, TORUS_2D), both mesh_rows and
        mesh_cols should be set (and their product should match
        unit_count); when topology is CROSSBAR, both should be None."""
        is_mesh = self.topology in (
            CGRANoCTopology.MESH_2D, CGRANoCTopology.TORUS_2D,
        )
        if is_mesh:
            if self.mesh_rows is None or self.mesh_cols is None:
                raise ValueError(
                    f"topology={self.topology.value} requires both "
                    f"mesh_rows and mesh_cols to be set; got "
                    f"rows={self.mesh_rows}, cols={self.mesh_cols}"
                )
            if self.mesh_rows * self.mesh_cols != self.unit_count:
                raise ValueError(
                    f"topology={self.topology.value}: mesh_rows * "
                    f"mesh_cols ({self.mesh_rows} * {self.mesh_cols} = "
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
# Thermal profile (single operating point on most research CGRAs)
# ---------------------------------------------------------------------------

class CGRAThermalProfile(BaseModel):
    """One CGRA operating point. Research-class CGRAs typically ship
    a single profile (Plasticine v2: 15W passive air, no DVFS); GPU-
    style multi-profile DVFS is rare. The schema uses a scalar
    ``clock_mhz`` plus ``dvfs_enabled`` flag instead of the GPU/CPU
    ``ClockDomain`` (base/boost/sustained) so the common case stays
    clean. Mirrors ``NPUThermalProfile`` field-by-field."""

    name: str = Field(...)
    tdp_watts: float = Field(..., gt=0)
    cooling_solution_id: str = Field(...)

    clock_mhz: float = Field(..., gt=0, description="Operating frequency")
    dvfs_enabled: bool = Field(
        False,
        description=(
            "False is the CGRA default (single fixed operating point). "
            "True only for the rare CGRA with multiple thermal profiles "
            "and frequency scaling between them."
        ),
    )

    # Per-precision empirical numbers, same shape as GPU/CPU/KPU/NPU
    efficiency_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    instruction_efficiency_by_precision: dict[str, float] = Field(default_factory=dict)
    memory_bottleneck_factor_by_precision: dict[str, float] = Field(default_factory=dict)

    vdd_v: float | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "CGRAThermalProfile":
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

class CGRATheoreticalPerformance(BaseModel):
    """Roll-up peak ops/sec per precision for the CGRA. Same shape as
    GPU/CPU/NPU. CGRAs commonly include both INT and emulated-FP
    entries (Plasticine v2: INT8 + emulated FP16 + emulated FP32),
    distinguishing them from INT-only NPUs."""

    peak_ops_per_sec_by_precision: dict[str, float] = Field(...)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_positive(self) -> "CGRATheoreticalPerformance":
        for prec, value in self.peak_ops_per_sec_by_precision.items():
            if value < 0:
                raise ValueError(
                    f"peak_ops_per_sec_by_precision[{prec!r}] = {value} "
                    f"must be >= 0"
                )
        return self


# ---------------------------------------------------------------------------
# CGRABlock (the discriminated union member)
# ---------------------------------------------------------------------------

class CGRABlock(BaseModel):
    """CGRA compute block. Carries the CGRA-specific architectural
    description: PCU + PMU hierarchy, fabric reconfiguration overhead,
    PMU + L2 + host-DRAM memory subsystem, PCU mesh on-die fabric,
    and CGRA-only scheduler attributes.

    The discriminator value ``BlockKind.CGRA`` is wired in
    ``compute_product.py``. Imports are arranged so this module does
    not import from ``compute_product`` (compute_product imports from
    here). Same pattern as GPUBlock / CPUBlock / NPUBlock.
    """

    kind: Literal["cgra"] = Field(
        "cgra",
        description="Discriminator -- always 'cgra' for CGRABlock",
    )

    # PCU hierarchy
    num_pcus: int = Field(
        ..., gt=0,
        description=(
            "Number of Pattern Compute Units. Plasticine v2: 32. "
            "SambaNova RDU: ~1000+ (future YAML). Cerebras WSE: "
            "850000+ (future YAML; wafer-scale)."
        ),
    )
    macs_per_pcu: int = Field(
        ..., gt=0,
        description=(
            "Multiply-Accumulate units per PCU. Medium granularity is "
            "the Plasticine v2 design point (8 MACs/PCU); fine-grain "
            "CGRAs (Wave Computing) use 1-2; coarse-grain (SambaNova) "
            "use 64+."
        ),
    )

    # Reconfiguration overhead -- the defining CGRA characteristic
    reconfig_overhead_cycles: int = Field(
        ..., ge=0,
        description=(
            "Cycles to fully reconfigure the fabric between compiled "
            "programs. Plasticine v2: ~1000 cycles. The Achilles heel "
            "of CGRAs vs fixed-function NPUs -- downstream estimators "
            "should penalize workloads with high kernel-switching "
            "frequency."
        ),
    )
    supports_partial_reconfig: bool = Field(
        False,
        description=(
            "Forward-compat hint for PCU-region-level dynamic remap. "
            "Plasticine v2 is whole-fabric only (False); future CGRAs "
            "may support this."
        ),
    )

    # Single compute fabric in the common case (Plasticine); could
    # grow to multiple fabrics for a CGRA with heterogeneous PCU types.
    compute_fabrics: list[CGRAComputeFabric] = Field(..., min_length=1)

    # Precisions supported chip-wide -- union of compute_fabrics[*].ops_per_unit_per_clock
    multi_precision_alu: list[str] = Field(default_factory=list)

    memory: CGRAMemorySubsystem = Field(...)
    noc: CGRAOnDieFabric = Field(...)

    # CGRA-only scheduler / mapper attributes
    min_occupancy: float = Field(
        0.3, ge=0.0, le=1.0,
        description=(
            "Lower default (0.3) than NPU (0.8) because reconfig "
            "overhead amortizes only on long-running kernels; PCU "
            "utilization varies more than fixed-function dataflow."
        ),
    )
    max_concurrent_models: int = Field(
        1, gt=0,
        description=(
            "Maximum number of compiled programs the CGRA can switch "
            "between without recompilation. Most CGRAs run one program "
            "at a time (max_concurrent_models=1); future partial-reconfig "
            "SKUs may support more."
        ),
    )
    wave_quantization: int = Field(
        1, gt=0,
        description="CGRAs don't wave-quantize; default 1 keeps shape consistent",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_noc_unit_count_matches(self) -> "CGRABlock":
        """noc.unit_count should equal num_pcus. Catches YAMLs where
        the NoC was authored against a different PCU count than the
        block declared. Mirrors NPUBlock's validator."""
        if self.noc.unit_count != self.num_pcus:
            raise ValueError(
                f"noc.unit_count ({self.noc.unit_count}) must equal "
                f"num_pcus ({self.num_pcus})"
            )
        return self
