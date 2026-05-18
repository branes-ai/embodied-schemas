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
``CGRAMemorySubsystem.external_dram_type`` reuses ``MemoryType`` from
``gpu`` (same as NPU's ``external_dram_type``; v11 rename matches the
5-of-6 convention, with ``dram_attachment=host_bus`` preserving the
PCIe-DRAM semantic the original ``host_dram_*`` naming captured);
``CGRAComputeFabric.
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

from embodied_schemas.compute_block_common import DramAttachment
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
    """CGRA memory hierarchy: PMU-dominant + small shared L2 + external
    DRAM. Plasticine v2 has 64 KB PMU per PCU + 2 MB shared L2 +
    4 GB DDR4 (accessed via host bus, NOT chip-attached -- captured
    via ``dram_attachment=host_bus``).

    **v11 (graphs#219) rename**: the field family was previously named
    ``has_host_dram`` / ``host_dram_*`` to signal the bus-mediated
    nature. v11 reconciled the naming to match the 5-of-6 convention
    (``has_external_dram`` / ``external_dram_*``) used by
    NPU/DPU/TPU/DSP, AND added the explicit ``dram_attachment``
    discriminator (CHIP_ATTACHED | HOST_BUS) so the semantic
    distinction the old naming captured is preserved.

    Plasticine v2 YAML migrated atomically in this PR: renamed fields
    + ``dram_attachment: host_bus``.
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

    # External DRAM. Gated by has_external_dram bool; when False all
    # the external_dram_* fields must be None / 0 (validator enforces).
    # Plasticine v2: True with DDR4 / 4 GB via host bus (set
    # dram_attachment=host_bus on the YAML). Future chip-attached
    # CGRAs (Cerebras WSE) would set dram_attachment=chip_attached.
    has_external_dram: bool = Field(False)
    external_dram_type: MemoryType | None = Field(default=None)
    external_dram_size_gb: float | None = Field(default=None, ge=0)
    external_dram_bandwidth_gbps: float | None = Field(default=None, ge=0)

    # v11 (graphs#219): DRAM-attachment discriminator. Optional in v11
    # for backward compat. Plasticine v2 YAML populates host_bus
    # explicitly to preserve the PCIe-DRAM semantic the original
    # has_host_dram naming captured.
    dram_attachment: DramAttachment | None = Field(default=None)

    # Energy per byte for the dominant on-chip memory tier (PMU + L2).
    # ~12 pJ/B for Plasticine 28nm; cheaper than DRAM access.
    pmu_access_energy_pj_per_byte: float = Field(..., gt=0)

    # Energy per byte for external DRAM access. Only meaningful when
    # has_external_dram=True. ~20 pJ/B for Plasticine DDR4 host-bus
    # path on 28nm (includes PCIe transit + host DRAM access).
    external_dram_access_energy_pj_per_byte: float = Field(0.0, ge=0)

    # Cache coherence. CGRA default is "none" since compiler-routed
    # spatial dataflow has no host-coherent cache. Free-form string
    # for future host-coherent CGRAs (rare).
    coherence_protocol: str = Field(
        "none",
        description="Common: 'none' (CGRA default), 'pcie' (host DMA)",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_external_dram_consistency(self) -> "CGRAMemorySubsystem":
        """When ``has_external_dram=True`` the external_dram_* fields
        must all be populated; when False they must all be None / 0.
        Catches typo'd YAMLs that toggled one field without the other.
        Mirrors the external_dram validator on ``NPUMemorySubsystem``."""
        if self.has_external_dram:
            missing = []
            if self.external_dram_type is None:
                missing.append("external_dram_type")
            if self.external_dram_size_gb is None or self.external_dram_size_gb <= 0:
                missing.append("external_dram_size_gb")
            if (self.external_dram_bandwidth_gbps is None
                    or self.external_dram_bandwidth_gbps <= 0):
                missing.append("external_dram_bandwidth_gbps")
            # v12 (graphs#222): dram_attachment required when external DRAM present.
            if self.dram_attachment is None:
                missing.append("dram_attachment")
            if missing:
                raise ValueError(
                    f"has_external_dram=True requires all of "
                    f"external_dram_type, external_dram_size_gb, "
                    f"external_dram_bandwidth_gbps, dram_attachment "
                    f"to be populated; missing/zero: {missing}"
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
            if self.dram_attachment is not None:
                extras.append("dram_attachment")
            if extras:
                raise ValueError(
                    f"has_external_dram=False requires external_dram_* "
                    f"fields to be None / 0; got populated: {extras}"
                )
        return self


# ---------------------------------------------------------------------------
# On-die fabric (PCU mesh, often low-confidence for research SKUs)
# ---------------------------------------------------------------------------

# CGRAOnDieFabric now inherits from the v10 ``OnDieFabric`` base
# (graphs#217 PR 3). The base provides the 7 shared fields + optional
# mesh dims + confidence; the subclass contributes only the typed
# CGRA-specific topology enum and the CGRA mesh-dim validator (handles
# both MESH_2D and TORUS_2D mesh topologies).
from embodied_schemas.compute_block_common import OnDieFabric


class CGRAOnDieFabric(OnDieFabric):
    """CGRA on-die interconnect between PCUs + PMUs + memory
    controllers. Most CGRAs use 2D meshes (Plasticine v2: 4x8 mesh of
    32 PCUs, estimated).

    Inherits all shared NoC fields from ``OnDieFabric``; contributes
    the CGRA-specific topology enum (MESH_2D / TORUS_2D / CROSSBAR)
    and the CGRA mesh-dim consistency validator.
    """

    topology: CGRANoCTopology = Field(...)

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

# CGRAThermalProfile is now an alias of the unified ``ThermalProfile``
# from ``compute_block_common`` (v9 sprint PR 3 -- branes-ai/graphs#215).
# The class body was byte-identical to 4 other per-block-kind classes
# (NPU/DPU/TPU/DSP); the unified type accepts the same data shape.
#
# Backward-compat: ``isinstance(x, CGRAThermalProfile)`` AND
# ``isinstance(x, ThermalProfile)`` both work (same class object).
from embodied_schemas.compute_block_common import ThermalProfile
CGRAThermalProfile = ThermalProfile


# ---------------------------------------------------------------------------
# Theoretical performance roll-up
# ---------------------------------------------------------------------------

# CGRATheoreticalPerformance is now an alias of the unified
# ``TheoreticalPerformance`` from ``compute_block_common`` (v8 follow-up
# -- branes-ai/graphs#210). The class body was byte-identical to 4
# other per-block-kind classes (NPU/CPU/DPU/TPU); the unified type
# accepts the same data shape. CGRAs commonly include both INT and
# emulated-FP entries (Plasticine v2: INT8 + emulated FP16 + emulated
# FP32); the optional ``sparse_peak_ops_per_sec_by_precision`` field
# (GPU-specific today) defaults to None for CGRA SKUs.
from embodied_schemas.compute_block_common import TheoreticalPerformance
CGRATheoreticalPerformance = TheoreticalPerformance


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
