"""DSP compute block for ``ComputeProduct`` (v9 schema, additive).

PR 2 of the DSP mini-sprint scoped at ``graphs#211``. Adds the eighth
+ **final** member of the ``Block`` discriminated union after KPU
(v1), GPU (v2), CPU (v3), NPU (v4), CGRA (v5), DPU (v6), and TPU
(v7); closes the last category schema gap. Modeled directly off the
field set audited in the paper exercise at
``graphs/docs/designs/dsp-compute-product-schema-extension.md``.

Design choice: **first block kind authored after v8 unification**.
DSPBlock uses ``compute_block_common``'s ``TheoreticalPerformance``
/ ``MemoryType`` / ``CircuitClass`` / ``ClockDomain`` /
``DataConfidence`` from day 1 rather than redefining per-architecture
duplicates. ``DSPTheoreticalPerformance = TheoreticalPerformance``
is kept as a symmetry alias (matches the CPU/GPU/NPU/CGRA/DPU/TPU
pattern) for any future caller; the alias IS the same class object,
not a subclass.

DSP-specific architectural patterns the schema covers:

  - **Multi-fabric is the rule, not the exception** -- AI-era DSPs
    canonically pair a vector core (HVX, C7x, CEVA-XM6, Cadence SIMD)
    with a tensor co-processor (HMX, MMA, NeuPro tensor units).
    ``compute_fabrics: list[DSPComputeFabric]`` with ``min_length=1``
    accommodates 1 (Cadence Vision Q8) through 2 (all SoCs + CEVA);
    no upper bound for hypothetical 3-fabric SKUs.
  - **Deployment kind discriminator** (``standalone_ip`` vs
    ``soc_integrated``) -- load-bearing for energy apportionment.
    IP cores quote typical-integration bandwidth; SoC SKUs quote
    measured datasheet bandwidth.
  - **External DRAM bandwidth kind** (``typical`` vs ``measured``)
    -- explicit discriminator preventing false precision in roofline
    analysis when mixing IP-core estimates with SoC measurements.
  - **Multi-profile DVFS** (1 profile for Cadence Q8 / CEVA / Synopsys;
    2 for TDA4VM; 3 for Qualcomm SA8775P) -- ``thermal_profiles: list``
    + ``default_thermal_profile_name`` mirrors the GPU / TPU pattern.
  - **Mixed integer + float precision** (most DSPs ship native
    INT8 / INT16 + FP16 / FP32; newer ones add INT4 / FP8) -- the
    fabric validator requires at least one of {INT8, INT16, FP16,
    FP32} (less strict than TPU's INT8+BF16-only because DSPs span
    signal processing into ML).
  - **VLIW issue width** documented but not load-bearing -- recorded
    as ``vliw_issue_width: int | None`` informational field.
  - **No DSP-specific tile energy decomposition** -- DSPs don't have
    centralized weight FIFOs / unified buffers like TPUs. Per-fabric
    ``energy_per_op_fp32_pj`` + per-precision ``energy_scaling`` is
    sufficient.

Cadence Tensilica Vision Q8 (reference SKU for this sprint) specifics:

  - Pure IP core (``deployment_kind=STANDALONE_IP``)
  - Single fabric (1024-bit SIMD, 32 units, vision-optimized)
  - INT8 / INT16 / FP16 / FP32 (vision-typical precision mix)
  - 1 thermal profile (1W passive)
  - 40 GB/s typical-integration bandwidth (NOT measured)
  - 16nm process

Multi-SKU schema coverage (all 10 DSP SKUs from the paper exercise):

  - **cadence_vision_q8 (this sprint)**: N16, SIMD 32x, 1W, 40 GB/s typical
  - ceva_neupro_npm11: N16, tensor 64 + vector 64, 2W, 50 GB/s typical
  - synopsys_arc_ev7x: N16, (vector + DNN engine), (TBD)
  - qualcomm_sa8775p: N5, HVX 4 + HMX 2, 20/30/45W (3 modes), 90 GB/s measured
  - ti_tda4vm: N16, C7x DSP 8 + MMAv1, 10/20W (2 modes), 60 GB/s measured
  - ti_tda4{al,vh,vl}: N16, C7x DSP (subset configurations)
  - qrb5165 / qualcomm_qcs6490: N7/N6, HVX, edge SoC bandwidths

The schema accommodates all 10 SKUs; only Cadence Vision Q8 ships
in this sprint. Other SKUs land as pure data PRs after this
umbrella closes.

Future-deferred (v10+):

  - Multi-fabric scheduling models (which fabric runs which op,
    fabric-to-fabric handoff costs). Runtime / mapper concern.
  - SoC-level integration validation (which DSPs share which fabric
    NoC with adjacent NPUs / CPUs). Chiplet / system-level scope.
  - v9 NEAR_UNIFIABLE patterns (``ThermalProfile``, ``OnDieFabric``
    cross-block-kind unification). Separate concern.
  - ``has_external_dram`` vs ``has_host_dram`` naming reconciliation.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.compute_block_common import (
    CircuitClass,
    ClockDomain,
    DataConfidence,
    MemoryType,
    TheoreticalPerformance,
)


# ---------------------------------------------------------------------------
# Helper enums
# ---------------------------------------------------------------------------

class DSPFabricKind(str, Enum):
    """The DSP fabric role within the chip.

    Multi-fabric DSPs canonically pair a VECTOR_SIMD fabric (general
    SIMD compute -- conv activations, pooling, pre/post-processing)
    with a TENSOR_MATRIX fabric (matrix accelerator -- conv, matmul).
    VLIW_SCALAR captures the small per-cycle scalar slots that some
    VLIW DSPs ship as a separate fabric (e.g. TI C7x). HYBRID covers
    rare cases where one physical fabric runs both modes (uncommon
    today but reserved).
    """

    VECTOR_SIMD = "vector_simd"
    TENSOR_MATRIX = "tensor_matrix"
    VLIW_SCALAR = "vliw_scalar"
    HYBRID = "hybrid"


class DSPDeploymentKind(str, Enum):
    """How the DSP is delivered. Load-bearing for energy / bandwidth
    interpretation:

    - ``STANDALONE_IP``: licensable IP core (Cadence, CEVA, Synopsys).
      External DRAM bandwidth in the SKU YAML is necessarily "typical
      SoC integration" -- the actual bandwidth depends on the system
      integrator's DDR choice. SoC TDP is the IP core's own envelope.
    - ``SOC_INTEGRATED``: shipping silicon SoC with DSP integrated
      alongside other blocks (CPU + GPU + NPU + ISP). External DRAM
      bandwidth in the YAML is measured; SoC TDP is shared across
      blocks (mapper apportions).
    """

    STANDALONE_IP = "standalone_ip"
    SOC_INTEGRATED = "soc_integrated"


# ---------------------------------------------------------------------------
# Compute fabric (always list; can be length 1 for Cadence-style or
# 2 for multi-fabric SoCs)
# ---------------------------------------------------------------------------

class DSPComputeFabric(BaseModel):
    """One compute fabric on a DSP. Most DSPs ship 2 fabrics
    (vector + tensor); Cadence Vision Q8 is the atypical single-fabric
    reference SKU.

    Mirrors ``DPUComputeFabric`` / ``TPUComputeFabric`` field-by-field
    where possible. The shape differences are (1) the discriminator
    (``fabric_kind`` is DSP-specific -- vector vs tensor vs scalar
    matters for routing decisions) and (2) the energy baseline (FP32,
    matching most DSP vendors' published energy figures, vs DPU/NPU's
    INT8 / TPU's BF16).
    """

    fabric_kind: DSPFabricKind = Field(...)
    circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library used for this fabric"
    )
    num_units: int = Field(
        ..., gt=0,
        description=(
            "Number of fabric units (SIMD lanes / MAC array elements / "
            "VLIW slots). Cadence Vision Q8: 32 SIMD units. TI TDA4VM "
            "C7x fabric: 8 cores. SA8775P HVX: 4 vector engines."
        ),
    )
    ops_per_unit_per_clock: dict[str, int] = Field(
        ...,
        description=(
            "Ops per fabric unit per clock keyed on precision name. "
            "Per-chip peak for a fabric = num_units * "
            "ops_per_unit_per_clock[precision] * clock_hz. Cadence "
            "Vision Q8 SIMD fabric: {'int8': 119, 'int16': 119, "
            "'fp32': 4, 'fp16': 8} -- 1024-bit SIMD divided by element "
            "width gives the ops/cycle per unit."
        ),
    )
    energy_per_op_fp32_pj: float = Field(
        ..., gt=0,
        description=(
            "Energy per FP32 op in picojoules at the fabric's nominal "
            "operating point. FP32 is the canonical DSP baseline "
            "(matches vendor energy publications). Cadence Vision Q8 "
            "(16nm SIMD): ~2.43 pJ per FP32 op."
        ),
    )
    energy_scaling: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Multiplier on ``energy_per_op_fp32_pj`` for each precision. "
            "Cadence Vision Q8: {'int8': 0.15, 'int16': 0.15, "
            "'fp16': 0.50} (INT8 ~7x cheaper than FP32; FP16 half-cost). "
            "FP32 baseline is 1.0 implicitly (omit from this dict)."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_precision_required(self) -> "DSPComputeFabric":
        """DSPs must ship at least one of {INT8, INT16, FP16, FP32} --
        the precisions that justify a DSP's existence (signal processing
        through ML inference). Less strict than TPU's INT8+BF16-only
        because DSPs span a wider workload range.

        Also rejects non-positive ops values (catches the same class
        of bug the DPU loader fix in ``graphs#202`` addressed)."""
        non_positive = [
            precision for precision, value in self.ops_per_unit_per_clock.items()
            if value <= 0
        ]
        if non_positive:
            raise ValueError(
                f"DSPComputeFabric.ops_per_unit_per_clock values must be "
                f"positive; got non-positive entries for: {sorted(non_positive)}"
            )
        precisions = {k.lower() for k in self.ops_per_unit_per_clock}
        canonical = {"int8", "int16", "fp16", "fp32"}
        if not (canonical & precisions):
            raise ValueError(
                "DSPComputeFabric.ops_per_unit_per_clock must include at "
                "least one of {'int8', 'int16', 'fp16', 'fp32'} (the "
                "canonical DSP precision set spanning signal processing "
                f"to ML); got: {sorted(precisions)}"
            )
        return self


# ---------------------------------------------------------------------------
# Memory subsystem (L1 per-unit + optional L2 shared + external DRAM)
# ---------------------------------------------------------------------------

class DSPMemorySubsystem(BaseModel):
    """DSP memory hierarchy: L1 per-unit + optional L2 shared + external
    DRAM. Unlike TPU's Unified Buffer collapse, DSPs preserve the L1/L2
    distinction.

    The ``external_dram_bandwidth_kind`` discriminator is **load-bearing**:
    IP cores (Cadence, CEVA, Synopsys) quote "typical SoC integration"
    bandwidth because the actual bandwidth depends on the system
    integrator's DDR choice. SoC SKUs (Qualcomm SA8775P, TI TDA4) quote
    measured datasheet bandwidth. Mixing them as if equivalent leads
    to false precision in roofline analysis.
    """

    # L1 per-unit
    l1_size_bytes_per_unit: int = Field(
        ..., gt=0,
        description=(
            "Per-unit L1 data cache + scratchpad size in bytes. Cadence "
            "Vision Q8: 32 KiB. TI C7x: 48 KiB (32 KiB cache + 16 KiB "
            "SRAM)."
        ),
    )

    # L2 shared (optional)
    l2_size_bytes_total: int | None = Field(
        default=None, ge=0,
        description=(
            "Total L2 size in bytes (shared across units). Cadence "
            "Vision Q8: 1 MiB. TDA4VM MSMC: 8 MiB. None for IP cores "
            "that don't ship an L2."
        ),
    )
    l2_bandwidth_gbps: float | None = Field(
        default=None, ge=0,
        description="On-chip L2 bandwidth. None when l2_size_bytes_total is None.",
    )

    # External DRAM (gated by has_external_dram bool)
    has_external_dram: bool = Field(False)
    external_dram_type: MemoryType | None = Field(default=None)
    external_dram_size_gb: float | None = Field(default=None, ge=0)
    external_dram_bandwidth_gbps: float | None = Field(default=None, ge=0)
    external_dram_bandwidth_kind: Literal["typical", "measured"] | None = Field(
        default=None,
        description=(
            "Load-bearing discriminator. 'typical' = quoted assuming a "
            "typical SoC integration (IP cores); 'measured' = from the "
            "shipping silicon datasheet (SoCs). Required when "
            "has_external_dram=True."
        ),
    )
    external_dram_access_energy_pj_per_byte: float = Field(
        0.0, ge=0,
        description="Energy per byte for external DRAM access. ~12 pJ/byte for LPDDR5.",
    )

    # Cache coherence
    coherence_protocol: str = Field(
        "none",
        description="Common: 'none' (most DSPs), 'mesi' (SoC L3 shared)",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_external_dram_consistency(self) -> "DSPMemorySubsystem":
        """When ``has_external_dram=True`` all external_dram_* fields
        must be populated (including the explicit ``typical|measured``
        discriminator); when False they must all be None / 0. Mirrors
        the NPU / DPU / TPU pattern + adds the bandwidth_kind
        requirement specific to DSPs."""
        if self.has_external_dram:
            missing = []
            if self.external_dram_type is None:
                missing.append("external_dram_type")
            if self.external_dram_size_gb is None or self.external_dram_size_gb <= 0:
                missing.append("external_dram_size_gb")
            if (self.external_dram_bandwidth_gbps is None
                    or self.external_dram_bandwidth_gbps <= 0):
                missing.append("external_dram_bandwidth_gbps")
            if self.external_dram_bandwidth_kind is None:
                missing.append("external_dram_bandwidth_kind")
            if missing:
                raise ValueError(
                    f"has_external_dram=True requires all of "
                    f"external_dram_type, external_dram_size_gb, "
                    f"external_dram_bandwidth_gbps, "
                    f"external_dram_bandwidth_kind to be populated; "
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
            if self.external_dram_bandwidth_kind is not None:
                extras.append("external_dram_bandwidth_kind")
            if extras:
                raise ValueError(
                    f"has_external_dram=False requires external_dram_* "
                    f"fields to be None / 0; got populated: {extras}"
                )
        return self

    @model_validator(mode="after")
    def _validate_l2_consistency(self) -> "DSPMemorySubsystem":
        """If ``l2_size_bytes_total`` is set, ``l2_bandwidth_gbps``
        must also be set (and vice versa). Catches typo'd YAMLs that
        declare one without the other."""
        has_size = self.l2_size_bytes_total is not None and self.l2_size_bytes_total > 0
        has_bw = self.l2_bandwidth_gbps is not None and self.l2_bandwidth_gbps > 0
        if has_size and not has_bw:
            raise ValueError(
                "l2_size_bytes_total is set but l2_bandwidth_gbps is None/0"
            )
        if has_bw and not has_size:
            raise ValueError(
                "l2_bandwidth_gbps is set but l2_size_bytes_total is None/0"
            )
        return self


# ---------------------------------------------------------------------------
# Thermal profile (1-3 profiles per chip; multi-profile common on
# automotive SoCs)
# ---------------------------------------------------------------------------

class DSPThermalProfile(BaseModel):
    """One DSP operating point. Cardinality varies by SKU:

      - Cadence Vision Q8 / CEVA / Synopsys IP cores: 1 profile
      - TI TDA4VM: 2 profiles (10W / 20W)
      - Qualcomm SA8775P: 3 profiles (20W / 30W / 45W)

    Mirrors ``DPUThermalProfile`` / ``TPUThermalProfile`` field-by-field.
    The DSPBlock holds a ``thermal_profiles: list[DSPThermalProfile]``
    + ``default_thermal_profile_name`` -- same pattern as GPU/TPU
    multi-profile DVFS."""

    name: str = Field(...)
    tdp_watts: float = Field(..., gt=0)
    cooling_solution_id: str = Field(...)

    clock_mhz: float = Field(..., gt=0, description="Operating frequency")
    dvfs_enabled: bool = Field(
        False,
        description=(
            "False is the IP-core default (single fixed operating point). "
            "True for SoC-integrated SKUs with multiple thermal profiles."
        ),
    )

    # Per-precision empirical numbers, same shape as other block kinds
    efficiency_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    instruction_efficiency_by_precision: dict[str, float] = Field(default_factory=dict)
    memory_bottleneck_factor_by_precision: dict[str, float] = Field(default_factory=dict)

    vdd_v: float | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "DSPThermalProfile":
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

# DSPTheoreticalPerformance is an alias of the v8-unified
# ``TheoreticalPerformance`` from ``compute_block_common``. DSPBlock
# is the first block kind authored after v8 unification, so it reuses
# from day 1 rather than redefining the byte-identical type. The alias
# IS the same class object (assignment, not subclass); preserved for
# symmetry with the other 6 modern block kinds (CPU/GPU/NPU/CGRA/DPU/TPU).
DSPTheoreticalPerformance = TheoreticalPerformance


# ---------------------------------------------------------------------------
# DSPBlock (the discriminated union member)
# ---------------------------------------------------------------------------

class DSPBlock(BaseModel):
    """DSP compute block. Carries the DSP-specific architectural
    description: multi-fabric vector / tensor compute, deployment-kind
    discriminator (IP core vs SoC-integrated), memory hierarchy
    (L1 + optional L2 + external DRAM), multi-profile DVFS, and
    DSP-only scheduler attributes.

    The discriminator value ``BlockKind.DSP`` is wired in
    ``compute_product.py``. Imports are arranged so this module does
    not import from ``compute_product`` (compute_product imports from
    here). Same pattern as the other 7 block kinds.
    """

    kind: Literal["dsp"] = Field(
        "dsp",
        description="Discriminator -- always 'dsp' for DSPBlock",
    )

    # Deployment kind (load-bearing for energy / bandwidth interpretation)
    deployment_kind: DSPDeploymentKind = Field(
        ...,
        description=(
            "STANDALONE_IP for licensable IP cores (Cadence, CEVA, "
            "Synopsys); SOC_INTEGRATED for shipping SoCs (Qualcomm, TI). "
            "Consumers use this to decide energy apportionment and how "
            "to interpret external_dram_bandwidth_gbps."
        ),
    )

    # Multi-fabric architecture (always plural, length 1 for Cadence,
    # length 2 for all multi-fabric SoCs + CEVA)
    compute_fabrics: list[DSPComputeFabric] = Field(..., min_length=1)

    # Memory subsystem (L1 per-unit + optional L2 + external DRAM)
    memory: DSPMemorySubsystem = Field(...)

    # Clock domain (reused from compute_block_common -- first block kind
    # to do so from day 1 rather than via v8 follow-up alias)
    clock_domain: ClockDomain = Field(
        ...,
        description=(
            "Chip-level clock domain (base / sustained / boost). "
            "Per-profile clock overrides live on DSPThermalProfile."
        ),
    )

    # Multi-profile thermals (1-3 profiles)
    thermal_profiles: list[DSPThermalProfile] = Field(..., min_length=1)
    default_thermal_profile_name: str = Field(
        ...,
        description=(
            "Name of the default thermal profile (must match one of the "
            "entries in thermal_profiles)."
        ),
    )

    # Theoretical performance roll-up (v8-unified type)
    theoretical_performance: TheoreticalPerformance = Field(
        ...,
        description=(
            "Chip-level peak ops/sec by precision. v8-unified type from "
            "compute_block_common -- the alias DSPTheoreticalPerformance "
            "is also available for symmetry."
        ),
    )
    default_precision: str = Field(
        ...,
        description=(
            "Default precision used by mappers when not otherwise "
            "specified. Cadence Vision Q8: 'int8' (vision-optimized "
            "INT8 is the canonical workload). C7x DSPs: 'fp32' (signal "
            "processing legacy)."
        ),
    )

    # Precisions supported chip-wide -- union of fabrics' ops_per_unit_per_clock
    multi_precision_alu: list[str] = Field(default_factory=list)

    # DSP-only scheduler / mapper attributes
    min_occupancy: float = Field(
        0.7, ge=0.0, le=1.0,
        description=(
            "DSPs are moderately efficient (better than CGRA/DPU FPGAs "
            "but less rigid than fixed TPUs/NPUs). Default 0.7 matches "
            "the Cadence Vision Q8 hand-coded factory."
        ),
    )
    max_concurrent_kernels: int = Field(
        4, gt=0,
        description=(
            "DSPs typically run 1-4 concurrent kernels. SoC-integrated "
            "DSPs may run more if SDK supports kernel queuing."
        ),
    )
    wave_quantization: int = Field(
        4, gt=0,
        description=(
            "SIMD lane group quantization. Informs roofline. Cadence "
            "Vision Q8 + most DSPs: 4 (matches threads_per_unit in the "
            "graphs hand-coded factories)."
        ),
    )
    vliw_issue_width: int | None = Field(
        default=None,
        description=(
            "VLIW issue width if the DSP is VLIW. Informational only -- "
            "matters for compiler scheduling but not analytical roofline. "
            "TI C7x: 8. Cadence Vision Q8: None (pure SIMD, not VLIW)."
        ),
    )

    # Optional NoC confidence indicator (for downstream filtering)
    noc_confidence: DataConfidence = Field(
        DataConfidence.THEORETICAL,
        description=(
            "Confidence in the DSP's on-die fabric numbers. DSP vendors "
            "rarely publish NoC details, so THEORETICAL is the dominant "
            "case."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_default_thermal_profile_exists(self) -> "DSPBlock":
        """default_thermal_profile_name must match one of the profile
        names in thermal_profiles. Catches typo'd YAMLs."""
        profile_names = {p.name for p in self.thermal_profiles}
        if self.default_thermal_profile_name not in profile_names:
            raise ValueError(
                f"default_thermal_profile_name "
                f"({self.default_thermal_profile_name!r}) does not match "
                f"any entry in thermal_profiles "
                f"(available: {sorted(profile_names)})"
            )
        return self

    @model_validator(mode="after")
    def _validate_deployment_dram_kind_consistency(self) -> "DSPBlock":
        """If deployment_kind=STANDALONE_IP and has_external_dram=True,
        the bandwidth_kind must be 'typical' (an IP core cannot have
        measured bandwidth -- the actual bandwidth depends on the SoC
        integration). Mirror constraint for SOC_INTEGRATED is NOT
        enforced because some SoC YAMLs may legitimately quote
        conservative 'typical' numbers when the datasheet doesn't
        publish a single value."""
        if (self.deployment_kind == DSPDeploymentKind.STANDALONE_IP
                and self.memory.has_external_dram
                and self.memory.external_dram_bandwidth_kind != "typical"):
            raise ValueError(
                f"deployment_kind=STANDALONE_IP requires "
                f"external_dram_bandwidth_kind='typical' (IP cores "
                f"cannot have measured bandwidth -- depends on SoC "
                f"integration); got "
                f"{self.memory.external_dram_bandwidth_kind!r}"
            )
        return self
