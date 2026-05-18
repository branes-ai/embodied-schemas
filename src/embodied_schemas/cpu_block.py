"""CPU compute block for ``ComputeProduct`` (v3 schema, additive).

PR 2 of the CPU sprint scoped at ``graphs#182``. Adds the third
member of the ``Block`` discriminated union after KPU (v1) and GPU
(v2). Modeled directly off the field set audited in
``graphs/docs/designs/cpu-compute-product-schema-extension.md``.

Design choice -- this module defines CPU-specific sub-types
(``CPUComputeFabric``, ``CoreClusterSpec``, ``CPUMemorySubsystem``,
``CPUOnDieFabric``, ``CPUThermalProfile``, ``CPUTheoreticalPerformance``)
rather than generalizing the KPU/GPU sub-types in place. Same
rationale as the GPU sprint: with only 3 architectures the right
unification shape isn't obvious yet. Defer rename + unify to v4 when
a 4th architecture (NPU? DSP?) lands.

**First cross-block-kind type sharing**: ``CPUThermalProfile`` reuses
``ClockDomain`` from ``gpu_block``. ``ClockDomain`` is the right
shape for any per-thermal-profile clock specification (base / boost /
sustained); it doesn't make sense to duplicate it. This is the first
data point in what should eventually become a vendor-neutral shared
``compute_block_common`` module.

Future-deferred (v4+):

  - Multi-socket NUMA topology (NUMA domain count, per-domain memory
    controllers, NUMA-aware bisection bandwidth)
  - AMD chiplet topology (IOD + multiple CCDs as separate ``Die``
    entries with ``Packaging.kind=CHIPLET``)
  - Apple M-series unified memory (zero-copy CPU/GPU sharing) --
    needs a cross-block memory link concept
  - AMX as a first-class fabric (Sapphire Rapids server-only; out of
    scope for the i7-12700K reference SKU but the
    ``CPUISAExtension.AMX_TILE`` enum value is already declared so
    Xeon SKUs can use it without schema churn)
  - SMT performance counter abstractions (separate from raw SMT
    threads)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.gpu import MemoryType
from embodied_schemas.gpu_block import ClockDomain  # cross-block-kind reuse
from embodied_schemas.process_node import CircuitClass


# ---------------------------------------------------------------------------
# Helper enums
# ---------------------------------------------------------------------------

class CoreClusterKind(str, Enum):
    """What kind of core cluster this is. PERFORMANCE / EFFICIENT
    cover Intel hybrid (Alder Lake, Raptor Lake, Meteor Lake) and
    Apple M-series. BIG / LITTLE cover ARM big.LITTLE. HOMOGENEOUS
    is the catch-all for AmpereOne / EPYC / Xeon non-hybrid SKUs."""

    PERFORMANCE = "performance"
    EFFICIENT = "efficient"
    BIG = "big"
    LITTLE = "little"
    HOMOGENEOUS = "homogeneous"


class CPUISAExtension(str, Enum):
    """Which SIMD / matrix-extension a fabric implements. A single CPU
    core can host multiple fabrics (Sapphire Rapids: AVX2 + AVX-512 +
    AMX simultaneously) -- model each as a separate ``CPUComputeFabric``
    entry on the cluster. Add new values when new SKUs need them; this
    enum is intentionally not exhaustive (the schema has no way to
    enumerate every future ISA extension up front)."""

    # x86
    SCALAR_X86 = "scalar_x86"
    SSE2 = "sse2"
    AVX2 = "avx2"
    AVX_VNNI = "avx_vnni"
    AVX512 = "avx512"
    AVX512_VNNI = "avx512_vnni"
    AVX512_BF16 = "avx512_bf16"
    AVX512_FP16 = "avx512_fp16"
    AMX_TILE = "amx_tile"
    AMX_BF16 = "amx_bf16"
    AMX_INT8 = "amx_int8"

    # ARM
    SCALAR_ARM = "scalar_arm"
    NEON = "neon"
    SVE = "sve"
    SVE2 = "sve2"
    SVE_BF16 = "sve_bf16"
    SVE_INT8 = "sve_int8"
    SME = "sme"  # Scalable Matrix Extension


class L2Layout(str, Enum):
    """How L2 is shared inside a cluster. PRIVATE_PER_CORE is
    Intel P-core / AmpereOne / Xeon style. SHARED_PER_CLUSTER is
    Intel E-core (4 E-cores share 2 MB) / Apple M-series E-cluster
    style. SHARED_GLOBAL means the cluster's "L2" is actually a
    chip-wide structure (uncommon but possible on some embedded
    SoCs)."""

    PRIVATE_PER_CORE = "private_per_core"
    SHARED_PER_CLUSTER = "shared_per_cluster"
    SHARED_GLOBAL = "shared_global"


class CPUNoCTopology(str, Enum):
    """On-die interconnect topology. Intel client uses RING; Intel
    server uses MESH_2D; AMD uses IOD_PLUS_CCD (separate IO die
    fanning out to compute chiplets); Apple M-series uses fabric
    closer to MESH_2D. DOUBLE_RING captures Intel client SKUs that
    run two parallel ring buses for snoop and data."""

    RING = "ring"
    DOUBLE_RING = "double_ring"
    MESH_2D = "mesh_2d"
    IO_DIE_PLUS_CCD = "io_die_plus_ccd"
    INFINITY_FABRIC = "infinity_fabric"


# ---------------------------------------------------------------------------
# Compute fabric (per-fabric SIMD / matrix unit on a CPU core)
# ---------------------------------------------------------------------------

class CPUComputeFabric(BaseModel):
    """One compute fabric on a CPU core (AVX2, AVX-512, AVX-VNNI, AMX,
    NEON, SVE, SME). A single core can host multiple fabrics; carry
    each as a separate entry on ``CoreClusterSpec.compute_fabrics``.

    Mirrors ``GPUComputeFabric`` field-by-field so the v4 unification
    is mechanical. The shape difference is in the discriminator
    (``isa_extension`` instead of ``fabric_kind``) -- CPUs identify
    fabrics by the ISA extension they implement, GPUs by the
    architectural unit kind (CUDA core vs Tensor core).
    """

    isa_extension: CPUISAExtension = Field(...)
    circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library used for this fabric"
    )
    ops_per_core_per_clock: dict[str, int] = Field(
        ...,
        description=(
            "Ops per core per clock keyed on precision name "
            "('fp64','fp32','fp16','bf16','int8','int4'). For an AVX2 "
            "P-core: FP32=16 (8 lanes * 2 FMA pipes), INT8=64 (AVX-VNNI "
            "VPDPBUSD: 4 INT8 ops/lane * 8 lanes * 2 pipes). For an "
            "AMX tile: BF16=1024 ops/clock (16x16x32 tile * 2 ops)."
        ),
    )
    energy_per_flop_fp32_pj: float = Field(
        ..., gt=0,
        description=(
            "Energy per FP32 FMA in picojoules at the fabric's nominal "
            "operating point. AVX-VNNI INT8 typically 0.3x of FP32 "
            "energy (denser packing); AMX further reduces by ~2x for "
            "matrix-shaped workloads."
        ),
    )
    energy_scaling: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Multiplier on ``energy_per_flop_fp32_pj`` for each "
            "precision. Empty means scale linearly by bytes_per_element."
        ),
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Core cluster (the new abstraction for hybrid CPUs)
# ---------------------------------------------------------------------------

class CoreClusterSpec(BaseModel):
    """One core cluster within a CPU. The first sub-type that maps
    cleanly to ARM big.LITTLE, Apple M-series performance/efficiency
    clusters, AND Intel hybrid (Alder/Raptor/Meteor Lake) cores.

    AmpereOne / EPYC / Xeon non-hybrid SKUs have exactly one cluster
    with ``cluster_kind=HOMOGENEOUS``. Alder Lake i7-12700K has two:
    one PERFORMANCE (8 P-cores SMT'd) and one EFFICIENT (4 E-cores
    no SMT, shared 2 MB L2).

    Memory hierarchy: ``l1_kib_per_core`` covers the per-core L1
    (always private). L2 is layout-dependent: PRIVATE_PER_CORE uses
    ``l2_kib_per_core``, SHARED_PER_CLUSTER uses ``l2_kib_shared``,
    SHARED_GLOBAL uses neither (the L2 is on ``CPUMemorySubsystem``).
    """

    cluster_kind: CoreClusterKind = Field(...)
    num_cores: int = Field(..., gt=0)
    smt_threads: int = Field(
        1, ge=1,
        description="Hardware threads per core (SMT/HyperThreading). 1=no SMT.",
    )
    aggregate_weight: float = Field(
        1.0, gt=0,
        description=(
            "Multiplier when rolling effective_cores across clusters. "
            "1.0 for PERFORMANCE / HOMOGENEOUS / BIG; ~0.6 for "
            "EFFICIENT (Intel E-core empirical performance ratio); "
            "~0.5 for LITTLE (ARM big.LITTLE)."
        ),
    )

    compute_fabrics: list[CPUComputeFabric] = Field(
        ..., min_length=1,
        description=(
            "All SIMD / matrix fabrics this cluster's cores can issue "
            "into. A Sapphire Rapids P-cluster carries AVX2 + AVX-512 "
            "+ AMX (3 entries); an Alder Lake P-cluster carries just "
            "AVX2 + AVX-VNNI."
        ),
    )

    # Per-core L1 (always private)
    l1_kib_per_core: int = Field(
        ..., gt=0,
        description="Per-core L1 capacity in KiB (data + instruction split aggregated)",
    )

    # Per-cluster L2 (layout-dependent)
    l2_layout: L2Layout = Field(...)
    l2_kib_per_core: int = Field(
        0, ge=0,
        description=(
            "Per-core L2 in KiB. Set when l2_layout == PRIVATE_PER_CORE; "
            "0 otherwise. Intel P-core: 1280 (1.25 MB)."
        ),
    )
    l2_kib_shared: int = Field(
        0, ge=0,
        description=(
            "Cluster-shared L2 in KiB. Set when l2_layout == "
            "SHARED_PER_CLUSTER; 0 otherwise. Intel E-cluster of 4 "
            "shares 2048 (2 MB)."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_l2_layout_consistency(self) -> "CoreClusterSpec":
        """Enforce that l2_kib_per_core / l2_kib_shared match the
        declared l2_layout. Catches typo'd YAMLs where someone set the
        wrong field for the layout (e.g. PRIVATE_PER_CORE with
        l2_kib_shared=2048)."""
        if self.l2_layout == L2Layout.PRIVATE_PER_CORE:
            if self.l2_kib_per_core <= 0:
                raise ValueError(
                    "l2_layout=PRIVATE_PER_CORE requires l2_kib_per_core > 0; "
                    f"got {self.l2_kib_per_core}"
                )
            if self.l2_kib_shared > 0:
                raise ValueError(
                    "l2_layout=PRIVATE_PER_CORE must have l2_kib_shared == 0; "
                    f"got {self.l2_kib_shared}"
                )
        elif self.l2_layout == L2Layout.SHARED_PER_CLUSTER:
            if self.l2_kib_shared <= 0:
                raise ValueError(
                    "l2_layout=SHARED_PER_CLUSTER requires l2_kib_shared > 0; "
                    f"got {self.l2_kib_shared}"
                )
            if self.l2_kib_per_core > 0:
                raise ValueError(
                    "l2_layout=SHARED_PER_CLUSTER must have "
                    f"l2_kib_per_core == 0; got {self.l2_kib_per_core}"
                )
        # SHARED_GLOBAL: both should be 0 (L2 is reported on CPUMemorySubsystem)
        elif self.l2_layout == L2Layout.SHARED_GLOBAL:
            if self.l2_kib_per_core > 0 or self.l2_kib_shared > 0:
                raise ValueError(
                    "l2_layout=SHARED_GLOBAL must have both l2_kib_per_core "
                    f"and l2_kib_shared == 0 (L2 lives on CPUMemorySubsystem); "
                    f"got per_core={self.l2_kib_per_core}, "
                    f"shared={self.l2_kib_shared}"
                )
        return self


# ---------------------------------------------------------------------------
# Memory subsystem (chip-shared L3 + DRAM only; per-cluster L1/L2 is on CoreClusterSpec)
# ---------------------------------------------------------------------------

class CPUMemorySubsystem(BaseModel):
    """CPU memory hierarchy at the chip-shared level. Per-cluster L1
    and L2 live on ``CoreClusterSpec``; this type carries only the
    chip-wide L3 LLC + DRAM.

    Real cache coherence is the CPU defining feature vs GPU/KPU
    (which model coherence as 'none' for SIMT). The protocol field is
    free-form because the protocol space is rich (MESI / MOESI / MESIF
    / Directory-MESI / snoopy / source-snoop) and pinning to an enum
    forces a schema bump for every variant.
    """

    # Off-chip (main) memory
    memory_type: MemoryType = Field(..., description="Main-memory technology")
    memory_size_gb: float = Field(..., gt=0)
    memory_bus_bits: int = Field(..., gt=0)
    memory_bandwidth_gbps: float = Field(..., gt=0)
    memory_controllers: int = Field(..., gt=0)

    # Chip-wide L3 (the LLC on most CPUs)
    l3_present: bool = Field(True, description="False on rare cache-light CPUs")
    l3_total_kib: int = Field(0, ge=0)

    # Optional L4 (eDRAM on some Intel client SKUs; HBM on Xeon Max)
    l4_present: bool = Field(False)
    l4_total_kib: int = Field(0, ge=0)
    l4_kind: str = Field(
        "",
        description=(
            "'edram' (Intel Crystalwell), 'hbm' (Xeon Max), or '' "
            "when l4_present=False"
        ),
    )

    # Cache coherence protocol -- free-form per the design doc
    coherence_protocol: str = Field(
        "snoopy_mesi",
        description=(
            "Coherence protocol. Common: 'snoopy_mesi', 'snoopy_moesi', "
            "'mesif', 'directory_mesi', 'source_snoop', 'cxl', or 'none' "
            "for non-coherent (rare on CPUs)."
        ),
    )

    # Per-byte energy
    read_energy_pj_per_byte: float | None = Field(default=None, ge=0)
    write_energy_pj_per_byte: float | None = Field(default=None, ge=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_l3_consistency(self) -> "CPUMemorySubsystem":
        """l3_present and l3_total_kib must agree (mirrors GPUMemorySubsystem)."""
        if self.l3_present and self.l3_total_kib <= 0:
            raise ValueError(
                f"l3_present=True requires l3_total_kib > 0; got {self.l3_total_kib}"
            )
        if not self.l3_present and self.l3_total_kib > 0:
            raise ValueError(
                f"l3_present=False requires l3_total_kib == 0; got {self.l3_total_kib}"
            )
        return self

    @model_validator(mode="after")
    def _validate_l4_consistency(self) -> "CPUMemorySubsystem":
        """Same shape as l3 consistency. l4_kind must be set if present,
        and must be empty when not present (catches typo'd YAMLs that
        toggled one field without the other)."""
        if self.l4_present and self.l4_total_kib <= 0:
            raise ValueError(
                f"l4_present=True requires l4_total_kib > 0; got {self.l4_total_kib}"
            )
        if not self.l4_present and self.l4_total_kib > 0:
            raise ValueError(
                f"l4_present=False requires l4_total_kib == 0; got {self.l4_total_kib}"
            )
        if self.l4_present and not self.l4_kind:
            raise ValueError(
                "l4_present=True requires non-empty l4_kind ('edram', 'hbm', etc.)"
            )
        if not self.l4_present and self.l4_kind:
            raise ValueError(
                f"l4_present=False requires empty l4_kind; got {self.l4_kind!r}"
            )
        return self


# ---------------------------------------------------------------------------
# On-die fabric
# ---------------------------------------------------------------------------

# CPUOnDieFabric now inherits from the v10 ``OnDieFabric`` base
# (graphs#217 PR 3). The base provides the 7 shared fields + optional
# mesh dims + confidence; the subclass contributes only the typed
# CPU-specific topology enum.
#
# **v10 field rename**: the field previously called ``stop_count``
# is now ``unit_count`` (inherited from the base). The base's field
# description documents the per-arch meaning: "ring stops" for CPU.
# SKU YAMLs migrated atomically in this PR.
from embodied_schemas.compute_block_common import OnDieFabric


class CPUOnDieFabric(OnDieFabric):
    """CPU on-die interconnect. Intel client uses RING; Intel server
    MESH_2D; AMD IO_DIE_PLUS_CCD (multi-die, modeled here as a single
    fabric -- chiplet support split deferred).

    Inherits all shared NoC fields from ``OnDieFabric``. The inherited
    ``unit_count`` field counts ring stops on RING topology
    (i7-12700K: 8 P + 4 E = 12), or row * col on MESH, or CCD count
    on IO_DIE_PLUS_CCD. Pre-v10 YAMLs used ``stop_count`` -- the
    field was renamed for cross-block-kind consistency.
    """

    topology: CPUNoCTopology = Field(...)


# ---------------------------------------------------------------------------
# Thermal profile -- first cross-block-kind type sharing (reuses ClockDomain)
# ---------------------------------------------------------------------------

class CPUThermalProfile(BaseModel):
    """One CPU thermal/power-limit operating point.

    CPUs use Intel-style power-limit windowing: PL1 (long-term TDP),
    PL2 (short-term boost), optionally PL3/PL4 (instantaneous).
    Each ``CPUThermalProfile`` represents one operating point.

    Because hybrid CPUs run different clocks per cluster (P-core can
    boost higher than E-core), the ``per_cluster_clock_domain`` field
    keys each cluster name to its own ``ClockDomain`` (base / boost /
    sustained). This is the **first cross-block-kind type sharing** in
    the schema -- ``ClockDomain`` lives in ``gpu_block.py`` and is
    reused here verbatim. Future v4 generalization will move
    ``ClockDomain`` to a vendor-neutral module.
    """

    name: str = Field(..., description="Profile label, e.g., '125W-PL1', '241W-PL2'")
    tdp_watts: float = Field(..., gt=0, description="Power limit at this profile")
    cooling_solution_id: str = Field(
        ..., description="References data/cooling-solutions/<id>.yaml"
    )

    # Per-cluster DVFS. Cluster name (matches CoreClusterSpec.cluster_kind.value
    # OR a free-form id when the CPU has multiple clusters of the same kind).
    per_cluster_clock_domain: dict[str, ClockDomain] = Field(
        ...,
        description=(
            "Cluster name -> ClockDomain mapping. For Alder Lake: "
            "{'performance': ClockDomain(...), 'efficient': ClockDomain(...)}. "
            "For homogeneous CPUs: {'homogeneous': ClockDomain(...)}."
        ),
    )

    # Optional per-precision empirical efficiency (matches GPU/KPU shape)
    efficiency_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    vdd_v: float | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "CPUThermalProfile":
        for prec, value in self.efficiency_factor_by_precision.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"efficiency_factor_by_precision[{prec!r}] = {value} "
                    f"is outside [0, 1]"
                )
        return self


# ---------------------------------------------------------------------------
# Theoretical performance roll-up
# ---------------------------------------------------------------------------

# CPUTheoreticalPerformance is now an alias of the unified
# ``TheoreticalPerformance`` from ``compute_block_common`` (v8 follow-up
# -- branes-ai/graphs#210). The class body was byte-identical to 4
# other per-block-kind classes (NPU/CGRA/DPU/TPU); the unified type
# accepts the same data shape. CPU's ``peak_ops_per_sec_by_precision``
# typically carries scalar/SIMD throughput per precision; the optional
# ``sparse_peak_ops_per_sec_by_precision`` field (GPU-specific today)
# defaults to None for CPU SKUs.
from embodied_schemas.compute_block_common import TheoreticalPerformance
CPUTheoreticalPerformance = TheoreticalPerformance


# ---------------------------------------------------------------------------
# CPUBlock (the discriminated union member)
# ---------------------------------------------------------------------------

class CPUBlock(BaseModel):
    """CPU compute block. Carries the CPU-specific architectural
    description: hybrid core clusters, chip-shared memory hierarchy,
    on-die interconnect, and CPU-only scheduler / mapper attributes
    (SIMD width, simd_efficiency by op-kind).

    The discriminator value ``BlockKind.CPU`` is wired in
    ``compute_product.py``. Imports are arranged so this module does
    not import from ``compute_product`` (compute_product imports from
    here).
    """

    # IMPORTANT: kind is set as ``Literal["cpu"]`` (not BlockKind.CPU)
    # to avoid circular import with compute_product. compute_product.py
    # validates that this matches BlockKind.CPU.value at the
    # discriminator level. Same pattern as GPUBlock.
    kind: Literal["cpu"] = Field(
        "cpu",
        description="Discriminator -- always 'cpu' for CPUBlock",
    )

    # Core hierarchy
    core_clusters: list[CoreClusterSpec] = Field(
        ..., min_length=1,
        description=(
            "All core clusters on this CPU. Homogeneous CPUs have one "
            "(cluster_kind=HOMOGENEOUS); Intel hybrid has 2 "
            "(PERFORMANCE + EFFICIENT); ARM big.LITTLE has 2 (BIG + "
            "LITTLE); a future heterogeneous SKU could have 3+."
        ),
    )
    total_effective_cores: int = Field(
        ..., gt=0,
        description=(
            "Roll-up effective core count, weighted by "
            "core_clusters[].aggregate_weight. For Alder Lake i7-12700K: "
            "8 P + int(4 E * 0.6) = 10. Used by downstream consumers "
            "that need a single 'cores' number."
        ),
    )
    simd_width_lanes: int = Field(
        ..., gt=0,
        description=(
            "FP32 SIMD lane count of the dominant fabric. AVX2: 8 "
            "(256-bit / 32-bit lane). AVX-512: 16. NEON: 4. SVE: "
            "implementation-defined (128/256/512/1024 etc.)."
        ),
    )

    # Multi-precision support, chip-wide (union of all clusters' fabrics)
    multi_precision_alu: list[str] = Field(default_factory=list)

    # Memory hierarchy (chip-shared parts; per-cluster L1/L2 lives on CoreClusterSpec)
    memory: CPUMemorySubsystem = Field(...)

    # On-die interconnect
    noc: CPUOnDieFabric = Field(...)

    # CPU-only scheduler / mapper attributes
    simd_efficiency_by_op_kind: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Vectorization-friendliness per op kind. Reference values "
            "from CPUMapper analytical defaults: 'elementwise'=0.95, "
            "'matrix'=0.80, 'default'=0.70 for AVX2 hybrid Alder Lake."
        ),
    )
    min_occupancy: float = Field(0.4, ge=0.0, le=1.0)
    max_concurrent_threads: int = Field(
        ..., gt=0,
        description=(
            "Maximum runnable threads. = sum(num_cores * smt_threads) "
            "across clusters. For i7-12700K: 8*2 + 4*1 = 20."
        ),
    )
    wave_quantization: int = Field(
        1, gt=0,
        description="CPUs don't wave-quantize; default 1 keeps shape consistent with GPU",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_simd_efficiency_ranges(self) -> "CPUBlock":
        for op_kind, value in self.simd_efficiency_by_op_kind.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"simd_efficiency_by_op_kind[{op_kind!r}] = {value} "
                    f"must be in [0, 1]"
                )
        return self
