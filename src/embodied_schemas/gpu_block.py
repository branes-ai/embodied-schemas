"""GPU compute block for ``ComputeProduct`` (v2 schema, additive).

PR 2 of the GPU sprint scoped at ``graphs#171``. Adds the second member
of the ``Block`` discriminated union so a ``ComputeProduct`` can carry
GPU dies in addition to KPU dies. Modeled directly off the field set
audited in
``graphs/docs/designs/gpu-compute-product-schema-extension.md``.

Design choice -- this module defines GPU-specific sub-types
(``GPUComputeFabric``, ``GPUMemorySubsystem``, ``GPUOnDieFabric``,
``GPUThermalProfile``, ``GPUTheoreticalPerformance``) rather than
generalizing the KPU sub-types in place. Rationale:

  - v1 (KPU) shipped its own KPU-shaped types; v2 (GPU) ships its own
    GPU-shaped types. The pattern is consistent and additive.
  - The "rename + generalize" plan in the design doc is the right
    long-term shape, but premature with only 2 architectures. Defer
    to a v3 consolidation PR once a third architecture (CPU? NPU?)
    lands and the right unification is obvious. Forcing the rename
    now would touch all 12 existing KPU YAMLs for no concrete payoff.
  - Existing KPU YAMLs validate identically -- this PR adds new types
    only.

Future-deferred (v3+):

  - Multi-GPU NVLink fabric (``Interconnect`` at the ``DIE_TO_DIE``
    or ``PACKAGE_TO_PACKAGE`` level)
  - MIG (Multi-Instance GPU) partitioning
  - Chiplet GPUs (Blackwell B200's GPU-to-GPU NVHBI)
  - DLA / PVA / ARM-CPU-complex on Tegra SoCs (separate block kinds)
  - Per-precision sparsity throughput multipliers (Ampere 2:4)
  - RT cores as a third compute fabric kind (currently a Tensor /
    CUDA core slot would have to suffice if anyone needs it)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.gpu import MemoryType
from embodied_schemas.process_node import CircuitClass


# ---------------------------------------------------------------------------
# Helper enums + value types
# ---------------------------------------------------------------------------

class GPUFabricKind(str, Enum):
    """What kind of compute fabric this is. Drives per-precision ops/clock
    and energy lookup. v2 covers CUDA cores and Tensor cores; RT cores
    fold in here when raytracing modeling lands."""

    CUDA_CORE = "cuda_core"
    TENSOR_CORE = "tensor_core"
    RT_CORE = "rt_core"


class GPUL1Kind(str, Enum):
    """Storage discipline of the L1 / shared-memory unified store on a
    GPU SM. NVIDIA Ampere SMs use a unified L1 / shared store that can
    be configured at runtime; classified as ``CACHE`` because the
    dominant deployment uses unified L1 cache mode."""

    CACHE = "cache"
    SCRATCHPAD = "scratchpad"
    UNIFIED = "unified"


class GPUL2Topology(str, Enum):
    """Topology of the L2 cache. Ampere SoCs have a single shared L2
    that acts as the LLC. Datacenter-class GPUs may have a banked or
    distributed L2."""

    SHARED_LLC = "shared_llc"
    BANKED = "banked"
    DISTRIBUTED = "distributed"


class GPUNoCTopology(str, Enum):
    """On-die fabric connecting SMs to L2 / L3 / memory. Edge GPUs use
    crossbars; datacenter GPUs use 2D meshes. Mirrors a subset of
    ``compute_product.TopologyKind`` so the block-level type stays
    GPU-flavored without dragging in unused topologies."""

    CROSSBAR = "crossbar"
    MESH_2D = "mesh_2d"
    RING = "ring"
    HIERARCHICAL = "hierarchical"


class ClockDomain(BaseModel):
    """A clock domain with base / boost / sustained clocks and DVFS
    flag. Lives per-thermal-profile because the same chip clocks
    differently at 15W vs 50W vs MAXN. ``sustained_hz`` is the
    empirical clock under thermal load and is what roofline analyses
    should use; ``boost_hz`` is the datasheet / nameplate value."""

    base_hz: float = Field(..., gt=0, description="Guaranteed minimum clock")
    boost_hz: float = Field(..., gt=0, description="Datasheet / nameplate boost clock")
    sustained_hz: float = Field(..., gt=0, description="Empirical clock under thermal load")
    dvfs_enabled: bool = Field(True)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_ordering(self) -> "ClockDomain":
        if not (self.base_hz <= self.sustained_hz <= self.boost_hz):
            raise ValueError(
                f"Clock ordering must be base ({self.base_hz}) <= sustained "
                f"({self.sustained_hz}) <= boost ({self.boost_hz})"
            )
        return self


# ---------------------------------------------------------------------------
# Compute fabric (per-fabric SM-internal compute units)
# ---------------------------------------------------------------------------

class GPUComputeFabric(BaseModel):
    """One compute fabric within a GPU SM (CUDA core array, Tensor core
    array, or RT core array). A typical Ampere SM has 128 CUDA cores
    and 4 Tensor cores -- two fabrics with different per-precision
    throughput and different energy profiles.

    Mirrors the ``ComputeFabric`` dataclass in
    ``graphs.hardware.resource_model`` field-for-field so the adapter
    in PR 4 of the sprint can map mechanically.
    """

    fabric_kind: GPUFabricKind = Field(...)
    circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library used for this fabric"
    )
    units_per_sm: int = Field(
        ..., gt=0,
        description=(
            "Number of compute units of this fabric per SM. "
            "Ampere example: 128 CUDA cores per SM, 4 Tensor cores per SM."
        ),
    )
    ops_per_unit_per_clock: dict[str, int] = Field(
        ...,
        description=(
            "Ops per unit per clock keyed on precision name "
            "('fp64','fp32','fp16','bf16','int8','int4'). Single CUDA "
            "core typically does 2 FP32 FMA ops/clock; an Ampere Tensor "
            "core does 256 FP16 ops/clock or 512 INT8 ops/clock."
        ),
    )
    energy_per_flop_fp32_pj: float = Field(
        ..., gt=0,
        description=(
            "Energy per FP32 FMA in picojoules at the fabric's nominal "
            "operating point. Tensor cores are ~15% more energy-"
            "efficient than CUDA cores for fused MAC + accumulate."
        ),
    )
    energy_scaling: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Multiplier on ``energy_per_flop_fp32_pj`` for each precision. "
            "FP64 typically 2.0x, FP32 baseline 1.0, FP16 0.5, INT8 "
            "0.125. Empty dict means scale linearly by bytes_per_element."
        ),
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Memory subsystem (GPU-shaped: cache hierarchy, coherence, energy/byte)
# ---------------------------------------------------------------------------

class GPUMemorySubsystem(BaseModel):
    """GPU memory hierarchy. Ampere example: per-SM unified L1 / shared
    (128 KiB usable as L1), chip-shared L2 (4 MiB on Orin AGX, the LLC),
    no L3, LPDDR5 main memory.

    The cache-hierarchy fields here are richer than ``KPUMemorySubsystem``
    today because GPUs distinguish L1 storage discipline (cache vs
    scratchpad), L2 topology (shared vs banked), explicit L3 absence,
    and per-byte read/write energy. KPUs may want to grow into this
    shape later (the design doc's "generalize" recommendation), but
    this PR keeps the two types separate.
    """

    # Off-chip (main) memory
    memory_type: MemoryType = Field(..., description="Main-memory technology")
    memory_size_gb: float = Field(..., gt=0, description="Off-chip memory capacity")
    memory_bus_bits: int = Field(..., gt=0)
    memory_bandwidth_gbps: float = Field(..., gt=0)
    memory_controllers: int = Field(..., gt=0)

    # Per-SM L1 / shared unified store
    l1_kib_per_sm: int = Field(..., gt=0, description="Per-SM L1 capacity in KiB")
    l1_kind: GPUL1Kind = Field(GPUL1Kind.CACHE)

    # Chip-wide L2 (the LLC on Ampere SoCs)
    l2_total_kib: int = Field(..., gt=0, description="Chip-wide L2 capacity in KiB")
    l2_topology: GPUL2Topology = Field(GPUL2Topology.SHARED_LLC)

    # L3: absent on Ampere SoCs but present on some datacenter GPUs and
    # all Hopper/Blackwell. ``l3_present`` is explicit so consumers can
    # distinguish "absent by design" from "we forgot to fill it in".
    l3_present: bool = Field(False)
    l3_total_kib: int = Field(0, ge=0)

    # Cache coherence among SMs / SMs <-> CPU. SIMT ordering is "none"
    # in the snoopy CPU sense; GPU-CPU coherent products (Grace-Hopper)
    # use a real protocol.
    coherence_protocol: str = Field(
        "none",
        description=(
            "Coherence protocol or 'none'. Values: 'none' (SIMT memory "
            "model, no snoopy coherence), 'msi'/'mesi'/'moesi' (CPU-style), "
            "'nvlink-c2c' (Grace-Hopper coherent), 'cxl' (CXL.cache)."
        ),
    )

    # Per-byte energy (used by the energy estimator for memory traffic)
    read_energy_pj_per_byte: float | None = Field(None, ge=0)
    write_energy_pj_per_byte: float | None = Field(None, ge=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_l3_consistency(self) -> "GPUMemorySubsystem":
        """``l3_present`` and ``l3_total_kib`` must agree.

        Catches typo'd YAMLs where someone toggled one field without the
        other -- e.g., ``l3_present=True, l3_total_kib=0`` (claims an L3
        of zero bytes) or ``l3_present=False, l3_total_kib=4096`` (claims
        4 MiB of L3 in a chip that says it has none).
        """
        if self.l3_present and self.l3_total_kib <= 0:
            raise ValueError(
                f"l3_present=True requires l3_total_kib > 0; got "
                f"l3_total_kib={self.l3_total_kib}"
            )
        if not self.l3_present and self.l3_total_kib > 0:
            raise ValueError(
                f"l3_present=False requires l3_total_kib == 0; got "
                f"l3_total_kib={self.l3_total_kib}"
            )
        return self


# ---------------------------------------------------------------------------
# On-die fabric (SM <-> L2 interconnect)
# ---------------------------------------------------------------------------

class GPUOnDieFabric(BaseModel):
    """GPU on-die interconnect. Edge Tegra SoCs use a CROSSBAR between
    SMs and L2; datacenter GPUs (H100, B200) use a 2D mesh. Edge GPUs
    are crossbar-shaped because num_sms is small (tens), so the
    quadratic crossbar cost is acceptable in exchange for single-hop
    SM-to-L2 latency. Datacenter GPUs (hundreds of SMs) require
    mesh / hierarchical fabrics.
    """

    topology: GPUNoCTopology = Field(...)
    bisection_bandwidth_gbps: float = Field(..., gt=0)
    controller_count: int = Field(
        ..., gt=0,
        description=(
            "Number of fabric ports. For a CROSSBAR this typically "
            "equals num_sms (each SM is a port). For a MESH this is "
            "the row * col count."
        ),
    )
    flit_size_bytes: int = Field(..., gt=0)
    hop_latency_ns: float = Field(..., ge=0)
    pj_per_flit_per_hop: float = Field(..., ge=0)
    routing_distance_factor: float = Field(
        1.0, gt=0,
        description=(
            "Average hop count per transaction relative to the topology "
            "diameter. 1.0 for a single-hop crossbar; >1 for meshes."
        ),
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Thermal profile (per-precision DVFS + efficiency)
# ---------------------------------------------------------------------------

class GPUThermalProfile(BaseModel):
    """One GPU operating point: clock-domain, TDP, cooling, and
    per-precision sustained efficiency / memory bottleneck factors.

    GPUs run different clocks at different power profiles (Jetson
    nvpmodel: 15W / 30W / 50W / MAXN). Each profile carries its own
    ClockDomain and per-precision empirical-efficiency map. The
    ``efficiency_factor`` is the fraction of theoretical peak that
    real workloads achieve at this profile; the ``memory_bottleneck_factor``
    captures additional throughput loss from memory traffic that
    saturates LPDDR5 / GDDR.

    Compared to ``KPUThermalProfile``, this type adds:
      - Full ClockDomain (base / boost / sustained) per profile rather
        than a single ``clock_mhz``.
      - ``memory_clock_mhz`` (memory clock is gated together with the
        SM clock by NVIDIA's nvpmodel).
      - ``instruction_efficiency_by_precision`` (separate from
        ``efficiency_factor`` -- this is the SM-internal pipeline
        fill efficiency, e.g., 0.85 baseline).
      - ``memory_bottleneck_factor_by_precision`` (the LPDDR5
        contention multiplier).
      - ``native_acceleration_by_precision`` (boolean: does the
        Tensor core natively accelerate this precision, or is it
        emulated via lower-precision MACs).
    """

    name: str = Field(..., description="Profile label, e.g., '15W-passive', 'MAXN'")
    tdp_watts: float = Field(..., gt=0)
    cooling_solution_id: str = Field(...)

    # GPU clocks (per profile due to DVFS)
    clock_domain: ClockDomain = Field(...)
    memory_clock_mhz: float = Field(..., gt=0)

    # Per-precision empirical numbers (all keyed on lowercase precision
    # strings: 'fp64', 'fp32', 'fp16', 'bf16', 'int8', 'int4')
    efficiency_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    instruction_efficiency_by_precision: dict[str, float] = Field(default_factory=dict)
    memory_bottleneck_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    native_acceleration_by_precision: dict[str, bool] = Field(default_factory=dict)

    # Optional: profile-wide voltage (for V^2 * f power scaling)
    vdd_v: float | None = Field(None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "GPUThermalProfile":
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
# Theoretical performance (per-precision peak)
# ---------------------------------------------------------------------------

# GPUTheoreticalPerformance is now an alias of the unified
# ``TheoreticalPerformance`` from ``compute_block_common`` (v8 follow-up
# -- branes-ai/graphs#210). The unified type already includes the
# optional ``sparse_peak_ops_per_sec_by_precision`` field (originally
# GPU-specific for Ampere/Hopper 2:4 structured sparsity) -- v8 PR 2
# included it in the unified type precisely so GPU could alias cleanly
# without losing the sparsity capability.
#
# GPUs report more precision points than KPUs (FP64, FP32, FP16,
# BF16, INT8, INT4, TF32 sometimes), and Tensor cores can amplify
# throughput well beyond the CUDA-core baseline at low precision.
# The flexible dict in TheoreticalPerformance carries whichever
# precisions the vendor reports.
from embodied_schemas.compute_block_common import TheoreticalPerformance
GPUTheoreticalPerformance = TheoreticalPerformance


# ---------------------------------------------------------------------------
# GPUBlock (the discriminated union member)
# ---------------------------------------------------------------------------

class GPUBlock(BaseModel):
    """GPU compute block. Carries the GPU-specific architectural
    description: SM hierarchy, per-fabric compute, memory hierarchy,
    on-die fabric, and the GPU-only scheduler attributes (occupancy,
    concurrent-kernel limit, wave quantization).

    The discriminator value ``BlockKind.GPU`` is set in
    ``compute_product.py`` so the AnyBlock union can dispatch on it.
    Imports are arranged so this module does not import from
    ``compute_product`` (compute_product imports from here).
    """

    # IMPORTANT: kind is set as ``Literal["gpu"]`` here because
    # importing ``BlockKind`` from compute_product would create a
    # circular import. compute_product.py validates that this matches
    # ``BlockKind.GPU.value`` at the discriminator level.
    kind: Literal["gpu"] = Field(
        "gpu",
        description="Discriminator -- always 'gpu' for GPUBlock",
    )

    # SM hierarchy
    num_sms: int = Field(..., gt=0)
    cuda_cores_per_sm: int = Field(..., gt=0)
    tensor_cores_per_sm: int = Field(0, ge=0)
    threads_per_sm: int = Field(..., gt=0)
    warps_per_sm: int = Field(..., gt=0)
    warp_size: int = Field(32, gt=0)

    # Compute fabrics (CUDA cores, Tensor cores, future RT cores)
    compute_fabrics: list[GPUComputeFabric] = Field(..., min_length=1)

    # Multi-precision support, chip-wide. Should be the union of
    # precisions present in any ``compute_fabrics[*].ops_per_unit_per_clock``.
    multi_precision_alu: list[str] = Field(default_factory=list)

    # Memory hierarchy and on-die fabric
    memory: GPUMemorySubsystem = Field(...)
    noc: GPUOnDieFabric = Field(...)

    # GPU-specific scheduler attributes
    min_occupancy: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Minimum scheduler occupancy below which kernels stall",
    )
    max_concurrent_kernels: int = Field(
        ..., gt=0,
        description="Maximum number of concurrent CUDA streams the GPU can dispatch",
    )
    wave_quantization: int = Field(
        ..., gt=0,
        description=(
            "Wave granularity: kernels round up to multiples of this many "
            "warps. Drives wave-quantization losses in roofline analysis."
        ),
    )

    model_config = {"extra": "forbid"}
