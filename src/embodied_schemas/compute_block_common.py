"""Vendor-neutral primitives shared across compute block kinds.

Originally PR 2 of the v8 unification sprint scoped at ``graphs#208``.
Centralizes the shared primitives that 7 prior sprints (KPU v1, GPU
v2, CPU v3, NPU v4, CGRA v5, DPU v6, TPU v7) accumulated as ad-hoc
cross-block-kind reuse, and collapses byte-identical types into
single canonical definitions.

Unified types landed so far:
  - ``TheoreticalPerformance`` -- v8 sprint (graphs#208). Collapses
    ``*TheoreticalPerformance`` across 6 block kinds (CPU/GPU/NPU/CGRA/
    DPU/TPU); DSP added in v9 catalog as a day-1 reuser. Alias-based.
  - ``ThermalProfile`` -- v9 sprint (graphs#215). Collapses the 5
    byte-identical inference-accelerator ``*ThermalProfile`` classes
    (NPU/CGRA/DPU/TPU/DSP). CPU and GPU stay separate (different
    shapes for legitimate architectural reasons). Alias-based.
  - ``OnDieFabric`` -- v10 sprint (graphs#217). Shared base for the
    6 ``*OnDieFabric`` classes; per-block-kind subclasses contribute
    only the architecture-specific ``topology`` enum. Inheritance-
    based (vs alias) because each block kind has its own topology
    enum that must stay validated.

This module is **additive only**. Existing block modules continue to
work unchanged:

  - The 4 re-exported primitives still live in their source modules
    (``process_node``, ``gpu``, ``gpu_block``); this module just gives
    them a single canonical import point.
  - Per-block-kind aliases (``CPUTheoreticalPerformance =
    TheoreticalPerformance``, ``NPUThermalProfile = ThermalProfile``,
    etc.) preserve backward compat for existing callers. The aliases
    land in follow-up PRs.

Backward-compat guarantees:

  1. Every existing YAML in ``data/compute_products/`` validates
     unchanged.
  2. Every existing per-block-kind type name remains importable.
  3. Every ``isinstance`` check continues to work.
  4. Every serialized JSON round-trips unchanged.
  5. graphs-side YAML loaders work unchanged.

Out of scope for v10:

  - KPU schema unification (oldest module; pre-dates the pattern;
    12 SKUs would need migration; KPUThermalProfile is doubly-purposed
    as chip-level ``Power.thermal_profiles`` -- defer to v12+)
  - ``MemorySubsystem`` / ``ComputeFabric`` unification (KEEP_SEPARATE
    -- architectural variations are meaningful)
  - Per-architecture fabric kind enums (NPUDataflowKind, NPUNoCTopology
    etc.) -- intentionally architecture-specific
  - ``has_external_dram`` vs ``has_host_dram`` naming reconciliation
    (touches SKU YAMLs; defer to v11)

See:
  - ``graphs/docs/designs/v8-compute-block-common-unification.md``
    (original v8 paper exercise: 4 primitives + TheoreticalPerformance)
  - ``graphs/docs/designs/v9-thermal-profile-unification.md``
    (v9 paper exercise: 7-class audit + ThermalProfile unification)
  - ``graphs/docs/designs/v10-on-die-fabric-unification.md``
    (v10 paper exercise: 6-class audit + OnDieFabric base + endpoint-
    count naming reconciliation)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-exports: the 4 shared primitives 7 prior sprints reused ad-hoc.
#
# Re-exporting from the source modules (not redefining) means existing
# imports from the source modules continue to work, AND new code can
# use this module as a single canonical import point.
# ---------------------------------------------------------------------------

from embodied_schemas.process_node import (
    CircuitClass,
    DataConfidence,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.gpu_block import ClockDomain


# ---------------------------------------------------------------------------
# Unified TheoreticalPerformance
#
# The 6 modern block kinds (CPU/GPU/NPU/CGRA/DPU/TPU) each defined a
# ``*TheoreticalPerformance`` class with a byte-identical body: one
# ``peak_ops_per_sec_by_precision`` field + one validator that rejects
# negative values. v8 collapses these into a single definition; per-
# block-kind follow-up PRs will alias the existing names to this type.
#
# (KPU's ``KPUTheoreticalPerformance`` has a different shape -- explicit
# per-precision fields like ``int8_tops``, ``bf16_tflops`` -- and is
# excluded from v8 unification. v10+ may revisit.)
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field, model_validator


class TheoreticalPerformance(BaseModel):
    """Per-precision peak ops/sec roll-up. Shared shape across all 6
    modern block kinds (CPU/GPU/NPU/CGRA/DPU/TPU); collapsed into one
    definition here.

    The mandatory ``peak_ops_per_sec_by_precision`` field is shared by
    all 6 block kinds. The optional ``sparse_peak_ops_per_sec_by_precision``
    is GPU-specific today (Ampere/Hopper 2:4 structured sparsity gives
    2x speedup on supporting Tensor cores) but is included here as
    Optional[None] so non-GPU block kinds pay no cost; future
    architectures with sparsity acceleration can populate it.

    Per-block-kind aliases (e.g. ``NPUTheoreticalPerformance =
    TheoreticalPerformance``) preserve backward compat for existing
    callers. The aliases land in follow-up PRs:

      - PR 3 of v8 sprint (graphs#208): NPU migration (proof of concept)
      - Follow-up issues: CPU, GPU, CGRA, DPU, TPU migrations
    """

    peak_ops_per_sec_by_precision: dict[str, float] = Field(
        ...,
        description=(
            "Roll-up of chip-level peak ops/sec keyed by precision "
            "name (e.g. {'bf16': 275e12, 'int8': 550e12, 'fp32': "
            "137.5e12} for a TPU v4-shaped SKU). Negative values are "
            "rejected; zero is allowed (some chips genuinely don't "
            "support a given precision and prefer to declare it as 0 "
            "rather than omit the key)."
        ),
    )

    # Optional sparsity-amplified peaks. Ampere 2:4 structured sparsity
    # gives 2x speedup on supporting Tensor cores; Hopper adds INT4
    # sparsity. None means sparsity not supported / not reported.
    # Today only GPUs populate this; included here so the unified type
    # accommodates GPU's data shape without subclassing.
    sparse_peak_ops_per_sec_by_precision: dict[str, float] | None = Field(None)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_positive(self) -> "TheoreticalPerformance":
        for prec, value in self.peak_ops_per_sec_by_precision.items():
            if value < 0:
                raise ValueError(
                    f"peak_ops_per_sec_by_precision[{prec!r}] = {value} "
                    f"must be >= 0"
                )
        if self.sparse_peak_ops_per_sec_by_precision is not None:
            for prec, value in self.sparse_peak_ops_per_sec_by_precision.items():
                if value < 0:
                    raise ValueError(
                        f"sparse_peak_ops_per_sec_by_precision[{prec!r}] "
                        f"= {value} must be >= 0"
                    )
        return self


# ---------------------------------------------------------------------------
# Unified ThermalProfile (v9 unification, graphs#215 PR 2)
#
# The 5 modern inference-accelerator block kinds (NPU/CGRA/DPU/TPU/DSP)
# each defined a ``*ThermalProfile`` class with a byte-identical body:
# 9 fields + 1 validator that rejects out-of-range efficiency values.
# v9 collapses these into a single definition; per-block-kind follow-
# up PR aliases the existing names to this type (same mechanical
# pattern v8 follow-up used for TheoreticalPerformance).
#
# CPU and GPU thermal profiles have different shapes for legitimate
# architectural reasons (per-cluster ClockDomain on CPU; ClockDomain
# + memory_clock + native_acceleration on GPU) and stay separate.
# KPU is the oldest module + doubly-purposed (also chip-level Power)
# and is deferred to v11+ KPU unification.
#
# See ``graphs/docs/designs/v9-thermal-profile-unification.md`` for the
# full paper exercise + 7-class audit + risk analysis.
# ---------------------------------------------------------------------------

class ThermalProfile(BaseModel):
    """Per-precision thermal operating point. Shared shape across the
    5 modern inference-accelerator block kinds (NPU/CGRA/DPU/TPU/DSP);
    collapsed into one definition here.

    CPU and GPU use different shapes (per-cluster ClockDomain on CPU;
    ClockDomain + memory_clock + native_acceleration on GPU) and stay
    separate. KPUThermalProfile (oldest module; also used for chip-
    level ``Power.thermal_profiles`` across the whole catalog) is also
    excluded; v11+ KPU unification may revisit.

    Cardinality varies by SKU: 1 profile for IP cores / single-mode
    accelerators; 2-3 profiles for automotive SoCs with multi-mode
    DVFS (e.g., Qualcomm SA8775P at 20/30/45W).

    Per-block-kind aliases (e.g. ``NPUThermalProfile = ThermalProfile``)
    preserve backward compat for existing callers. The aliases land
    in PR 3 of the v9 sprint (graphs#215).
    """

    name: str = Field(...)
    tdp_watts: float = Field(..., gt=0)
    cooling_solution_id: str = Field(...)

    clock_mhz: float = Field(..., gt=0, description="Operating frequency")
    dvfs_enabled: bool = Field(
        False,
        description=(
            "False is the IP-core / single-profile default; True for "
            "SKUs with multiple thermal profiles (multi-mode DVFS)."
        ),
    )

    # Per-precision empirical numbers. Each is a unit fraction in [0, 1].
    efficiency_factor_by_precision: dict[str, float] = Field(default_factory=dict)
    instruction_efficiency_by_precision: dict[str, float] = Field(default_factory=dict)
    memory_bottleneck_factor_by_precision: dict[str, float] = Field(default_factory=dict)

    vdd_v: float | None = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_efficiency_ranges(self) -> "ThermalProfile":
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
# Inheritance base: OnDieFabric (v10 unification, graphs#217 PR 2)
#
# The 6 ``*OnDieFabric`` classes (CPU/GPU/NPU/CGRA/DPU/TPU) share 7
# fields but each carries an architecture-specific ``topology`` enum
# (CPU's infinity_fabric, DPU's aie_mesh, NPU's systolic, etc.). The
# topology enums are genuinely different -- only ``crossbar`` overlaps
# across multiple kinds. So v10 uses **inheritance**, not aliasing:
# the base holds the shared fields; per-block-kind subclasses contribute
# only the typed ``topology`` field.
#
# This is the first inheritance-based unification (vs v8/v9 alias-based).
# The cost is one inheritance level; the benefit is per-kind topology
# enum validation preserved + ~150 LOC saved across the 6 subclasses.
#
# CPU and GPU gain a ``confidence`` field (with THEORETICAL default)
# in v10; previously these two block kinds had no provenance hook on
# their fabric description.
#
# See ``graphs/docs/designs/v10-on-die-fabric-unification.md`` for the
# full paper exercise + 6-class audit + risk analysis.
# ---------------------------------------------------------------------------

class OnDieFabric(BaseModel):
    """Base for on-die fabric descriptions. Per-block-kind subclasses
    contribute the architecture-specific ``topology`` enum; this base
    holds the 7 fields all 6 block-kind NoCs share + 2 optional mesh
    dims + optional confidence.

    Inheritance (not alias) because the topology field is genuinely
    architecture-specific. Each per-kind subclass looks like::

        class NPUOnDieFabric(OnDieFabric):
            topology: NPUNoCTopology = Field(...)

    ``isinstance(x, NPUOnDieFabric)`` AND ``isinstance(x, OnDieFabric)``
    both work (proper subclass relationship).

    Endpoint-count semantics by block kind:
      - NPU/DPU/CGRA/TPU: ``unit_count`` = number of compute units
      - CPU: ``unit_count`` = number of ring stops (formerly stop_count)
      - GPU: ``unit_count`` = number of memory controllers (formerly controller_count)

    The v10 sprint renamed CPU's ``stop_count`` and GPU's
    ``controller_count`` to ``unit_count`` for shared-base
    compatibility. Pre-rename data shapes are NOT supported; the
    YAML migration is atomic with the schema change.
    """

    bisection_bandwidth_gbps: float = Field(..., gt=0)
    unit_count: int = Field(
        ..., gt=0,
        description=(
            "Number of fabric endpoints. Meaning varies by block kind: "
            "compute units (NPU/DPU/CGRA/TPU); ring stops (CPU); "
            "memory controllers (GPU)."
        ),
    )
    flit_size_bytes: int = Field(..., gt=0)
    hop_latency_ns: float = Field(..., ge=0)
    pj_per_flit_per_hop: float = Field(..., ge=0)
    routing_distance_factor: float = Field(1.0, gt=0)

    # Mesh-specific (optional; only populated when topology is a
    # mesh-like one). NPU/CGRA/DPU populate these for 2D meshes;
    # TPU/CPU/GPU leave them None.
    mesh_rows: int | None = Field(default=None, gt=0)
    mesh_cols: int | None = Field(default=None, gt=0)

    # NoC provenance. NPU/CGRA/DPU/TPU populate explicitly; CPU/GPU
    # gain this field in v10 with THEORETICAL default.
    confidence: DataConfidence = Field(
        DataConfidence.THEORETICAL,
        description=(
            "Provenance of NoC numbers. Vendors rarely publish full "
            "NoC details, so THEORETICAL is the common case."
        ),
    )

    model_config = {"extra": "forbid"}


__all__ = [
    # Shared primitives (re-exported from source modules)
    "CircuitClass",
    "DataConfidence",
    "MemoryType",
    "ClockDomain",
    # Unified types (alias-based, v8/v9)
    "TheoreticalPerformance",
    "ThermalProfile",
    # Inheritance base (v10)
    "OnDieFabric",
]
