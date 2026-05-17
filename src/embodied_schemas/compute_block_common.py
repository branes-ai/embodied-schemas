"""Vendor-neutral primitives shared across compute block kinds.

PR 2 of the v8 unification sprint scoped at ``graphs#208``. Centralizes
the shared primitives that 7 prior sprints (KPU v1, GPU v2, CPU v3,
NPU v4, CGRA v5, DPU v6, TPU v7) accumulated as ad-hoc cross-block-kind
reuse, and collapses the byte-identical ``*TheoreticalPerformance``
type into a single ``TheoreticalPerformance`` definition.

This module is **additive only**. Existing block modules continue to
work unchanged:

  - The 4 re-exported primitives still live in their source modules
    (``process_node``, ``gpu``, ``gpu_block``); this module just gives
    them a single canonical import point.
  - ``TheoreticalPerformance`` is a new type. The per-block-kind
    aliases (``CPUTheoreticalPerformance``, ``GPUTheoreticalPerformance``,
    ``NPUTheoreticalPerformance``, ``CGRATheoreticalPerformance``,
    ``DPUTheoreticalPerformance``, ``TPUTheoreticalPerformance``)
    will be aliased to ``TheoreticalPerformance`` in follow-up PRs
    (PR 3 of this sprint migrates NPU as proof of concept; per-block-
    kind follow-up issues migrate the rest).

Backward-compat guarantees:

  1. Every existing YAML in ``data/compute_products/`` validates
     unchanged (this PR doesn't touch any block module yet).
  2. Every existing per-block-kind type name remains importable.
  3. Every ``isinstance`` check continues to work.
  4. Every serialized JSON round-trips unchanged.
  5. graphs-side YAML loaders work unchanged.

Out of scope for v8:

  - KPU schema unification (oldest module; pre-dates the pattern;
    12 SKUs would need migration -- defer to v10+)
  - ``ThermalProfile`` / ``OnDieFabric`` unification (NEAR_UNIFIABLE
    patterns; defer to v9)
  - ``MemorySubsystem`` / ``ComputeFabric`` unification (KEEP_SEPARATE
    -- architectural variations are meaningful)
  - Per-architecture fabric kind enums (NPUDataflowKind etc.) --
    intentionally architecture-specific
  - ``has_external_dram`` vs ``has_host_dram`` naming reconciliation
    (touches SKU YAMLs; defer to v9)

See ``graphs/docs/designs/v8-compute-block-common-unification.md`` for
the full paper exercise + migration strategy + risk analysis.
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


__all__ = [
    # Shared primitives (re-exported from source modules)
    "CircuitClass",
    "DataConfidence",
    "MemoryType",
    "ClockDomain",
    # Unified types
    "TheoreticalPerformance",
]
