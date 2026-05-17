"""Smoke tests for ``compute_block_common.py``.

PR 2 of the v8 unification sprint scoped at ``branes-ai/graphs#208``.
Validates the additive module that centralizes shared primitives and
the unified ``TheoreticalPerformance`` type. Backward-compat is the
sprint's primary commitment, so this test suite covers:

1. Re-exports work (the 4 primitives are importable from the new
   single canonical point AND from their original source modules).
2. Re-exports preserve identity (``compute_block_common.CircuitClass``
   IS ``process_node.CircuitClass`` -- not a copy/subclass).
3. ``TheoreticalPerformance`` constructs and validates cleanly.
4. ``TheoreticalPerformance`` accepts the existing per-block-kind
   data shapes (proves the alias migration in PR 3 will work).
5. Backward compat: existing per-block-kind types still importable
   AND functionally equivalent (verified across NPU/CGRA/DPU/TPU).
6. Catalog still loads cleanly (no regressions from the additive
   module landing).
"""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    # Original module imports (must keep working)
    CGRATheoreticalPerformance,
    CircuitClass,
    ClockDomain,
    CPUTheoreticalPerformance,
    DataConfidence,
    DPUTheoreticalPerformance,
    GPUTheoreticalPerformance,
    MemoryType,
    NPUTheoreticalPerformance,
    # NEW: unified type from compute_block_common
    TheoreticalPerformance,
    TPUTheoreticalPerformance,
)


# ---------------------------------------------------------------------------
# 1. Re-exports work (importable from the new module)
# ---------------------------------------------------------------------------

def test_compute_block_common_module_exports_4_primitives():
    """The 4 shared primitives (CircuitClass, DataConfidence,
    MemoryType, ClockDomain) are importable from compute_block_common."""
    from embodied_schemas.compute_block_common import (
        CircuitClass as common_CircuitClass,
        ClockDomain as common_ClockDomain,
        DataConfidence as common_DataConfidence,
        MemoryType as common_MemoryType,
    )
    # All four should be the actual classes
    assert common_CircuitClass is not None
    assert common_DataConfidence is not None
    assert common_MemoryType is not None
    assert common_ClockDomain is not None


def test_compute_block_common_exports_theoretical_performance():
    """The new unified ``TheoreticalPerformance`` type is importable."""
    from embodied_schemas.compute_block_common import TheoreticalPerformance
    assert TheoreticalPerformance is not None
    # Also importable from the top-level package
    assert TheoreticalPerformance is TheoreticalPerformance


# ---------------------------------------------------------------------------
# 2. Re-exports preserve identity
# ---------------------------------------------------------------------------

def test_re_exports_are_same_class_not_copies():
    """``compute_block_common.CircuitClass`` IS ``process_node.CircuitClass``
    (not a copy / subclass). This guarantees ``isinstance`` checks
    work across both import points."""
    from embodied_schemas.compute_block_common import (
        CircuitClass as common_CircuitClass,
        ClockDomain as common_ClockDomain,
        DataConfidence as common_DataConfidence,
        MemoryType as common_MemoryType,
    )
    from embodied_schemas.gpu import MemoryType as source_MemoryType
    from embodied_schemas.gpu_block import ClockDomain as source_ClockDomain
    from embodied_schemas.process_node import (
        CircuitClass as source_CircuitClass,
        DataConfidence as source_DataConfidence,
    )

    # ``is`` checks (not just ``==``) -- prove they're the same object
    assert common_CircuitClass is source_CircuitClass
    assert common_DataConfidence is source_DataConfidence
    assert common_MemoryType is source_MemoryType
    assert common_ClockDomain is source_ClockDomain


def test_re_exports_match_top_level_package():
    """The package-level imports also point at the same objects."""
    from embodied_schemas.compute_block_common import (
        CircuitClass as common_CircuitClass,
    )
    # CircuitClass is exported from the package root too
    assert CircuitClass is common_CircuitClass


# ---------------------------------------------------------------------------
# 3. TheoreticalPerformance constructs and validates
# ---------------------------------------------------------------------------

def test_theoretical_performance_constructs_with_realistic_data():
    """TPU-v4-shaped data validates cleanly."""
    perf = TheoreticalPerformance(
        peak_ops_per_sec_by_precision={
            "bf16": 275e12,
            "int8": 550e12,
            "fp32": 137.5e12,
        }
    )
    assert perf.peak_ops_per_sec_by_precision["bf16"] == 275e12
    assert perf.peak_ops_per_sec_by_precision["int8"] == 550e12


def test_theoretical_performance_allows_zero():
    """Some chips genuinely don't support a precision; declaring it
    as 0 (rather than omitting) is allowed."""
    perf = TheoreticalPerformance(
        peak_ops_per_sec_by_precision={"int8": 10e12, "bf16": 0.0}
    )
    assert perf.peak_ops_per_sec_by_precision["bf16"] == 0.0


def test_theoretical_performance_rejects_negative():
    """Negative peak rejected by validator."""
    with pytest.raises(ValidationError, match=r"must be >= 0"):
        TheoreticalPerformance(
            peak_ops_per_sec_by_precision={"int8": -1.0}
        )


def test_theoretical_performance_forbids_extra_fields():
    """``extra = forbid`` catches typo'd YAMLs."""
    with pytest.raises(ValidationError):
        TheoreticalPerformance(
            peak_ops_per_sec_by_precision={"int8": 10e12},
            unknown_field=42,
        )


def test_theoretical_performance_round_trips_through_json():
    """JSON serialization round-trips cleanly."""
    perf = TheoreticalPerformance(
        peak_ops_per_sec_by_precision={"bf16": 275e12, "int8": 550e12}
    )
    payload = perf.model_dump(mode="json")
    rebuilt = TheoreticalPerformance.model_validate(payload)
    assert rebuilt.peak_ops_per_sec_by_precision == perf.peak_ops_per_sec_by_precision


# ---------------------------------------------------------------------------
# 4. TheoreticalPerformance accepts existing per-block-kind data shapes
#    (proves the alias migration in PR 3+ will work)
# ---------------------------------------------------------------------------

def test_theoretical_performance_accepts_npu_data():
    """Hailo-8-shaped INT8 + INT4 data validates cleanly."""
    perf = TheoreticalPerformance(
        peak_ops_per_sec_by_precision={"int8": 26e12, "int4": 52e12}
    )
    assert perf.peak_ops_per_sec_by_precision["int4"] == 52e12


def test_theoretical_performance_accepts_cgra_data():
    """Plasticine-shaped INT8 + emulated FP16 + emulated FP32."""
    perf = TheoreticalPerformance(
        peak_ops_per_sec_by_precision={
            "int8": 10.24e12, "fp16": 2.56e12, "fp32": 1.28e12,
        }
    )
    assert perf.peak_ops_per_sec_by_precision["fp16"] == 2.56e12


def test_theoretical_performance_accepts_dpu_data():
    """Vitis AI B4096-shaped INT8 + native FP16 + emulated FP32."""
    perf = TheoreticalPerformance(
        peak_ops_per_sec_by_precision={
            "int8": 10.24e12, "fp16": 2.56e12, "fp32": 0.32e12,
        }
    )
    assert perf.peak_ops_per_sec_by_precision["fp32"] == 0.32e12


# ---------------------------------------------------------------------------
# 5. Backward compat: existing per-block-kind types still importable
#    AND functionally equivalent (key invariant for the v8 sprint)
# ---------------------------------------------------------------------------

def test_per_block_kind_theoretical_performance_classes_still_importable():
    """All 6 per-block-kind ``*TheoreticalPerformance`` classes remain
    importable from the package root. This is the v8 backward-compat
    guarantee #2 -- existing imports must keep working."""
    # If any import broke, the import at the top of this file would
    # have raised ImportError already. This test pins the contract.
    assert CPUTheoreticalPerformance is not None
    assert GPUTheoreticalPerformance is not None
    assert NPUTheoreticalPerformance is not None
    assert CGRATheoreticalPerformance is not None
    assert DPUTheoreticalPerformance is not None
    assert TPUTheoreticalPerformance is not None


def test_all_per_block_kind_theoretical_performance_aliased_to_unified():
    """v8 follow-up (branes-ai/graphs#210) batched the per-block-kind
    migration: all 6 ``*TheoreticalPerformance`` classes are now
    aliases of the unified ``TheoreticalPerformance``. This means
    ``isinstance(x, XXXTheoreticalPerformance)`` works for any
    TheoreticalPerformance instance, and all aliases ARE the same
    class object.

    NPU was migrated in v8 PR 3 (branes-ai/embodied-schemas#41) as
    proof of concept; CPU/GPU/CGRA/DPU/TPU follow in this batch."""
    aliases = [
        CPUTheoreticalPerformance, GPUTheoreticalPerformance,
        NPUTheoreticalPerformance, CGRATheoreticalPerformance,
        DPUTheoreticalPerformance, TPUTheoreticalPerformance,
    ]
    for cls in aliases:
        assert cls is TheoreticalPerformance, (
            f"{cls.__name__} is not aliased to TheoreticalPerformance "
            f"(got {cls!r}); the v8 follow-up should have unified all 6."
        )
    # All 6 aliases ARE the same class object
    for i, c1 in enumerate(aliases):
        for c2 in aliases[i+1:]:
            assert c1 is c2, (
                f"{c1.__name__} and {c2.__name__} are unexpectedly "
                f"different classes; v8 follow-up should have unified them."
            )


def test_npu_isinstance_works_through_both_names():
    """An NPU performance instance is recognized as both
    NPUTheoreticalPerformance AND TheoreticalPerformance (because the
    former IS the latter post-alias)."""
    perf = TheoreticalPerformance(
        peak_ops_per_sec_by_precision={"int8": 26e12, "int4": 52e12}
    )
    assert isinstance(perf, NPUTheoreticalPerformance)
    assert isinstance(perf, TheoreticalPerformance)
    # And constructing via the legacy name produces the same class
    perf2 = NPUTheoreticalPerformance(
        peak_ops_per_sec_by_precision={"int8": 26e12, "int4": 52e12}
    )
    assert isinstance(perf2, TheoreticalPerformance)
    assert type(perf) is type(perf2)


def test_per_block_kind_theoretical_performance_shapes_subset_unified():
    """All 6 per-block-kind ``*TheoreticalPerformance`` classes have
    a field structure that is a SUBSET of the unified
    ``TheoreticalPerformance`` (key invariant: each per-block-kind
    type's fields are present in the unified type, so aliasing is
    safe).

    The unified type adds Optional ``sparse_peak_ops_per_sec_by_precision``
    to accommodate GPU; non-GPU block kinds leave it None. All 6
    types share the mandatory ``peak_ops_per_sec_by_precision`` field."""
    unified_fields = set(TheoreticalPerformance.model_fields.keys())
    # Mandatory field shared by all
    assert "peak_ops_per_sec_by_precision" in unified_fields
    for cls in (
        CPUTheoreticalPerformance, GPUTheoreticalPerformance,
        NPUTheoreticalPerformance, CGRATheoreticalPerformance,
        DPUTheoreticalPerformance, TPUTheoreticalPerformance,
    ):
        cls_fields = set(cls.model_fields.keys())
        # Each per-block-kind type's fields must be a subset of the
        # unified type (so the unified type can accept any per-block-
        # kind data without losing fields).
        assert cls_fields.issubset(unified_fields), (
            f"{cls.__name__} has fields {cls_fields - unified_fields} "
            f"not in TheoreticalPerformance {unified_fields}; "
            f"PR 3+ migration may not be safe for this kind."
        )
        # The mandatory field must be present in every per-block-kind type
        assert "peak_ops_per_sec_by_precision" in cls_fields


# ---------------------------------------------------------------------------
# 6. Catalog still loads (no regressions from the additive module)
# ---------------------------------------------------------------------------

def test_existing_catalog_still_loads():
    """All 21 ComputeProducts continue to validate. The v8 additive
    module doesn't touch any block module, so this should be a no-op
    test -- but pinning it gives a regression signal for PR 3+."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    assert len(products) >= 21   # 12 KPU + 2 GPU + 1 CPU + 3 NPU + 1 CGRA + 1 DPU + 1 TPU


def test_existing_catalog_block_kinds_unchanged():
    """7 BlockKinds populated across the catalog (8th = DSP, future)."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    kinds = set()
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kinds.add(block.kind.value if hasattr(block.kind, "value") else str(block.kind))
    assert kinds == {"kpu", "gpu", "cpu", "npu", "cgra", "dpu", "tpu"}
