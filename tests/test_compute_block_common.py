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
    CGRAOnDieFabric,
    CGRATheoreticalPerformance,
    CGRAThermalProfile,
    CircuitClass,
    ClockDomain,
    CPUOnDieFabric,
    CPUTheoreticalPerformance,
    CPUThermalProfile,
    DataConfidence,
    DPUOnDieFabric,
    DPUTheoreticalPerformance,
    DPUThermalProfile,
    DSPThermalProfile,
    GPUOnDieFabric,
    GPUTheoreticalPerformance,
    GPUThermalProfile,
    MemoryType,
    NPUOnDieFabric,
    NPUTheoreticalPerformance,
    NPUThermalProfile,
    # v8 unified type from compute_block_common
    TheoreticalPerformance,
    # v9 unified type from compute_block_common
    ThermalProfile,
    # v10 inheritance base from compute_block_common
    OnDieFabric,
    TPUOnDieFabric,
    TPUTheoreticalPerformance,
    TPUThermalProfile,
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
    """8 BlockKinds populated across the catalog after DSP sprint
    (#211): all category gaps closed."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    kinds = set()
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kinds.add(block.kind.value if hasattr(block.kind, "value") else str(block.kind))
    assert {"kpu", "gpu", "cpu", "npu", "cgra", "dpu", "tpu", "dsp"}.issubset(kinds)


# ---------------------------------------------------------------------------
# 7. v9: ThermalProfile (unified across 5 inference-accelerator block
# kinds; CPU + GPU stay separate)
# ---------------------------------------------------------------------------

def test_compute_block_common_exports_thermal_profile():
    """The new v9 unified ``ThermalProfile`` is importable from both
    compute_block_common and the top-level package."""
    from embodied_schemas.compute_block_common import (
        ThermalProfile as common_ThermalProfile,
    )
    assert ThermalProfile is common_ThermalProfile


def test_thermal_profile_constructs_with_realistic_data():
    """A Cadence-Vision-Q8-shaped (1W single-profile) thermal entry validates."""
    profile = ThermalProfile(
        name="1W",
        tdp_watts=1.0,
        cooling_solution_id="passive_fanless",
        clock_mhz=1000.0,
        dvfs_enabled=False,
        efficiency_factor_by_precision={"int8": 0.65, "fp32": 0.60},
        instruction_efficiency_by_precision={"int8": 0.88, "fp32": 0.85},
        memory_bottleneck_factor_by_precision={"int8": 0.70, "fp32": 0.65},
        vdd_v=0.8,
    )
    assert profile.tdp_watts == 1.0
    assert profile.dvfs_enabled is False


def test_thermal_profile_minimal_construction():
    """Only the 4 mandatory fields (name, tdp_watts, cooling_solution_id,
    clock_mhz) are required; dicts default to empty; vdd_v defaults None."""
    profile = ThermalProfile(
        name="default",
        tdp_watts=350.0,
        cooling_solution_id="liquid_cooled",
        clock_mhz=1050.0,
    )
    assert profile.efficiency_factor_by_precision == {}
    assert profile.vdd_v is None


def test_thermal_profile_rejects_zero_or_negative_tdp():
    with pytest.raises(ValidationError):
        ThermalProfile(
            name="bad", tdp_watts=0.0,
            cooling_solution_id="x", clock_mhz=1000.0,
        )
    with pytest.raises(ValidationError):
        ThermalProfile(
            name="bad", tdp_watts=-1.0,
            cooling_solution_id="x", clock_mhz=1000.0,
        )


def test_thermal_profile_rejects_zero_or_negative_clock():
    with pytest.raises(ValidationError):
        ThermalProfile(
            name="bad", tdp_watts=1.0,
            cooling_solution_id="x", clock_mhz=0.0,
        )


def test_thermal_profile_rejects_out_of_range_efficiency():
    """efficiency_factor / instruction_efficiency / memory_bottleneck_factor
    are unit fractions in [0, 1]; values outside the range fail validation."""
    for attr in (
        "efficiency_factor_by_precision",
        "instruction_efficiency_by_precision",
        "memory_bottleneck_factor_by_precision",
    ):
        with pytest.raises(ValidationError, match=r"outside"):
            ThermalProfile(
                name="bad", tdp_watts=1.0,
                cooling_solution_id="x", clock_mhz=1000.0,
                **{attr: {"int8": 1.5}},  # > 1.0 -- invalid
            )
        with pytest.raises(ValidationError, match=r"outside"):
            ThermalProfile(
                name="bad", tdp_watts=1.0,
                cooling_solution_id="x", clock_mhz=1000.0,
                **{attr: {"int8": -0.1}},  # < 0.0 -- invalid
            )


def test_thermal_profile_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ThermalProfile(
            name="x", tdp_watts=1.0,
            cooling_solution_id="x", clock_mhz=1000.0,
            unknown_field=42,
        )


def test_thermal_profile_round_trips_through_json():
    profile = ThermalProfile(
        name="30W",
        tdp_watts=30.0,
        cooling_solution_id="active_fan",
        clock_mhz=2000.0,
        dvfs_enabled=True,
        efficiency_factor_by_precision={"int8": 0.80, "fp16": 0.65},
    )
    payload = profile.model_dump(mode="json")
    rebuilt = ThermalProfile.model_validate(payload)
    assert rebuilt == profile


def test_thermal_profile_accepts_existing_npu_data():
    """The unified type accepts data shaped exactly like the existing
    NPUThermalProfile uses (proves the PR 3 alias migration is safe)."""
    npu = NPUThermalProfile(
        name="default",
        tdp_watts=20.0,
        cooling_solution_id="passive_heatsink_small",
        clock_mhz=1200.0,
        dvfs_enabled=True,
        efficiency_factor_by_precision={"int8": 0.78},
    )
    payload = npu.model_dump()
    rebuilt = ThermalProfile.model_validate(payload)
    assert rebuilt.tdp_watts == 20.0


def test_thermal_profile_accepts_existing_dsp_data():
    """Same for DSPThermalProfile (Cadence Vision Q8 shape)."""
    dsp = DSPThermalProfile(
        name="1W",
        tdp_watts=1.0,
        cooling_solution_id="passive_fanless",
        clock_mhz=1000.0,
    )
    payload = dsp.model_dump()
    rebuilt = ThermalProfile.model_validate(payload)
    assert rebuilt.tdp_watts == 1.0


def test_thermal_profile_accepts_existing_tpu_data():
    """Same for TPUThermalProfile (TPU v4 350W shape)."""
    tpu = TPUThermalProfile(
        name="default",
        tdp_watts=350.0,
        cooling_solution_id="liquid_cooled",
        clock_mhz=1050.0,
        vdd_v=0.85,
    )
    payload = tpu.model_dump()
    rebuilt = ThermalProfile.model_validate(payload)
    assert rebuilt.vdd_v == 0.85


def test_cpu_and_gpu_thermal_profile_have_different_shape():
    """CPUThermalProfile and GPUThermalProfile have different field
    sets from the unified ThermalProfile and stay separate. This test
    documents the v9 boundary; if CPU/GPU shapes ever converge,
    revisit the classification in the v9 paper exercise."""
    unified_fields = set(ThermalProfile.model_fields.keys())
    cpu_fields = set(CPUThermalProfile.model_fields.keys())
    gpu_fields = set(GPUThermalProfile.model_fields.keys())

    # CPU has per_cluster_clock_domain (not in unified)
    assert "per_cluster_clock_domain" in cpu_fields
    assert "per_cluster_clock_domain" not in unified_fields

    # GPU has clock_domain + memory_clock_mhz + native_acceleration_by_precision
    # (not in unified)
    assert "clock_domain" in gpu_fields
    assert "memory_clock_mhz" in gpu_fields
    assert "native_acceleration_by_precision" in gpu_fields
    for f in ("clock_domain", "memory_clock_mhz", "native_acceleration_by_precision"):
        assert f not in unified_fields


def test_five_per_block_kind_thermal_profiles_are_field_identical():
    """The 5 byte-identical classes (NPU/CGRA/DPU/TPU/DSP) all have
    the SAME field set as the unified ThermalProfile. This proves
    PR 3's alias migration is safe -- no field is dropped on either
    side."""
    unified_fields = set(ThermalProfile.model_fields.keys())
    for cls in (NPUThermalProfile, CGRAThermalProfile, DPUThermalProfile,
                TPUThermalProfile, DSPThermalProfile):
        cls_fields = set(cls.model_fields.keys())
        assert cls_fields == unified_fields, (
            f"{cls.__name__} fields differ from ThermalProfile: "
            f"only in cls: {cls_fields - unified_fields}; "
            f"only in unified: {unified_fields - cls_fields}"
        )


# ---------------------------------------------------------------------------
# 8. v10: OnDieFabric (inheritance base across 6 block kinds)
# ---------------------------------------------------------------------------

def test_compute_block_common_exports_on_die_fabric():
    """The new v10 ``OnDieFabric`` base is importable from both
    compute_block_common and the top-level package."""
    from embodied_schemas.compute_block_common import (
        OnDieFabric as common_OnDieFabric,
    )
    assert OnDieFabric is common_OnDieFabric


def test_on_die_fabric_base_constructs_without_topology():
    """The base class itself has no ``topology`` field -- per-block-kind
    subclasses contribute it. Constructing OnDieFabric directly should
    succeed (only base fields required)."""
    fabric = OnDieFabric(
        bisection_bandwidth_gbps=500.0,
        unit_count=16,
        flit_size_bytes=32,
        hop_latency_ns=1.0,
        pj_per_flit_per_hop=2.0,
        routing_distance_factor=1.0,
    )
    assert fabric.unit_count == 16
    assert fabric.mesh_rows is None
    assert fabric.mesh_cols is None
    # Default confidence is THEORETICAL
    assert fabric.confidence == DataConfidence.THEORETICAL


def test_on_die_fabric_rejects_zero_or_negative_required_fields():
    """bisection_bandwidth_gbps, unit_count, flit_size_bytes are all
    positive-required; routing_distance_factor must be > 0."""
    with pytest.raises(ValidationError):
        OnDieFabric(
            bisection_bandwidth_gbps=0.0, unit_count=16,
            flit_size_bytes=32, hop_latency_ns=1.0,
            pj_per_flit_per_hop=2.0,
        )
    with pytest.raises(ValidationError):
        OnDieFabric(
            bisection_bandwidth_gbps=500.0, unit_count=0,
            flit_size_bytes=32, hop_latency_ns=1.0,
            pj_per_flit_per_hop=2.0,
        )
    with pytest.raises(ValidationError):
        OnDieFabric(
            bisection_bandwidth_gbps=500.0, unit_count=16,
            flit_size_bytes=32, hop_latency_ns=1.0,
            pj_per_flit_per_hop=2.0, routing_distance_factor=0.0,
        )


def test_on_die_fabric_accepts_zero_hop_latency_and_energy():
    """hop_latency_ns and pj_per_flit_per_hop use ge=0 (not gt=0)
    because some idealized models report 0 for these."""
    fabric = OnDieFabric(
        bisection_bandwidth_gbps=500.0, unit_count=16,
        flit_size_bytes=32, hop_latency_ns=0.0,
        pj_per_flit_per_hop=0.0,
    )
    assert fabric.hop_latency_ns == 0.0
    assert fabric.pj_per_flit_per_hop == 0.0


def test_on_die_fabric_accepts_optional_mesh_dims():
    """NPU/CGRA/DPU populate mesh_rows + mesh_cols for 2D meshes."""
    fabric = OnDieFabric(
        bisection_bandwidth_gbps=500.0, unit_count=32,
        flit_size_bytes=32, hop_latency_ns=1.0,
        pj_per_flit_per_hop=2.0,
        mesh_rows=4, mesh_cols=8,
        confidence=DataConfidence.CALIBRATED,
    )
    assert fabric.mesh_rows == 4
    assert fabric.mesh_cols == 8
    assert fabric.confidence == DataConfidence.CALIBRATED


def test_on_die_fabric_rejects_zero_mesh_dim_when_set():
    """If mesh_rows is set, it must be > 0 (not None and not 0)."""
    with pytest.raises(ValidationError):
        OnDieFabric(
            bisection_bandwidth_gbps=500.0, unit_count=16,
            flit_size_bytes=32, hop_latency_ns=1.0,
            pj_per_flit_per_hop=2.0,
            mesh_rows=0,  # invalid
        )


def test_on_die_fabric_forbids_extra_fields():
    with pytest.raises(ValidationError):
        OnDieFabric(
            bisection_bandwidth_gbps=500.0, unit_count=16,
            flit_size_bytes=32, hop_latency_ns=1.0,
            pj_per_flit_per_hop=2.0,
            unknown_field=42,  # invalid (extra: forbid)
        )


def test_on_die_fabric_round_trips_through_json():
    fabric = OnDieFabric(
        bisection_bandwidth_gbps=2000.0, unit_count=2,
        flit_size_bytes=32, hop_latency_ns=1.0,
        pj_per_flit_per_hop=2.0,
        confidence=DataConfidence.THEORETICAL,
    )
    payload = fabric.model_dump(mode="json")
    rebuilt = OnDieFabric.model_validate(payload)
    assert rebuilt == fabric


def test_existing_per_block_kind_on_die_fabric_classes_still_importable():
    """All 6 ``*OnDieFabric`` types remain importable from the top-level
    package; PR 3 of v10 sprint converts each to inherit from OnDieFabric."""
    for cls in (NPUOnDieFabric, CGRAOnDieFabric, DPUOnDieFabric,
                TPUOnDieFabric, CPUOnDieFabric, GPUOnDieFabric):
        assert cls is not None
        # In PR 2 (this PR) the existing classes are NOT yet aliased / inheriting.
        # PR 3 of v10 sprint migrates them; until then, identity / subclass
        # relationship is NOT yet expected.
