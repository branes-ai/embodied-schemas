"""Tests for v9 ComputeProduct schema additions: DSPBlock and friends.

Validates that:
1. DSPBlock + supporting types (DSPComputeFabric, DSPMemorySubsystem,
   DSPThermalProfile) construct cleanly with realistic Cadence
   Tensilica Vision Q8 numbers.
2. The AnyBlock discriminated union now dispatches all EIGHT block
   kinds (KPU + GPU + CPU + NPU + CGRA + DPU + TPU + DSP) via the
   ``kind`` field.
3. ComputeProduct round-trips a DSP SKU through serialize / deserialize.
4. Schema invariants: external_dram consistency (including the new
   ``external_dram_bandwidth_kind`` required field), L2 size/bw
   consistency, default_thermal_profile_name must exist, deployment
   STANDALONE_IP requires typical bandwidth_kind, INT/FP precision
   required on fabrics, ``extra: forbid``, non-positive ops rejected.
5. Existing KPU + GPU + CPU + NPU + CGRA + DPU + TPU YAMLs continue
   to validate (additive guarantee).
6. **DSPTheoreticalPerformance IS the same object as TheoreticalPerformance**
   from compute_block_common (the first block kind to use v8 unification
   from day 1 rather than via follow-up alias).
"""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    AnyBlock,
    BlockKind,
    CGRABlock,
    ComputeProduct,
    CPUBlock,
    Die,
    DieRole,
    DPUBlock,
    DSPBlock,
    DSPComputeFabric,
    DSPDeploymentKind,
    DSPFabricKind,
    DSPMemorySubsystem,
    DSPTheoreticalPerformance,
    DSPThermalProfile,
    GPUBlock,
    KPUBlock,
    LifecycleStatus,
    Market,
    NPUBlock,
    Packaging,
    PackagingKind,
    Power,
    ProductKind,
    TheoreticalPerformance,
    TPUBlock,
)
from embodied_schemas.compute_block_common import ClockDomain
from embodied_schemas.gpu import MemoryType
from embodied_schemas.kpu import (
    KPUClocks,
    KPUSiliconBin,
    KPUTheoreticalPerformance,
    KPUThermalProfile,
    SiliconBinBlock,
    TransistorSource,
    TransistorSourceKind,
)
from embodied_schemas.process_node import CircuitClass, DataConfidence


# ---------------------------------------------------------------------------
# Fixtures: Cadence-Tensilica-Vision-Q8-shaped DSP block
# ---------------------------------------------------------------------------

@pytest.fixture
def cadence_q8_simd_fabric() -> DSPComputeFabric:
    """32 SIMD units, 1024-bit, INT8/INT16/FP32/FP16."""
    return DSPComputeFabric(
        fabric_kind=DSPFabricKind.VECTOR_SIMD,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        num_units=32,
        ops_per_unit_per_clock={
            "int8": 119,
            "int16": 119,
            "fp32": 4,
            "fp16": 8,
        },
        energy_per_op_fp32_pj=2.43,
        energy_scaling={"int8": 0.15, "int16": 0.15, "fp16": 0.50},
    )


@pytest.fixture
def cadence_q8_memory() -> DSPMemorySubsystem:
    """32 KiB L1/unit + 1 MiB L2 + 4 GB LPDDR4 (typical SoC integration)."""
    return DSPMemorySubsystem(
        l1_size_bytes_per_unit=32 * 1024,
        l2_size_bytes_total=1 * 1024 * 1024,
        l2_bandwidth_gbps=80.0,
        has_external_dram=True,
        external_dram_type=MemoryType.LPDDR4,
        external_dram_size_gb=4.0,
        external_dram_bandwidth_gbps=40.0,
        external_dram_bandwidth_kind="typical",
        external_dram_access_energy_pj_per_byte=12.0,
        coherence_protocol="none",
    )


@pytest.fixture
def cadence_q8_thermal() -> DSPThermalProfile:
    """1W vision profile."""
    return DSPThermalProfile(
        name="1W",
        tdp_watts=1.0,
        cooling_solution_id="passive-mobile",
        clock_mhz=1000.0,
        dvfs_enabled=False,
    )


@pytest.fixture
def cadence_q8_clock() -> ClockDomain:
    """600 MHz min / 1.0 GHz sustained / 1.2 GHz boost."""
    return ClockDomain(
        base_hz=600e6,
        boost_hz=1.2e9,
        sustained_hz=1.0e9,
    )


@pytest.fixture
def cadence_q8_perf() -> TheoreticalPerformance:
    """3.8 TOPS INT8, 129 GFLOPS FP32."""
    return TheoreticalPerformance(
        peak_ops_per_sec_by_precision={
            "int8": 3.8e12,
            "int16": 3.8e12,
            "fp32": 129e9,
        }
    )


@pytest.fixture
def dsp_block(
    cadence_q8_simd_fabric,
    cadence_q8_memory,
    cadence_q8_thermal,
    cadence_q8_clock,
    cadence_q8_perf,
) -> DSPBlock:
    return DSPBlock(
        deployment_kind=DSPDeploymentKind.STANDALONE_IP,
        compute_fabrics=[cadence_q8_simd_fabric],
        memory=cadence_q8_memory,
        clock_domain=cadence_q8_clock,
        thermal_profiles=[cadence_q8_thermal],
        default_thermal_profile_name="1W",
        theoretical_performance=cadence_q8_perf,
        default_precision="int8",
        multi_precision_alu=["int8", "int16", "fp16", "fp32"],
        min_occupancy=0.7,
        max_concurrent_kernels=4,
        wave_quantization=4,
        vliw_issue_width=None,
        noc_confidence=DataConfidence.THEORETICAL,
    )


# ---------------------------------------------------------------------------
# 1. DSPBlock constructs and dispatches via discriminator
# ---------------------------------------------------------------------------

def test_block_kind_dsp_added_in_v9():
    assert BlockKind.DSP.value == "dsp"
    kinds = {k.value for k in BlockKind}
    assert {"kpu", "gpu", "cpu", "npu", "cgra", "dpu", "tpu", "dsp"}.issubset(kinds)


def test_dsp_block_constructs(dsp_block):
    assert dsp_block.kind == "dsp"
    assert dsp_block.deployment_kind == DSPDeploymentKind.STANDALONE_IP
    assert len(dsp_block.compute_fabrics) == 1
    assert dsp_block.compute_fabrics[0].num_units == 32
    assert dsp_block.default_precision == "int8"
    assert dsp_block.default_thermal_profile_name == "1W"


def test_anyblock_dispatches_to_dsp_block(dsp_block):
    """AnyBlock discriminated union should validate DSPBlock dict-form data."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(AnyBlock)
    payload = dsp_block.model_dump()
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, DSPBlock)
    assert parsed.kind == "dsp"


def test_anyblock_still_dispatches_other_block_kinds():
    """v9 must not break dispatch of KPU/GPU/CPU/NPU/CGRA/DPU/TPU."""
    from pydantic import TypeAdapter
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    adapter = TypeAdapter(AnyBlock)
    by_kind = {}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        by_kind.setdefault(kind, []).append(block)
    for kind in ("kpu", "gpu", "cpu", "npu", "cgra", "dpu", "tpu"):
        assert by_kind.get(kind), f"no {kind} block in catalog"
        block = by_kind[kind][0]
        parsed = adapter.validate_python(block.model_dump())
        assert (parsed.kind.value if hasattr(parsed.kind, "value")
                else str(parsed.kind)) == kind


# ---------------------------------------------------------------------------
# 2. Schema invariants
# ---------------------------------------------------------------------------

def test_external_dram_true_requires_all_fields(cadence_q8_memory):
    """has_external_dram=True with missing external_dram_* fields must fail."""
    payload = cadence_q8_memory.model_dump()
    payload["external_dram_type"] = None
    payload["external_dram_size_gb"] = None
    payload["external_dram_bandwidth_gbps"] = None
    payload["external_dram_bandwidth_kind"] = None
    with pytest.raises(ValidationError, match=r"has_external_dram=True requires"):
        DSPMemorySubsystem(**payload)


def test_external_dram_true_requires_bandwidth_kind(cadence_q8_memory):
    """has_external_dram=True without bandwidth_kind must fail (load-bearing field)."""
    payload = cadence_q8_memory.model_dump()
    payload["external_dram_bandwidth_kind"] = None
    with pytest.raises(ValidationError,
                       match=r"external_dram_bandwidth_kind"):
        DSPMemorySubsystem(**payload)


def test_external_dram_false_rejects_populated_fields(cadence_q8_memory):
    """has_external_dram=False with populated fields must fail."""
    payload = cadence_q8_memory.model_dump()
    payload["has_external_dram"] = False
    with pytest.raises(ValidationError, match=r"has_external_dram=False requires"):
        DSPMemorySubsystem(**payload)


def test_external_dram_true_requires_access_energy(cadence_q8_memory):
    """has_external_dram=True with energy_pj_per_byte=0 must fail
    (unrealistic: SKU declares DRAM but claims free access)."""
    payload = cadence_q8_memory.model_dump()
    payload["external_dram_access_energy_pj_per_byte"] = 0.0
    with pytest.raises(ValidationError,
                       match=r"external_dram_access_energy_pj_per_byte"):
        DSPMemorySubsystem(**payload)


def test_external_dram_false_rejects_populated_access_energy():
    """has_external_dram=False with energy_pj_per_byte > 0 must fail."""
    with pytest.raises(ValidationError, match=r"has_external_dram=False requires"):
        DSPMemorySubsystem(
            l1_size_bytes_per_unit=32 * 1024,
            has_external_dram=False,
            external_dram_access_energy_pj_per_byte=12.0,  # invalid -- no DRAM
        )


def test_l2_size_without_bandwidth_rejected():
    """L2 size set but bandwidth missing must fail."""
    with pytest.raises(ValidationError,
                       match=r"l2_size_bytes_total is set but l2_bandwidth_gbps"):
        DSPMemorySubsystem(
            l1_size_bytes_per_unit=32 * 1024,
            l2_size_bytes_total=1 * 1024 * 1024,
            l2_bandwidth_gbps=None,
            has_external_dram=False,
        )


def test_l2_bandwidth_without_size_rejected():
    """L2 bandwidth set but size missing must fail."""
    with pytest.raises(ValidationError,
                       match=r"l2_bandwidth_gbps is set but l2_size_bytes_total"):
        DSPMemorySubsystem(
            l1_size_bytes_per_unit=32 * 1024,
            l2_size_bytes_total=None,
            l2_bandwidth_gbps=80.0,
            has_external_dram=False,
        )


def test_dsp_compute_fabric_requires_canonical_precision():
    """DSPs must ship at least one of {INT8, INT16, FP16, FP32}."""
    with pytest.raises(ValidationError,
                       match=r"at least one of.*int8.*int16.*fp16.*fp32"):
        DSPComputeFabric(
            fabric_kind=DSPFabricKind.VECTOR_SIMD,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            num_units=32,
            ops_per_unit_per_clock={"int4": 256},  # invalid -- not in canonical set
            energy_per_op_fp32_pj=2.43,
        )


def test_dsp_compute_fabric_rejects_non_positive_ops():
    """Zero / negative ops/unit/clock are not meaningful capacity numbers."""
    with pytest.raises(ValidationError, match=r"values must be positive"):
        DSPComputeFabric(
            fabric_kind=DSPFabricKind.VECTOR_SIMD,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            num_units=32,
            ops_per_unit_per_clock={"int8": 0},  # invalid -- zero cap
            energy_per_op_fp32_pj=2.43,
        )


def test_dsp_block_requires_at_least_one_fabric(
    cadence_q8_memory, cadence_q8_thermal, cadence_q8_clock, cadence_q8_perf,
):
    with pytest.raises(ValidationError, match="compute_fabrics"):
        DSPBlock(
            deployment_kind=DSPDeploymentKind.STANDALONE_IP,
            compute_fabrics=[],  # empty
            memory=cadence_q8_memory,
            clock_domain=cadence_q8_clock,
            thermal_profiles=[cadence_q8_thermal],
            default_thermal_profile_name="1W",
            theoretical_performance=cadence_q8_perf,
            default_precision="int8",
        )


def test_dsp_block_default_thermal_profile_must_exist(dsp_block):
    """default_thermal_profile_name must match a profile in the list."""
    payload = dsp_block.model_dump()
    payload["default_thermal_profile_name"] = "nonexistent"
    with pytest.raises(ValidationError,
                       match=r"does not match any entry in thermal_profiles"):
        DSPBlock(**payload)


def test_dsp_block_standalone_ip_requires_typical_bandwidth(
    cadence_q8_simd_fabric, cadence_q8_thermal, cadence_q8_clock, cadence_q8_perf,
):
    """STANDALONE_IP with external DRAM must have bandwidth_kind='typical'.
    IP cores cannot have measured bandwidth -- depends on SoC integration."""
    bad_memory = DSPMemorySubsystem(
        l1_size_bytes_per_unit=32 * 1024,
        has_external_dram=True,
        external_dram_type=MemoryType.LPDDR4,
        external_dram_size_gb=4.0,
        external_dram_bandwidth_gbps=40.0,
        external_dram_bandwidth_kind="measured",  # wrong for IP core
        external_dram_access_energy_pj_per_byte=12.0,
    )
    with pytest.raises(ValidationError,
                       match=r"STANDALONE_IP requires.*external_dram_bandwidth_kind"):
        DSPBlock(
            deployment_kind=DSPDeploymentKind.STANDALONE_IP,
            compute_fabrics=[cadence_q8_simd_fabric],
            memory=bad_memory,
            clock_domain=cadence_q8_clock,
            thermal_profiles=[cadence_q8_thermal],
            default_thermal_profile_name="1W",
            theoretical_performance=cadence_q8_perf,
            default_precision="int8",
        )


def test_dsp_block_soc_integrated_allows_measured_or_typical(
    cadence_q8_simd_fabric, cadence_q8_thermal, cadence_q8_clock, cadence_q8_perf,
):
    """SOC_INTEGRATED can use either 'typical' or 'measured' (some SoC
    YAMLs may quote conservative typical numbers when datasheet is
    silent on a single value)."""
    for kind in ("typical", "measured"):
        memory = DSPMemorySubsystem(
            l1_size_bytes_per_unit=32 * 1024,
            has_external_dram=True,
            external_dram_type=MemoryType.LPDDR5,
            external_dram_size_gb=8.0,
            external_dram_bandwidth_gbps=90.0,
            external_dram_bandwidth_kind=kind,
            external_dram_access_energy_pj_per_byte=12.0,
        )
        block = DSPBlock(
            deployment_kind=DSPDeploymentKind.SOC_INTEGRATED,
            compute_fabrics=[cadence_q8_simd_fabric],
            memory=memory,
            clock_domain=cadence_q8_clock,
            thermal_profiles=[cadence_q8_thermal],
            default_thermal_profile_name="1W",
            theoretical_performance=cadence_q8_perf,
            default_precision="int8",
        )
        assert block.memory.external_dram_bandwidth_kind == kind


def test_dsp_block_accepts_multi_fabric(
    cadence_q8_memory, cadence_q8_thermal, cadence_q8_clock, cadence_q8_perf,
):
    """SoC DSPs typically have 2 fabrics (HVX vector + HMX tensor;
    C7x + MMA). Schema must accept length >= 2."""
    vector = DSPComputeFabric(
        fabric_kind=DSPFabricKind.VECTOR_SIMD,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        num_units=4,
        ops_per_unit_per_clock={"int8": 256, "int16": 128, "fp16": 64},
        energy_per_op_fp32_pj=1.35,
    )
    tensor = DSPComputeFabric(
        fabric_kind=DSPFabricKind.TENSOR_MATRIX,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        num_units=2,
        ops_per_unit_per_clock={"int8": 1024, "int16": 512},
        energy_per_op_fp32_pj=0.50,
    )
    block = DSPBlock(
        deployment_kind=DSPDeploymentKind.SOC_INTEGRATED,
        compute_fabrics=[vector, tensor],
        memory=cadence_q8_memory,
        clock_domain=cadence_q8_clock,
        thermal_profiles=[cadence_q8_thermal],
        default_thermal_profile_name="1W",
        theoretical_performance=cadence_q8_perf,
        default_precision="int8",
    )
    assert len(block.compute_fabrics) == 2
    assert block.compute_fabrics[0].fabric_kind == DSPFabricKind.VECTOR_SIMD
    assert block.compute_fabrics[1].fabric_kind == DSPFabricKind.TENSOR_MATRIX


def test_dsp_block_accepts_multi_thermal_profile(
    cadence_q8_simd_fabric, cadence_q8_memory, cadence_q8_clock, cadence_q8_perf,
):
    """SA8775P ships 3 profiles (20W/30W/45W). Schema must accept."""
    profiles = [
        DSPThermalProfile(name="20W", tdp_watts=20.0,
                          cooling_solution_id="passive-auto",
                          clock_mhz=1500.0, dvfs_enabled=True),
        DSPThermalProfile(name="30W", tdp_watts=30.0,
                          cooling_solution_id="active-auto",
                          clock_mhz=2000.0, dvfs_enabled=True),
        DSPThermalProfile(name="45W", tdp_watts=45.0,
                          cooling_solution_id="active-auto",
                          clock_mhz=2400.0, dvfs_enabled=True),
    ]
    block = DSPBlock(
        deployment_kind=DSPDeploymentKind.SOC_INTEGRATED,
        compute_fabrics=[cadence_q8_simd_fabric],
        memory=cadence_q8_memory,
        clock_domain=cadence_q8_clock,
        thermal_profiles=profiles,
        default_thermal_profile_name="30W",
        theoretical_performance=cadence_q8_perf,
        default_precision="int8",
    )
    assert len(block.thermal_profiles) == 3
    assert block.default_thermal_profile_name == "30W"


def test_dsp_block_vliw_issue_width_must_be_positive(
    cadence_q8_simd_fabric, cadence_q8_memory, cadence_q8_thermal,
    cadence_q8_clock, cadence_q8_perf,
):
    """vliw_issue_width must be >= 1 when set (None is allowed for
    non-VLIW DSPs like Cadence Vision Q8)."""
    with pytest.raises(ValidationError, match=r"vliw_issue_width"):
        DSPBlock(
            deployment_kind=DSPDeploymentKind.STANDALONE_IP,
            compute_fabrics=[cadence_q8_simd_fabric],
            memory=cadence_q8_memory,
            clock_domain=cadence_q8_clock,
            thermal_profiles=[cadence_q8_thermal],
            default_thermal_profile_name="1W",
            theoretical_performance=cadence_q8_perf,
            default_precision="int8",
            vliw_issue_width=0,  # invalid -- must be >= 1 or None
        )


def test_dsp_block_vliw_issue_width_accepts_vliw_value(
    cadence_q8_simd_fabric, cadence_q8_memory, cadence_q8_thermal,
    cadence_q8_clock, cadence_q8_perf,
):
    """vliw_issue_width=8 (TI C7x) should be accepted."""
    block = DSPBlock(
        deployment_kind=DSPDeploymentKind.SOC_INTEGRATED,
        compute_fabrics=[cadence_q8_simd_fabric],
        memory=cadence_q8_memory,
        clock_domain=cadence_q8_clock,
        thermal_profiles=[cadence_q8_thermal],
        default_thermal_profile_name="1W",
        theoretical_performance=cadence_q8_perf,
        default_precision="fp32",
        vliw_issue_width=8,  # TI C7x
    )
    assert block.vliw_issue_width == 8


def test_thermal_profile_efficiency_must_be_unit_fraction():
    with pytest.raises(ValidationError, match="outside"):
        DSPThermalProfile(
            name="1W",
            tdp_watts=1.0,
            cooling_solution_id="passive-mobile",
            clock_mhz=1000.0,
            efficiency_factor_by_precision={"int8": 1.5},  # invalid
        )


def test_extra_fields_forbidden_on_dsp_block(dsp_block):
    with pytest.raises(ValidationError):
        DSPBlock(**dsp_block.model_dump(), unknown_field=42)


def test_dsp_theoretical_performance_negative_value_rejected():
    with pytest.raises(ValidationError, match=r"must be >= 0"):
        DSPTheoreticalPerformance(
            peak_ops_per_sec_by_precision={"int8": -1.0}
        )


# ---------------------------------------------------------------------------
# 3. First-of-its-kind: v8 unification used from day 1
# ---------------------------------------------------------------------------

def test_dsp_theoretical_performance_is_unified_type():
    """DSPBlock is the first block kind authored AFTER v8 unification.
    DSPTheoreticalPerformance IS the same class object as
    TheoreticalPerformance from compute_block_common (alias, not subclass)."""
    assert DSPTheoreticalPerformance is TheoreticalPerformance


def test_dsp_block_uses_compute_block_common_clock_domain(dsp_block):
    """DSPBlock.clock_domain uses ClockDomain from compute_block_common
    directly (the re-export from gpu_block)."""
    from embodied_schemas.gpu_block import ClockDomain as GPUClockDomain
    assert ClockDomain is GPUClockDomain  # same re-exported object
    assert isinstance(dsp_block.clock_domain, ClockDomain)


def test_dsp_memory_reuses_memory_type(cadence_q8_memory):
    """DSPMemorySubsystem.external_dram_type reuses MemoryType from
    compute_block_common (re-exported from gpu)."""
    assert isinstance(cadence_q8_memory.external_dram_type, MemoryType)


# ---------------------------------------------------------------------------
# 4. ComputeProduct end-to-end with a DSP die
# ---------------------------------------------------------------------------

def _cadence_q8_silicon_bin() -> KPUSiliconBin:
    return KPUSiliconBin(
        blocks=[
            SiliconBinBlock(
                name="simd_engine",
                circuit_class=CircuitClass.BALANCED_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=100.0,  # ~0.1B for IP core
                ),
            ),
        ]
    )


def _cadence_q8_thermal_profile() -> KPUThermalProfile:
    """Chip-level Power.thermal_profiles still uses KPUThermalProfile."""
    return KPUThermalProfile(
        name="1W",
        tdp_watts=1.0,
        clock_mhz=1000.0,
        cooling_solution_id="passive-mobile",
    )


def test_compute_product_with_dsp_die_round_trips(dsp_block):
    """Build a Cadence-Vision-Q8-shaped ComputeProduct end-to-end and
    round-trip through JSON."""
    product = ComputeProduct(
        id="cadence_tensilica_vision_q8",
        name="Cadence Tensilica Vision Q8",
        vendor="cadence",
        kind=ProductKind.CHIP,
        packaging=Packaging(kind=PackagingKind.MONOLITHIC, num_dies=1,
                            package_type="ip_core"),
        lifecycle=LifecycleStatus.PRODUCTION,
        dies=[
            Die(
                die_id="vision_q8_die",
                die_role=DieRole.COMPUTE,
                process_node_id="tsmc_n16",
                die_size_mm2=5.0,
                transistors_billion=0.1,
                silicon_bin=_cadence_q8_silicon_bin(),
                clocks=KPUClocks(base_clock_mhz=1000.0, boost_clock_mhz=1200.0),
                blocks=[dsp_block],
            )
        ],
        performance=KPUTheoreticalPerformance(
            int8_tops=3.8,
            bf16_tflops=0.0,
            fp32_tflops=0.129,
            int4_tops=0.0,
        ),
        power=Power(
            tdp_watts=1.0,
            max_power_watts=1.2,
            min_power_watts=0.5,
            default_thermal_profile="1W",
            thermal_profiles=[_cadence_q8_thermal_profile()],
        ),
        market=Market(
            launch_date="2021-01-01",
            target_market="embodied",
            product_family="Tensilica-Vision-DSP",
            model_tier="ip_core",
            is_available=True,
        ),
        last_updated="2026-05-17",
    )

    assert product.dies[0].blocks[0].kind == "dsp"
    assert isinstance(product.dies[0].blocks[0], DSPBlock)

    payload = product.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    assert isinstance(rebuilt.dies[0].blocks[0], DSPBlock)
    assert rebuilt.dies[0].blocks[0].deployment_kind == DSPDeploymentKind.STANDALONE_IP
    assert len(rebuilt.dies[0].blocks[0].compute_fabrics) == 1


# ---------------------------------------------------------------------------
# 5. Existing v1-v7 YAMLs still validate (additive guarantee)
# ---------------------------------------------------------------------------

def test_v9_does_not_break_existing_catalog():
    """v9 must keep the v7 catalog loading cleanly (every prior block
    kind continues to load). DSP count is 0 at the schema PR baseline
    (data PR adds the first one)."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()

    counts = {"kpu": 0, "gpu": 0, "cpu": 0, "npu": 0, "cgra": 0,
              "dpu": 0, "tpu": 0, "dsp": 0}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts[kind] = counts.get(kind, 0) + 1

    # Each prior block kind has at least one entry in the catalog
    assert counts["kpu"] >= 1
    assert counts["gpu"] >= 1
    assert counts["cpu"] >= 1
    assert counts["npu"] >= 1
    assert counts["cgra"] >= 1
    assert counts["dpu"] >= 1
    assert counts["tpu"] >= 1
    # DSP count is 0 at schema-PR baseline; data PR (#211 PR 3) will bump to 1
    assert counts["dsp"] == 0
