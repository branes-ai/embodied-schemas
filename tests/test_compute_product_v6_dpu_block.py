"""Tests for v6 ComputeProduct schema additions: DPUBlock and friends.

Validates that:
1. DPUBlock + supporting types (DPUComputeFabric, DPUMemorySubsystem,
   DPUOnDieFabric, DPUThermalProfile, DPUTheoreticalPerformance)
   construct cleanly with realistic Xilinx Vitis AI B4096 numbers.
2. The AnyBlock discriminated union now dispatches all SIX block kinds
   (KPU + GPU + CPU + NPU + CGRA + DPU) via the ``kind`` field.
3. ComputeProduct round-trips a DPU SKU through serialize / deserialize.
4. Schema invariants: external_dram consistency, mesh-dim consistency
   (AIE_MESH vs CROSSBAR), NoC unit_count matches num_aie_tiles, INT
   precision required on fabrics, efficiency-factor ranges,
   ``extra: forbid``, fpga_fabric_overhead_factor >= 1.0.
5. Existing KPU + GPU + CPU + NPU + CGRA YAMLs continue to validate
   (additive guarantee).
6. Fourth cross-block-kind type reuse works (DPUOnDieFabric.confidence
   uses DataConfidence; DPUMemorySubsystem.external_dram_type uses
   MemoryType -- same as NPU).
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
    DPUComputeFabric,
    DPUFabricKind,
    DPUMemorySubsystem,
    DPUNoCTopology,
    DPUOnDieFabric,
    DPUTheoreticalPerformance,
    DPUThermalProfile,
    GPUBlock,
    KPUBlock,
    LifecycleStatus,
    Market,
    NPUBlock,
    Packaging,
    PackagingKind,
    Power,
    ProductKind,
)
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
# Fixtures: Vitis-AI-B4096-shaped DPU block
# ---------------------------------------------------------------------------

@pytest.fixture
def vitis_ai_fabric() -> DPUComputeFabric:
    """64 AIE tiles, 64 MACs each, 128 INT8 + 32 FP16 ops/tile/clock."""
    return DPUComputeFabric(
        fabric_kind=DPUFabricKind.AIE_ML_V1,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        ops_per_unit_per_clock={"int8": 128, "fp16": 32},
        energy_per_op_int8_pj=0.4,
        energy_scaling={"fp16": 2.5, "fp32": 5.0},
        fpga_fabric_overhead_factor=1.25,
    )


@pytest.fixture
def vitis_ai_memory() -> DPUMemorySubsystem:
    """Scratchpad + L2 + chip-attached DDR4."""
    return DPUMemorySubsystem(
        on_chip_bandwidth_gbps=80.0,
        scratchpad_kib_per_tile=64,
        shared_sram_kib=4096,    # 4 MiB shared L2
        shared_sram_layout="shared",
        has_external_dram=True,
        external_dram_type=MemoryType.DDR4,
        external_dram_size_gb=8.0,
        external_dram_bandwidth_gbps=50.0,
        scratchpad_access_energy_pj_per_byte=5.0,
        external_dram_access_energy_pj_per_byte=15.0,
        coherence_protocol="none",
    )


@pytest.fixture
def vitis_ai_noc() -> DPUOnDieFabric:
    """8x8 AIE_MESH of 64 tiles."""
    return DPUOnDieFabric(
        topology=DPUNoCTopology.AIE_MESH,
        bisection_bandwidth_gbps=80.0,
        unit_count=64,
        flit_size_bytes=32,
        mesh_rows=8,
        mesh_cols=8,
        hop_latency_ns=1.0,
        pj_per_flit_per_hop=1.5,
        routing_distance_factor=1.1,
        confidence=DataConfidence.THEORETICAL,
    )


@pytest.fixture
def dpu_block(vitis_ai_fabric, vitis_ai_memory, vitis_ai_noc) -> DPUBlock:
    return DPUBlock(
        num_aie_tiles=64,
        macs_per_tile=64,
        simd_lanes_per_tile=8,
        is_statically_reconfigurable=True,
        bitstream_load_time_ms=2000.0,
        compute_fabrics=[vitis_ai_fabric],
        multi_precision_alu=["int8", "fp16", "fp32"],
        memory=vitis_ai_memory,
        noc=vitis_ai_noc,
        min_occupancy=0.3,
        max_concurrent_models=4,
        wave_quantization=2,
    )


# ---------------------------------------------------------------------------
# 1. DPUBlock constructs and dispatches via discriminator
# ---------------------------------------------------------------------------

def test_block_kind_dpu_added_in_v6():
    assert BlockKind.DPU.value == "dpu"
    kinds = {k.value for k in BlockKind}
    assert {"kpu", "gpu", "cpu", "npu", "cgra", "dpu"}.issubset(kinds)


def test_dpu_block_constructs(dpu_block):
    assert dpu_block.kind == "dpu"
    assert dpu_block.num_aie_tiles == 64
    assert dpu_block.macs_per_tile == 64
    assert dpu_block.simd_lanes_per_tile == 8
    assert dpu_block.is_statically_reconfigurable is True
    assert dpu_block.bitstream_load_time_ms == 2000.0
    assert dpu_block.max_concurrent_models == 4   # higher than NPU/CGRA
    assert dpu_block.wave_quantization == 2       # Xilinx pair-quantization
    assert dpu_block.compute_fabrics[0].fabric_kind == DPUFabricKind.AIE_ML_V1


def test_anyblock_dispatches_to_dpu_block(dpu_block):
    """AnyBlock discriminated union should validate DPUBlock dict-form data."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(AnyBlock)
    payload = dpu_block.model_dump()
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, DPUBlock)
    assert parsed.kind == "dpu"


def test_anyblock_still_dispatches_other_block_kinds():
    """v6 must not break dispatch of KPU/GPU/CPU/NPU/CGRA. Loads each
    from the catalog and round-trips via AnyBlock."""
    from pydantic import TypeAdapter
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    adapter = TypeAdapter(AnyBlock)
    by_kind = {}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        by_kind.setdefault(kind, []).append(block)
    for kind in ("kpu", "gpu", "cpu", "npu", "cgra"):
        assert by_kind.get(kind), f"no {kind} block in catalog"
        block = by_kind[kind][0]
        parsed = adapter.validate_python(block.model_dump())
        assert (parsed.kind.value if hasattr(parsed.kind, "value")
                else str(parsed.kind)) == kind


# ---------------------------------------------------------------------------
# 2. Schema invariants
# ---------------------------------------------------------------------------

def test_external_dram_true_requires_all_fields(vitis_ai_memory):
    """has_external_dram=True with missing external_dram_* fields must fail."""
    payload = vitis_ai_memory.model_dump()
    payload["external_dram_type"] = None
    payload["external_dram_size_gb"] = None
    payload["external_dram_bandwidth_gbps"] = None
    with pytest.raises(ValidationError, match="has_external_dram=True requires"):
        DPUMemorySubsystem(**payload)


def test_external_dram_false_rejects_populated_fields(vitis_ai_memory):
    """has_external_dram=False with populated fields must fail."""
    payload = vitis_ai_memory.model_dump()
    payload["has_external_dram"] = False
    with pytest.raises(ValidationError, match="has_external_dram=False requires"):
        DPUMemorySubsystem(**payload)


def test_external_dram_false_with_all_cleared_validates(vitis_ai_memory):
    """SRAM-only DPU (rare; would be a hard-DPU variant)."""
    payload = vitis_ai_memory.model_dump()
    payload["has_external_dram"] = False
    payload["external_dram_type"] = None
    payload["external_dram_size_gb"] = None
    payload["external_dram_bandwidth_gbps"] = None
    mem = DPUMemorySubsystem(**payload)
    assert mem.has_external_dram is False


def test_aie_mesh_requires_dimensions(vitis_ai_noc):
    """topology=AIE_MESH must have mesh_rows + mesh_cols set."""
    payload = vitis_ai_noc.model_dump()
    payload["mesh_rows"] = None
    payload["mesh_cols"] = None
    with pytest.raises(ValidationError, match="requires both mesh_rows and mesh_cols"):
        DPUOnDieFabric(**payload)


def test_mesh_dimensions_must_multiply_to_unit_count(vitis_ai_noc):
    """mesh_rows * mesh_cols must equal unit_count."""
    payload = vitis_ai_noc.model_dump()
    payload["mesh_rows"] = 6  # 6 * 8 = 48, doesn't match unit_count=64
    payload["mesh_cols"] = 8
    with pytest.raises(ValidationError, match="must equal unit_count"):
        DPUOnDieFabric(**payload)


def test_crossbar_topology_rejects_dimensions(vitis_ai_noc):
    """topology=CROSSBAR shouldn't set mesh_rows/mesh_cols."""
    payload = vitis_ai_noc.model_dump()
    payload["topology"] = DPUNoCTopology.CROSSBAR
    with pytest.raises(ValidationError, match=r"requires mesh_rows.*to be None"):
        DPUOnDieFabric(**payload)


def test_dpu_compute_fabric_requires_int_precision():
    """DPUs must ship INT4 or INT8; FP-only fabric is wrong for DNN."""
    with pytest.raises(ValidationError, match=r"at least one of.*int4.*int8"):
        DPUComputeFabric(
            fabric_kind=DPUFabricKind.AIE_ML_V1,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            ops_per_unit_per_clock={"fp16": 32},  # invalid -- no INT
            energy_per_op_int8_pj=0.4,
        )


def test_dpu_compute_fabric_rejects_non_positive_ops():
    """Zero / negative ops/unit/clock values are not meaningful
    capacity numbers and must be rejected (the int-precision-required
    validator catches FP-only fabrics; this validator catches zero-cap
    fabrics)."""
    with pytest.raises(ValidationError, match=r"values must be positive"):
        DPUComputeFabric(
            fabric_kind=DPUFabricKind.AIE_ML_V1,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            ops_per_unit_per_clock={"int8": 0},   # invalid -- zero cap
            energy_per_op_int8_pj=0.4,
        )
    with pytest.raises(ValidationError, match=r"values must be positive"):
        DPUComputeFabric(
            fabric_kind=DPUFabricKind.AIE_ML_V1,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            ops_per_unit_per_clock={"int8": 128, "fp16": -1},   # invalid -- negative cap
            energy_per_op_int8_pj=0.4,
        )


def test_fpga_fabric_overhead_factor_must_be_at_least_1():
    """fpga_fabric_overhead_factor < 1.0 makes no physical sense
    (would mean FPGA is more efficient than ASIC)."""
    with pytest.raises(ValidationError):
        DPUComputeFabric(
            fabric_kind=DPUFabricKind.AIE_ML_V1,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            ops_per_unit_per_clock={"int8": 128},
            energy_per_op_int8_pj=0.4,
            fpga_fabric_overhead_factor=0.8,  # invalid
        )


def test_fpga_fabric_overhead_factor_defaults_to_1():
    """Default 1.0 for fully-hardened DPU variants."""
    fabric = DPUComputeFabric(
        fabric_kind=DPUFabricKind.AIE_HD,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        ops_per_unit_per_clock={"int8": 256},
        energy_per_op_int8_pj=0.3,
    )
    assert fabric.fpga_fabric_overhead_factor == 1.0


def test_dpu_block_noc_unit_count_must_match(dpu_block):
    """noc.unit_count must equal num_aie_tiles."""
    payload = dpu_block.model_dump()
    payload["noc"]["unit_count"] = 16  # doesn't match num_aie_tiles=64
    payload["noc"]["mesh_rows"] = 4
    payload["noc"]["mesh_cols"] = 4
    with pytest.raises(ValidationError, match=r"noc\.unit_count.*must equal num_aie_tiles"):
        DPUBlock(**payload)


def test_thermal_profile_efficiency_must_be_unit_fraction():
    with pytest.raises(ValidationError, match="outside"):
        DPUThermalProfile(
            name="20W",
            tdp_watts=20.0,
            cooling_solution_id="active_fan",
            clock_mhz=1250.0,
            dvfs_enabled=False,
            efficiency_factor_by_precision={"int8": 1.5},  # invalid
        )


def test_extra_fields_forbidden_on_dpu_block(dpu_block):
    with pytest.raises(ValidationError):
        DPUBlock(**dpu_block.model_dump(), unknown_field=42)


def test_dpu_block_requires_at_least_one_fabric(vitis_ai_memory, vitis_ai_noc):
    with pytest.raises(ValidationError, match="compute_fabrics"):
        DPUBlock(
            num_aie_tiles=64, macs_per_tile=64,
            compute_fabrics=[],
            memory=vitis_ai_memory, noc=vitis_ai_noc,
        )


def test_dpu_theoretical_performance_negative_value_rejected():
    with pytest.raises(ValidationError, match="must be >= 0"):
        DPUTheoreticalPerformance(
            peak_ops_per_sec_by_precision={"int8": -1.0}
        )


def test_bitstream_load_time_is_optional():
    """bitstream_load_time_ms is Optional; the legacy doesn't carry it."""
    fabric = DPUComputeFabric(
        fabric_kind=DPUFabricKind.AIE_ML_V1,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        ops_per_unit_per_clock={"int8": 128},
        energy_per_op_int8_pj=0.4,
    )
    mem = DPUMemorySubsystem(
        on_chip_bandwidth_gbps=80.0,
        scratchpad_kib_per_tile=64, shared_sram_kib=4096,
        has_external_dram=False,
        scratchpad_access_energy_pj_per_byte=5.0,
    )
    noc = DPUOnDieFabric(
        topology=DPUNoCTopology.AIE_MESH,
        bisection_bandwidth_gbps=80.0, unit_count=64, flit_size_bytes=32,
        mesh_rows=8, mesh_cols=8,
        hop_latency_ns=1.0, pj_per_flit_per_hop=1.5,
    )
    block = DPUBlock(
        num_aie_tiles=64, macs_per_tile=64,
        compute_fabrics=[fabric], memory=mem, noc=noc,
    )
    assert block.bitstream_load_time_ms is None


# ---------------------------------------------------------------------------
# 3. Fourth cross-block-kind type reuse
# ---------------------------------------------------------------------------

def test_dpu_on_die_fabric_reuses_data_confidence(vitis_ai_noc):
    """DPUOnDieFabric.confidence reuses DataConfidence from process_node
    (same as NPU/CGRA -- fourth cross-block-kind reuse, building
    evidence for v7 compute_block_common unification)."""
    assert isinstance(vitis_ai_noc.confidence, DataConfidence)
    assert vitis_ai_noc.confidence == DataConfidence.THEORETICAL


def test_dpu_memory_reuses_memory_type(vitis_ai_memory):
    """DPUMemorySubsystem.external_dram_type reuses MemoryType from
    gpu (same as NPU's external_dram_type). CGRA uses host_dram_type
    with the same underlying enum but different field name -- v7
    unification will resolve."""
    assert isinstance(vitis_ai_memory.external_dram_type, MemoryType)
    assert vitis_ai_memory.external_dram_type == MemoryType.DDR4


# ---------------------------------------------------------------------------
# 4. ComputeProduct end-to-end with a DPU die
# ---------------------------------------------------------------------------

def _vitis_ai_silicon_bin() -> KPUSiliconBin:
    """Reuse KPUSiliconBin shape (silicon_bin is general per design doc)."""
    return KPUSiliconBin(
        blocks=[
            SiliconBinBlock(
                name="aie_tile_array",
                circuit_class=CircuitClass.BALANCED_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=800.0,
                ),
            ),
        ]
    )


def _vitis_ai_thermal_profile() -> KPUThermalProfile:
    """Chip-level Power.thermal_profiles still uses KPUThermalProfile
    shape across all v1-v6 sprints (v7 generalization deferred)."""
    return KPUThermalProfile(
        name="default",
        tdp_watts=20.0,
        clock_mhz=1250.0,
        cooling_solution_id="active_fan",
    )


def test_compute_product_with_dpu_die_round_trips(dpu_block):
    """Build a Vitis-AI-shaped ComputeProduct end-to-end and round-trip."""
    product = ComputeProduct(
        id="xilinx_vitis_ai_b4096",
        name="Xilinx Vitis AI B4096 (Versal VE2302)",
        vendor="xilinx",
        kind=ProductKind.CHIP,
        packaging=Packaging(kind=PackagingKind.MONOLITHIC, num_dies=1,
                            package_type="bga"),
        lifecycle=LifecycleStatus.PRODUCTION,
        dies=[
            Die(
                die_id="ve2302_die",
                die_role=DieRole.COMPUTE,
                process_node_id="tsmc_n16",   # already in catalog from KPU sprint
                die_size_mm2=600.0,    # Versal VE2302 SoC die size (estimate)
                transistors_billion=3.7,
                silicon_bin=_vitis_ai_silicon_bin(),
                clocks=KPUClocks(base_clock_mhz=1250.0, boost_clock_mhz=1250.0),
                blocks=[dpu_block],
            )
        ],
        performance=KPUTheoreticalPerformance(
            int8_tops=10.24,    # 64 tiles * 128 ops/clk * 1.25 GHz
            bf16_tflops=0.0,
            fp32_tflops=0.32,   # emulated; ~1/32 INT8
            int4_tops=0.0,
        ),
        power=Power(
            tdp_watts=20.0,
            max_power_watts=22.0,
            min_power_watts=3.0,
            default_thermal_profile="default",
            thermal_profiles=[_vitis_ai_thermal_profile()],
        ),
        market=Market(
            launch_date="2022-11-01",
            target_market="edge",
            product_family="Vitis AI",
            model_tier="mid",
            is_available=True,
        ),
        last_updated="2026-05-17",
    )

    assert product.dies[0].blocks[0].kind == "dpu"
    assert isinstance(product.dies[0].blocks[0], DPUBlock)

    payload = product.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    assert isinstance(rebuilt.dies[0].blocks[0], DPUBlock)
    assert rebuilt.dies[0].blocks[0].num_aie_tiles == 64
    assert rebuilt.dies[0].blocks[0].is_statically_reconfigurable is True


# ---------------------------------------------------------------------------
# 5. Existing v1-v5 YAMLs still validate (additive guarantee)
# ---------------------------------------------------------------------------

def test_v6_does_not_break_existing_catalog():
    """v6 must keep the v5 catalog loading cleanly (12 KPU + 2 GPU +
    1 CPU + 3 NPU + 1 CGRA = 19 ComputeProducts)."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()

    counts = {"kpu": 0, "gpu": 0, "cpu": 0, "npu": 0, "cgra": 0, "dpu": 0}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts[kind] = counts.get(kind, 0) + 1

    assert counts["kpu"] == 12
    assert counts["gpu"] == 2
    assert counts["cpu"] == 1
    assert counts["npu"] == 3
    assert counts["cgra"] == 1
    # dpu_count was 0 at the v6 schema PR baseline; the Vitis AI B4096
    # data PR bumped it to 1 (first xilinx/ vendor SKU; first
    # FPGA-based architecture in the catalog).
    assert counts["dpu"] == 1
    assert sum(counts.values()) == len(products)
