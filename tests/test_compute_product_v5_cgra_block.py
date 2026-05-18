"""Tests for v5 ComputeProduct schema additions: CGRABlock and friends.

Validates that:
1. CGRABlock + supporting types (CGRAComputeFabric, CGRAMemorySubsystem,
   CGRAOnDieFabric, CGRAThermalProfile, CGRATheoreticalPerformance)
   construct cleanly with realistic Plasticine v2 numbers.
2. The AnyBlock discriminated union now dispatches all FIVE block
   kinds (KPU + GPU + CPU + NPU + CGRA) via the ``kind`` field.
3. ComputeProduct round-trips a CGRA SKU through serialize / deserialize.
4. Schema invariants: external_dram consistency, mesh-dim consistency
   (including TORUS_2D), NoC unit_count matches num_pcus, INT precision
   required on fabrics, efficiency-factor ranges, ``extra: forbid``.
5. Existing KPU + GPU + CPU + NPU YAMLs continue to validate (additive
   guarantee).
6. Third cross-block-kind type reuse works (CGRAOnDieFabric.confidence
   uses DataConfidence from process_node; CGRAMemorySubsystem.external_dram_type
   uses MemoryType from gpu).
"""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    AnyBlock,
    BlockKind,
    CGRABlock,
    CGRAComputeFabric,
    CGRAFabricKind,
    CGRAMemorySubsystem,
    CGRANoCTopology,
    CGRAOnDieFabric,
    CGRATheoreticalPerformance,
    CGRAThermalProfile,
    ComputeProduct,
    CPUBlock,
    Die,
    DieRole,
    DramAttachment,
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
# Fixtures: Plasticine-v2-shaped CGRA block
# ---------------------------------------------------------------------------

@pytest.fixture
def plasticine_fabric() -> CGRAComputeFabric:
    """32 PCUs, 8 MACs each, 320 INT8 ops/PCU/clock, 80 FP16 ops/PCU/clock."""
    return CGRAComputeFabric(
        fabric_kind=CGRAFabricKind.PCU_SPATIAL_DATAFLOW,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        ops_per_unit_per_clock={"int8": 320, "fp16": 80},
        energy_per_op_int8_pj=0.6,
        energy_scaling={"fp16": 3.3, "fp32": 6.7},
    )


@pytest.fixture
def plasticine_memory() -> CGRAMemorySubsystem:
    """PMU + shared L2 + host DDR4."""
    return CGRAMemorySubsystem(
        on_chip_bandwidth_gbps=40.0,
        pmu_kib_per_pcu=64,
        shared_sram_kib=2048,    # 2 MB shared L2
        shared_sram_layout="shared",
        has_external_dram=True,
        dram_attachment=DramAttachment.HOST_BUS,
        external_dram_type=MemoryType.DDR4,
        external_dram_size_gb=4.0,
        external_dram_bandwidth_gbps=12.8,
        pmu_access_energy_pj_per_byte=12.0,
        external_dram_access_energy_pj_per_byte=20.0,
        coherence_protocol="none",
    )


@pytest.fixture
def plasticine_noc() -> CGRAOnDieFabric:
    """4x8 mesh of 32 PCUs, low-confidence per design doc."""
    return CGRAOnDieFabric(
        topology=CGRANoCTopology.MESH_2D,
        bisection_bandwidth_gbps=40.0,
        unit_count=32,
        flit_size_bytes=16,
        mesh_rows=4,
        mesh_cols=8,
        hop_latency_ns=1.0,
        pj_per_flit_per_hop=1.2,
        routing_distance_factor=1.1,
        confidence=DataConfidence.THEORETICAL,
    )


@pytest.fixture
def cgra_block(plasticine_fabric, plasticine_memory, plasticine_noc) -> CGRABlock:
    return CGRABlock(
        num_pcus=32,
        macs_per_pcu=8,
        reconfig_overhead_cycles=1000,
        supports_partial_reconfig=False,
        compute_fabrics=[plasticine_fabric],
        multi_precision_alu=["int8", "fp16", "fp32"],
        memory=plasticine_memory,
        noc=plasticine_noc,
        min_occupancy=0.3,
        max_concurrent_models=1,
        wave_quantization=1,
    )


# ---------------------------------------------------------------------------
# 1. CGRABlock constructs and dispatches via discriminator
# ---------------------------------------------------------------------------

def test_block_kind_cgra_added_in_v5():
    assert BlockKind.CGRA.value == "cgra"
    kinds = {k.value for k in BlockKind}
    assert {"kpu", "gpu", "cpu", "npu", "cgra"}.issubset(kinds)


def test_cgra_block_constructs(cgra_block):
    assert cgra_block.kind == "cgra"
    assert cgra_block.num_pcus == 32
    assert cgra_block.macs_per_pcu == 8
    assert cgra_block.reconfig_overhead_cycles == 1000
    assert cgra_block.supports_partial_reconfig is False
    assert len(cgra_block.compute_fabrics) == 1
    assert cgra_block.compute_fabrics[0].fabric_kind == CGRAFabricKind.PCU_SPATIAL_DATAFLOW


def test_anyblock_dispatches_to_cgra_block(cgra_block):
    """AnyBlock discriminated union should validate CGRABlock dict-form data."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(AnyBlock)
    payload = cgra_block.model_dump()
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, CGRABlock)
    assert parsed.kind == "cgra"


def test_anyblock_still_dispatches_other_block_kinds():
    """v5 must not break dispatch of KPU/GPU/CPU/NPU. Loads each from
    the catalog and round-trips via AnyBlock."""
    from pydantic import TypeAdapter
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    adapter = TypeAdapter(AnyBlock)
    by_kind = {}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        by_kind.setdefault(kind, []).append(block)
    # Confirm at least one of each known v4 kind round-trips
    for kind in ("kpu", "gpu", "cpu", "npu"):
        assert by_kind.get(kind), f"no {kind} block in catalog"
        block = by_kind[kind][0]
        parsed = adapter.validate_python(block.model_dump())
        assert (parsed.kind.value if hasattr(parsed.kind, "value")
                else str(parsed.kind)) == kind


# ---------------------------------------------------------------------------
# 2. Schema invariants
# ---------------------------------------------------------------------------

def test_external_dram_true_requires_all_fields(plasticine_memory):
    """has_external_dram=True with missing external_dram_* fields must fail."""
    payload = plasticine_memory.model_dump()
    payload["external_dram_type"] = None
    payload["external_dram_size_gb"] = None
    payload["external_dram_bandwidth_gbps"] = None
    with pytest.raises(ValidationError, match="has_external_dram=True requires"):
        CGRAMemorySubsystem(**payload)


def test_external_dram_false_rejects_populated_fields(plasticine_memory):
    """has_external_dram=False with populated external_dram_* fields must fail."""
    payload = plasticine_memory.model_dump()
    payload["has_external_dram"] = False
    # leave external_dram_type populated (DDR4) -- validator should reject
    with pytest.raises(ValidationError, match="has_external_dram=False requires"):
        CGRAMemorySubsystem(**payload)


def test_external_dram_false_with_all_cleared_validates(plasticine_memory):
    """SRAM-only CGRA (Cerebras-style; no host DRAM)."""
    payload = plasticine_memory.model_dump()
    payload["has_external_dram"] = False
    payload["external_dram_type"] = None
    payload["external_dram_size_gb"] = None
    payload["external_dram_bandwidth_gbps"] = None
    payload["dram_attachment"] = None   # v12 requirement: clear when no external_dram
    mem = CGRAMemorySubsystem(**payload)
    assert mem.has_external_dram is False


def test_mesh_2d_requires_dimensions(plasticine_noc):
    """topology=MESH_2D must have mesh_rows + mesh_cols set."""
    payload = plasticine_noc.model_dump()
    payload["mesh_rows"] = None
    payload["mesh_cols"] = None
    with pytest.raises(ValidationError, match="requires both mesh_rows and mesh_cols"):
        CGRAOnDieFabric(**payload)


def test_torus_2d_requires_dimensions(plasticine_noc):
    """topology=TORUS_2D must also have mesh_rows + mesh_cols set."""
    payload = plasticine_noc.model_dump()
    payload["topology"] = CGRANoCTopology.TORUS_2D
    payload["mesh_rows"] = None
    payload["mesh_cols"] = None
    with pytest.raises(ValidationError, match="requires both mesh_rows and mesh_cols"):
        CGRAOnDieFabric(**payload)


def test_mesh_dimensions_must_multiply_to_unit_count(plasticine_noc):
    """mesh_rows * mesh_cols must equal unit_count."""
    payload = plasticine_noc.model_dump()
    payload["mesh_rows"] = 6  # 6 * 8 = 48, doesn't match unit_count=32
    payload["mesh_cols"] = 8
    with pytest.raises(ValidationError, match="must equal unit_count"):
        CGRAOnDieFabric(**payload)


def test_crossbar_topology_rejects_dimensions(plasticine_noc):
    """topology=CROSSBAR shouldn't set mesh_rows/mesh_cols."""
    payload = plasticine_noc.model_dump()
    payload["topology"] = CGRANoCTopology.CROSSBAR
    with pytest.raises(ValidationError, match="requires mesh_rows.*to be None"):
        CGRAOnDieFabric(**payload)


def test_cgra_compute_fabric_requires_int_precision():
    """CGRAs must ship INT4 or INT8; FP-only fabric is wrong for DNN."""
    with pytest.raises(ValidationError, match="at least one of.*int4.*int8"):
        CGRAComputeFabric(
            fabric_kind=CGRAFabricKind.PCU_SPATIAL_DATAFLOW,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            ops_per_unit_per_clock={"fp32": 80},  # invalid -- no INT
            energy_per_op_int8_pj=0.6,
        )


def test_cgra_block_noc_unit_count_must_match(cgra_block):
    """noc.unit_count must equal num_pcus."""
    payload = cgra_block.model_dump()
    payload["noc"]["unit_count"] = 16  # doesn't match num_pcus=32
    payload["noc"]["mesh_rows"] = 4    # to keep mesh-dim invariant satisfied
    payload["noc"]["mesh_cols"] = 4
    with pytest.raises(ValidationError, match="noc.unit_count.*must equal num_pcus"):
        CGRABlock(**payload)


def test_thermal_profile_efficiency_must_be_unit_fraction():
    with pytest.raises(ValidationError, match="outside"):
        CGRAThermalProfile(
            name="15W",
            tdp_watts=15.0,
            cooling_solution_id="passive_heatsink_large",
            clock_mhz=1000.0,
            dvfs_enabled=False,
            efficiency_factor_by_precision={"int8": 1.5},  # invalid
        )


def test_extra_fields_forbidden_on_cgra_block(cgra_block):
    with pytest.raises(ValidationError):
        CGRABlock(**cgra_block.model_dump(), unknown_field=42)


def test_cgra_block_requires_at_least_one_fabric(plasticine_memory, plasticine_noc):
    with pytest.raises(ValidationError, match="compute_fabrics"):
        CGRABlock(
            num_pcus=32, macs_per_pcu=8, reconfig_overhead_cycles=1000,
            compute_fabrics=[],
            memory=plasticine_memory, noc=plasticine_noc,
            min_occupancy=0.3, max_concurrent_models=1, wave_quantization=1,
        )


def test_cgra_theoretical_performance_negative_value_rejected():
    with pytest.raises(ValidationError, match="must be >= 0"):
        CGRATheoreticalPerformance(
            peak_ops_per_sec_by_precision={"int8": -1.0}
        )


def test_reconfig_overhead_can_be_zero():
    """Future CGRAs may achieve zero-cycle reconfig (rare); the schema
    should not forbid it."""
    fabric = CGRAComputeFabric(
        fabric_kind=CGRAFabricKind.PCU_SPATIAL_DATAFLOW,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        ops_per_unit_per_clock={"int8": 320},
        energy_per_op_int8_pj=0.6,
    )
    mem = CGRAMemorySubsystem(
        on_chip_bandwidth_gbps=40.0,
        pmu_kib_per_pcu=64, shared_sram_kib=2048,
        has_external_dram=False,
        pmu_access_energy_pj_per_byte=12.0,
    )
    noc = CGRAOnDieFabric(
        topology=CGRANoCTopology.MESH_2D,
        bisection_bandwidth_gbps=40.0, unit_count=32, flit_size_bytes=16,
        mesh_rows=4, mesh_cols=8,
        hop_latency_ns=1.0, pj_per_flit_per_hop=1.2,
    )
    block = CGRABlock(
        num_pcus=32, macs_per_pcu=8, reconfig_overhead_cycles=0,
        compute_fabrics=[fabric], memory=mem, noc=noc,
    )
    assert block.reconfig_overhead_cycles == 0


# ---------------------------------------------------------------------------
# 3. Third cross-block-kind type reuse (DataConfidence + MemoryType)
# ---------------------------------------------------------------------------

def test_cgra_on_die_fabric_reuses_data_confidence(plasticine_noc):
    """CGRAOnDieFabric.confidence reuses DataConfidence from process_node
    (same pattern as NPU). Pinning the shared usage so a v6 cleanup
    that moves DataConfidence to compute_block_common can chase callers
    safely."""
    assert isinstance(plasticine_noc.confidence, DataConfidence)
    assert plasticine_noc.confidence == DataConfidence.THEORETICAL


def test_cgra_memory_reuses_memory_type(plasticine_memory):
    """CGRAMemorySubsystem.external_dram_type reuses MemoryType from gpu
    (same pattern as NPU's external_dram_type)."""
    assert isinstance(plasticine_memory.external_dram_type, MemoryType)
    assert plasticine_memory.external_dram_type == MemoryType.DDR4


# ---------------------------------------------------------------------------
# 4. ComputeProduct end-to-end with a CGRA die
# ---------------------------------------------------------------------------

def _plasticine_silicon_bin() -> KPUSiliconBin:
    """Reuse KPUSiliconBin shape (silicon_bin is general per design doc)."""
    return KPUSiliconBin(
        blocks=[
            SiliconBinBlock(
                name="pcu_array",
                circuit_class=CircuitClass.BALANCED_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=160.0,
                ),
            ),
        ]
    )


def _plasticine_thermal_profile() -> KPUThermalProfile:
    """Chip-level Power.thermal_profiles still uses KPUThermalProfile
    shape across all five sprints (v6 generalization deferred)."""
    return KPUThermalProfile(
        name="default",
        tdp_watts=15.0,
        clock_mhz=1000.0,
        cooling_solution_id="passive_heatsink_large",
    )


def test_compute_product_with_cgra_die_round_trips(cgra_block):
    """Build a Plasticine-v2-shaped ComputeProduct end-to-end and
    round-trip through JSON."""
    product = ComputeProduct(
        id="stanford_plasticine_v2",
        name="Stanford Plasticine v2",
        vendor="stanford",
        kind=ProductKind.CHIP,
        packaging=Packaging(kind=PackagingKind.MONOLITHIC, num_dies=1,
                            package_type="bare_die"),
        lifecycle=LifecycleStatus.PRODUCTION,
        dies=[
            Die(
                die_id="plasticine_v2_die",
                die_role=DieRole.COMPUTE,
                process_node_id="gf_28nm",   # landed in embodied-schemas#32
                die_size_mm2=45.0,
                transistors_billion=0.52,
                silicon_bin=_plasticine_silicon_bin(),
                clocks=KPUClocks(base_clock_mhz=1000.0, boost_clock_mhz=1000.0),
                blocks=[cgra_block],
            )
        ],
        performance=KPUTheoreticalPerformance(
            int8_tops=10.0,    # 32 PCUs * 320 ops/clk * 1 GHz = 10.24 TOPS
            bf16_tflops=0.0,
            fp32_tflops=0.77,  # emulated, ~1/8 INT8
            int4_tops=0.0,
        ),
        power=Power(
            tdp_watts=15.0,
            max_power_watts=18.0,
            min_power_watts=2.0,
            default_thermal_profile="default",
            thermal_profiles=[_plasticine_thermal_profile()],
        ),
        market=Market(
            launch_date="2017-06-01",     # Plasticine ISCA paper
            target_market="research",
            product_family="Plasticine",
            model_tier="research",
            is_available=False,           # academic prototype
        ),
        last_updated="2026-05-16",
    )

    assert product.dies[0].blocks[0].kind == "cgra"
    assert isinstance(product.dies[0].blocks[0], CGRABlock)

    payload = product.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    assert isinstance(rebuilt.dies[0].blocks[0], CGRABlock)
    assert rebuilt.dies[0].blocks[0].num_pcus == 32
    assert rebuilt.dies[0].blocks[0].reconfig_overhead_cycles == 1000


# ---------------------------------------------------------------------------
# 5. Existing KPU + GPU + CPU + NPU YAMLs still validate (additive guarantee)
# ---------------------------------------------------------------------------

def test_v5_does_not_break_existing_catalog():
    """v5 must keep the v4 catalog loading cleanly (12 KPU + 2 GPU +
    1 CPU + 3 NPU = 18 ComputeProducts)."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()

    counts = {"kpu": 0, "gpu": 0, "cpu": 0, "npu": 0, "cgra": 0}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts[kind] = counts.get(kind, 0) + 1

    assert counts["kpu"] == 12
    assert counts["gpu"] == 2
    assert counts["cpu"] == 1
    assert counts["npu"] == 3
    # cgra_count was 0 at the v5 schema PR baseline; the Plasticine v2
    # data PR bumped it to 1 (first stanford/ vendor SKU; first
    # ENGINEERING_SAMPLE lifecycle SKU). DPU additions (#36+) don't
    # affect cgra_count but contribute to len(products) -- relax the
    # closure to subset.
    assert counts["cgra"] == 1
    assert counts["kpu"] + counts["gpu"] + counts["cpu"] + counts["npu"] + counts["cgra"] <= len(products)
