"""Tests for v3 ComputeProduct schema additions: CPUBlock and friends.

Validates that:
1. CPUBlock + supporting types (CPUComputeFabric, CoreClusterSpec,
   CPUMemorySubsystem, CPUOnDieFabric, CPUThermalProfile,
   CPUTheoreticalPerformance) construct cleanly with realistic Intel
   Core i7-12700K numbers.
2. The AnyBlock discriminated union now dispatches all THREE block
   kinds (KPU + GPU + CPU) via the ``kind`` field.
3. ComputeProduct round-trips a CPU SKU through serialize / deserialize.
4. Schema invariants: L2 layout consistency, L3 / L4 present-vs-capacity,
   efficiency factor ranges, ClockDomain ordering, ``extra: forbid``.
5. Existing KPU + GPU YAMLs continue to validate (additive guarantee).
6. ClockDomain cross-block-kind sharing works (CPUThermalProfile reuses
   it from gpu_block).
"""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    AnyBlock,
    BlockKind,
    ClockDomain,
    ComputeProduct,
    CoreClusterKind,
    CoreClusterSpec,
    CPUBlock,
    CPUComputeFabric,
    CPUISAExtension,
    CPUMemorySubsystem,
    CPUNoCTopology,
    CPUOnDieFabric,
    CPUTheoreticalPerformance,
    CPUThermalProfile,
    Die,
    DieRole,
    GPUBlock,
    KPUBlock,
    L2Layout,
    LifecycleStatus,
    Market,
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
from embodied_schemas.process_node import CircuitClass


# ---------------------------------------------------------------------------
# Fixtures: Intel i7-12700K hybrid Alder Lake P+E clusters
# ---------------------------------------------------------------------------

@pytest.fixture
def p_core_avx2_fabric() -> CPUComputeFabric:
    """Golden Cove P-core AVX2 + AVX-VNNI."""
    return CPUComputeFabric(
        isa_extension=CPUISAExtension.AVX_VNNI,
        circuit_class=CircuitClass.HP_LOGIC,
        ops_per_core_per_clock={
            "fp64": 8,
            "fp32": 16,
            "fp16": 4,
            "bf16": 4,
            "int8": 64,
            "int4": 128,
        },
        energy_per_flop_fp32_pj=0.85,
        energy_scaling={
            "fp64": 2.0, "fp32": 1.0, "fp16": 1.0, "bf16": 1.0,
            "int8": 0.30, "int4": 0.18,
        },
    )


@pytest.fixture
def e_core_avx2_fabric() -> CPUComputeFabric:
    """Gracemont E-core AVX2 + AVX-VNNI (single FMA pipe vs P's two)."""
    return CPUComputeFabric(
        isa_extension=CPUISAExtension.AVX_VNNI,
        circuit_class=CircuitClass.LP_LOGIC,
        ops_per_core_per_clock={
            "fp64": 4, "fp32": 8, "fp16": 2, "bf16": 2,
            "int8": 32, "int4": 64,
        },
        energy_per_flop_fp32_pj=0.65,
        energy_scaling={
            "fp64": 2.0, "fp32": 1.0, "fp16": 1.0, "bf16": 1.0,
            "int8": 0.30, "int4": 0.18,
        },
    )


@pytest.fixture
def p_cluster(p_core_avx2_fabric) -> CoreClusterSpec:
    """8 P-cores, SMT=2, 1.25 MB private L2 each."""
    return CoreClusterSpec(
        cluster_kind=CoreClusterKind.PERFORMANCE,
        num_cores=8,
        smt_threads=2,
        aggregate_weight=1.0,
        compute_fabrics=[p_core_avx2_fabric],
        l1_kib_per_core=48,
        l2_layout=L2Layout.PRIVATE_PER_CORE,
        l2_kib_per_core=1280,  # 1.25 MB private
        l2_kib_shared=0,
    )


@pytest.fixture
def e_cluster(e_core_avx2_fabric) -> CoreClusterSpec:
    """4 E-cores, no SMT, 2 MB shared L2 across the cluster."""
    return CoreClusterSpec(
        cluster_kind=CoreClusterKind.EFFICIENT,
        num_cores=4,
        smt_threads=1,
        aggregate_weight=0.6,
        compute_fabrics=[e_core_avx2_fabric],
        l1_kib_per_core=32,
        l2_layout=L2Layout.SHARED_PER_CLUSTER,
        l2_kib_per_core=0,
        l2_kib_shared=2048,  # 2 MB shared across the 4 E-cores
    )


@pytest.fixture
def i7_memory() -> CPUMemorySubsystem:
    """64 GB DDR5-4800 dual-channel; 25 MB shared L3 LLC; no L4."""
    return CPUMemorySubsystem(
        memory_type=MemoryType.DDR5,
        memory_size_gb=64.0,
        memory_bus_bits=128,
        memory_bandwidth_gbps=76.8,
        memory_controllers=2,
        l3_present=True,
        l3_total_kib=25 * 1024,
        coherence_protocol="snoopy_mesi",
        read_energy_pj_per_byte=25.0,
        write_energy_pj_per_byte=30.0,
    )


@pytest.fixture
def i7_noc() -> CPUOnDieFabric:
    """Alder Lake double ring bus."""
    return CPUOnDieFabric(
        topology=CPUNoCTopology.DOUBLE_RING,
        bisection_bandwidth_gbps=512.0,
        stop_count=12,  # 8 P + 4 E
        flit_size_bytes=32,
        hop_latency_ns=1.5,
        pj_per_flit_per_hop=5.0,
        routing_distance_factor=1.0,
    )


@pytest.fixture
def cpu_block(p_cluster, e_cluster, i7_memory, i7_noc) -> CPUBlock:
    return CPUBlock(
        core_clusters=[p_cluster, e_cluster],
        total_effective_cores=10,  # 8 + int(4*0.6)
        simd_width_lanes=8,
        multi_precision_alu=["fp64", "fp32", "fp16", "bf16", "int8", "int4"],
        memory=i7_memory,
        noc=i7_noc,
        simd_efficiency_by_op_kind={
            "elementwise": 0.95, "matrix": 0.80, "default": 0.70,
        },
        min_occupancy=0.4,
        max_concurrent_threads=20,  # 8*2 + 4*1
        wave_quantization=1,
    )


# ---------------------------------------------------------------------------
# 1. CPUBlock constructs and dispatches via discriminator
# ---------------------------------------------------------------------------

def test_block_kind_cpu_added_in_v3():
    assert BlockKind.CPU.value == "cpu"
    kinds = {k.value for k in BlockKind}
    assert {"kpu", "gpu", "cpu"}.issubset(kinds)


def test_cpu_block_constructs(cpu_block):
    assert cpu_block.kind == "cpu"
    assert len(cpu_block.core_clusters) == 2
    assert cpu_block.total_effective_cores == 10
    assert cpu_block.max_concurrent_threads == 20


def test_cpu_block_clusters_have_distinct_kinds(cpu_block):
    kinds = {c.cluster_kind for c in cpu_block.core_clusters}
    assert kinds == {CoreClusterKind.PERFORMANCE, CoreClusterKind.EFFICIENT}


def test_anyblock_dispatches_to_cpu_block(cpu_block):
    """AnyBlock discriminated union should validate CPUBlock dict-form data."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(AnyBlock)
    payload = cpu_block.model_dump()
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, CPUBlock)
    assert parsed.kind == "cpu"


def test_anyblock_still_dispatches_to_kpu_block():
    """v3 must not break v1 KPU dispatch."""
    from pydantic import TypeAdapter
    from embodied_schemas import KPUMemorySubsystem, KPUNoCSpec
    from embodied_schemas.kpu import KPUTileSpec
    kpu_block = KPUBlock(
        total_tiles=64,
        multi_precision_alu=["int8"],
        tiles=[KPUTileSpec(
            tile_type="INT8", num_tiles=64,
            pe_array_rows=32, pe_array_cols=32,
            pe_circuit_class=CircuitClass.HP_LOGIC,
            ops_per_tile_per_clock={"int8": 1024.0},
        )],
        noc=KPUNoCSpec(topology="mesh_2d", mesh_rows=8, mesh_cols=8, flit_bytes=16),
        memory=KPUMemorySubsystem(
            memory_type=MemoryType.LPDDR5, memory_size_gb=2.0,
            memory_bus_bits=64, memory_bandwidth_gbps=64.0,
            memory_controllers=2, l3_kib_per_tile=64,
        ),
    )
    adapter = TypeAdapter(AnyBlock)
    parsed = adapter.validate_python(kpu_block.model_dump())
    assert isinstance(parsed, KPUBlock)


def test_anyblock_still_dispatches_to_gpu_block():
    """v3 must not break v2 GPU dispatch -- specifically when CPUBlock
    is added as a third union arm."""
    from pydantic import TypeAdapter
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    orin = products.get("nvidia_jetson_agx_orin_64gb")
    assert orin is not None
    block = orin.dies[0].blocks[0]
    assert isinstance(block, GPUBlock)
    # Round-trip the GPU payload through AnyBlock
    adapter = TypeAdapter(AnyBlock)
    parsed = adapter.validate_python(block.model_dump())
    assert isinstance(parsed, GPUBlock)


# ---------------------------------------------------------------------------
# 2. Schema invariants
# ---------------------------------------------------------------------------

def test_l2_layout_private_requires_per_core_capacity(e_core_avx2_fabric):
    """L2Layout.PRIVATE_PER_CORE with l2_kib_per_core=0 is a contradiction."""
    with pytest.raises(ValidationError, match="PRIVATE_PER_CORE requires"):
        CoreClusterSpec(
            cluster_kind=CoreClusterKind.PERFORMANCE,
            num_cores=8, smt_threads=2, aggregate_weight=1.0,
            compute_fabrics=[e_core_avx2_fabric],
            l1_kib_per_core=48,
            l2_layout=L2Layout.PRIVATE_PER_CORE,
            l2_kib_per_core=0,    # invalid -- layout says private but no capacity
            l2_kib_shared=0,
        )


def test_l2_layout_private_rejects_shared_capacity(e_core_avx2_fabric):
    with pytest.raises(ValidationError, match="must have l2_kib_shared"):
        CoreClusterSpec(
            cluster_kind=CoreClusterKind.PERFORMANCE,
            num_cores=8, smt_threads=2, aggregate_weight=1.0,
            compute_fabrics=[e_core_avx2_fabric],
            l1_kib_per_core=48,
            l2_layout=L2Layout.PRIVATE_PER_CORE,
            l2_kib_per_core=1280,
            l2_kib_shared=2048,   # invalid -- both can't be set
        )


def test_l2_layout_shared_requires_shared_capacity(e_core_avx2_fabric):
    with pytest.raises(ValidationError, match="SHARED_PER_CLUSTER requires"):
        CoreClusterSpec(
            cluster_kind=CoreClusterKind.EFFICIENT,
            num_cores=4, smt_threads=1, aggregate_weight=0.6,
            compute_fabrics=[e_core_avx2_fabric],
            l1_kib_per_core=32,
            l2_layout=L2Layout.SHARED_PER_CLUSTER,
            l2_kib_per_core=0,
            l2_kib_shared=0,   # invalid -- shared layout needs shared capacity
        )


def test_l3_present_invariant_inherited_from_gpu_pattern(i7_memory):
    """CPUMemorySubsystem must enforce l3_present consistency."""
    payload = i7_memory.model_dump()
    payload["l3_present"] = True
    payload["l3_total_kib"] = 0
    with pytest.raises(ValidationError, match="l3_present=True requires"):
        CPUMemorySubsystem(**payload)


def test_l4_present_invariant(i7_memory):
    """l4_present=True needs l4_total_kib>0 AND l4_kind set."""
    payload = i7_memory.model_dump()
    payload["l4_present"] = True
    payload["l4_total_kib"] = 0
    payload["l4_kind"] = "edram"
    with pytest.raises(ValidationError, match="l4_present=True requires"):
        CPUMemorySubsystem(**payload)


def test_l4_present_requires_kind_string(i7_memory):
    payload = i7_memory.model_dump()
    payload["l4_present"] = True
    payload["l4_total_kib"] = 128 * 1024
    payload["l4_kind"] = ""
    with pytest.raises(ValidationError, match="non-empty l4_kind"):
        CPUMemorySubsystem(**payload)


def test_l4_absent_rejects_l4_kind(i7_memory):
    """If l4_present=False, l4_kind must also be empty -- catches
    typo'd YAMLs that toggled one field without the other (e.g. set
    l4_kind='edram' but forgot to set l4_present=True)."""
    payload = i7_memory.model_dump()
    payload["l4_present"] = False
    payload["l4_total_kib"] = 0
    payload["l4_kind"] = "edram"   # invalid -- present=False but kind set
    with pytest.raises(ValidationError, match="l4_present=False requires empty"):
        CPUMemorySubsystem(**payload)


def test_simd_efficiency_must_be_unit_fraction(cpu_block):
    payload = cpu_block.model_dump()
    payload["simd_efficiency_by_op_kind"]["matrix"] = 1.5
    with pytest.raises(ValidationError, match="must be in"):
        CPUBlock(**payload)


def test_thermal_profile_efficiency_must_be_unit_fraction():
    cd = ClockDomain(base_hz=800e6, boost_hz=4.7e9, sustained_hz=4.7e9)
    with pytest.raises(ValidationError, match="outside"):
        CPUThermalProfile(
            name="125W-PL1",
            tdp_watts=125.0,
            cooling_solution_id="active_fan",
            per_cluster_clock_domain={"performance": cd},
            efficiency_factor_by_precision={"int8": 1.5},
        )


def test_extra_fields_forbidden_on_cpu_block(cpu_block):
    with pytest.raises(ValidationError):
        CPUBlock(**cpu_block.model_dump(), unknown_field=42)


def test_cpu_block_requires_at_least_one_cluster(i7_memory, i7_noc):
    """Isolate the empty-clusters invariant. ``total_effective_cores``
    set to a valid positive value so the only invalid field is
    ``core_clusters=[]``; this way the assertion can't pass spuriously
    on a different validation."""
    with pytest.raises(ValidationError, match="core_clusters"):
        CPUBlock(
            core_clusters=[],   # the one invariant under test
            total_effective_cores=10,
            simd_width_lanes=8,
            memory=i7_memory, noc=i7_noc,
            min_occupancy=0.4, max_concurrent_threads=10, wave_quantization=1,
        )


def test_cpu_theoretical_performance_negative_value_rejected():
    with pytest.raises(ValidationError, match="must be >= 0"):
        CPUTheoreticalPerformance(
            peak_ops_per_sec_by_precision={"fp32": -1.0}
        )


# ---------------------------------------------------------------------------
# 3. Cross-block-kind type sharing (the v3 design's first instance)
# ---------------------------------------------------------------------------

def test_cpu_thermal_profile_reuses_clock_domain():
    """ClockDomain is defined in gpu_block; CPUThermalProfile's
    per_cluster_clock_domain dict carries it directly. This test pins
    the shared usage so the v4 cleanup (move ClockDomain to a
    vendor-neutral module) can rename callers safely."""
    cd_p = ClockDomain(base_hz=800e6, boost_hz=4.9e9, sustained_hz=4.7e9)
    cd_e = ClockDomain(base_hz=600e6, boost_hz=3.8e9, sustained_hz=3.6e9)
    profile = CPUThermalProfile(
        name="125W-PL1",
        tdp_watts=125.0,
        cooling_solution_id="active_fan",
        per_cluster_clock_domain={"performance": cd_p, "efficient": cd_e},
    )
    assert profile.per_cluster_clock_domain["performance"].sustained_hz == 4.7e9
    assert isinstance(profile.per_cluster_clock_domain["efficient"], ClockDomain)


# ---------------------------------------------------------------------------
# 4. ComputeProduct end-to-end with a CPU die
# ---------------------------------------------------------------------------

def _i7_silicon_bin() -> KPUSiliconBin:
    """Reuse KPUSiliconBin shape (silicon_bin is general per design doc)."""
    return KPUSiliconBin(
        blocks=[
            SiliconBinBlock(
                name="p_cores",
                circuit_class=CircuitClass.HP_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=2400.0,
                ),
            ),
            SiliconBinBlock(
                name="e_cores",
                circuit_class=CircuitClass.LP_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=400.0,
                ),
            ),
            SiliconBinBlock(
                name="l3_sram",
                circuit_class=CircuitClass.SRAM_HD,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=600.0,
                ),
            ),
        ]
    )


def _i7_thermal_profile() -> KPUThermalProfile:
    """Chip-level Power.thermal_profiles still uses KPUThermalProfile shape
    in v3 (full generalization is v4 scope). The richer per-cluster
    DVFS lives inside CPUBlock via CPUThermalProfile (the v3 design
    chose this trade-off; the chip-wide thermal profiles stay simple)."""
    return KPUThermalProfile(
        name="125W",
        tdp_watts=125.0,
        clock_mhz=4700.0,
        cooling_solution_id="active_fan",
    )


def test_compute_product_with_cpu_die_round_trips(cpu_block):
    """Build an i7-12700K-shaped ComputeProduct end-to-end and round-trip."""
    product = ComputeProduct(
        id="intel_core_i7_12700k",
        name="Intel Core i7-12700K",
        vendor="intel",
        kind=ProductKind.CHIP,
        packaging=Packaging(kind=PackagingKind.MONOLITHIC, num_dies=1),
        lifecycle=LifecycleStatus.PRODUCTION,
        dies=[
            Die(
                die_id="alder_lake_die",
                die_role=DieRole.COMPUTE,
                process_node_id="intel_7",  # not in catalog yet (PR 3 will add)
                die_size_mm2=215.0,
                transistors_billion=22.0,
                silicon_bin=_i7_silicon_bin(),
                clocks=KPUClocks(base_clock_mhz=3600.0, boost_clock_mhz=5000.0),
                blocks=[cpu_block],
            )
        ],
        performance=KPUTheoreticalPerformance(
            int8_tops=4.8,    # GPU-side roll-up shape; CPU peaks land here for v3
            bf16_tflops=0.6,
            fp32_tflops=1.2,
        ),
        power=Power(
            tdp_watts=125.0,
            max_power_watts=190.0,
            min_power_watts=10.0,
            default_thermal_profile="125W",
            thermal_profiles=[_i7_thermal_profile()],
        ),
        market=Market(
            launch_date="2021-11-04",
            target_market="desktop",
            product_family="Core i7",
            model_tier="high",
            is_available=True,
        ),
        last_updated="2026-05-15",
    )

    assert product.dies[0].blocks[0].kind == "cpu"
    assert isinstance(product.dies[0].blocks[0], CPUBlock)

    payload = product.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    assert isinstance(rebuilt.dies[0].blocks[0], CPUBlock)
    assert rebuilt.dies[0].blocks[0].total_effective_cores == 10


# ---------------------------------------------------------------------------
# 5. Existing KPU + GPU YAMLs still validate (additive guarantee)
# ---------------------------------------------------------------------------

def test_v3_does_not_break_existing_catalog():
    """Both the 12-SKU KPU catalog and the 2-SKU GPU catalog must
    continue to load cleanly after v3 schema additions."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()

    kpu_count = sum(
        1 for cp in products.values()
        if isinstance(cp.dies[0].blocks[0], KPUBlock)
    )
    gpu_count = sum(
        1 for cp in products.values()
        if isinstance(cp.dies[0].blocks[0], GPUBlock)
    )
    cpu_count = sum(
        1 for cp in products.values()
        if isinstance(cp.dies[0].blocks[0], CPUBlock)
    )

    assert kpu_count == 12
    assert gpu_count == 2
    # cpu_count was 0 at the v3 schema PR (#22) baseline; v3 data
    # PR adds intel_core_i7_12700k as the first CPU SKU.
    assert cpu_count == 1
    assert kpu_count + gpu_count + cpu_count == len(products)
