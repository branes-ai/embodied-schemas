"""Tests for v2 ComputeProduct schema additions: GPUBlock and friends.

Validates that:
1. GPUBlock and its supporting types (GPUComputeFabric, GPUMemorySubsystem,
   GPUOnDieFabric, GPUThermalProfile, GPUTheoreticalPerformance,
   ClockDomain) construct cleanly with realistic Jetson AGX Orin numbers.
2. The AnyBlock discriminated union dispatches GPUBlock correctly via
   the ``kind`` field, alongside the existing KPUBlock dispatch.
3. ComputeProduct round-trips a GPU SKU through serialize / deserialize.
4. Schema invariants hold: clock ordering, efficiency factor ranges,
   "extra fields forbidden" on every new type.
5. v1 KPU YAMLs continue to validate (additive guarantee).
"""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    AnyBlock,
    BlockKind,
    ClockDomain,
    ComputeProduct,
    Die,
    DieRole,
    GPUBlock,
    GPUComputeFabric,
    GPUFabricKind,
    GPUL1Kind,
    GPUL2Topology,
    GPUMemorySubsystem,
    GPUNoCTopology,
    GPUOnDieFabric,
    GPUTheoreticalPerformance,
    GPUThermalProfile,
    KPUBlock,
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
# Fixtures: Jetson AGX Orin 64GB-shaped GPU block
# ---------------------------------------------------------------------------

@pytest.fixture
def cuda_fabric() -> GPUComputeFabric:
    """128 CUDA cores per SM, 2 FP32 ops/clock per core (FMA)."""
    return GPUComputeFabric(
        fabric_kind=GPUFabricKind.CUDA_CORE,
        circuit_class=CircuitClass.HP_LOGIC,
        units_per_sm=128,
        ops_per_unit_per_clock={"fp64": 2, "fp32": 2, "fp16": 4, "int8": 8},
        energy_per_flop_fp32_pj=1.9,
        energy_scaling={"fp64": 2.0, "fp32": 1.0, "fp16": 0.5, "int8": 0.125},
    )


@pytest.fixture
def tensor_fabric() -> GPUComputeFabric:
    """4 Tensor cores per SM, 256 FP16 ops/clock and 512 INT8 ops/clock."""
    return GPUComputeFabric(
        fabric_kind=GPUFabricKind.TENSOR_CORE,
        circuit_class=CircuitClass.HP_LOGIC,
        units_per_sm=4,
        ops_per_unit_per_clock={"fp16": 256, "int8": 512},
        energy_per_flop_fp32_pj=1.62,
        energy_scaling={"fp16": 0.5, "int8": 0.125},
    )


@pytest.fixture
def agx_orin_memory() -> GPUMemorySubsystem:
    """64 GB LPDDR5, 4 MB shared L2 (LLC), no L3, 256-bit bus."""
    return GPUMemorySubsystem(
        memory_type=MemoryType.LPDDR5,
        memory_size_gb=64.0,
        memory_bus_bits=256,
        memory_bandwidth_gbps=204.8,
        memory_controllers=8,
        l1_kib_per_sm=128,
        l1_kind=GPUL1Kind.UNIFIED,
        l2_total_kib=4 * 1024,
        l2_topology=GPUL2Topology.SHARED_LLC,
        l3_present=False,
        l3_total_kib=0,
        coherence_protocol="none",
        read_energy_pj_per_byte=15.0,
        write_energy_pj_per_byte=18.0,
    )


@pytest.fixture
def agx_orin_noc() -> GPUOnDieFabric:
    """SM-to-L2 crossbar, 16 SMs as ports."""
    return GPUOnDieFabric(
        topology=GPUNoCTopology.CROSSBAR,
        bisection_bandwidth_gbps=2048.0,
        controller_count=16,
        flit_size_bytes=32,
        hop_latency_ns=2.0,
        pj_per_flit_per_hop=8.0,
        routing_distance_factor=1.0,
    )


@pytest.fixture
def gpu_block(cuda_fabric, tensor_fabric, agx_orin_memory, agx_orin_noc) -> GPUBlock:
    """Jetson AGX Orin 64GB shaped GPUBlock."""
    return GPUBlock(
        num_sms=16,
        cuda_cores_per_sm=128,
        tensor_cores_per_sm=4,
        threads_per_sm=64,
        warps_per_sm=2,
        warp_size=32,
        compute_fabrics=[cuda_fabric, tensor_fabric],
        multi_precision_alu=["fp64", "fp32", "fp16", "bf16", "int8"],
        memory=agx_orin_memory,
        noc=agx_orin_noc,
        min_occupancy=0.3,
        max_concurrent_kernels=8,
        wave_quantization=4,
    )


# ---------------------------------------------------------------------------
# 1. GPUBlock constructs and dispatches via discriminator
# ---------------------------------------------------------------------------

def test_block_kind_gpu_added_in_v2():
    assert BlockKind.GPU.value == "gpu"
    kinds = {k.value for k in BlockKind}
    assert {"kpu", "gpu"}.issubset(kinds)


def test_gpu_block_constructs(gpu_block):
    assert gpu_block.kind == "gpu"
    assert gpu_block.num_sms == 16
    assert gpu_block.cuda_cores_per_sm == 128
    assert gpu_block.tensor_cores_per_sm == 4
    assert len(gpu_block.compute_fabrics) == 2


def test_gpu_block_total_compute_units(gpu_block):
    """Cross-check that fabric units * num_sms gives the chip totals."""
    cuda = gpu_block.compute_fabrics[0]
    tensor = gpu_block.compute_fabrics[1]
    assert cuda.units_per_sm * gpu_block.num_sms == 2048
    assert tensor.units_per_sm * gpu_block.num_sms == 64


def test_anyblock_discriminator_dispatches_to_gpu_block(gpu_block):
    """AnyBlock should validate dict-form GPUBlock data."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(AnyBlock)
    payload = gpu_block.model_dump()
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, GPUBlock)
    assert parsed.kind == "gpu"


def test_anyblock_discriminator_still_dispatches_to_kpu_block():
    """v2 must not break v1 KPU dispatch on the AnyBlock union."""
    from pydantic import TypeAdapter
    from embodied_schemas import (
        KPUMemorySubsystem,
        KPUNoCSpec,
    )
    from embodied_schemas.kpu import KPUTileSpec
    kpu_block = KPUBlock(
        total_tiles=64,
        multi_precision_alu=["int8", "bf16"],
        tiles=[
            KPUTileSpec(
                tile_type="INT8",
                num_tiles=64,
                pe_array_rows=32,
                pe_array_cols=32,
                pe_circuit_class=CircuitClass.HP_LOGIC,
                ops_per_tile_per_clock={"int8": 1024.0},
            )
        ],
        noc=KPUNoCSpec(
            topology="mesh_2d",
            mesh_rows=8,
            mesh_cols=8,
            flit_bytes=16,
        ),
        memory=KPUMemorySubsystem(
            memory_type=MemoryType.LPDDR5,
            memory_size_gb=2.0,
            memory_bus_bits=64,
            memory_bandwidth_gbps=64.0,
            memory_controllers=2,
            l3_kib_per_tile=64,
        ),
    )
    adapter = TypeAdapter(AnyBlock)
    parsed = adapter.validate_python(kpu_block.model_dump())
    assert isinstance(parsed, KPUBlock)
    assert parsed.kind == BlockKind.KPU


# ---------------------------------------------------------------------------
# 2. Schema invariants
# ---------------------------------------------------------------------------

def test_clock_domain_rejects_inverted_ordering():
    """sustained_hz must lie between base_hz and boost_hz."""
    with pytest.raises(ValidationError, match="Clock ordering"):
        ClockDomain(base_hz=1e9, boost_hz=500e6, sustained_hz=750e6)


def test_clock_domain_accepts_equal_values():
    """A locked-clock GPU (no DVFS) sets all three equal."""
    cd = ClockDomain(base_hz=1e9, boost_hz=1e9, sustained_hz=1e9, dvfs_enabled=False)
    assert cd.base_hz == cd.sustained_hz == cd.boost_hz


def test_thermal_profile_efficiency_must_be_unit_fraction():
    with pytest.raises(ValidationError, match="outside"):
        GPUThermalProfile(
            name="30W",
            tdp_watts=30.0,
            cooling_solution_id="active_fan",
            clock_domain=ClockDomain(
                base_hz=612e6, boost_hz=1.15e9, sustained_hz=650e6
            ),
            memory_clock_mhz=3200.0,
            efficiency_factor_by_precision={"int8": 1.5},  # invalid > 1
        )


def test_extra_fields_forbidden_on_gpu_block(gpu_block):
    """``extra: forbid`` should reject typo'd / unknown fields."""
    with pytest.raises(ValidationError):
        GPUBlock(
            **gpu_block.model_dump(),
            unknown_field=42,
        )


def test_gpu_block_requires_at_least_one_compute_fabric(
    agx_orin_memory, agx_orin_noc
):
    with pytest.raises(ValidationError):
        GPUBlock(
            num_sms=16,
            cuda_cores_per_sm=128,
            tensor_cores_per_sm=4,
            threads_per_sm=64,
            warps_per_sm=2,
            warp_size=32,
            compute_fabrics=[],   # empty -- violates min_length=1
            multi_precision_alu=["fp32"],
            memory=agx_orin_memory,
            noc=agx_orin_noc,
            min_occupancy=0.3,
            max_concurrent_kernels=8,
            wave_quantization=4,
        )


def test_gpu_theoretical_performance_negative_value_rejected():
    with pytest.raises(ValidationError, match="must be >= 0"):
        GPUTheoreticalPerformance(
            peak_ops_per_sec_by_precision={"int8": -1.0}
        )


# ---------------------------------------------------------------------------
# 3. ComputeProduct end-to-end with a GPU die
# ---------------------------------------------------------------------------

def _agx_orin_silicon_bin() -> KPUSiliconBin:
    """Reuse KPUSiliconBin shape for v2 -- silicon_bin is general-purpose,
    only the count_ref strings are KPU-flavored. For a GPU we'd use
    ``count_ref="sm.cuda_core"`` etc.; the schema doesn't validate the
    string content. v3 may introduce a vendor-neutral SiliconBin type
    if convenient."""
    return KPUSiliconBin(
        blocks=[
            SiliconBinBlock(
                name="cuda_cores",
                circuit_class=CircuitClass.HP_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=4000.0
                ),
            ),
        ]
    )


def _agx_orin_thermal_profile() -> KPUThermalProfile:
    """Use KPUThermalProfile in chip-wide Power for now -- the v2
    GPUThermalProfile is referenced from inside GPUBlock; the chip-level
    Power.thermal_profiles list still uses KPU-shaped profiles for
    schema-compatibility. Generalizing that list is v3 scope (see the
    design doc's "open question 7")."""
    return KPUThermalProfile(
        name="30W",
        tdp_watts=30.0,
        clock_mhz=650.0,
        cooling_solution_id="active_fan",
    )


def test_compute_product_with_gpu_die_round_trips(gpu_block):
    """Build a Jetson-AGX-Orin-shaped ComputeProduct end-to-end and round
    trip through model_dump / model_validate."""
    product = ComputeProduct(
        id="nvidia_jetson_agx_orin_64gb",
        name="NVIDIA Jetson AGX Orin 64GB",
        vendor="nvidia",
        kind=ProductKind.CHIP,
        packaging=Packaging(kind=PackagingKind.MONOLITHIC, num_dies=1),
        lifecycle=LifecycleStatus.PRODUCTION,
        dies=[
            Die(
                die_id="ga10b_gpu",
                die_role=DieRole.COMPUTE,
                process_node_id="samsung_8lpp",
                die_size_mm2=455.0,
                transistors_billion=17.0,
                silicon_bin=_agx_orin_silicon_bin(),
                clocks=KPUClocks(base_clock_mhz=612.0, boost_clock_mhz=1300.0),
                blocks=[gpu_block],
            )
        ],
        performance=KPUTheoreticalPerformance(
            int8_tops=31.95,  # GPU-side roll-up
            bf16_tflops=15.97,
            fp32_tflops=2.66,
        ),
        power=Power(
            tdp_watts=30.0,
            max_power_watts=60.0,
            min_power_watts=15.0,
            default_thermal_profile="30W",
            thermal_profiles=[_agx_orin_thermal_profile()],
        ),
        market=Market(
            launch_date="2022-11-01",
            target_market="edge",
            product_family="Jetson Orin",
            model_tier="high",
            is_available=True,
        ),
        last_updated="2026-05-15",
    )

    assert product.dies[0].blocks[0].kind == "gpu"
    assert isinstance(product.dies[0].blocks[0], GPUBlock)

    payload = product.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    assert isinstance(rebuilt.dies[0].blocks[0], GPUBlock)
    assert rebuilt.dies[0].blocks[0].num_sms == 16


# ---------------------------------------------------------------------------
# 4. Existing KPU YAMLs still validate (additive guarantee)
# ---------------------------------------------------------------------------

def test_v2_does_not_break_existing_kpu_catalog():
    """The full 12-SKU KPU ComputeProduct catalog must continue to load
    cleanly after v2 schema additions."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    assert len(products) >= 12
    for sku_id, product in products.items():
        assert isinstance(product, ComputeProduct)
        # All v1 SKUs are KPU
        assert product.dies[0].blocks[0].kind == BlockKind.KPU
