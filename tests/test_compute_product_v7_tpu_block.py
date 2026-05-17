"""Tests for v7 ComputeProduct schema additions: TPUBlock and friends.

Validates that:
1. TPUBlock + supporting types (TPUComputeFabric,
   TPUTileEnergyCoefficients, TPUMemorySubsystem, TPUOnDieFabric,
   TPUThermalProfile, TPUTheoreticalPerformance) construct cleanly
   with realistic TPU v4 numbers.
2. The AnyBlock discriminated union now dispatches all SEVEN block
   kinds (KPU + GPU + CPU + NPU + CGRA + DPU + TPU) via the ``kind``
   field.
3. ComputeProduct round-trips a TPU SKU through serialize / deserialize.
4. Schema invariants: external_dram consistency, NoC unit_count matches
   num_mxus, INT/BF16 precision required on fabrics, MXU dimension
   allow-list, ICI port/bandwidth consistency, efficiency-factor ranges,
   ``extra: forbid``, non-positive ops rejected.
5. Existing KPU + GPU + CPU + NPU + CGRA + DPU YAMLs continue to
   validate (additive guarantee).
6. Fifth cross-block-kind type reuse works (TPUOnDieFabric.confidence
   uses DataConfidence; TPUMemorySubsystem.external_dram_type uses
   MemoryType -- same as NPU/DPU).
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
    GPUBlock,
    KPUBlock,
    LifecycleStatus,
    Market,
    NPUBlock,
    Packaging,
    PackagingKind,
    Power,
    ProductKind,
    TPUBlock,
    TPUComputeFabric,
    TPUFabricKind,
    TPUMemorySubsystem,
    TPUNoCTopology,
    TPUOnDieFabric,
    TPUTheoreticalPerformance,
    TPUThermalProfile,
    TPUTileEnergyCoefficients,
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
# Fixtures: TPU-v4-shaped TPU block
# ---------------------------------------------------------------------------

@pytest.fixture
def tpu_v4_fabric() -> TPUComputeFabric:
    """2 MXUs, 128x128 each, BF16 + INT8 at 2 ops/MAC."""
    return TPUComputeFabric(
        fabric_kind=TPUFabricKind.TPU_V2_PLUS,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        ops_per_unit_per_clock={"bf16": 2, "int8": 2},
        energy_per_op_bf16_pj=0.225,
        energy_scaling={"int8": 0.25, "fp32": 2.0},
    )


@pytest.fixture
def tpu_v4_tile_energies() -> TPUTileEnergyCoefficients:
    """The 9 canonical tile energy coefficients for TPU v4 7nm HBM2e."""
    return TPUTileEnergyCoefficients(
        mac_energy_pj=0.25,
        weight_memory_energy_pj_per_byte=10.0,
        weight_fifo_energy_pj_per_byte=0.5,
        unified_buffer_read_energy_pj_per_byte=0.5,
        unified_buffer_write_energy_pj_per_byte=0.5,
        accumulator_read_energy_pj_per_element=0.3,
        accumulator_write_energy_pj_per_element=0.4,
        weight_shift_in_energy_pj_per_element=0.3,
        activation_stream_energy_pj_per_element=0.2,
    )


@pytest.fixture
def tpu_v4_memory() -> TPUMemorySubsystem:
    """32 MiB UB + 32 GiB HBM2e at 1.2 TB/s."""
    return TPUMemorySubsystem(
        on_chip_bandwidth_gbps=2000.0,
        unified_buffer_size_kib=32 * 1024,    # 32 MiB
        unified_buffer_access_energy_pj_per_byte=0.5,
        has_external_dram=True,
        external_dram_type=MemoryType.HBM2,
        external_dram_size_gb=32.0,
        external_dram_bandwidth_gbps=1200.0,
        external_dram_access_energy_pj_per_byte=10.0,
        coherence_protocol="none",
    )


@pytest.fixture
def tpu_v4_noc() -> TPUOnDieFabric:
    """MULTI_CROSSBAR -- 2 MXUs share the UB."""
    return TPUOnDieFabric(
        topology=TPUNoCTopology.MULTI_CROSSBAR,
        bisection_bandwidth_gbps=2000.0,
        unit_count=2,
        flit_size_bytes=32,
        hop_latency_ns=1.0,
        pj_per_flit_per_hop=2.0,
        routing_distance_factor=1.0,
        confidence=DataConfidence.THEORETICAL,
    )


@pytest.fixture
def tpu_block(tpu_v4_fabric, tpu_v4_tile_energies, tpu_v4_memory, tpu_v4_noc) -> TPUBlock:
    return TPUBlock(
        num_mxus=2,
        mxu_dim_rows=128,
        mxu_dim_cols=128,
        weight_tile_size_kib=32,
        weight_fifo_depth=2,
        pipeline_fill_cycles=128,
        accumulator_size_kib_per_mxu=2048,
        tile_energy_coefficients=tpu_v4_tile_energies,
        ici_port_count=6,
        ici_bandwidth_per_port_gbps=400.0,
        ici_topology_hint="3d_torus_2x2x2",
        compute_fabrics=[tpu_v4_fabric],
        multi_precision_alu=["bf16", "int8", "fp32"],
        memory=tpu_v4_memory,
        noc=tpu_v4_noc,
        min_occupancy=0.5,
        max_concurrent_models=1,
        wave_quantization=1,
    )


# ---------------------------------------------------------------------------
# 1. TPUBlock constructs and dispatches via discriminator
# ---------------------------------------------------------------------------

def test_block_kind_tpu_added_in_v7():
    assert BlockKind.TPU.value == "tpu"
    kinds = {k.value for k in BlockKind}
    assert {"kpu", "gpu", "cpu", "npu", "cgra", "dpu", "tpu"}.issubset(kinds)


def test_tpu_block_constructs(tpu_block):
    assert tpu_block.kind == "tpu"
    assert tpu_block.num_mxus == 2
    assert tpu_block.mxu_dim_rows == 128
    assert tpu_block.mxu_dim_cols == 128
    assert tpu_block.weight_tile_size_kib == 32
    assert tpu_block.weight_fifo_depth == 2
    assert tpu_block.pipeline_fill_cycles == 128
    assert tpu_block.accumulator_size_kib_per_mxu == 2048
    assert tpu_block.ici_port_count == 6
    assert tpu_block.ici_bandwidth_per_port_gbps == 400.0
    assert tpu_block.ici_topology_hint == "3d_torus_2x2x2"


def test_anyblock_dispatches_to_tpu_block(tpu_block):
    """AnyBlock discriminated union should validate TPUBlock dict-form data."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(AnyBlock)
    payload = tpu_block.model_dump()
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, TPUBlock)
    assert parsed.kind == "tpu"


def test_anyblock_still_dispatches_other_block_kinds():
    """v7 must not break dispatch of KPU/GPU/CPU/NPU/CGRA/DPU."""
    from pydantic import TypeAdapter
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    adapter = TypeAdapter(AnyBlock)
    by_kind = {}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        by_kind.setdefault(kind, []).append(block)
    for kind in ("kpu", "gpu", "cpu", "npu", "cgra", "dpu"):
        assert by_kind.get(kind), f"no {kind} block in catalog"
        block = by_kind[kind][0]
        parsed = adapter.validate_python(block.model_dump())
        assert (parsed.kind.value if hasattr(parsed.kind, "value")
                else str(parsed.kind)) == kind


# ---------------------------------------------------------------------------
# 2. Schema invariants
# ---------------------------------------------------------------------------

def test_external_dram_true_requires_all_fields(tpu_v4_memory):
    """has_external_dram=True with missing external_dram_* fields must fail."""
    payload = tpu_v4_memory.model_dump()
    payload["external_dram_type"] = None
    payload["external_dram_size_gb"] = None
    payload["external_dram_bandwidth_gbps"] = None
    with pytest.raises(ValidationError, match=r"has_external_dram=True requires"):
        TPUMemorySubsystem(**payload)


def test_external_dram_false_rejects_populated_fields(tpu_v4_memory):
    """has_external_dram=False with populated fields must fail."""
    payload = tpu_v4_memory.model_dump()
    payload["has_external_dram"] = False
    with pytest.raises(ValidationError, match=r"has_external_dram=False requires"):
        TPUMemorySubsystem(**payload)


def test_tpu_compute_fabric_requires_int_or_bf16():
    """TPUs must ship INT8 or BF16; FP-only fabric is wrong (TPUs
    emulate FP32 from BF16 building blocks)."""
    with pytest.raises(ValidationError, match=r"at least one of.*int8.*bf16"):
        TPUComputeFabric(
            fabric_kind=TPUFabricKind.TPU_V2_PLUS,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            ops_per_unit_per_clock={"fp32": 2},   # invalid -- no INT/BF16
            energy_per_op_bf16_pj=0.225,
        )


def test_tpu_compute_fabric_rejects_non_positive_ops():
    """Zero / negative ops/unit/clock are not meaningful capacity
    numbers and must be rejected (same class of bug the DPU loader
    fix in graphs#202 addressed)."""
    with pytest.raises(ValidationError, match=r"values must be positive"):
        TPUComputeFabric(
            fabric_kind=TPUFabricKind.TPU_V2_PLUS,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            ops_per_unit_per_clock={"bf16": 0},   # invalid -- zero cap
            energy_per_op_bf16_pj=0.225,
        )


def test_tpu_block_noc_unit_count_must_match(tpu_block):
    """noc.unit_count must equal num_mxus."""
    payload = tpu_block.model_dump()
    payload["noc"]["unit_count"] = 4   # doesn't match num_mxus=2
    with pytest.raises(ValidationError, match=r"noc\.unit_count.*must equal num_mxus"):
        TPUBlock(**payload)


def test_tpu_block_mxu_dim_rejects_non_canonical_values(tpu_block):
    """MXU dimensions must be in the canonical TPU set {64, 128, 256, 512}."""
    payload = tpu_block.model_dump()
    payload["mxu_dim_rows"] = 100   # not in canonical set
    with pytest.raises(ValidationError, match=r"mxu_dim_rows.*not in the canonical TPU set"):
        TPUBlock(**payload)


def test_tpu_block_accepts_v1_size_256x256(tpu_v4_fabric, tpu_v4_tile_energies, tpu_v4_memory):
    """TPU v1 uses 256x256 MXU; schema must accept this."""
    v1_noc = TPUOnDieFabric(
        topology=TPUNoCTopology.CROSSBAR,
        bisection_bandwidth_gbps=500.0,
        unit_count=1,   # v1 has 1 MXU
        flit_size_bytes=32,
        hop_latency_ns=1.0,
        pj_per_flit_per_hop=2.0,
    )
    block = TPUBlock(
        num_mxus=1, mxu_dim_rows=256, mxu_dim_cols=256,
        weight_tile_size_kib=64, weight_fifo_depth=4,
        pipeline_fill_cycles=256,
        accumulator_size_kib_per_mxu=4096,
        tile_energy_coefficients=tpu_v4_tile_energies,
        compute_fabrics=[tpu_v4_fabric],
        memory=tpu_v4_memory, noc=v1_noc,
    )
    assert block.mxu_dim_rows == 256
    assert block.mxu_dim_cols == 256


def test_tpu_block_ici_consistency_port_without_bandwidth(tpu_block):
    """If ici_port_count > 0, ici_bandwidth_per_port_gbps must be > 0."""
    payload = tpu_block.model_dump()
    payload["ici_bandwidth_per_port_gbps"] = 0.0
    with pytest.raises(ValidationError, match=r"requires ici_bandwidth_per_port_gbps > 0"):
        TPUBlock(**payload)


def test_tpu_block_ici_consistency_bandwidth_without_port(tpu_block):
    """If ici_bandwidth_per_port_gbps > 0, ici_port_count must be > 0."""
    payload = tpu_block.model_dump()
    payload["ici_port_count"] = 0
    with pytest.raises(ValidationError, match=r"requires ici_port_count > 0"):
        TPUBlock(**payload)


def test_tpu_block_no_ici_is_valid(tpu_v4_fabric, tpu_v4_tile_energies, tpu_v4_memory, tpu_v4_noc):
    """tpu_edge_pro will have ici_port_count=0 (no pod). Schema must accept this."""
    block = TPUBlock(
        num_mxus=2, mxu_dim_rows=128, mxu_dim_cols=128,
        weight_tile_size_kib=32, weight_fifo_depth=2,
        pipeline_fill_cycles=128,
        accumulator_size_kib_per_mxu=2048,
        tile_energy_coefficients=tpu_v4_tile_energies,
        ici_port_count=0,   # no ICI
        ici_bandwidth_per_port_gbps=0.0,
        compute_fabrics=[tpu_v4_fabric],
        memory=tpu_v4_memory, noc=tpu_v4_noc,
    )
    assert block.ici_port_count == 0
    assert block.ici_bandwidth_per_port_gbps == 0.0


def test_thermal_profile_efficiency_must_be_unit_fraction():
    with pytest.raises(ValidationError, match="outside"):
        TPUThermalProfile(
            name="default",
            tdp_watts=350.0,
            cooling_solution_id="liquid_cooled",
            clock_mhz=1050.0,
            dvfs_enabled=False,
            efficiency_factor_by_precision={"bf16": 1.5},   # invalid
        )


def test_extra_fields_forbidden_on_tpu_block(tpu_block):
    with pytest.raises(ValidationError):
        TPUBlock(**tpu_block.model_dump(), unknown_field=42)


def test_tpu_block_requires_at_least_one_fabric(tpu_v4_tile_energies, tpu_v4_memory, tpu_v4_noc):
    with pytest.raises(ValidationError, match="compute_fabrics"):
        TPUBlock(
            num_mxus=2, mxu_dim_rows=128, mxu_dim_cols=128,
            weight_tile_size_kib=32, weight_fifo_depth=2,
            pipeline_fill_cycles=128, accumulator_size_kib_per_mxu=2048,
            tile_energy_coefficients=tpu_v4_tile_energies,
            compute_fabrics=[],   # empty
            memory=tpu_v4_memory, noc=tpu_v4_noc,
        )


def test_tpu_theoretical_performance_negative_value_rejected():
    with pytest.raises(ValidationError, match=r"must be >= 0"):
        TPUTheoreticalPerformance(
            peak_ops_per_sec_by_precision={"bf16": -1.0}
        )


# ---------------------------------------------------------------------------
# 3. Fifth cross-block-kind type reuse
# ---------------------------------------------------------------------------

def test_tpu_on_die_fabric_reuses_data_confidence(tpu_v4_noc):
    """TPUOnDieFabric.confidence reuses DataConfidence (same as
    NPU/CGRA/DPU -- fifth cross-block-kind reuse pattern)."""
    assert isinstance(tpu_v4_noc.confidence, DataConfidence)
    assert tpu_v4_noc.confidence == DataConfidence.THEORETICAL


def test_tpu_memory_reuses_memory_type(tpu_v4_memory):
    """TPUMemorySubsystem.external_dram_type reuses MemoryType from
    gpu (same as NPU/DPU). MemoryType covers HBM/HBM2/HBM3/DDR3/LPDDR4X
    -- all 5 TPU SKUs without enum extension."""
    assert isinstance(tpu_v4_memory.external_dram_type, MemoryType)
    assert tpu_v4_memory.external_dram_type == MemoryType.HBM2


def test_memory_type_covers_all_5_tpu_sku_memories():
    """Cover-test: MemoryType has all the memories the 5 TPU SKUs use."""
    expected = {"hbm", "hbm2", "hbm3", "ddr3", "lpddr4x"}
    available = {m.value.lower() for m in MemoryType}
    assert expected.issubset(available), (
        f"MemoryType missing: {expected - available}"
    )


# ---------------------------------------------------------------------------
# 4. ComputeProduct end-to-end with a TPU die
# ---------------------------------------------------------------------------

def _tpu_v4_silicon_bin() -> KPUSiliconBin:
    return KPUSiliconBin(
        blocks=[
            SiliconBinBlock(
                name="mxu_array",
                circuit_class=CircuitClass.BALANCED_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=50000.0,   # ~50B for full v4 chip
                ),
            ),
        ]
    )


def _tpu_v4_thermal_profile() -> KPUThermalProfile:
    """Chip-level Power.thermal_profiles still uses KPUThermalProfile
    shape across all v1-v7 sprints (v8 generalization deferred)."""
    return KPUThermalProfile(
        name="default",
        tdp_watts=350.0,
        clock_mhz=1050.0,
        cooling_solution_id="liquid_cooled",
    )


def test_compute_product_with_tpu_die_round_trips(tpu_block):
    """Build a TPU-v4-shaped ComputeProduct end-to-end and round-trip
    through JSON."""
    product = ComputeProduct(
        id="google_tpu_v4",
        name="Google TPU v4",
        vendor="google",
        kind=ProductKind.CHIP,
        packaging=Packaging(kind=PackagingKind.MONOLITHIC, num_dies=1,
                            package_type="datacenter_oam"),
        lifecycle=LifecycleStatus.PRODUCTION,
        dies=[
            Die(
                die_id="tpu_v4_die",
                die_role=DieRole.COMPUTE,
                process_node_id="tsmc_n7",   # already in catalog
                die_size_mm2=600.0,
                transistors_billion=50.0,
                silicon_bin=_tpu_v4_silicon_bin(),
                clocks=KPUClocks(base_clock_mhz=1050.0, boost_clock_mhz=1050.0),
                blocks=[tpu_block],
            )
        ],
        performance=KPUTheoreticalPerformance(
            int8_tops=550.0,
            bf16_tflops=275.0,
            fp32_tflops=137.5,
            int4_tops=0.0,
        ),
        power=Power(
            tdp_watts=350.0,
            max_power_watts=400.0,
            min_power_watts=50.0,
            default_thermal_profile="default",
            thermal_profiles=[_tpu_v4_thermal_profile()],
        ),
        market=Market(
            launch_date="2021-12-01",
            target_market="datacenter",
            product_family="TPU",
            model_tier="datacenter",
            is_available=True,
        ),
        last_updated="2026-05-17",
    )

    assert product.dies[0].blocks[0].kind == "tpu"
    assert isinstance(product.dies[0].blocks[0], TPUBlock)

    payload = product.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    assert isinstance(rebuilt.dies[0].blocks[0], TPUBlock)
    assert rebuilt.dies[0].blocks[0].num_mxus == 2
    assert rebuilt.dies[0].blocks[0].mxu_dim_rows == 128


# ---------------------------------------------------------------------------
# 5. Existing v1-v6 YAMLs still validate (additive guarantee)
# ---------------------------------------------------------------------------

def test_v7_does_not_break_existing_catalog():
    """v7 must keep the v6 catalog loading cleanly (12 KPU + 2 GPU +
    1 CPU + 3 NPU + 1 CGRA + 1 DPU = 20 ComputeProducts)."""
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()

    counts = {"kpu": 0, "gpu": 0, "cpu": 0, "npu": 0, "cgra": 0, "dpu": 0, "tpu": 0}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts[kind] = counts.get(kind, 0) + 1

    assert counts["kpu"] == 12
    assert counts["gpu"] == 2
    assert counts["cpu"] == 1
    assert counts["npu"] == 3
    assert counts["cgra"] == 1
    assert counts["dpu"] == 1
    # tpu_count is 0 at the v7 schema PR baseline; the next data PR
    # will add Google TPU v4 as the first TPU SKU.
    assert counts["tpu"] == 0
    assert sum(counts.values()) == len(products)
