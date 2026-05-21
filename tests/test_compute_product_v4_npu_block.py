"""Tests for v4 ComputeProduct schema additions: NPUBlock and friends.

Validates that:
1. NPUBlock + supporting types (NPUComputeFabric, NPUMemorySubsystem,
   NPUOnDieFabric, NPUThermalProfile, NPUTheoreticalPerformance)
   construct cleanly with realistic Hailo-8 numbers.
2. The AnyBlock discriminated union now dispatches all FOUR block
   kinds (KPU + GPU + CPU + NPU) via the ``kind`` field.
3. ComputeProduct round-trips an NPU SKU through serialize / deserialize.
4. Schema invariants: external DRAM consistency, mesh-dim consistency,
   NoC unit_count matches num_dataflow_units, INT precision required
   on fabrics, efficiency-factor ranges, ``extra: forbid``.
5. Existing KPU + GPU + CPU YAMLs continue to validate (additive guarantee).
6. Second cross-block-kind type reuse works (NPUOnDieFabric.confidence
   uses DataConfidence from process_node).
"""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    AnyBlock,
    BlockKind,
    ComputeProduct,
    CPUBlock,
    Die,
    DieRole,
    DramAttachment,
    GPUBlock,
    KPUBlock,
    KVCacheSpec,
    KVCacheStreamingKind,
    LifecycleStatus,
    Market,
    NPUBlock,
    NPUComputeFabric,
    NPUDataflowKind,
    NPUMemorySubsystem,
    NPUNoCTopology,
    NPUOnDieFabric,
    NPUSramLayout,
    NPUTheoreticalPerformance,
    NPUThermalProfile,
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
# Fixtures: Hailo-8-shaped NPU block
# ---------------------------------------------------------------------------

@pytest.fixture
def hailo8_fabric() -> NPUComputeFabric:
    """32 dataflow units, 500 INT8 ops/unit/clock, 1000 INT4."""
    return NPUComputeFabric(
        dataflow_kind=NPUDataflowKind.STRUCTURE_DRIVEN,
        circuit_class=CircuitClass.BALANCED_LOGIC,
        ops_per_unit_per_clock={"int8": 500, "int4": 1000},
        energy_per_op_int8_pj=0.34,
        energy_scaling={"int4": 0.5},
    )


@pytest.fixture
def hailo8_memory() -> NPUMemorySubsystem:
    """All on-chip SRAM, no external DRAM."""
    return NPUMemorySubsystem(
        on_chip_bandwidth_gbps=200.0,
        sram_kib_per_unit=512,
        shared_sram_kib=8 * 1024,
        shared_sram_layout=NPUSramLayout.SHARED,
        has_external_dram=False,
        sram_access_energy_pj_per_byte=2.0,
        coherence_protocol="none",
    )


@pytest.fixture
def hailo8_noc() -> NPUOnDieFabric:
    """8x4 mesh of 32 dataflow units, low-confidence per design doc."""
    return NPUOnDieFabric(
        topology=NPUNoCTopology.MESH_2D,
        bisection_bandwidth_gbps=64.0,
        unit_count=32,
        flit_size_bytes=16,
        mesh_rows=8,
        mesh_cols=4,
        hop_latency_ns=1.5,
        pj_per_flit_per_hop=1.5,
        routing_distance_factor=1.1,
        confidence=DataConfidence.THEORETICAL,
    )


@pytest.fixture
def npu_block(hailo8_fabric, hailo8_memory, hailo8_noc) -> NPUBlock:
    return NPUBlock(
        num_dataflow_units=32,
        lanes_per_unit=1,
        compute_fabrics=[hailo8_fabric],
        multi_precision_alu=["int8", "int4"],
        memory=hailo8_memory,
        noc=hailo8_noc,
        min_occupancy=0.85,
        max_concurrent_models=1,
        wave_quantization=1,
    )


# ---------------------------------------------------------------------------
# 1. NPUBlock constructs and dispatches via discriminator
# ---------------------------------------------------------------------------

def test_block_kind_npu_added_in_v4():
    assert BlockKind.NPU.value == "npu"
    kinds = {k.value for k in BlockKind}
    assert {"kpu", "gpu", "cpu", "npu"}.issubset(kinds)


def test_npu_block_constructs(npu_block):
    assert npu_block.kind == "npu"
    assert npu_block.num_dataflow_units == 32
    assert len(npu_block.compute_fabrics) == 1
    assert npu_block.compute_fabrics[0].dataflow_kind == NPUDataflowKind.STRUCTURE_DRIVEN


def test_anyblock_dispatches_to_npu_block(npu_block):
    """AnyBlock discriminated union should validate NPUBlock dict-form data."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(AnyBlock)
    payload = npu_block.model_dump()
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, NPUBlock)
    assert parsed.kind == "npu"


def test_anyblock_still_dispatches_other_block_kinds():
    """v4 must not break dispatch of KPU/GPU/CPU. Loads each from the
    catalog (KPU + GPU SKUs exist; CPU has intel_core_i7_12700k) and
    round-trips via AnyBlock."""
    from pydantic import TypeAdapter
    from embodied_schemas.loaders import load_compute_products
    products = load_compute_products()
    adapter = TypeAdapter(AnyBlock)
    by_kind = {}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        by_kind.setdefault(kind, []).append(block)
    # Confirm at least one of each known kind round-trips
    for kind in ("kpu", "gpu", "cpu"):
        assert by_kind.get(kind), f"no {kind} block in catalog"
        block = by_kind[kind][0]
        parsed = adapter.validate_python(block.model_dump())
        assert (parsed.kind.value if hasattr(parsed.kind, "value")
                else str(parsed.kind)) == kind


# ---------------------------------------------------------------------------
# 2. Schema invariants
# ---------------------------------------------------------------------------

def test_external_dram_true_requires_all_fields(hailo8_memory):
    """has_external_dram=True with missing dram_* fields must fail."""
    payload = hailo8_memory.model_dump()
    payload["has_external_dram"] = True
    # Don't populate any of the dram_* fields
    with pytest.raises(ValidationError, match="has_external_dram=True requires"):
        NPUMemorySubsystem(**payload)


def test_external_dram_false_rejects_populated_fields(hailo8_memory):
    """has_external_dram=False with populated dram_* fields must fail."""
    payload = hailo8_memory.model_dump()
    payload["has_external_dram"] = False
    payload["external_dram_type"] = MemoryType.LPDDR4X
    with pytest.raises(ValidationError, match="has_external_dram=False requires"):
        NPUMemorySubsystem(**payload)


def test_external_dram_true_with_full_fields_validates(hailo8_memory):
    """Hailo-10H-shaped: 4 GB LPDDR4X, 12 GB/s bandwidth."""
    payload = hailo8_memory.model_dump()
    payload["has_external_dram"] = True
    payload["external_dram_type"] = MemoryType.LPDDR4X
    payload["external_dram_size_gb"] = 4.0
    payload["external_dram_bandwidth_gbps"] = 12.8
    payload["dram_attachment"] = DramAttachment.CHIP_ATTACHED   # v12 requirement
    mem = NPUMemorySubsystem(**payload)
    assert mem.has_external_dram is True
    assert mem.external_dram_size_gb == 4.0


def test_mesh_2d_requires_dimensions(hailo8_noc):
    """topology=MESH_2D must have mesh_rows + mesh_cols set."""
    payload = hailo8_noc.model_dump()
    payload["mesh_rows"] = None
    payload["mesh_cols"] = None
    with pytest.raises(ValidationError, match="topology=MESH_2D requires"):
        NPUOnDieFabric(**payload)


def test_mesh_2d_dimensions_must_multiply_to_unit_count(hailo8_noc):
    """mesh_rows * mesh_cols must equal unit_count."""
    payload = hailo8_noc.model_dump()
    payload["mesh_rows"] = 6  # 6 * 4 = 24, doesn't match unit_count=32
    payload["mesh_cols"] = 4
    with pytest.raises(ValidationError, match="must equal unit_count"):
        NPUOnDieFabric(**payload)


def test_non_mesh_topology_rejects_dimensions(hailo8_noc):
    """topology=CROSSBAR shouldn't set mesh_rows/mesh_cols."""
    payload = hailo8_noc.model_dump()
    payload["topology"] = NPUNoCTopology.CROSSBAR
    # mesh_rows / mesh_cols still set from the fixture
    with pytest.raises(ValidationError, match="requires mesh_rows.*to be None"):
        NPUOnDieFabric(**payload)


def test_npu_compute_fabric_requires_int_precision():
    """NPUs must ship INT4 or INT8; FP-only fabric is wrong."""
    with pytest.raises(ValidationError, match="at least one of.*int4.*int8"):
        NPUComputeFabric(
            dataflow_kind=NPUDataflowKind.STRUCTURE_DRIVEN,
            circuit_class=CircuitClass.BALANCED_LOGIC,
            ops_per_unit_per_clock={"fp32": 100},  # invalid -- no INT
            energy_per_op_int8_pj=0.34,
        )


def test_npu_block_noc_unit_count_must_match(npu_block, hailo8_noc):
    """noc.unit_count must equal num_dataflow_units."""
    payload = npu_block.model_dump()
    # Tweak the NoC unit_count to mismatch
    payload["noc"]["unit_count"] = 16  # doesn't match num_dataflow_units=32
    payload["noc"]["mesh_rows"] = 4    # to keep mesh-dim invariant satisfied
    payload["noc"]["mesh_cols"] = 4
    with pytest.raises(ValidationError, match="noc.unit_count.*must equal num_dataflow_units"):
        NPUBlock(**payload)


def test_thermal_profile_efficiency_must_be_unit_fraction():
    with pytest.raises(ValidationError, match="outside"):
        NPUThermalProfile(
            name="2.5W",
            tdp_watts=2.5,
            cooling_solution_id="passive_heatsink_small",
            clock_mhz=1600.0,
            dvfs_enabled=False,
            efficiency_factor_by_precision={"int8": 1.5},  # invalid
        )


def test_extra_fields_forbidden_on_npu_block(npu_block):
    with pytest.raises(ValidationError):
        NPUBlock(**npu_block.model_dump(), unknown_field=42)


def test_npu_block_requires_at_least_one_fabric(hailo8_memory, hailo8_noc):
    with pytest.raises(ValidationError, match="compute_fabrics"):
        NPUBlock(
            num_dataflow_units=32,
            lanes_per_unit=1,
            compute_fabrics=[],   # empty -- min_length=1
            memory=hailo8_memory, noc=hailo8_noc,
            min_occupancy=0.85, max_concurrent_models=1, wave_quantization=1,
        )


def test_npu_theoretical_performance_negative_value_rejected():
    with pytest.raises(ValidationError, match="must be >= 0"):
        NPUTheoreticalPerformance(
            peak_ops_per_sec_by_precision={"int8": -1.0}
        )


# ---------------------------------------------------------------------------
# 3. Second cross-block-kind type reuse (DataConfidence from process_node)
# ---------------------------------------------------------------------------

def test_npu_on_die_fabric_reuses_data_confidence(hailo8_noc):
    """NPUOnDieFabric.confidence uses DataConfidence from process_node
    rather than an NPU-specific enum. This is the second cross-block-
    kind type reuse (CPU's ClockDomain from gpu_block was the first).

    Pinning the shared usage so a v5 cleanup that moves DataConfidence
    to a vendor-neutral compute_block_common module can chase callers
    safely."""
    assert isinstance(hailo8_noc.confidence, DataConfidence)
    # DataConfidence enum has THEORETICAL among its values
    assert hailo8_noc.confidence == DataConfidence.THEORETICAL


# ---------------------------------------------------------------------------
# 4. ComputeProduct end-to-end with an NPU die
# ---------------------------------------------------------------------------

def _hailo8_silicon_bin() -> KPUSiliconBin:
    """Reuse KPUSiliconBin shape (silicon_bin is general per design doc)."""
    return KPUSiliconBin(
        blocks=[
            SiliconBinBlock(
                name="dataflow_units",
                circuit_class=CircuitClass.BALANCED_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED, mtx=1200.0,
                ),
            ),
        ]
    )


def _hailo8_thermal_profile() -> KPUThermalProfile:
    """Chip-level Power.thermal_profiles still uses KPUThermalProfile
    shape across all four sprints (v5 generalization deferred).
    The richer per-precision NPU thermal data lives inside NPUBlock
    via NPUThermalProfile (the per-block detail) when needed."""
    return KPUThermalProfile(
        name="2.5W",
        tdp_watts=2.5,
        clock_mhz=1600.0,
        cooling_solution_id="passive_heatsink_small",
    )


def test_compute_product_with_npu_die_round_trips(npu_block):
    """Build a Hailo-8-shaped ComputeProduct end-to-end and round-trip."""
    product = ComputeProduct(
        id="hailo_hailo_8",
        name="Hailo-8",
        vendor="hailo",
        kind=ProductKind.CHIP,
        packaging=Packaging(kind=PackagingKind.MONOLITHIC, num_dies=1,
                            package_type="m_dot_2"),
        lifecycle=LifecycleStatus.PRODUCTION,
        dies=[
            Die(
                die_id="hailo_8_die",
                die_role=DieRole.COMPUTE,
                process_node_id="tsmc_n16",  # exists in catalog from KPU sprint
                die_size_mm2=8.0,
                transistors_billion=1.2,
                silicon_bin=_hailo8_silicon_bin(),
                clocks=KPUClocks(base_clock_mhz=1600.0, boost_clock_mhz=1600.0),
                blocks=[npu_block],
            )
        ],
        performance=KPUTheoreticalPerformance(
            int8_tops=26.0,
            bf16_tflops=0.0,
            fp32_tflops=0.0,
            int4_tops=52.0,
        ),
        power=Power(
            tdp_watts=2.5,
            max_power_watts=3.0,
            min_power_watts=2.0,
            default_thermal_profile="2.5W",
            thermal_profiles=[_hailo8_thermal_profile()],
        ),
        market=Market(
            launch_date="2019-05-01",
            target_market="edge",
            product_family="Hailo-8",
            model_tier="entry",
            is_available=True,
        ),
        last_updated="2026-05-16",
    )

    assert product.dies[0].blocks[0].kind == "npu"
    assert isinstance(product.dies[0].blocks[0], NPUBlock)

    payload = product.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    assert isinstance(rebuilt.dies[0].blocks[0], NPUBlock)
    assert rebuilt.dies[0].blocks[0].num_dataflow_units == 32


# ---------------------------------------------------------------------------
# 5. Existing KPU + GPU + CPU YAMLs still validate (additive guarantee)
# ---------------------------------------------------------------------------

def test_v4_does_not_break_existing_catalog():
    """v4 must keep the 12-KPU + 2-GPU + 1-CPU catalog loading cleanly."""
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
    npu_count = sum(
        1 for cp in products.values()
        if isinstance(cp.dies[0].blocks[0], NPUBlock)
    )

    assert kpu_count == 12
    assert gpu_count == 2
    assert cpu_count == 7   # i7 + 3 EPYC + 2 AmpereOne (sprint #62) + Xeon 8490H Sapphire Rapids (sprint #68 PR 1)
    # NPU count history: 0 at the v4 schema PR (#25); Hailo-8 (#26)
    # brought it to 1; Hailo-10H (#31, first KVCacheSpec user from #30)
    # brought it to 2; Coral Edge TPU (#33) brought it to 3. CGRA
    # additions (#35+, Plasticine v2 onwards) don't affect npu_count
    # but contribute to len(products) -- relax the closure to subset.
    assert npu_count == 3
    assert kpu_count + gpu_count + cpu_count + npu_count <= len(products)


# ---------------------------------------------------------------------------
# 6. KVCacheSpec (issue #27) -- transformer-capable NPU surface
# ---------------------------------------------------------------------------

@pytest.fixture
def hailo10h_kv_cache() -> KVCacheSpec:
    """Hailo-10H-shaped KV cache: ring-buffer streaming with LPDDR4X
    offload, INT8/INT4 asymmetric per-tier quantization."""
    return KVCacheSpec(
        max_context_length=8192,
        kv_cache_kib_per_layer=256,
        num_layers_supported=32,
        quantization={"k": "int8", "v": "int4"},
        streaming_strategy=KVCacheStreamingKind.RING_BUFFER,
        has_offload_to_dram=True,
    )


def test_kv_cache_spec_constructs(hailo10h_kv_cache):
    assert hailo10h_kv_cache.max_context_length == 8192
    assert hailo10h_kv_cache.streaming_strategy == KVCacheStreamingKind.RING_BUFFER
    assert hailo10h_kv_cache.quantization == {"k": "int8", "v": "int4"}
    assert hailo10h_kv_cache.has_offload_to_dram is True


def test_kv_cache_spec_quantization_keys_validator():
    """quantization must have exactly keys {'k', 'v'} -- catches typo'd
    YAMLs that named the tiers 'key'/'value' or omitted one tier."""
    with pytest.raises(ValidationError, match="must have exactly keys"):
        KVCacheSpec(
            max_context_length=8192,
            kv_cache_kib_per_layer=256,
            num_layers_supported=32,
            quantization={"key": "int8", "value": "int8"},   # wrong tier names
            streaming_strategy=KVCacheStreamingKind.RING_BUFFER,
            has_offload_to_dram=True,
        )

    with pytest.raises(ValidationError, match="must have exactly keys"):
        KVCacheSpec(
            max_context_length=8192,
            kv_cache_kib_per_layer=256,
            num_layers_supported=32,
            quantization={"k": "int8"},   # missing 'v'
            streaming_strategy=KVCacheStreamingKind.RING_BUFFER,
            has_offload_to_dram=True,
        )


def test_kv_cache_spec_extra_fields_forbidden(hailo10h_kv_cache):
    with pytest.raises(ValidationError):
        KVCacheSpec(**hailo10h_kv_cache.model_dump(), unknown_field=1)


def test_kv_cache_streaming_kinds_present():
    kinds = {k.value for k in KVCacheStreamingKind}
    assert kinds == {"ring_buffer", "sliding_window", "page_based", "precomputed"}


def test_npu_block_kv_cache_defaults_to_none(npu_block):
    """Hailo-8 (CNN-class) NPUBlock leaves kv_cache None -- the additive
    field must not break existing-shape construction."""
    assert npu_block.kv_cache is None


def test_npu_block_accepts_kv_cache_when_dram_present(
    hailo8_fabric, hailo8_noc, hailo10h_kv_cache
):
    """KV cache with offload=True is valid when memory.has_external_dram=True
    (Hailo-10H shape)."""
    hailo10h_memory = NPUMemorySubsystem(
        on_chip_bandwidth_gbps=200.0,
        sram_kib_per_unit=512,
        shared_sram_kib=12 * 1024,
        shared_sram_layout=NPUSramLayout.SHARED,
        has_external_dram=True,
        dram_attachment=DramAttachment.CHIP_ATTACHED,
        external_dram_type=MemoryType.LPDDR4X,
        external_dram_size_gb=4.0,
        external_dram_bandwidth_gbps=12.8,
        sram_access_energy_pj_per_byte=2.0,
        coherence_protocol="none",
    )
    block = NPUBlock(
        num_dataflow_units=32,
        lanes_per_unit=1,
        compute_fabrics=[hailo8_fabric],
        multi_precision_alu=["int8", "int4"],
        memory=hailo10h_memory,
        noc=hailo8_noc,
        min_occupancy=0.85,
        max_concurrent_models=1,
        wave_quantization=1,
        kv_cache=hailo10h_kv_cache,
    )
    assert block.kv_cache is not None
    assert block.kv_cache.has_offload_to_dram is True
    assert block.memory.has_external_dram is True


def test_npu_block_rejects_dram_offload_without_external_dram(
    hailo8_fabric, hailo8_memory, hailo8_noc, hailo10h_kv_cache
):
    """KV cache with offload=True on an SRAM-only NPU must fail --
    the cache cannot offload to nonexistent external DRAM."""
    with pytest.raises(ValidationError, match="cannot offload to nonexistent external DRAM"):
        NPUBlock(
            num_dataflow_units=32,
            lanes_per_unit=1,
            compute_fabrics=[hailo8_fabric],
            multi_precision_alu=["int8", "int4"],
            memory=hailo8_memory,        # has_external_dram=False
            noc=hailo8_noc,
            min_occupancy=0.85,
            max_concurrent_models=1,
            wave_quantization=1,
            kv_cache=hailo10h_kv_cache,  # has_offload_to_dram=True
        )


def test_npu_block_accepts_in_sram_kv_cache_without_dram(
    hailo8_fabric, hailo8_memory, hailo8_noc
):
    """Groq-LPU shape: KV cache lives entirely in on-chip SRAM
    (has_offload_to_dram=False), so SRAM-only NPU memory is OK."""
    in_sram_cache = KVCacheSpec(
        max_context_length=2048,
        kv_cache_kib_per_layer=128,
        num_layers_supported=12,
        quantization={"k": "int8", "v": "int8"},
        streaming_strategy=KVCacheStreamingKind.SLIDING_WINDOW,
        has_offload_to_dram=False,
    )
    block = NPUBlock(
        num_dataflow_units=32,
        lanes_per_unit=1,
        compute_fabrics=[hailo8_fabric],
        multi_precision_alu=["int8", "int4"],
        memory=hailo8_memory,    # no external DRAM
        noc=hailo8_noc,
        min_occupancy=0.85,
        max_concurrent_models=1,
        wave_quantization=1,
        kv_cache=in_sram_cache,
    )
    assert block.kv_cache.has_offload_to_dram is False


def test_npu_block_with_kv_cache_round_trips_through_anyblock(
    hailo8_fabric, hailo8_noc, hailo10h_kv_cache
):
    """AnyBlock dispatch + KVCacheSpec round-trip through JSON."""
    from pydantic import TypeAdapter
    hailo10h_memory = NPUMemorySubsystem(
        on_chip_bandwidth_gbps=200.0,
        sram_kib_per_unit=512,
        shared_sram_kib=12 * 1024,
        shared_sram_layout=NPUSramLayout.SHARED,
        has_external_dram=True,
        dram_attachment=DramAttachment.CHIP_ATTACHED,
        external_dram_type=MemoryType.LPDDR4X,
        external_dram_size_gb=4.0,
        external_dram_bandwidth_gbps=12.8,
        sram_access_energy_pj_per_byte=2.0,
        coherence_protocol="none",
    )
    block = NPUBlock(
        num_dataflow_units=32,
        lanes_per_unit=1,
        compute_fabrics=[hailo8_fabric],
        multi_precision_alu=["int8", "int4"],
        memory=hailo10h_memory,
        noc=hailo8_noc,
        min_occupancy=0.85,
        max_concurrent_models=1,
        wave_quantization=1,
        kv_cache=hailo10h_kv_cache,
    )

    adapter = TypeAdapter(AnyBlock)
    payload = block.model_dump(mode="json")
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, NPUBlock)
    assert parsed.kv_cache is not None
    assert parsed.kv_cache.streaming_strategy == KVCacheStreamingKind.RING_BUFFER
    assert parsed.kv_cache.quantization == {"k": "int8", "v": "int4"}
