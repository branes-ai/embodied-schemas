"""Tests for the second NPU SKU YAML: hailo_hailo_10h.

Catalog gain: 12 KPU + 2 GPU + 1 CPU + 2 NPU = 17 products. First
SKU to exercise the v4+ ``KVCacheSpec`` extension (transformer-
capable NPU; landed in embodied-schemas#30) and the first NPU with
external DRAM (LPDDR4X) populating the ``has_external_dram=True`` path.

Same shape as ``test_compute_product_v4_hailo_8_yaml.py``.
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    KVCacheStreamingKind,
    LifecycleStatus,
    NPUBlock,
    NPUDataflowKind,
    NPUNoCTopology,
    NPUSramLayout,
    PackagingKind,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import DataConfidence


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def hailo10h(all_products) -> ComputeProduct:
    cp = all_products.get("hailo_hailo_10h")
    if cp is None:
        pytest.fail("hailo_hailo_10h missing from catalog")
    return cp


@pytest.fixture(scope="module")
def hailo10h_compute_die(hailo10h):
    die = next((d for d in hailo10h.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Hailo-10H has no compute die")
    return die


@pytest.fixture(scope="module")
def hailo10h_npu_block(hailo10h_compute_die) -> NPUBlock:
    block = next(
        (b for b in hailo10h_compute_die.blocks if isinstance(b, NPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("Hailo-10H compute die has no NPUBlock")
    return block


# ---------------------------------------------------------------------------
# Catalog: Hailo-10H is the second NPU SKU (joins Hailo-8 under hailo/)
# ---------------------------------------------------------------------------

def test_catalog_now_includes_both_hailo_skus(all_products):
    hailo_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "hailo")
    assert hailo_skus == ["hailo_hailo_10h", "hailo_hailo_8"]


def test_catalog_has_17_total_products(all_products):
    """Tight: 12 KPU + 2 GPU + 1 CPU + 2 NPU = 17. A future addition
    fires this test as a deliberate-update reminder."""
    counts_by_kind: dict[str, int] = {}
    for cp in all_products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    assert counts_by_kind == {"kpu": 12, "gpu": 2, "cpu": 1, "npu": 2}, (
        f"unexpected catalog composition: {counts_by_kind}"
    )


def test_other_vendors_unaffected_by_hailo10h_addition(all_products):
    """Additive guarantee: adding the second hailo SKU must not perturb
    stillwater/, nvidia/, or intel/ loading."""
    stillwater = [s for s, cp in all_products.items() if cp.vendor == "stillwater"]
    nvidia = [s for s, cp in all_products.items() if cp.vendor == "nvidia"]
    intel = [s for s, cp in all_products.items() if cp.vendor == "intel"]
    assert len(stillwater) == 12
    assert len(nvidia) == 2
    assert len(intel) == 1


# ---------------------------------------------------------------------------
# Hailo-10H identity / die / silicon_bin
# ---------------------------------------------------------------------------

def test_hailo10h_identity(hailo10h):
    assert hailo10h.id == "hailo_hailo_10h"
    assert hailo10h.vendor == "hailo"
    assert hailo10h.packaging.kind == PackagingKind.MONOLITHIC
    assert hailo10h.packaging.package_type == "m_dot_2"
    assert hailo10h.lifecycle == LifecycleStatus.PRODUCTION


def test_hailo10h_die_references_tsmc_n16(hailo10h_compute_die):
    """Hailo-10H also ships on TSMC N16 (same process as Hailo-8)."""
    die = hailo10h_compute_die
    assert die.process_node_id == "tsmc_n16"
    assert die.die_role.value == "compute"
    # Slightly larger than Hailo-8 (30 mm^2) due to more units + larger SRAM + LPDDR4X PHY
    assert die.die_size_mm2 == pytest.approx(38.0, rel=0.05)
    assert die.transistors_billion == pytest.approx(2.8, rel=0.1)


def test_hailo10h_silicon_bin_reconciles_with_die_total(hailo10h_compute_die):
    die = hailo10h_compute_die
    sb_total_mtx = sum(
        b.transistor_source.mtx
        for b in die.silicon_bin.blocks
        if b.transistor_source.mtx is not None
    )
    declared_mtx = die.transistors_billion * 1000.0
    rel_err = abs(sb_total_mtx - declared_mtx) / declared_mtx
    assert rel_err < 0.10, (
        f"silicon_bin sum {sb_total_mtx:.0f} Mtx differs from declared "
        f"{declared_mtx:.0f} Mtx by {rel_err*100:.1f}%"
    )


# ---------------------------------------------------------------------------
# NPUBlock structural fields (transformer-specific)
# ---------------------------------------------------------------------------

def test_hailo10h_block_is_npu(hailo10h_npu_block):
    assert isinstance(hailo10h_npu_block, NPUBlock)
    assert hailo10h_npu_block.kind == "npu"


def test_hailo10h_dataflow_hierarchy(hailo10h_npu_block):
    """40 dataflow units (vs Hailo-8's 32), scalar (lanes_per_unit=1)."""
    block = hailo10h_npu_block
    assert block.num_dataflow_units == 40
    assert block.lanes_per_unit == 1
    assert block.max_concurrent_models == 1
    # Lower than Hailo-8 (0.85) -- transformer workloads vary more
    assert block.min_occupancy == pytest.approx(0.75, rel=0.05)


def test_hailo10h_structure_driven_dataflow_fabric(hailo10h_npu_block):
    """Same per-unit ops/clock as Hailo-8: 500 INT8, 1000 INT4."""
    block = hailo10h_npu_block
    assert len(block.compute_fabrics) == 1
    fabric = block.compute_fabrics[0]
    assert fabric.dataflow_kind == NPUDataflowKind.STRUCTURE_DRIVEN
    assert fabric.ops_per_unit_per_clock["int8"] == 500
    assert fabric.ops_per_unit_per_clock["int4"] == 1000
    # Same N16 energy as Hailo-8
    assert fabric.energy_per_op_int8_pj == pytest.approx(0.34, rel=0.05)


def test_hailo10h_no_fp_precisions(hailo10h_npu_block):
    """INT8 / INT4 only -- no FP."""
    block = hailo10h_npu_block
    assert set(block.multi_precision_alu) == {"int8", "int4"}
    fabric_precisions = set(block.compute_fabrics[0].ops_per_unit_per_clock)
    assert fabric_precisions == {"int8", "int4"}


# ---------------------------------------------------------------------------
# Memory: this is the first NPU SKU with external DRAM
# ---------------------------------------------------------------------------

def test_hailo10h_memory_has_lpddr4x(hailo10h_npu_block):
    """Hailo-10H is the first NPU SKU to populate has_external_dram=True."""
    mem = hailo10h_npu_block.memory
    assert mem.has_external_dram is True
    assert mem.external_dram_type == MemoryType.LPDDR4X
    assert mem.external_dram_size_gb == pytest.approx(8.0, rel=0.01)
    assert mem.external_dram_bandwidth_gbps == pytest.approx(40.0, rel=0.1)


def test_hailo10h_on_chip_sram(hailo10h_npu_block):
    """40 * 512 KiB private + 12 MiB shared = 32 MiB total."""
    mem = hailo10h_npu_block.memory
    assert mem.sram_kib_per_unit == 512
    assert mem.shared_sram_kib == 12 * 1024   # 12 MiB shared L2 (KV-cache-sized)
    assert mem.shared_sram_layout == NPUSramLayout.SHARED
    assert mem.coherence_protocol == "none"


# ---------------------------------------------------------------------------
# KVCacheSpec: first SKU to populate the v4+ extension
# ---------------------------------------------------------------------------

def test_hailo10h_kv_cache_populated(hailo10h_npu_block):
    """Hailo-10H is the first SKU to populate kv_cache (the
    transformer-capable extension landed in embodied-schemas#30)."""
    kv = hailo10h_npu_block.kv_cache
    assert kv is not None
    assert kv.max_context_length == 8192
    assert kv.num_layers_supported == 32
    assert kv.streaming_strategy == KVCacheStreamingKind.RING_BUFFER
    assert kv.has_offload_to_dram is True


def test_hailo10h_kv_cache_asymmetric_quantization(hailo10h_npu_block):
    """Hailo-10H pattern: K=INT8, V=INT4 (asymmetric, lossy V is OK)."""
    assert hailo10h_npu_block.kv_cache.quantization == {"k": "int8", "v": "int4"}


def test_hailo10h_kv_cache_dram_offload_consistent_with_memory(hailo10h_npu_block):
    """Cross-field invariant from NPUBlock model_validator: when
    kv_cache.has_offload_to_dram=True, memory.has_external_dram=True.
    Loading the SKU at all means the invariant held; this test pins it."""
    block = hailo10h_npu_block
    assert block.kv_cache.has_offload_to_dram is True
    assert block.memory.has_external_dram is True


# ---------------------------------------------------------------------------
# NoC: 8x5 mesh for 40 units (vs Hailo-8's 8x4 for 32)
# ---------------------------------------------------------------------------

def test_hailo10h_noc_is_mesh_2d_8x5_low_confidence(hailo10h_npu_block):
    noc = hailo10h_npu_block.noc
    assert noc.topology == NPUNoCTopology.MESH_2D
    assert noc.unit_count == 40
    assert noc.mesh_rows == 8
    assert noc.mesh_cols == 5
    assert noc.confidence == DataConfidence.THEORETICAL


# ---------------------------------------------------------------------------
# Power / performance / market
# ---------------------------------------------------------------------------

def test_hailo10h_single_thermal_profile(hailo10h):
    """Same 2.5W envelope as Hailo-8 (more units balanced by lower clock)."""
    power = hailo10h.power
    assert power.tdp_watts == 2.5
    assert power.default_thermal_profile == "2.5W"
    profile_names = {p.name for p in power.thermal_profiles}
    assert profile_names == {"2.5W"}
    assert power.thermal_profiles[0].cooling_solution_id == "passive_heatsink_small"
    # Clock is 1.0 GHz (vs Hailo-8's 1.6 GHz)
    assert power.thermal_profiles[0].clock_mhz == pytest.approx(1000.0)


def test_hailo10h_chip_level_performance_matches_marketing(hailo10h):
    """40 TOPS INT4 (primary GenAI), 20 TOPS INT8 (CV). No FP."""
    perf = hailo10h.performance
    assert perf.int4_tops == 40.0
    assert perf.int8_tops == 20.0
    assert perf.fp32_tflops == 0.0
    assert perf.bf16_tflops == 0.0


def test_hailo10h_round_trips_through_serialize(hailo10h):
    """Full ComputeProduct -> JSON -> ComputeProduct round-trip preserves
    NPUBlock discriminator dispatch AND the KVCacheSpec sub-model."""
    payload = hailo10h.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    rebuilt_block = next(
        b for d in rebuilt.dies
        if d.die_role.value == "compute"
        for b in d.blocks
        if isinstance(b, NPUBlock)
    )
    assert rebuilt_block.num_dataflow_units == 40
    assert rebuilt_block.memory.has_external_dram is True
    assert rebuilt_block.kv_cache is not None
    assert rebuilt_block.kv_cache.streaming_strategy == KVCacheStreamingKind.RING_BUFFER
    assert rebuilt_block.kv_cache.quantization == {"k": "int8", "v": "int4"}
