"""Tests for the Intel Xeon Platinum 8592+ (Emerald Rapids) ComputeProduct YAML
(sprint #68 PR 2).

Third Intel SKU and second datacenter Xeon. Same Intel 7 process as
Sapphire Rapids 8490H (#69), refined Emerald Rapids microarch with
substantially larger L3 LLC and faster memory.

Tests focus on the deltas from 8490H (tile count, L3 size, memory
speed, core count) and the same-architecture invariants (AVX-512 +
AMX fabrics, energy coefficients, Intel 7 node).
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    CoreClusterKind,
    CPUBlock,
    CPUISAExtension,
    CPUNoCTopology,
    L2Layout,
    LifecycleStatus,
    PackagingKind,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products, load_process_nodes


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def xeon(all_products) -> ComputeProduct:
    cp = all_products.get("intel_xeon_platinum_8592plus")
    if cp is None:
        pytest.fail("intel_xeon_platinum_8592plus missing from catalog")
    return cp


@pytest.fixture(scope="module")
def xeon_compute_die(xeon):
    die = next((d for d in xeon.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Xeon 8592+ has no compute die")
    return die


@pytest.fixture(scope="module")
def xeon_cpu_block(xeon_compute_die) -> CPUBlock:
    block = next(
        (b for b in xeon_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("Xeon 8592+ compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Identity + packaging
# ---------------------------------------------------------------------------

def test_identity(xeon):
    assert xeon.id == "intel_xeon_platinum_8592plus"
    assert xeon.vendor == "intel"
    assert xeon.lifecycle == LifecycleStatus.PRODUCTION


def test_packaging_is_2_tile_xcc(xeon):
    """Emerald Rapids XCC = 2 tiles (vs SPR's 4-tile XCC)."""
    assert xeon.packaging.kind == PackagingKind.CHIPLET
    assert xeon.packaging.num_dies == 2     # Emerald consolidated SPR's 4 tiles
    assert xeon.packaging.package_type == "lga4677"


def test_die_geometry(xeon_compute_die):
    """2 tiles * ~763 mm^2 = 1526 mm^2; 2 * 31 B tx = 62 B."""
    assert xeon_compute_die.die_size_mm2 == 1526.0
    assert xeon_compute_die.transistors_billion == 62.0
    assert xeon_compute_die.process_node_id == "intel_7"   # same node as 8490H


# ---------------------------------------------------------------------------
# 2-tile vs 4-tile cross-SKU comparison with 8490H
# ---------------------------------------------------------------------------

def test_emerald_consolidated_to_fewer_tiles_than_sapphire(all_products):
    """Emerald Rapids reduced tile count from SPR's 4 -> 2 while
    growing per-tile size. Schema-level invariant cross-SKU check."""
    emr = all_products.get("intel_xeon_platinum_8592plus")
    spr = all_products.get("intel_xeon_platinum_8490h")
    if emr is None or spr is None:
        pytest.skip("both Xeons needed for cross-SKU check")
    # Fewer tiles
    assert emr.packaging.num_dies < spr.packaging.num_dies
    # More cores
    assert emr.dies[0].blocks[0].total_effective_cores > spr.dies[0].blocks[0].total_effective_cores
    # Total package silicon slightly smaller (fewer tiles, but each
    # tile is larger -- the net is a slight reduction)
    assert emr.dies[0].die_size_mm2 < spr.dies[0].die_size_mm2


# ---------------------------------------------------------------------------
# CPUBlock shape
# ---------------------------------------------------------------------------

def test_cpu_block_64_cores(xeon_cpu_block):
    """64 cores -> 128 threads with SMT=2."""
    assert xeon_cpu_block.total_effective_cores == 64
    assert xeon_cpu_block.max_concurrent_threads == 128
    assert xeon_cpu_block.simd_width_lanes == 16


def test_cpu_block_homogeneous_cluster(xeon_cpu_block):
    assert len(xeon_cpu_block.core_clusters) == 1
    cluster = xeon_cpu_block.core_clusters[0]
    assert cluster.cluster_kind == CoreClusterKind.HOMOGENEOUS
    assert cluster.num_cores == 64
    assert cluster.smt_threads == 2
    # Same L1D / L2 per core as 8490H (same core microarch)
    assert cluster.l1_kib_per_core == 48
    assert cluster.l2_kib_per_core == 2048
    assert cluster.l2_layout == L2Layout.PRIVATE_PER_CORE


# ---------------------------------------------------------------------------
# Multi-fabric: AVX-512 + AMX (same as 8490H)
# ---------------------------------------------------------------------------

def test_avx512_and_amx_fabrics_present(xeon_cpu_block):
    """Same dual-fabric layout as 8490H (Emerald Rapids inherits SPR's
    AMX silicon path unchanged)."""
    cluster = xeon_cpu_block.core_clusters[0]
    assert len(cluster.compute_fabrics) == 2
    fabric_isas = {f.isa_extension for f in cluster.compute_fabrics}
    assert fabric_isas == {
        CPUISAExtension.AVX512_BF16,
        CPUISAExtension.AMX_BF16,
    }


def test_amx_ops_unchanged_from_sapphire(xeon_cpu_block, all_products):
    """AMX silicon path is identical between SPR and EMR -- same 2
    tiles per core, same 16x16 BF16 / INT8 matmul. The Emerald Rapids
    speedup comes from clock and L3, not AMX path itself."""
    cluster = xeon_cpu_block.core_clusters[0]
    amx = next(
        f for f in cluster.compute_fabrics
        if f.isa_extension == CPUISAExtension.AMX_BF16
    )
    assert amx.ops_per_core_per_clock["bf16"] == 2048   # 2 tiles * 1024
    assert amx.ops_per_core_per_clock["int8"] == 4096

    # Cross-SKU same-path check
    spr = all_products.get("intel_xeon_platinum_8490h")
    if spr is not None:
        spr_amx = next(
            f for f in spr.dies[0].blocks[0].core_clusters[0].compute_fabrics
            if f.isa_extension == CPUISAExtension.AMX_BF16
        )
        assert amx.ops_per_core_per_clock == spr_amx.ops_per_core_per_clock


# ---------------------------------------------------------------------------
# Emerald Rapids deltas: bigger L3, faster memory
# ---------------------------------------------------------------------------

def test_l3_is_5mb_per_core(xeon_cpu_block):
    """The defining Emerald Rapids feature: 5 MB L3 per core, vs
    Sapphire Rapids' 1.875 MB. Total 320 MB."""
    assert xeon_cpu_block.memory.l3_total_kib == 327680   # 320 MiB


def test_l3_2_85x_larger_than_sapphire(all_products):
    """Emerald's 320 MB L3 is ~2.85x Sapphire's 112.5 MB. The defining
    inter-generational uplift."""
    emr = all_products.get("intel_xeon_platinum_8592plus")
    spr = all_products.get("intel_xeon_platinum_8490h")
    if emr is None or spr is None:
        pytest.skip("both Xeons needed")
    emr_l3 = emr.dies[0].blocks[0].memory.l3_total_kib
    spr_l3 = spr.dies[0].blocks[0].memory.l3_total_kib
    ratio = emr_l3 / spr_l3
    assert 2.7 < ratio < 3.0, f"unexpected EMR/SPR L3 ratio: {ratio:.2f}"


def test_ddr5_5600(xeon_cpu_block):
    """Emerald Rapids supports DDR5-5600 (vs SPR's DDR5-4800)."""
    mem = xeon_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 8
    assert mem.memory_bandwidth_gbps == pytest.approx(358.4)   # 8 * 5600 * 8


def test_memory_bandwidth_faster_than_sapphire(all_products):
    """Emerald's DDR5-5600 should be +17% over Sapphire's DDR5-4800."""
    emr = all_products.get("intel_xeon_platinum_8592plus")
    spr = all_products.get("intel_xeon_platinum_8490h")
    if emr is None or spr is None:
        pytest.skip("both Xeons needed")
    emr_bw = emr.dies[0].blocks[0].memory.memory_bandwidth_gbps
    spr_bw = spr.dies[0].blocks[0].memory.memory_bandwidth_gbps
    assert emr_bw > spr_bw
    ratio = emr_bw / spr_bw
    assert 1.15 < ratio < 1.20


# ---------------------------------------------------------------------------
# Power + market
# ---------------------------------------------------------------------------

def test_same_tdp_as_sapphire(xeon, all_products):
    """Both 8490H and 8592+ at 350W. Power envelope is unchanged --
    Intel held the line on TDP while expanding cores + L3."""
    assert xeon.power.tdp_watts == 350.0
    spr = all_products.get("intel_xeon_platinum_8490h")
    if spr is not None:
        assert xeon.power.tdp_watts == spr.power.tdp_watts


def test_higher_all_core_clock_than_sapphire(xeon, all_products):
    """Emerald's process refinement gives +10% all-core clock at the
    same Intel 7 node and 350W TDP."""
    spr = all_products.get("intel_xeon_platinum_8490h")
    if spr is None:
        pytest.skip()
    emr_clock = xeon.power.thermal_profiles[0].clock_mhz
    spr_clock = spr.power.thermal_profiles[0].clock_mhz
    assert emr_clock > spr_clock
    assert emr_clock == 3200.0


def test_market(xeon):
    """Emerald Rapids launched December 2023; cheaper than SPR flagship."""
    assert xeon.market.launch_date == "2023-12-14"
    assert xeon.market.launch_msrp_usd == 11600.0


# ---------------------------------------------------------------------------
# Performance roll-up
# ---------------------------------------------------------------------------

def test_amx_bf16_uplift_over_sapphire(xeon, all_products):
    """64 cores * 3.2 GHz / (60 cores * 2.9 GHz) = 1.177, so Emerald's
    AMX BF16 peak should be ~17.7% higher than Sapphire's at the same
    AMX path."""
    spr = all_products.get("intel_xeon_platinum_8490h")
    if spr is None:
        pytest.skip()
    ratio = xeon.performance.bf16_tflops / spr.performance.bf16_tflops
    assert 1.15 < ratio < 1.20, f"unexpected EMR/SPR BF16 ratio: {ratio:.3f}"


def test_performance_rollup(xeon):
    assert xeon.performance.fp32_tflops == pytest.approx(6.55)
    assert xeon.performance.bf16_tflops == pytest.approx(419.4)
    assert xeon.performance.int8_tops == pytest.approx(838.9)


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums(xeon_compute_die):
    """62000 Mtx in emr_compute_tiles = 62 B tx total."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in xeon_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = xeon_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)
