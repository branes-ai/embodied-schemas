"""Tests for the Intel Xeon 6 6980P (Granite Rapids-AP) ComputeProduct
YAML (sprint #68 PR 3, closes sprint).

Fourth Intel SKU and FIRST Intel 3 SKU in the catalog. Granite Rapids
128-core flagship. Closes the 3-SKU Intel Xeon batch.

Tests focus on the Granite Rapids deltas vs the prior 2 Xeons (8490H
Sapphire Rapids and 8592+ Emerald Rapids), the new Intel 3 process
node, and the new AMX_FP16 path.
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
    cp = all_products.get("intel_xeon_6980p")
    if cp is None:
        pytest.fail("intel_xeon_6980p missing from catalog")
    return cp


@pytest.fixture(scope="module")
def xeon_compute_die(xeon):
    die = next((d for d in xeon.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("6980P has no compute die")
    return die


@pytest.fixture(scope="module")
def xeon_cpu_block(xeon_compute_die) -> CPUBlock:
    block = next(
        (b for b in xeon_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("6980P compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process node: intel_3 is NEW in this PR
# ---------------------------------------------------------------------------

def test_intel_3_process_node_added():
    """Intel 3 is the second Intel process node in the catalog
    (after intel_7 added with the i7-12700k in #24)."""
    nodes = load_process_nodes()
    assert "intel_3" in nodes
    n = nodes["intel_3"]
    assert n.foundry.value == "intel"
    assert n.node_nm == 5
    # +18% PPW vs Intel 4 / better than TSMC N5 logic density
    assert n.densities["hp_logic"].mtx_per_mm2 == 135.0
    # SRAM modestly better than Intel 7 (~0.024 vs 0.0312 um^2 cell)
    assert n.densities["sram_hd"].mtx_per_mm2 == 320.0
    # Lower nominal Vdd than Intel 7 (0.75V)
    assert n.nominal_vdd_v == 0.70
    # Still FinFET (Intel's last before 18A GAA)
    assert n.transistor_topology.value == "finfet"


# ---------------------------------------------------------------------------
# Identity + packaging
# ---------------------------------------------------------------------------

def test_identity(xeon):
    assert xeon.id == "intel_xeon_6980p"
    assert xeon.vendor == "intel"
    assert xeon.lifecycle == LifecycleStatus.PRODUCTION


def test_packaging_is_4_tile_mcm(xeon):
    """Granite Rapids-AP returns to 4-tile MCM (after Emerald Rapids'
    2-tile consolidation). New LGA-4710 socket -- not compatible with
    SPR/EMR's LGA-4677."""
    assert xeon.packaging.kind == PackagingKind.CHIPLET
    assert xeon.packaging.num_dies == 4
    assert xeon.packaging.package_type == "lga4710"   # NEW socket vs SPR/EMR


def test_die_geometry_on_intel_3(xeon_compute_die):
    """First Intel 3 SKU. 4 * 580 mm^2 tile estimate = 2320 mm^2."""
    assert xeon_compute_die.die_size_mm2 == 2320.0
    assert xeon_compute_die.transistors_billion == 80.0
    assert xeon_compute_die.process_node_id == "intel_3"   # NEW process node


# ---------------------------------------------------------------------------
# CPUBlock shape
# ---------------------------------------------------------------------------

def test_cpu_block_128_cores(xeon_cpu_block):
    """128 cores -> 256 threads with SMT=2 (first 256-thread Xeon
    in the catalog)."""
    assert xeon_cpu_block.total_effective_cores == 128
    assert xeon_cpu_block.max_concurrent_threads == 256
    assert xeon_cpu_block.simd_width_lanes == 16


def test_cpu_block_homogeneous_cluster(xeon_cpu_block):
    assert len(xeon_cpu_block.core_clusters) == 1
    cluster = xeon_cpu_block.core_clusters[0]
    assert cluster.cluster_kind == CoreClusterKind.HOMOGENEOUS
    assert cluster.num_cores == 128
    assert cluster.smt_threads == 2
    # Cache hierarchy unchanged from Emerald Rapids
    assert cluster.l1_kib_per_core == 48
    assert cluster.l2_kib_per_core == 2048
    assert cluster.l2_layout == L2Layout.PRIVATE_PER_CORE


# ---------------------------------------------------------------------------
# Multi-fabric: AVX-512 + AMX (Granite Rapids adds AMX_FP16)
# ---------------------------------------------------------------------------

def test_avx512_and_amx_fabrics_present(xeon_cpu_block):
    """Same dual-fabric layout as 8490H/8592+."""
    cluster = xeon_cpu_block.core_clusters[0]
    assert len(cluster.compute_fabrics) == 2
    fabric_isas = {f.isa_extension for f in cluster.compute_fabrics}
    assert fabric_isas == {
        CPUISAExtension.AVX512_BF16,
        CPUISAExtension.AMX_BF16,
    }


def test_amx_adds_fp16_path(xeon_cpu_block):
    """Granite Rapids adds AMX_FP16 mode -- the schema's CPUISAExtension.
    AMX_FP16 enum value (declared in v3) was unused until this YAML.
    The new FP16 path surfaces through the existing AMX fabric's
    ops_per_core_per_clock dict."""
    cluster = xeon_cpu_block.core_clusters[0]
    amx = next(
        f for f in cluster.compute_fabrics
        if f.isa_extension == CPUISAExtension.AMX_BF16
    )
    # NEW: Granite Rapids AMX accepts FP16 inputs natively
    assert amx.ops_per_core_per_clock["fp16"] == 2048
    # The pre-existing BF16 and INT8 paths are unchanged
    assert amx.ops_per_core_per_clock["bf16"] == 2048
    assert amx.ops_per_core_per_clock["int8"] == 4096


def test_amx_fp16_not_on_prior_xeons(all_products):
    """Cross-SKU invariant: AMX_FP16 ops are NOT present on the
    Sapphire Rapids / Emerald Rapids AMX fabrics."""
    for sku in ("intel_xeon_platinum_8490h", "intel_xeon_platinum_8592plus"):
        cp = all_products.get(sku)
        if cp is None:
            continue
        amx = next(
            (f for f in cp.dies[0].blocks[0].core_clusters[0].compute_fabrics
             if f.isa_extension == CPUISAExtension.AMX_BF16),
            None,
        )
        if amx is None:
            continue
        assert "fp16" not in amx.ops_per_core_per_clock, (
            f"{sku} unexpectedly has AMX_FP16 -- that's Granite Rapids' addition"
        )


def test_intel_3_lower_energy_than_intel_7(xeon_cpu_block, all_products):
    """Intel 3 +18% PPW vs Intel 4 translates to a tighter per-FMA
    energy at the same fabric width. AVX-512: 1.25 pJ on Intel 3 vs
    1.5 pJ on Intel 7. AMX: 0.65 pJ vs 0.75 pJ."""
    gnr_avx = next(
        f for f in xeon_cpu_block.core_clusters[0].compute_fabrics
        if f.isa_extension == CPUISAExtension.AVX512_BF16
    )
    assert gnr_avx.energy_per_flop_fp32_pj == pytest.approx(1.25)
    # Cross-SKU: compare to 8592+ on Intel 7
    emr = all_products.get("intel_xeon_platinum_8592plus")
    if emr is None:
        return
    emr_avx = next(
        f for f in emr.dies[0].blocks[0].core_clusters[0].compute_fabrics
        if f.isa_extension == CPUISAExtension.AVX512_BF16
    )
    assert gnr_avx.energy_per_flop_fp32_pj < emr_avx.energy_per_flop_fp32_pj


# ---------------------------------------------------------------------------
# Memory subsystem: 12-channel DDR5-6400 (return to 12-ch vs SPR/EMR's 8)
# ---------------------------------------------------------------------------

def test_memory_12_channel_ddr5_6400(xeon_cpu_block):
    """Granite Rapids-AP returns to 12-channel layout (SPR/EMR were
    8-channel). DDR5-6400 -> 614.4 GB/s peak."""
    mem = xeon_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 12
    assert mem.memory_bus_bits == 768
    assert mem.memory_bandwidth_gbps == pytest.approx(614.4)


def test_memory_channels_match_amd_epyc(all_products):
    """Cross-vendor invariant: Granite Rapids-AP's 12-channel DDR5
    matches AMD EPYC's 12-channel layout. Intel finally caught up
    on memory channel count."""
    gnr = all_products.get("intel_xeon_6980p")
    epyc = all_products.get("amd_epyc_9654_sp5")
    if gnr is None or epyc is None:
        pytest.skip("both needed for cross-vendor invariant")
    gnr_ch = gnr.dies[0].blocks[0].memory.memory_controllers
    epyc_ch = epyc.dies[0].blocks[0].memory.memory_controllers
    assert gnr_ch == epyc_ch == 12


def test_l3_504_mb(xeon_cpu_block):
    """Intel-published L3 for the 6980P: 504 MiB total."""
    assert xeon_cpu_block.memory.l3_total_kib == 516096   # 504 MiB


# ---------------------------------------------------------------------------
# Power + market
# ---------------------------------------------------------------------------

def test_500w_tdp(xeon):
    """6980P top bin: 500W (highest Xeon TDP in the catalog)."""
    assert xeon.power.tdp_watts == 500.0
    assert xeon.power.max_power_watts == 550.0
    prof = xeon.power.thermal_profiles[0]
    assert prof.clock_mhz == 3200.0


def test_market(xeon):
    """Xeon 6 brand launched September 2024."""
    assert xeon.market.launch_date == "2024-09-25"
    assert xeon.market.product_family == "Xeon 6"      # brand rename
    assert xeon.market.launch_msrp_usd == 17800.0


# ---------------------------------------------------------------------------
# Performance roll-up: 2x Emerald Rapids at peak
# ---------------------------------------------------------------------------

def test_perf_2x_emerald_rapids(xeon, all_products):
    """128 cores * 3.2 GHz / (64 cores * 3.2 GHz) = exactly 2x.
    Granite Rapids' core count doubling at the same clock and same
    AMX path gives a clean 2x chip-level peak vs Emerald."""
    emr = all_products.get("intel_xeon_platinum_8592plus")
    if emr is None:
        pytest.skip()
    ratio = xeon.performance.bf16_tflops / emr.performance.bf16_tflops
    assert 1.95 < ratio < 2.05, f"unexpected GNR/EMR BF16 ratio: {ratio:.3f}"


def test_performance_rollup(xeon):
    """128 cores * 32 ops/cycle * 3.2 GHz = 13.11 TFLOPS FP32."""
    assert xeon.performance.fp32_tflops == pytest.approx(13.11)
    assert xeon.performance.bf16_tflops == pytest.approx(838.9)
    assert xeon.performance.int8_tops == pytest.approx(1677.7)


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums(xeon_compute_die):
    """80000 Mtx in gnr_compute_tiles = 80 B tx total."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in xeon_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = xeon_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)
