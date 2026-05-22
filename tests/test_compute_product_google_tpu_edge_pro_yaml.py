"""Tests for the Google TPU Edge Pro ComputeProduct YAML (sprint #72 PR 6).

CLOSES sprint #72. The HYPOTHETICAL 30W edge TPU joins Plasticine v2
as the catalog's second engineering_sample SKU.

Tests focus on the hypothetical-SKU invariants (lifecycle, model_tier,
is_available), the unique single-MXU layout, LPDDR5 memory (vs other
TPUs' HBM), and the DVFS profile set (first DVFS-capable TPU in the
catalog).
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    LifecycleStatus,
    PackagingKind,
    TPUBlock,
    TPUFabricKind,
    TPUNoCTopology,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def tpu(all_products) -> ComputeProduct:
    cp = all_products.get("google_tpu_edge_pro")
    if cp is None:
        pytest.fail("google_tpu_edge_pro missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tpu_die(tpu):
    die = next((d for d in tpu.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("TPU Edge Pro has no compute die")
    return die


@pytest.fixture(scope="module")
def tpu_block(tpu_die) -> TPUBlock:
    block = next((b for b in tpu_die.blocks if isinstance(b, TPUBlock)), None)
    if block is None:
        pytest.fail("TPU Edge Pro die has no TPUBlock")
    return block


# ---------------------------------------------------------------------------
# Hypothetical-SKU invariants
# ---------------------------------------------------------------------------

def test_lifecycle_is_engineering_sample(tpu):
    """Edge Pro is HYPOTHETICAL -- no physical silicon exists. Uses
    engineering_sample lifecycle (Plasticine v2 precedent for
    research/aspirational SKUs)."""
    assert tpu.lifecycle == LifecycleStatus.ENGINEERING_SAMPLE


def test_model_tier_is_research(tpu):
    """Same model_tier as Plasticine v2: 'research' (the only
    catalog-approved tier for hypothetical / research SKUs)."""
    assert tpu.market.model_tier == "research"


def test_not_commercially_available(tpu):
    """is_available must be False -- Edge Pro is hypothetical."""
    assert tpu.market.is_available is False


def test_die_on_tsmc_n7(tpu_die):
    """Sibling to TPU v4 on the same N7 node, different scale
    (Edge Pro: ~150 mm^2 vs v4's 600 mm^2)."""
    assert tpu_die.process_node_id == "tsmc_n7"
    assert tpu_die.die_size_mm2 == 150.0
    assert tpu_die.transistors_billion == 10.0


# ---------------------------------------------------------------------------
# Single MXU layout (vs v3/v4/v5p's 2-MXU per-die convention)
# ---------------------------------------------------------------------------

def test_single_mxu_layout(tpu_block):
    """Edge Pro uses a single 128x128 systolic array, not the
    2-MXU per-die convention used by v3/v4/v5p. The graphs factory
    is explicit: 'Single large systolic array' (16,384 PEs)."""
    assert tpu_block.num_mxus == 1
    assert tpu_block.mxu_dim_rows == 128
    assert tpu_block.mxu_dim_cols == 128


# ---------------------------------------------------------------------------
# LPDDR5 memory (first non-HBM TPU since v1's DDR3)
# ---------------------------------------------------------------------------

def test_lpddr5_memory(tpu_block):
    """Edge Pro uses LPDDR5 (edge memory subsystem) at 128 GB/s,
    vs v3's HBM2 / v4's HBM2e / v5p's HBM3. v1 was the only other
    non-HBM TPU (DDR3)."""
    mem = tpu_block.memory
    assert mem.external_dram_type == MemoryType.LPDDR5
    assert mem.external_dram_size_gb == 32.0
    assert mem.external_dram_bandwidth_gbps == 128.0


def test_unified_buffer_2_mib(tpu_block):
    """Smaller UB than the datacenter TPUs: 2 MiB (vs v3: 16, v4:
    32, v5p: 48). Edge-class memory hierarchy."""
    assert tpu_block.memory.unified_buffer_size_kib == 2048


# ---------------------------------------------------------------------------
# DVFS profile set (first DVFS-capable TPU in the catalog)
# ---------------------------------------------------------------------------

def test_three_dvfs_profiles(tpu):
    """First TPU in the catalog with multiple thermal profiles.
    Datacenter TPUs (v1/v3/v4/v5p) are fixed-frequency designs;
    Edge Pro adds 15W / 30W / 45W DVFS knobs."""
    profile_names = sorted(p.name for p in tpu.power.thermal_profiles)
    assert profile_names == ["15W", "30W", "45W"]


def test_default_profile_is_30w(tpu):
    """30W is the default operating point (the 'fair comparison
    against KPU T256 / Jetson AGX' design point)."""
    assert tpu.power.default_thermal_profile == "30W"


def test_dvfs_clock_range(tpu):
    """Clock scales 500 -> 850 -> 900 MHz across the 15/30/45W
    profiles."""
    profiles = {p.name: p for p in tpu.power.thermal_profiles}
    assert profiles["15W"].clock_mhz == 500.0
    assert profiles["30W"].clock_mhz == 850.0
    assert profiles["45W"].clock_mhz == 900.0


def test_dvfs_vdd_range(tpu):
    """Vdd scales 0.65 -> 0.75 -> 0.85 V across the DVFS profiles."""
    profiles = {p.name: p for p in tpu.power.thermal_profiles}
    assert profiles["15W"].vdd_v == 0.65
    assert profiles["30W"].vdd_v == 0.75
    assert profiles["45W"].vdd_v == 0.85


# ---------------------------------------------------------------------------
# Precision support: BF16 + INT8 + FP32 (no FP8 / INT4)
# ---------------------------------------------------------------------------

def test_no_fp8_no_int4(tpu_block):
    """Edge Pro lacks v5p's FP8 + INT4 paths. v4-era ISA."""
    assert "fp8" not in tpu_block.multi_precision_alu
    assert "int4" not in tpu_block.multi_precision_alu


def test_supports_bf16_int8_fp32(tpu_block):
    """Edge Pro adds FP32 + BF16 over Coral (which was INT8 only)."""
    assert set(tpu_block.multi_precision_alu) == {"bf16", "int8", "fp32"}


# ---------------------------------------------------------------------------
# Cross-SKU invariants
# ---------------------------------------------------------------------------

def test_only_other_engineering_sample_is_plasticine(all_products):
    """Engineering-sample SKUs in the catalog: TPU Edge Pro
    (hypothetical TPU) + Stanford Plasticine v2 (academic CGRA)."""
    eng_samples = [
        cp.id for cp in all_products.values()
        if cp.lifecycle == LifecycleStatus.ENGINEERING_SAMPLE
    ]
    assert set(eng_samples) == {"google_tpu_edge_pro", "stanford_plasticine_v2"}


def test_edge_pro_smaller_than_v4(all_products):
    """Edge Pro is a 30W edge SKU; v4 is a 350W datacenter SKU.
    Die area should differ by >3x."""
    edge = all_products.get("google_tpu_edge_pro")
    v4 = all_products.get("google_tpu_v4")
    if edge is None or v4 is None:
        pytest.skip()
    assert edge.dies[0].die_size_mm2 < v4.dies[0].die_size_mm2 / 3


def test_edge_pro_first_non_hbm_tpu_since_v1(all_products):
    """v3, v4, v5p all use HBM. v1 was DDR3. Edge Pro is the only
    other non-HBM TPU."""
    edge_mem = all_products["google_tpu_edge_pro"].dies[0].blocks[0].memory.external_dram_type
    for sku in ("google_tpu_v3", "google_tpu_v4", "google_tpu_v5p"):
        cp = all_products.get(sku)
        if cp is None:
            continue
        mem_type = cp.dies[0].blocks[0].memory.external_dram_type
        assert mem_type != edge_mem, (
            f"{sku} unexpectedly shares memory type with Edge Pro ({edge_mem.value})"
        )


# ---------------------------------------------------------------------------
# Performance (derivable from the systolic array config)
# ---------------------------------------------------------------------------

def test_performance_rollup(tpu):
    """16,384 PEs * ops/cycle * 850 MHz.
    INT8: 2 MACs/PE/cycle * 2 ops/MAC = 4 ops/PE/cycle -> 55.7 TOPS
    BF16: 1 MAC/PE/cycle * 2 ops/MAC = 2 ops/PE/cycle -> 27.85 TFLOPS
    FP32: 0.5 MAC/PE/cycle * 2 ops/MAC = 1 op/PE/cycle -> 13.93 TFLOPS"""
    assert tpu.performance.int8_tops == pytest.approx(55.7)
    assert tpu.performance.bf16_tflops == pytest.approx(27.85)
    assert tpu.performance.fp32_tflops == pytest.approx(13.93)


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums_within_5_percent(tpu_die):
    total_mtx = sum(
        b.transistor_source.mtx
        for b in tpu_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = tpu_die.transistors_billion * 1000.0
    rel_err = abs(total_mtx - expected_mtx) / expected_mtx
    assert rel_err < 0.05, (
        f"silicon_bin sum {total_mtx:.0f} differs from declared "
        f"{expected_mtx:.0f} by {rel_err*100:.1f}%"
    )
