"""Tests for the Google TPU v5p ComputeProduct YAML (sprint #72 PR 5).

Joins TPU v1 (#75), v3 (#76), and v4 in the google/ vendor directory.
First FP8 / HBM3 / INT4 / N4P TPU SKU in the catalog.

Tests focus on the new precision support (FP8, INT4), the HBM3
upgrade, and cross-SKU invariants vs v4.
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    LifecycleStatus,
    PackagingKind,
    TPUBlock,
    TPUFabricKind,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def tpu(all_products) -> ComputeProduct:
    cp = all_products.get("google_tpu_v5p")
    if cp is None:
        pytest.fail("google_tpu_v5p missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tpu_die(tpu):
    die = next((d for d in tpu.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("TPU v5p has no compute die")
    return die


@pytest.fixture(scope="module")
def tpu_block(tpu_die) -> TPUBlock:
    block = next((b for b in tpu_die.blocks if isinstance(b, TPUBlock)), None)
    if block is None:
        pytest.fail("TPU v5p die has no TPUBlock")
    return block


# ---------------------------------------------------------------------------
# Identity + packaging
# ---------------------------------------------------------------------------

def test_identity(tpu):
    assert tpu.id == "google_tpu_v5p"
    assert tpu.vendor == "google"
    assert tpu.lifecycle == LifecycleStatus.PRODUCTION


def test_die_geometry_n4p(tpu_die):
    """TSMC N4P (first N4P TPU). Die size + transistor count are
    estimates -- Google hasn't published them."""
    assert tpu_die.process_node_id == "tsmc_n4p"
    assert tpu_die.die_size_mm2 == 720.0
    assert tpu_die.transistors_billion == 70.0


# ---------------------------------------------------------------------------
# New precision support: FP8 + INT4 (first in TPU catalog)
# ---------------------------------------------------------------------------

def test_fp8_supported(tpu_block):
    """v5p added native FP8 MAC support -- the headline v5p
    capability. First FP8 SKU in the TPU catalog."""
    assert "fp8" in tpu_block.multi_precision_alu
    fab = tpu_block.compute_fabrics[0]
    assert fab.ops_per_unit_per_clock.get("fp8") == 2


def test_int4_supported(tpu_block):
    """v5p added INT4 for quantized inference (Google's QAT path).
    First INT4 SKU in the TPU catalog."""
    assert "int4" in tpu_block.multi_precision_alu
    fab = tpu_block.compute_fabrics[0]
    assert fab.ops_per_unit_per_clock.get("int4") == 2


def test_full_precision_set(tpu_block):
    """v5p supports BF16, INT8, FP8, FP32 (emulated), INT4 -- the
    widest precision matrix of any TPU in the catalog."""
    expected = {"bf16", "int8", "fp8", "fp32", "int4"}
    assert set(tpu_block.multi_precision_alu) == expected


# ---------------------------------------------------------------------------
# HBM3 (first in TPU catalog)
# ---------------------------------------------------------------------------

def test_hbm3_first_in_tpu_catalog(tpu_block):
    """v5p upgraded to HBM3. v3 = HBM2, v4 = HBM2e."""
    mem = tpu_block.memory
    assert mem.external_dram_type == MemoryType.HBM3
    # 95 GiB capacity (vs v4's 32 GiB)
    assert mem.external_dram_size_gb == 95.0
    # 2.8 TB/s bandwidth (vs v4's 1.2 TB/s)
    assert mem.external_dram_bandwidth_gbps == 2800.0


def test_unified_buffer_48_mib(tpu_block):
    """48 MiB UB (1.5x v4's 32 MiB)."""
    assert tpu_block.memory.unified_buffer_size_kib == 49152


# ---------------------------------------------------------------------------
# Architecture lineage
# ---------------------------------------------------------------------------

def test_uses_tpu_v2_plus_fabric_kind(tpu_block):
    """v5p inherits the v2/v3/v4 fabric architecture (TPU_V2_PLUS)."""
    fab = tpu_block.compute_fabrics[0]
    assert fab.fabric_kind == TPUFabricKind.TPU_V2_PLUS


def test_6_ici_ports_3d_torus(tpu_block):
    """6 ICI ports for 3D torus (same as v4); higher per-port BW
    enables 8960-chip pod (vs v4's 4096)."""
    assert tpu_block.ici_port_count == 6
    assert tpu_block.ici_topology_hint == "3d_torus_2x2x2"
    # 2x v4's 400 GB/s per port
    assert tpu_block.ici_bandwidth_per_port_gbps == 800.0


# ---------------------------------------------------------------------------
# Cross-SKU invariants vs v4
# ---------------------------------------------------------------------------

def test_v5p_chip_perf_roughly_1_67x_v4(all_products):
    """Google's marketed v5p uplift: 459 TFLOPS BF16 vs v4's 275 =
    ~1.67x at the chip level. Pinned in [1.6, 1.8]."""
    v5p = all_products.get("google_tpu_v5p")
    v4 = all_products.get("google_tpu_v4")
    if v5p is None or v4 is None:
        pytest.skip()
    ratio = v5p.performance.bf16_tflops / v4.performance.bf16_tflops
    assert 1.6 < ratio < 1.8, f"unexpected v5p/v4 BF16 ratio: {ratio:.2f}"


def test_v5p_hbm_capacity_3x_v4(all_products):
    """v5p has 95 GiB HBM3 vs v4's 32 GiB HBM2e -- nearly 3x."""
    v5p = all_products.get("google_tpu_v5p")
    v4 = all_products.get("google_tpu_v4")
    if v5p is None or v4 is None:
        pytest.skip()
    v5p_mem = v5p.dies[0].blocks[0].memory.external_dram_size_gb
    v4_mem = v4.dies[0].blocks[0].memory.external_dram_size_gb
    assert 2.8 < v5p_mem / v4_mem < 3.2


def test_v5p_hbm_bandwidth_233x_v4(all_products):
    """v5p HBM3 bandwidth 2.8 TB/s vs v4 HBM2e 1.2 TB/s = ~2.33x."""
    v5p = all_products.get("google_tpu_v5p")
    v4 = all_products.get("google_tpu_v4")
    if v5p is None or v4 is None:
        pytest.skip()
    v5p_bw = v5p.dies[0].blocks[0].memory.external_dram_bandwidth_gbps
    v4_bw = v4.dies[0].blocks[0].memory.external_dram_bandwidth_gbps
    ratio = v5p_bw / v4_bw
    assert 2.2 < ratio < 2.5


def test_fp8_not_on_v3_or_v4(all_products):
    """Cross-SKU: FP8 path is v5p-only. v3/v4 should not declare FP8
    in multi_precision_alu."""
    for sku in ("google_tpu_v3", "google_tpu_v4"):
        cp = all_products.get(sku)
        if cp is None:
            continue
        assert "fp8" not in cp.dies[0].blocks[0].multi_precision_alu, (
            f"{sku} unexpectedly has fp8 -- v5p's exclusive feature"
        )


def test_int4_not_on_v3_or_v4(all_products):
    """Cross-SKU: INT4 is v5p-only (added for quantized inference)."""
    for sku in ("google_tpu_v3", "google_tpu_v4"):
        cp = all_products.get(sku)
        if cp is None:
            continue
        assert "int4" not in cp.dies[0].blocks[0].multi_precision_alu


# ---------------------------------------------------------------------------
# Performance roll-up
# ---------------------------------------------------------------------------

def test_chip_level_performance(tpu):
    """Google-published v5p chip-level peaks."""
    assert tpu.performance.bf16_tflops == 459.0
    assert tpu.performance.int8_tops == 918.0
    # INT4 = 2x FP8 = 4x BF16 = 1836 TOPS per Google's QAT path
    assert tpu.performance.int4_tops == 1836.0


def test_int8_is_2x_bf16(tpu):
    assert tpu.performance.int8_tops == pytest.approx(2 * tpu.performance.bf16_tflops)


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
