"""Tests for the Google TPU v3 ComputeProduct YAML (sprint #72 PR 4).

Joins TPU v1 (#75) and TPU v4 (#206) in the google/ vendor directory.
First HBM-equipped, liquid-cooled, 2D-torus ICI TPU SKU in the catalog.

Tests focus on the deltas from TPU v1 (HBM vs DDR3, BF16 added, larger
die) and TPU v4 (HBM2 vs HBM2e, smaller die, 2D vs 3D torus, N16 vs
N7, smaller Unified Buffer).
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
    cp = all_products.get("google_tpu_v3")
    if cp is None:
        pytest.fail("google_tpu_v3 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tpu_die(tpu):
    die = next((d for d in tpu.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("TPU v3 has no compute die")
    return die


@pytest.fixture(scope="module")
def tpu_block(tpu_die) -> TPUBlock:
    block = next((b for b in tpu_die.blocks if isinstance(b, TPUBlock)), None)
    if block is None:
        pytest.fail("TPU v3 die has no TPUBlock")
    return block


# ---------------------------------------------------------------------------
# Identity + packaging
# ---------------------------------------------------------------------------

def test_identity(tpu):
    assert tpu.id == "google_tpu_v3"
    assert tpu.vendor == "google"
    # v3 still available on Google Cloud (v3-8 / v3-32 / v3-1024)
    assert tpu.lifecycle == LifecycleStatus.MATURE


def test_monolithic_oam_form_factor(tpu):
    assert tpu.packaging.kind == PackagingKind.MONOLITHIC
    assert tpu.packaging.num_dies == 1
    # Liquid-cooled OAM (vs v1's PCIe card)
    assert tpu.packaging.package_type == "datacenter_oam"


def test_die_geometry_n16(tpu_die):
    """TSMC N16 with ~635 mm^2 / 21 B tx (estimate)."""
    assert tpu_die.die_size_mm2 == 635.0
    assert tpu_die.transistors_billion == 21.0
    assert tpu_die.process_node_id == "tsmc_n16"


# ---------------------------------------------------------------------------
# TPUBlock shape
# ---------------------------------------------------------------------------

def test_2_mxus_128x128_modeled(tpu_block):
    """Per-die model: 2 MXUs at 128x128. The chip-level performance
    roll-up reflects Google's marketed 4-MXU equivalent (2 TCs share
    the die); same modeling gap as the v4 entry."""
    assert tpu_block.num_mxus == 2
    assert tpu_block.mxu_dim_rows == 128
    assert tpu_block.mxu_dim_cols == 128


def test_uses_tpu_v2_plus_fabric_kind(tpu_block):
    """Same fabric kind as v4 (TPU v2/v3/v4 architecture line).
    Differentiated from v1's TPU_V1_STYLE."""
    fab = tpu_block.compute_fabrics[0]
    assert fab.fabric_kind == TPUFabricKind.TPU_V2_PLUS


def test_bf16_and_int8_and_fp32_supported(tpu_block):
    """v3 supports BF16 (added in v2), INT8, FP32 (emulated). The
    full TPU v2/v3/v4 precision matrix; v1 was INT8-only."""
    assert "bf16" in tpu_block.multi_precision_alu
    assert "int8" in tpu_block.multi_precision_alu
    assert "fp32" in tpu_block.multi_precision_alu


def test_hbm2_chip_attached(tpu_block):
    """v3 introduced HBM to the TPU family (32 GiB HBM2)."""
    mem = tpu_block.memory
    assert mem.has_external_dram is True
    assert mem.external_dram_type == MemoryType.HBM2   # NOT HBM2e (that's v4)
    assert mem.external_dram_size_gb == 32.0
    assert mem.external_dram_bandwidth_gbps == 900.0


def test_unified_buffer_16_mib(tpu_block):
    """v3 has 16 MiB UB (smaller than v4's 32 MiB; v3 had less
    generous on-chip SRAM, partly compensated by HBM intro)."""
    assert tpu_block.memory.unified_buffer_size_kib == 16384   # 16 MiB


def test_2d_torus_ici_topology(tpu_block):
    """v3 uses 4 ICI ports for 2D-torus pod (vs v4's 6 ports for 3D
    torus). The defining v3-vs-v4 interconnect difference."""
    assert tpu_block.ici_port_count == 4
    assert tpu_block.ici_topology_hint == "2d_torus"


# ---------------------------------------------------------------------------
# Cross-SKU invariants
# ---------------------------------------------------------------------------

def test_v3_uses_same_fabric_as_v4(all_products):
    """v3 and v4 share TPU_V2_PLUS fabric kind (same architecture
    family). v1 uses TPU_V1_STYLE."""
    v3 = all_products.get("google_tpu_v3")
    v4 = all_products.get("google_tpu_v4")
    if v3 is None or v4 is None:
        pytest.skip()
    v3_kind = v3.dies[0].blocks[0].compute_fabrics[0].fabric_kind
    v4_kind = v4.dies[0].blocks[0].compute_fabrics[0].fabric_kind
    assert v3_kind == v4_kind == TPUFabricKind.TPU_V2_PLUS


def test_v3_smaller_ub_than_v4(all_products):
    """v3 has 16 MiB UB; v4 doubled to 32 MiB."""
    v3 = all_products.get("google_tpu_v3")
    v4 = all_products.get("google_tpu_v4")
    if v3 is None or v4 is None:
        pytest.skip()
    v3_ub = v3.dies[0].blocks[0].memory.unified_buffer_size_kib
    v4_ub = v4.dies[0].blocks[0].memory.unified_buffer_size_kib
    assert v3_ub < v4_ub
    assert v4_ub == v3_ub * 2


def test_v3_fewer_ici_ports_than_v4(all_products):
    """v3: 4 ports (2D torus). v4: 6 ports (3D torus). Schema-level
    invariant of the TPU generation jump."""
    v3 = all_products.get("google_tpu_v3")
    v4 = all_products.get("google_tpu_v4")
    if v3 is None or v4 is None:
        pytest.skip()
    v3_ports = v3.dies[0].blocks[0].ici_port_count
    v4_ports = v4.dies[0].blocks[0].ici_port_count
    assert v3_ports == 4
    assert v4_ports == 6


def test_v3_hbm2_v4_hbm2e(all_products):
    """v3 uses HBM2; v4 upgraded to HBM2e (similar capacity per
    stack but ~33% higher bandwidth)."""
    v3 = all_products.get("google_tpu_v3")
    v4 = all_products.get("google_tpu_v4")
    if v3 is None or v4 is None:
        pytest.skip()
    assert v3.dies[0].blocks[0].memory.external_dram_type == MemoryType.HBM2
    assert v4.dies[0].blocks[0].memory.external_dram_type == MemoryType.HBM2E


# ---------------------------------------------------------------------------
# Performance roll-up
# ---------------------------------------------------------------------------

def test_chip_level_bf16_123_tflops(tpu):
    """Google-published per-chip BF16 throughput: 123 TFLOPS."""
    assert tpu.performance.bf16_tflops == pytest.approx(123.2)


def test_int8_double_bf16(tpu):
    """INT8 runs at 2x BF16 rate on the v2/v3/v4 fabric."""
    assert tpu.performance.int8_tops == pytest.approx(246.4)
    # 2x BF16 rate
    assert tpu.performance.int8_tops == pytest.approx(2 * tpu.performance.bf16_tflops)


def test_no_int4(tpu):
    """INT4 not added until TPU v5p; v3 / v4 report 0."""
    assert tpu.performance.int4_tops == 0.0


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums_within_5_percent(tpu_die):
    """Sum within 5% of declared transistor count."""
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
