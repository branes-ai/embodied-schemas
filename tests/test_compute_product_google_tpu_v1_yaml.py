"""Tests for the Google TPU v1 ComputeProduct YAML (sprint #72 PR 3).

First TPU v1 SKU in the catalog and first SKU using the
TPUFabricKind.TPU_V1_STYLE enum value (single large 256x256 MXU).
Companion process node tsmc_n28hpm is added in this PR.

Same test shape as the existing TPU v4 YAML tests.
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
from embodied_schemas.loaders import load_compute_products, load_process_nodes


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def tpu(all_products) -> ComputeProduct:
    cp = all_products.get("google_tpu_v1")
    if cp is None:
        pytest.fail("google_tpu_v1 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tpu_die(tpu):
    die = next((d for d in tpu.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("TPU v1 has no compute die")
    return die


@pytest.fixture(scope="module")
def tpu_block(tpu_die) -> TPUBlock:
    block = next((b for b in tpu_die.blocks if isinstance(b, TPUBlock)), None)
    if block is None:
        pytest.fail("TPU v1 die has no TPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process node: tsmc_n28hpm added by this PR
# ---------------------------------------------------------------------------

def test_tsmc_n28hpm_process_node_added():
    """tsmc_n28hpm joins gf_28nm in the catalog -- distinct TSMC vs GF
    28nm process characteristics for TPU v1 (TSMC) vs Coral Edge TPU
    (GF). Both are bulk planar HKMG, not FinFET."""
    nodes = load_process_nodes()
    assert "tsmc_n28hpm" in nodes
    n = nodes["tsmc_n28hpm"]
    assert n.foundry.value == "tsmc"
    assert n.node_nm == 28
    # Bulk planar (pre-FinFET) -- key invariant
    assert n.transistor_topology.value == "bulk_planar"
    # 28nm density ~10 Mtx/mm^2 HP (vs N16's ~30 Mtx/mm^2)
    assert n.densities["hp_logic"].mtx_per_mm2 == 10.0


def test_both_28nm_nodes_now_in_catalog():
    """tsmc_n28hpm (NEW) sits alongside gf_28nm. Distinct entries
    because TSMC and GF 28nm have materially different characteristics."""
    nodes = load_process_nodes()
    assert "tsmc_n28hpm" in nodes
    assert "gf_28nm" in nodes


# ---------------------------------------------------------------------------
# Identity + packaging
# ---------------------------------------------------------------------------

def test_identity(tpu):
    assert tpu.id == "google_tpu_v1"
    assert tpu.vendor == "google"
    # TPU v1 is end-of-life (production 2015-2018, replaced by v2/v3/v4)
    assert tpu.lifecycle == LifecycleStatus.EOL


def test_monolithic_pcie_card(tpu):
    """TPU v1 was a standalone PCIe Gen3 x16 add-in card. Single die."""
    assert tpu.packaging.kind == PackagingKind.MONOLITHIC
    assert tpu.packaging.num_dies == 1
    assert tpu.packaging.package_type == "pcie_card"


def test_die_geometry(tpu_die):
    """331 mm^2 (Jouppi et al.: 'less than half of Haswell-EX [662 mm^2]')
    and ~3 B transistors (estimated from 28nm density)."""
    assert tpu_die.die_size_mm2 == 331.0
    assert tpu_die.transistors_billion == 3.0
    assert tpu_die.process_node_id == "tsmc_n28hpm"   # NEW process node


# ---------------------------------------------------------------------------
# TPUBlock shape -- single large 256x256 MXU
# ---------------------------------------------------------------------------

def test_single_large_mxu_canonical_tpu_v1_shape(tpu_block):
    """The defining TPU v1 feature: single 256x256 MAC array
    (65,536 MACs total). v2+ shifted to multiple smaller MXUs."""
    assert tpu_block.num_mxus == 1
    assert tpu_block.mxu_dim_rows == 256
    assert tpu_block.mxu_dim_cols == 256
    # Total MAC count: 1 * 256 * 256 = 65,536
    total_macs = tpu_block.num_mxus * tpu_block.mxu_dim_rows * tpu_block.mxu_dim_cols
    assert total_macs == 65536


def test_tpu_v1_style_fabric_kind_first_use(tpu_block):
    """FIRST USE of TPUFabricKind.TPU_V1_STYLE in the catalog. The enum
    value was declared in v7 schema but unused until this YAML.
    Pre-existing TPU v4 entry uses TPU_V2_PLUS."""
    assert len(tpu_block.compute_fabrics) == 1
    fab = tpu_block.compute_fabrics[0]
    assert fab.fabric_kind == TPUFabricKind.TPU_V1_STYLE


def test_int8_only(tpu_block, tpu):
    """TPU v1 is INT8 only -- no FP support whatsoever. BF16 arrived
    in TPU v2; FP32 emulation arrived in TPU v4."""
    assert tpu_block.multi_precision_alu == ["int8"]
    fab = tpu_block.compute_fabrics[0]
    assert "int8" in fab.ops_per_unit_per_clock
    assert "bf16" not in fab.ops_per_unit_per_clock
    assert "fp32" not in fab.ops_per_unit_per_clock
    # Performance roll-up also reflects INT8-only nature
    assert tpu.performance.bf16_tflops == 0.0
    assert tpu.performance.fp32_tflops == 0.0


def test_memory_ddr3_chip_attached(tpu_block):
    """TPU v1 used 8 GiB chip-attached DDR3-2133 (predates HBM TPU adoption
    in v2+). 24 MiB Unified Buffer on-chip."""
    mem = tpu_block.memory
    # 24 MiB Unified Buffer
    assert mem.unified_buffer_size_kib == 24576
    # External DRAM is DDR3 (not HBM)
    assert mem.has_external_dram is True
    assert mem.external_dram_type == MemoryType.DDR3
    assert mem.external_dram_size_gb == 8.0
    assert mem.external_dram_bandwidth_gbps == pytest.approx(34.1)


def test_no_ici_topology(tpu_block):
    """TPU v1 had no inter-chip interconnect (predates ICI; each chip
    was a standalone PCIe card). The schema requires ICI fields so we
    populate with PCIe-as-ICI nominal values."""
    # Nominal port count (PCIe placeholder); not a real torus
    assert tpu_block.ici_port_count == 1
    assert tpu_block.ici_topology_hint == "standalone_pcie"


def test_noc_single_crossbar(tpu_block):
    """1 MXU means no NoC -- UB -> MXU direct path via crossbar."""
    assert tpu_block.noc.topology == TPUNoCTopology.CROSSBAR
    assert tpu_block.noc.unit_count == 1


# ---------------------------------------------------------------------------
# Performance roll-up
# ---------------------------------------------------------------------------

def test_int8_tops_91_75(tpu):
    """65,536 MACs * 2 ops/MAC * 700 MHz = 91.75 TOPS (Jouppi et al.)."""
    assert tpu.performance.int8_tops == pytest.approx(91.75)


def test_low_power_envelope(tpu):
    """TPU v1 is 75W card-level (28W chip-only per Jouppi et al.).
    Inference-focused at very low TDP -- the design point that
    proved systolic arrays could outperform GPUs at 1/30 the power."""
    assert tpu.power.tdp_watts == 75.0
    # 28W chip-only -- min_power_watts reflects the chip envelope
    assert tpu.power.min_power_watts == 28.0


# ---------------------------------------------------------------------------
# Cross-SKU: TPU v1 vs v4
# ---------------------------------------------------------------------------

def test_v1_uses_different_fabric_kind_than_v4(all_products):
    """v1 uses TPU_V1_STYLE (single 256x256 MXU); v4 uses TPU_V2_PLUS
    (2 smaller 128x128 MXUs)."""
    v1 = all_products.get("google_tpu_v1")
    v4 = all_products.get("google_tpu_v4")
    if v1 is None or v4 is None:
        pytest.skip()
    v1_kind = v1.dies[0].blocks[0].compute_fabrics[0].fabric_kind
    v4_kind = v4.dies[0].blocks[0].compute_fabrics[0].fabric_kind
    assert v1_kind == TPUFabricKind.TPU_V1_STYLE
    assert v4_kind == TPUFabricKind.TPU_V2_PLUS
    assert v1_kind != v4_kind


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums_within_10_percent(tpu_die):
    """Sum within 10% of declared transistor count (rougher tolerance
    than the newer SKUs because TPU v1 transistor count is itself an
    estimate -- Google never published the official figure)."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in tpu_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = tpu_die.transistors_billion * 1000.0
    rel_err = abs(total_mtx - expected_mtx) / expected_mtx
    assert rel_err < 0.10, (
        f"silicon_bin sum {total_mtx:.0f} differs from declared "
        f"{expected_mtx:.0f} by {rel_err*100:.1f}%"
    )
