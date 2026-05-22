"""Tests for the first TPU SKU YAML: google_tpu_v4.

Catalog gain: 12 KPU + 2 GPU + 1 CPU + 3 NPU + 1 CGRA + 1 DPU + 1 TPU
= 21 ComputeProducts. First SKU to exercise:
  - ``BlockKind.TPU`` and the full ``TPUBlock`` schema (landed in #38)
  - ``TPUFabricKind.TPU_V2_PLUS``
  - ``TPUNoCTopology.MULTI_CROSSBAR`` (2 MXUs share the UB)
  - ``TPUTileEnergyCoefficients`` sub-type (all 9 canonical fields)
  - Chip-attached HBM2e (vs DPU's DDR4, CGRA's DDR4 host-bus)
  - ICI single-chip port surface (6 ports for 3D-torus pod)
  - ``MemoryType.HBM2E`` populated

Same shape as ``test_compute_product_v6_xilinx_vitis_ai_yaml.py``.
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
from embodied_schemas.process_node import DataConfidence


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def tpu_v4(all_products) -> ComputeProduct:
    cp = all_products.get("google_tpu_v4")
    if cp is None:
        pytest.fail("google_tpu_v4 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tpu_v4_compute_die(tpu_v4):
    die = next((d for d in tpu_v4.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("TPU v4 has no compute die")
    return die


@pytest.fixture(scope="module")
def tpu_v4_block(tpu_v4_compute_die) -> TPUBlock:
    block = next(
        (b for b in tpu_v4_compute_die.blocks if isinstance(b, TPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("TPU v4 compute die has no TPUBlock")
    return block


# ---------------------------------------------------------------------------
# Catalog: TPU v4 joins Coral in google/ vendor directory
# ---------------------------------------------------------------------------

def test_catalog_includes_tpu_v4(all_products):
    google_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "google")
    # google_tpu_v1 added in sprint #72 PR 3 (Bucket A long-tail backfill)
    assert google_skus == ["google_coral_edge_tpu", "google_tpu_v1", "google_tpu_v3", "google_tpu_v4"]


def test_catalog_has_21_total_products(all_products):
    """At least 12 KPU + 2 GPU + 1 CPU + 3 NPU + 1 CGRA + 1 DPU +
    1 TPU = 21 ComputeProducts in the v7 baseline. Subset semantics
    so this test does not regress when future SKUs land (e.g., the
    v9 DSP sprint added a DSP entry which is fine here)."""
    counts_by_kind: dict[str, int] = {}
    for cp in all_products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    assert counts_by_kind.get("kpu", 0) >= 12
    assert counts_by_kind.get("gpu", 0) >= 2
    assert counts_by_kind.get("cpu", 0) >= 1
    assert counts_by_kind.get("npu", 0) >= 3
    assert counts_by_kind.get("cgra", 0) >= 1
    assert counts_by_kind.get("dpu", 0) >= 1
    assert counts_by_kind.get("tpu", 0) >= 1


def test_other_vendors_unaffected_by_tpu_v4_addition(all_products):
    """Additive guarantee: adding TPU v4 to google/ must not perturb
    other vendor directories."""
    counts_by_vendor: dict[str, int] = {}
    for cp in all_products.values():
        counts_by_vendor[cp.vendor] = counts_by_vendor.get(cp.vendor, 0) + 1
    assert counts_by_vendor.get("stillwater") == 12
    assert counts_by_vendor.get("nvidia") == 2
    assert counts_by_vendor.get("intel") == 4  # i7-12700k + 3 Xeons: 8490H + 8592+ + 6980P (sprint #68 PR 1-3)
    assert counts_by_vendor.get("hailo") == 2
    assert counts_by_vendor.get("stanford") == 1
    assert counts_by_vendor.get("xilinx") == 1
    # google grows from 1 (Coral) to 2 (Coral + TPU v4)
    assert counts_by_vendor.get("google") == 4  # +google_tpu_v3 (sprint #72 PR 4)


# ---------------------------------------------------------------------------
# TPU v4 identity / die / silicon_bin
# ---------------------------------------------------------------------------

def test_tpu_v4_identity(tpu_v4):
    assert tpu_v4.id == "google_tpu_v4"
    assert tpu_v4.vendor == "google"
    assert tpu_v4.packaging.kind == PackagingKind.MONOLITHIC
    assert tpu_v4.packaging.package_type == "datacenter_oam"
    assert tpu_v4.lifecycle == LifecycleStatus.PRODUCTION


def test_tpu_v4_die_references_tsmc_n7(tpu_v4_compute_die):
    """TPU v4 ships on TSMC N7. Already in catalog from KPU sprint."""
    die = tpu_v4_compute_die
    assert die.process_node_id == "tsmc_n7"
    assert die.die_role.value == "compute"
    assert die.die_size_mm2 == pytest.approx(600.0, rel=0.1)
    assert die.transistors_billion == pytest.approx(50.0, rel=0.15)


def test_tpu_v4_silicon_bin_reconciles_with_die_total(tpu_v4_compute_die):
    die = tpu_v4_compute_die
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


def test_tpu_v4_silicon_bin_has_tpu_count_refs(tpu_v4_compute_die):
    """Per the paper exercise, TPU silicon_bin uses TPU-specific blocks:
    mxu_array, accumulator_sram, unified_buffer_sram, weight_fifo_sram,
    hbm_phy, ici_phy. The ``unified_buffer_sram`` is the canonical TPU
    on-chip memory tier."""
    names = {b.name for b in tpu_v4_compute_die.silicon_bin.blocks}
    assert "mxu_array" in names
    assert "unified_buffer_sram" in names, \
        "TPU-specific unified_buffer_sram block missing"
    assert "hbm_phy" in names
    assert "ici_phy" in names


# ---------------------------------------------------------------------------
# TPUBlock structural fields
# ---------------------------------------------------------------------------

def test_tpu_v4_block_is_tpu(tpu_v4_block):
    assert isinstance(tpu_v4_block, TPUBlock)
    assert tpu_v4_block.kind == "tpu"


def test_tpu_v4_mxu_hierarchy(tpu_v4_block):
    """2 MXUs, each 128x128 systolic array (= 32,768 total MACs)."""
    block = tpu_v4_block
    assert block.num_mxus == 2
    assert block.mxu_dim_rows == 128
    assert block.mxu_dim_cols == 128


def test_tpu_v4_tile_energy_decomposition(tpu_v4_block):
    """All 9 canonical tile energy fields populated."""
    coeffs = tpu_v4_block.tile_energy_coefficients
    assert coeffs.mac_energy_pj == pytest.approx(0.25)
    assert coeffs.weight_memory_energy_pj_per_byte == pytest.approx(10.0)
    assert coeffs.weight_fifo_energy_pj_per_byte == pytest.approx(0.5)
    assert coeffs.unified_buffer_read_energy_pj_per_byte == pytest.approx(0.5)
    assert coeffs.unified_buffer_write_energy_pj_per_byte == pytest.approx(0.5)
    assert coeffs.accumulator_read_energy_pj_per_element == pytest.approx(0.3)
    assert coeffs.accumulator_write_energy_pj_per_element == pytest.approx(0.4)


def test_tpu_v4_other_tile_energy_fields(tpu_v4_block):
    """Per-MXU buffer + pipeline fields."""
    block = tpu_v4_block
    assert block.weight_tile_size_kib == 32
    assert block.weight_fifo_depth == 2
    assert block.pipeline_fill_cycles == 128
    assert block.accumulator_size_kib_per_mxu == 2048   # 2 MiB


def test_tpu_v4_ici_single_chip_surface(tpu_v4_block):
    """ICI 6 ports * 400 GB/s for 3D-torus pod. Pod-level topology
    deferred to v8+."""
    block = tpu_v4_block
    assert block.ici_port_count == 6
    assert block.ici_bandwidth_per_port_gbps == pytest.approx(400.0)
    assert block.ici_topology_hint == "3d_torus_2x2x2"


def test_tpu_v4_compute_fabric(tpu_v4_block):
    """First SKU exercising DPUFabricKind.TPU_V2_PLUS. 2 ops/MAC for
    both BF16 and INT8."""
    block = tpu_v4_block
    assert len(block.compute_fabrics) == 1
    fabric = block.compute_fabrics[0]
    assert fabric.fabric_kind == TPUFabricKind.TPU_V2_PLUS
    assert fabric.ops_per_unit_per_clock["bf16"] == 2
    assert fabric.ops_per_unit_per_clock["int8"] == 2
    # BF16 baseline; INT8 is 4x cheaper (uses BF16 MAC at higher throughput)
    assert fabric.energy_per_op_bf16_pj == pytest.approx(0.225)
    assert fabric.energy_scaling.get("int8") == pytest.approx(0.25)
    assert fabric.energy_scaling.get("fp32") == pytest.approx(2.0)


def test_tpu_v4_multi_precision_training_first(tpu_v4_block):
    """TPU v4 supports BF16 + INT8 + FP32 (training-first design).
    Distinguishes from inference-first NPU/DPU (INT-only)."""
    block = tpu_v4_block
    assert set(block.multi_precision_alu) == {"bf16", "int8", "fp32"}


# ---------------------------------------------------------------------------
# Memory: first SKU with chip-attached HBM2e
# ---------------------------------------------------------------------------

def test_tpu_v4_unified_buffer(tpu_v4_block):
    """32 MiB UB (collapses L1+L2 into one tier)."""
    mem = tpu_v4_block.memory
    assert mem.unified_buffer_size_kib == 32 * 1024
    assert mem.unified_buffer_access_energy_pj_per_byte == pytest.approx(0.5)


def test_tpu_v4_hbm2e_chip_attached(tpu_v4_block):
    """First SKU populating HBM2e. 32 GiB at 1.2 TB/s chip-attached
    (vs DPU's chip-attached DDR4, CGRA's host-bus DDR4)."""
    mem = tpu_v4_block.memory
    assert mem.has_external_dram is True
    assert mem.external_dram_type == MemoryType.HBM2E
    assert mem.external_dram_size_gb == pytest.approx(32.0)
    assert mem.external_dram_bandwidth_gbps == pytest.approx(1200.0)


# ---------------------------------------------------------------------------
# NoC: MULTI_CROSSBAR for 2 MXUs sharing UB
# ---------------------------------------------------------------------------

def test_tpu_v4_noc_is_multi_crossbar(tpu_v4_block):
    """First SKU with TPUNoCTopology.MULTI_CROSSBAR. 2 MXUs share UB."""
    noc = tpu_v4_block.noc
    assert noc.topology == TPUNoCTopology.MULTI_CROSSBAR
    assert noc.unit_count == 2
    assert noc.confidence == DataConfidence.THEORETICAL


# ---------------------------------------------------------------------------
# Scheduler: TPU-specific (high systolic occupancy)
# ---------------------------------------------------------------------------

def test_tpu_v4_min_occupancy_is_high(tpu_v4_block):
    """TPUs need high systolic utilization. Default 0.5 (vs CGRA/DPU 0.3)."""
    assert tpu_v4_block.min_occupancy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Power / performance / market
# ---------------------------------------------------------------------------

def test_tpu_v4_single_thermal_profile(tpu_v4):
    """350W datacenter envelope, liquid cooling."""
    power = tpu_v4.power
    assert power.tdp_watts == 350.0
    assert power.default_thermal_profile == "default"
    profile_names = {p.name for p in power.thermal_profiles}
    assert profile_names == {"default"}
    assert power.thermal_profiles[0].cooling_solution_id == "liquid_cooled"
    assert power.thermal_profiles[0].clock_mhz == pytest.approx(1050.0)


def test_tpu_v4_chip_level_performance_matches_marketing(tpu_v4):
    """275 TFLOPS BF16 (the marketed number), 550 TOPS INT8 (2x BF16),
    137.5 TFLOPS emulated FP32. No INT4."""
    perf = tpu_v4.performance
    assert perf.bf16_tflops == pytest.approx(275.0)
    assert perf.int8_tops == pytest.approx(550.0)
    assert perf.fp32_tflops == pytest.approx(137.5)
    assert perf.int4_tops == 0.0   # added in v5p


def test_tpu_v4_datacenter_market(tpu_v4):
    """Production datacenter SKU; Google Cloud TPU rental."""
    market = tpu_v4.market
    assert market.target_market == "datacenter"
    assert market.is_available is True
    assert market.product_family == "TPU"
    assert market.model_tier == "datacenter"


def test_tpu_v4_round_trips_through_serialize(tpu_v4):
    """Full ComputeProduct -> JSON -> ComputeProduct round-trip preserves
    TPUBlock + tile energy coefficients + ICI surface."""
    payload = tpu_v4.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    rebuilt_block = next(
        b for d in rebuilt.dies
        if d.die_role.value == "compute"
        for b in d.blocks
        if isinstance(b, TPUBlock)
    )
    assert rebuilt_block.num_mxus == 2
    assert rebuilt_block.mxu_dim_rows == 128
    assert rebuilt_block.ici_port_count == 6
    assert rebuilt_block.tile_energy_coefficients.mac_energy_pj == pytest.approx(0.25)
    assert rebuilt_block.memory.external_dram_type == MemoryType.HBM2E
    assert rebuilt_block.compute_fabrics[0].fabric_kind == TPUFabricKind.TPU_V2_PLUS
