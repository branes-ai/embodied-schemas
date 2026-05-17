"""Tests for the third NPU SKU YAML: google_coral_edge_tpu.

Catalog gain: 12 KPU + 2 GPU + 1 CPU + 3 NPU = 18 ComputeProducts.
First SKU to exercise:
  - ``NPUDataflowKind.SYSTOLIC`` (vs STRUCTURE_DRIVEN used by Hailo)
  - The newly-added ``gf_28nm`` process node (PR #32)
  - The ``google`` vendor directory

Same shape as ``test_compute_product_v4_hailo_8_yaml.py`` and
``test_compute_product_v4_hailo_10h_yaml.py``.
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    LifecycleStatus,
    NPUBlock,
    NPUDataflowKind,
    NPUNoCTopology,
    PackagingKind,
)
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import DataConfidence


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def coral(all_products) -> ComputeProduct:
    cp = all_products.get("google_coral_edge_tpu")
    if cp is None:
        pytest.fail("google_coral_edge_tpu missing from catalog")
    return cp


@pytest.fixture(scope="module")
def coral_compute_die(coral):
    die = next((d for d in coral.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Coral has no compute die")
    return die


@pytest.fixture(scope="module")
def coral_npu_block(coral_compute_die) -> NPUBlock:
    block = next(
        (b for b in coral_compute_die.blocks if isinstance(b, NPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("Coral compute die has no NPUBlock")
    return block


# ---------------------------------------------------------------------------
# Catalog: Coral is the third NPU SKU (first Google vendor SKU)
# ---------------------------------------------------------------------------

def test_catalog_includes_coral(all_products):
    """At-least-Coral check (subset semantics). The TPU v4 data PR
    grew google/ past the original single-SKU mark; this test just
    pins Coral as present."""
    google_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "google")
    assert "google_coral_edge_tpu" in google_skus


def test_catalog_has_three_npu_skus(all_products):
    """At-least-three-NPU check (subset semantics). The Plasticine v2
    CGRA follow-up grew the catalog past the original 18-product mark;
    the CGRA PR's own contract test pins the new total."""
    counts_by_kind: dict[str, int] = {}
    for cp in all_products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    # NPU count is the focus of the Coral PR; pin >= 3 (Hailo-8 +
    # Hailo-10H + Coral) without locking the total catalog size.
    assert counts_by_kind.get("npu", 0) >= 3


def test_other_vendors_unaffected_by_coral_addition(all_products):
    """Additive guarantee: adding google/ vendor directory must not
    perturb stillwater/, nvidia/, intel/, or hailo/ loading."""
    stillwater = [s for s, cp in all_products.items() if cp.vendor == "stillwater"]
    nvidia = [s for s, cp in all_products.items() if cp.vendor == "nvidia"]
    intel = [s for s, cp in all_products.items() if cp.vendor == "intel"]
    hailo = [s for s, cp in all_products.items() if cp.vendor == "hailo"]
    assert len(stillwater) == 12
    assert len(nvidia) == 2
    assert len(intel) == 1
    assert len(hailo) == 2


# ---------------------------------------------------------------------------
# Coral identity / die / silicon_bin
# ---------------------------------------------------------------------------

def test_coral_identity(coral):
    assert coral.id == "google_coral_edge_tpu"
    assert coral.vendor == "google"
    assert coral.packaging.kind == PackagingKind.MONOLITHIC
    assert coral.packaging.package_type == "m_dot_2"
    assert coral.lifecycle == LifecycleStatus.PRODUCTION


def test_coral_die_references_gf_28nm(coral_compute_die):
    """Coral ships on GF 28nm SLP per the precursor PR (#32)."""
    die = coral_compute_die
    assert die.process_node_id == "gf_28nm"
    assert die.die_role.value == "compute"
    assert die.die_size_mm2 == pytest.approx(25.0, rel=0.05)
    assert die.transistors_billion == pytest.approx(0.25, rel=0.1)


def test_coral_silicon_bin_reconciles_with_die_total(coral_compute_die):
    die = coral_compute_die
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
# NPUBlock structural fields (systolic-specific)
# ---------------------------------------------------------------------------

def test_coral_block_is_npu(coral_npu_block):
    assert isinstance(coral_npu_block, NPUBlock)
    assert coral_npu_block.kind == "npu"


def test_coral_single_dataflow_unit_with_wide_lanes(coral_npu_block):
    """Per NPUBlock schema convention: the 64x64 systolic array is
    ONE dataflow unit with 4096 SIMD lanes (vs Hailo's 32-unit
    multi-fabric layout). Pinned to enforce the convention."""
    block = coral_npu_block
    assert block.num_dataflow_units == 1
    assert block.lanes_per_unit == 4096
    assert block.max_concurrent_models == 1


def test_coral_systolic_dataflow_fabric(coral_npu_block):
    """First SKU in catalog with NPUDataflowKind.SYSTOLIC. Hailo SKUs
    are STRUCTURE_DRIVEN; Coral is the canonical systolic NPU."""
    block = coral_npu_block
    assert len(block.compute_fabrics) == 1
    fabric = block.compute_fabrics[0]
    assert fabric.dataflow_kind == NPUDataflowKind.SYSTOLIC
    # 4096 MACs * 2 ops/MAC = 8000 ops per unit per clock
    assert fabric.ops_per_unit_per_clock["int8"] == 8000


def test_coral_is_int8_only(coral_npu_block):
    """Edge TPU is hard-wired for INT8 only -- no INT4 (unlike Hailo)
    and no FP."""
    block = coral_npu_block
    assert set(block.multi_precision_alu) == {"int8"}
    fabric_precisions = set(block.compute_fabrics[0].ops_per_unit_per_clock)
    assert fabric_precisions == {"int8"}


# ---------------------------------------------------------------------------
# Memory: SRAM-only with no shared (only 1 unit, so nothing to share)
# ---------------------------------------------------------------------------

def test_coral_memory_is_sram_only(coral_npu_block):
    """Coral has NO external DRAM. Uses host memory via USB/PCIe."""
    mem = coral_npu_block.memory
    assert mem.has_external_dram is False
    assert mem.external_dram_type is None
    assert mem.external_dram_size_gb is None
    assert mem.external_dram_bandwidth_gbps is None
    # 512 KiB unified buffer; no shared SRAM (single dataflow unit)
    assert mem.sram_kib_per_unit == 512
    assert mem.shared_sram_kib == 0
    assert mem.coherence_protocol == "none"


def test_coral_kv_cache_is_none(coral_npu_block):
    """Coral is a CNN-class NPU (TFLite vision models); no KV cache."""
    assert coral_npu_block.kv_cache is None


# ---------------------------------------------------------------------------
# NoC: degenerate crossbar for single-unit dataflow
# ---------------------------------------------------------------------------

def test_coral_noc_is_crossbar(coral_npu_block):
    """With a single dataflow unit, the NoC degenerates to a trivial
    crossbar between the systolic array and the unified buffer."""
    noc = coral_npu_block.noc
    assert noc.topology == NPUNoCTopology.CROSSBAR
    assert noc.unit_count == 1
    # Crossbar -- no mesh dimensions
    assert noc.mesh_rows is None
    assert noc.mesh_cols is None
    assert noc.confidence == DataConfidence.THEORETICAL


# ---------------------------------------------------------------------------
# Power / performance / market
# ---------------------------------------------------------------------------

def test_coral_single_thermal_profile(coral):
    """2W envelope, passive cooling."""
    power = coral.power
    assert power.tdp_watts == 2.0
    assert power.default_thermal_profile == "2W"
    profile_names = {p.name for p in power.thermal_profiles}
    assert profile_names == {"2W"}
    assert power.thermal_profiles[0].cooling_solution_id == "passive_heatsink_small"
    # 500 MHz clock
    assert power.thermal_profiles[0].clock_mhz == pytest.approx(500.0)


def test_coral_chip_level_performance_matches_marketing(coral):
    """4 TOPS INT8 (the marketed number). No FP, no INT4."""
    perf = coral.performance
    assert perf.int8_tops == 4.0
    assert perf.int4_tops == 0.0   # not supported (vs Hailo SKUs that do INT4)
    assert perf.fp32_tflops == 0.0
    assert perf.bf16_tflops == 0.0


def test_coral_round_trips_through_serialize(coral):
    """Full ComputeProduct -> JSON -> ComputeProduct round-trip
    preserves NPUBlock discriminator dispatch."""
    payload = coral.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    rebuilt_block = next(
        b for d in rebuilt.dies
        if d.die_role.value == "compute"
        for b in d.blocks
        if isinstance(b, NPUBlock)
    )
    assert rebuilt_block.num_dataflow_units == 1
    assert rebuilt_block.lanes_per_unit == 4096
    assert rebuilt_block.compute_fabrics[0].dataflow_kind == NPUDataflowKind.SYSTOLIC
    assert rebuilt_block.kv_cache is None
