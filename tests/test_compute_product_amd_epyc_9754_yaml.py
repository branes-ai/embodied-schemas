"""Tests for the AMD EPYC 9754 (Bergamo) ComputeProduct YAML (sprint #62 PR 2).

Second AMD SKU in the catalog and first Zen 4c entry. Validates the
YAML loads as a fully-formed ``ComputeProduct`` + ``CPUBlock`` with
field values matching AMD's published Bergamo specs and the graphs-
side hand-coded mapper (the migration source).

Same test shape as ``test_compute_product_amd_epyc_9654_yaml.py``.
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
def epyc(all_products) -> ComputeProduct:
    cp = all_products.get("amd_epyc_9754_sp5")
    if cp is None:
        pytest.fail("amd_epyc_9754_sp5 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def epyc_compute_die(epyc):
    die = next((d for d in epyc.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("EPYC 9754 has no compute die")
    return die


@pytest.fixture(scope="module")
def epyc_cpu_block(epyc_compute_die) -> CPUBlock:
    block = next(
        (b for b in epyc_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("EPYC 9754 compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process nodes (both already in catalog from sprint #62 PR 1).
# ---------------------------------------------------------------------------

def test_process_nodes_present():
    nodes = load_process_nodes()
    assert "tsmc_n5" in nodes  # Zen 4c CCDs
    assert "tsmc_n6" in nodes  # Shared Genoa IOD


# ---------------------------------------------------------------------------
# Top-level identity and packaging
# ---------------------------------------------------------------------------

def test_identity(epyc):
    assert epyc.id == "amd_epyc_9754_sp5"
    assert epyc.vendor == "amd"
    assert epyc.lifecycle == LifecycleStatus.PRODUCTION


def test_packaging_is_chiplet(epyc):
    """9 dies (8 Zen 4c CCDs + 1 Genoa IOD) -- chiplet packaging."""
    assert epyc.packaging.kind == PackagingKind.CHIPLET
    assert epyc.packaging.num_dies == 9          # one fewer than 9654 (8+1 vs 12+1)
    assert epyc.packaging.package_type == "sp5"  # same SP5 socket as 9654


def test_die_geometry(epyc_compute_die):
    """Headline PhysicalSpec values: 8*72.7 (CCDs) + 397 (IOD) = 978.6 mm^2."""
    assert epyc_compute_die.die_size_mm2 == pytest.approx(978.6, rel=0.01)
    assert epyc_compute_die.transistors_billion == pytest.approx(89.6, rel=0.01)
    assert epyc_compute_die.process_node_id == "tsmc_n5"


def test_die_size_smaller_than_9654(all_products):
    """Cross-SKU sanity: Bergamo's package is ~17% smaller than Genoa's
    despite +33% cores (8 CCDs vs 12, sharing the same IOD)."""
    nine_seven = all_products.get("amd_epyc_9754_sp5")
    nine_six = all_products.get("amd_epyc_9654_sp5")
    if nine_seven is None or nine_six is None:
        pytest.skip("both SKUs needed; this is a cross-SKU sanity check")
    assert nine_seven.dies[0].die_size_mm2 < nine_six.dies[0].die_size_mm2


# ---------------------------------------------------------------------------
# CPUBlock shape -- single homogeneous cluster of 128 Zen 4c cores
# ---------------------------------------------------------------------------

def test_cpu_block_headline(epyc_cpu_block):
    assert epyc_cpu_block.total_effective_cores == 128   # vs 9654's 96
    assert epyc_cpu_block.max_concurrent_threads == 256  # SMT=2
    assert epyc_cpu_block.simd_width_lanes == 16         # full AVX-512


def test_cpu_block_homogeneous_cluster(epyc_cpu_block):
    """EPYC 9754 has one HOMOGENEOUS cluster of 128 cores with SMT=2."""
    assert len(epyc_cpu_block.core_clusters) == 1
    cluster = epyc_cpu_block.core_clusters[0]
    assert cluster.cluster_kind == CoreClusterKind.HOMOGENEOUS
    assert cluster.num_cores == 128
    assert cluster.smt_threads == 2
    assert cluster.l2_layout == L2Layout.PRIVATE_PER_CORE
    assert cluster.l1_kib_per_core == 32
    # Zen 4c keeps the same per-core L2 capacity as Zen 4 (1 MB)
    assert cluster.l2_kib_per_core == 1024


def test_cpu_block_avx512_fabric_matches_zen4(epyc_cpu_block):
    """Zen 4c shares Zen 4's ISA byte-for-byte. The fabric shape is
    identical (full AVX-512_BF16 with VNNI) -- only the underlying
    physical layout / max clock differ."""
    cluster = epyc_cpu_block.core_clusters[0]
    assert len(cluster.compute_fabrics) == 1
    fab = cluster.compute_fabrics[0]
    assert fab.isa_extension == CPUISAExtension.AVX512_BF16
    assert fab.ops_per_core_per_clock["fp32"] == 16
    assert fab.ops_per_core_per_clock["bf16"] == 32
    assert fab.ops_per_core_per_clock["int8"] == 64
    assert fab.ops_per_core_per_clock["int4"] == 128
    # Same per-FMA energy as Zen 4 (same node, same logic structure)
    assert fab.energy_per_flop_fp32_pj == pytest.approx(1.35)


def test_cpu_block_memory_smaller_l3(epyc_cpu_block):
    """Bergamo trades L3 for cores: 128 MB total vs Genoa's 384 MB."""
    mem = epyc_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 12             # same as 9654 (shared IOD)
    assert mem.memory_bandwidth_gbps == pytest.approx(460.8)
    assert mem.l3_present is True
    assert mem.l3_total_kib == 131072               # 128 MiB (vs 393216 on 9654)
    assert mem.coherence_protocol == "directory_mesi"


def test_cpu_block_noc_has_8_ccds(epyc_cpu_block):
    """unit_count counts CCDs; Bergamo has 8 (vs Genoa's 12)."""
    assert epyc_cpu_block.noc.topology == CPUNoCTopology.IO_DIE_PLUS_CCD
    assert epyc_cpu_block.noc.unit_count == 8


# ---------------------------------------------------------------------------
# Power + market
# ---------------------------------------------------------------------------

def test_power_envelope_same_as_9654(epyc):
    """Same SP5 thermal envelope as Genoa: 360W TDP, cTDP 320-400W."""
    assert epyc.power.tdp_watts == 360.0
    assert epyc.power.max_power_watts == 400.0
    assert epyc.power.min_power_watts == 320.0
    prof = epyc.power.thermal_profiles[0]
    assert prof.clock_mhz == 2650.0      # all-core sustained boost (lower than 9654's 3550)


def test_market(epyc):
    """Bergamo launched June 2023, slightly cheaper than Genoa."""
    assert epyc.market.launch_date == "2023-06-13"
    assert epyc.market.launch_msrp_usd == 10632.0
    assert epyc.market.product_family == "EPYC"
    assert epyc.market.target_market == "datacenter"


# ---------------------------------------------------------------------------
# Performance roll-up (peak at 2.65 GHz all-core)
# ---------------------------------------------------------------------------

def test_performance_rollup(epyc):
    """Bergamo's chip-level peak is within 1% of Genoa's despite +33%
    cores and -25% clock -- the trade-off cancels at the chip level."""
    assert epyc.performance.fp32_tflops == pytest.approx(5.43)
    assert epyc.performance.bf16_tflops == pytest.approx(10.86)
    assert epyc.performance.int8_tops == pytest.approx(21.71)


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums(epyc_compute_die):
    """77600 (8 Zen 4c CCDs) + 0 (informational) + 12000 (IOD) = 89600 Mtx."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = epyc_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)
