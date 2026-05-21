"""Tests for the AMD EPYC 9965 (Turin Dense) ComputeProduct YAML (sprint #62 PR 3).

Third AMD SKU in the catalog and first Zen 5 entry. Validates the
YAML loads as a fully-formed ``ComputeProduct`` + ``CPUBlock`` with
field values matching AMD's published Turin Dense specs and the
graphs-side hand-coded mapper (the migration source -- 192 cores,
500W TDP).

Same test shape as ``test_compute_product_amd_epyc_9{654,754}_yaml.py``.
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
    cp = all_products.get("amd_epyc_9965_sp5")
    if cp is None:
        pytest.fail("amd_epyc_9965_sp5 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def epyc_compute_die(epyc):
    die = next((d for d in epyc.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("EPYC 9965 has no compute die")
    return die


@pytest.fixture(scope="module")
def epyc_cpu_block(epyc_compute_die) -> CPUBlock:
    block = next(
        (b for b in epyc_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("EPYC 9965 compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process nodes (tsmc_n4p already in catalog; n3e is a known
# follow-up per the YAML header).
# ---------------------------------------------------------------------------

def test_process_node_present():
    nodes = load_process_nodes()
    assert "tsmc_n4p" in nodes
    # Note: TSMC N3E is the more-accurate node for Zen 5c per
    # AMD's Oct 2024 launch coverage but is not yet in the
    # catalog. This test only confirms the placeholder node we use.


# ---------------------------------------------------------------------------
# Identity and packaging
# ---------------------------------------------------------------------------

def test_identity(epyc):
    assert epyc.id == "amd_epyc_9965_sp5"
    assert epyc.vendor == "amd"
    assert epyc.lifecycle == LifecycleStatus.PRODUCTION


def test_packaging_is_chiplet(epyc):
    """13 dies (12 Zen 5c CCDs + 1 Turin IOD) -- chiplet packaging."""
    assert epyc.packaging.kind == PackagingKind.CHIPLET
    assert epyc.packaging.num_dies == 13       # same chiplet count as 9654 (12 CCDs + IOD)
    assert epyc.packaging.package_type == "sp5"


def test_die_geometry(epyc_compute_die):
    """Headline PhysicalSpec values: 12*73 (CCDs) + 400 (IOD) = 1276 mm^2."""
    assert epyc_compute_die.die_size_mm2 == 1276.0
    assert epyc_compute_die.transistors_billion == 145.0
    assert epyc_compute_die.process_node_id == "tsmc_n4p"


# ---------------------------------------------------------------------------
# CPUBlock shape -- single homogeneous cluster of 192 Zen 5c cores
# ---------------------------------------------------------------------------

def test_cpu_block_headline(epyc_cpu_block):
    assert epyc_cpu_block.total_effective_cores == 192
    assert epyc_cpu_block.max_concurrent_threads == 384  # SMT=2
    assert epyc_cpu_block.simd_width_lanes == 16


def test_cpu_block_zen5_homogeneous_cluster(epyc_cpu_block):
    """One HOMOGENEOUS cluster of 192 cores, SMT=2."""
    assert len(epyc_cpu_block.core_clusters) == 1
    cluster = epyc_cpu_block.core_clusters[0]
    assert cluster.cluster_kind == CoreClusterKind.HOMOGENEOUS
    assert cluster.num_cores == 192
    assert cluster.smt_threads == 2
    assert cluster.l2_layout == L2Layout.PRIVATE_PER_CORE


def test_zen5_l1d_enlarged_to_48kib(epyc_cpu_block):
    """Zen 5 enlarged L1D from 32 KB (Zen 4) to 48 KB. Cross-arch invariant."""
    cluster = epyc_cpu_block.core_clusters[0]
    assert cluster.l1_kib_per_core == 48
    assert cluster.l2_kib_per_core == 1024   # L2 unchanged


def test_zen5_avx512_fabric(epyc_cpu_block):
    """Same ops/clock convention as Zen 4 YAMLs, but Zen 5's full-width
    FMA datapath sustains these per cycle vs Zen 4's amortization over 2."""
    cluster = epyc_cpu_block.core_clusters[0]
    assert len(cluster.compute_fabrics) == 1
    fab = cluster.compute_fabrics[0]
    assert fab.isa_extension == CPUISAExtension.AVX512_BF16
    assert fab.ops_per_core_per_clock["fp32"] == 16
    assert fab.ops_per_core_per_clock["bf16"] == 32
    assert fab.ops_per_core_per_clock["int8"] == 64
    assert fab.energy_per_flop_fp32_pj == pytest.approx(1.35)


def test_cpu_block_memory_ddr5_6000(epyc_cpu_block):
    """Turin Dense bumps memory to DDR5-6000 (576 GB/s) -- faster
    than Genoa / Bergamo (DDR5-4800, 460.8 GB/s)."""
    mem = epyc_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 12
    assert mem.memory_bandwidth_gbps == 576.0       # 12 * 6000 MT/s * 8 B
    assert mem.l3_total_kib == 393216               # 384 MiB
    assert mem.coherence_protocol == "directory_mesi"


def test_cpu_block_noc_has_12_ccds(epyc_cpu_block):
    """Zen 5c CCDs hold 16 cores each, so 192 cores -> 12 CCDs."""
    assert epyc_cpu_block.noc.topology == CPUNoCTopology.IO_DIE_PLUS_CCD
    assert epyc_cpu_block.noc.unit_count == 12


# ---------------------------------------------------------------------------
# Power + market
# ---------------------------------------------------------------------------

def test_power_envelope_500w(epyc):
    """Turin Dense: 500W TDP, cTDP 400-550W (higher than Genoa / Bergamo's 360W)."""
    assert epyc.power.tdp_watts == 500.0
    assert epyc.power.max_power_watts == 550.0
    assert epyc.power.min_power_watts == 400.0
    prof = epyc.power.thermal_profiles[0]
    assert prof.clock_mhz == 3400.0


def test_market(epyc):
    """Turin Dense launched Oct 2024."""
    assert epyc.market.launch_date == "2024-10-10"
    assert epyc.market.launch_msrp_usd == 14813.0
    assert epyc.market.product_family == "EPYC"


# ---------------------------------------------------------------------------
# Performance roll-up + cross-SKU comparisons
# ---------------------------------------------------------------------------

def test_performance_rollup(epyc):
    """192 cores * 16 ops/cycle * 3.4 GHz = 10.44 TFLOPS FP32."""
    assert epyc.performance.fp32_tflops == pytest.approx(10.44)
    assert epyc.performance.bf16_tflops == pytest.approx(20.89)
    assert epyc.performance.int8_tops == pytest.approx(41.78)


def test_turin_dense_roughly_2x_bergamo(all_products):
    """Turin Dense should hit ~2x Bergamo's chip-level peak: +50% cores
    and Zen 5's full-width FMA cancels Bergamo's Zen 4c double-pumping."""
    nine_nine = all_products.get("amd_epyc_9965_sp5")
    nine_seven = all_products.get("amd_epyc_9754_sp5")
    if nine_nine is None or nine_seven is None:
        pytest.skip("both SKUs needed; this is a cross-SKU sanity check")
    ratio = nine_nine.performance.fp32_tflops / nine_seven.performance.fp32_tflops
    # Roughly 2x (10.44 / 5.43 = 1.92)
    assert 1.8 < ratio < 2.2, f"unexpected Turin/Bergamo FP32 ratio: {ratio:.2f}"


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums(epyc_compute_die):
    """132000 (12 Zen 5c CCDs) + 0 + 13000 (IOD) = 145000 Mtx = 145 B tx."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = epyc_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)
