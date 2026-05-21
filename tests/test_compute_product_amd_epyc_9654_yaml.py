"""Tests for the AMD EPYC 9654 ComputeProduct YAML (sprint #62 PR 1).

First chiplet CPU SKU in the catalog. Validates the YAML loads as a
fully-formed ``ComputeProduct`` + ``CPUBlock`` with field values
matching AMD's published Genoa specs and the graphs-side hand-coded
mapper (the migration source).

Same test shape as
``test_compute_product_v3_intel_i7_12700k_yaml.py``.
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
    cp = all_products.get("amd_epyc_9654_sp5")
    if cp is None:
        pytest.fail("amd_epyc_9654_sp5 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def epyc_compute_die(epyc):
    """Pick the compute die by role (same pattern as i7 / Thor)."""
    die = next((d for d in epyc.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("EPYC 9654 has no compute die")
    return die


@pytest.fixture(scope="module")
def epyc_cpu_block(epyc_compute_die) -> CPUBlock:
    block = next(
        (b for b in epyc_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("EPYC 9654 compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process node: tsmc_n5 already in catalog (CCDs); IOD on tsmc_n6 is
# documented in YAML comments but not at the Die level (schema gap --
# no IOBlock kind yet; see embodied-schemas#62).
# ---------------------------------------------------------------------------

def test_process_nodes_present():
    nodes = load_process_nodes()
    assert "tsmc_n5" in nodes  # CCD process
    assert "tsmc_n6" in nodes  # IOD process (referenced in YAML comments)


# ---------------------------------------------------------------------------
# Top-level identity and packaging
# ---------------------------------------------------------------------------

def test_identity(epyc):
    assert epyc.id == "amd_epyc_9654_sp5"
    assert epyc.vendor == "amd"
    assert epyc.lifecycle == LifecycleStatus.PRODUCTION


def test_packaging_is_chiplet(epyc):
    """13 dies (12 CCDs + 1 IOD) -- chiplet packaging."""
    assert epyc.packaging.kind == PackagingKind.CHIPLET
    assert epyc.packaging.num_dies == 13
    assert epyc.packaging.package_type == "sp5"


def test_die_geometry(epyc_compute_die):
    """Headline PhysicalSpec values: total die area + transistors
    summed across the chiplets (12*66 CCDs + 397 IOD)."""
    assert epyc_compute_die.die_size_mm2 == 1189.0
    assert epyc_compute_die.transistors_billion == pytest.approx(90.84)
    # Primary process node = CCDs (TSMC N5). IOD on N6 is captured
    # only in silicon_bin notes pending an IOBlock kind.
    assert epyc_compute_die.process_node_id == "tsmc_n5"


# ---------------------------------------------------------------------------
# CPUBlock shape -- single homogeneous cluster of 96 Zen 4 cores
# ---------------------------------------------------------------------------

def test_cpu_block_headline(epyc_cpu_block):
    assert epyc_cpu_block.total_effective_cores == 96
    assert epyc_cpu_block.simd_width_lanes == 16  # 512-bit AVX-512


def test_cpu_block_homogeneous_cluster(epyc_cpu_block):
    """EPYC 9654 has one HOMOGENEOUS cluster of 96 cores with SMT=2."""
    assert len(epyc_cpu_block.core_clusters) == 1
    cluster = epyc_cpu_block.core_clusters[0]
    assert cluster.cluster_kind == CoreClusterKind.HOMOGENEOUS
    assert cluster.num_cores == 96
    assert cluster.smt_threads == 2
    assert cluster.l2_layout == L2Layout.PRIVATE_PER_CORE
    assert cluster.l1_kib_per_core == 32     # 32 KB L1D per core (Zen 4)
    assert cluster.l2_kib_per_core == 1024   # 1 MB private L2 per core


def test_cpu_block_avx512_fabric(epyc_cpu_block):
    """Single AVX-512_BF16 fabric (Zen 4's AVX-512 lite double-pumped 256b)."""
    cluster = epyc_cpu_block.core_clusters[0]
    assert len(cluster.compute_fabrics) == 1
    fab = cluster.compute_fabrics[0]
    assert fab.isa_extension == CPUISAExtension.AVX512_BF16
    # FP32: 16 lanes * 1 FMA = 16 ops/clock; INT8 VNNI: 64; BF16: 32
    assert fab.ops_per_core_per_clock["fp32"] == 16
    assert fab.ops_per_core_per_clock["bf16"] == 32
    assert fab.ops_per_core_per_clock["int8"] == 64


def test_cpu_block_memory(epyc_cpu_block):
    """12-channel DDR5-4800 + 384 MB shared L3 LLC."""
    mem = epyc_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 12
    assert mem.memory_bus_bits == 768          # 12 * 64
    assert mem.memory_bandwidth_gbps == pytest.approx(460.8)
    assert mem.l3_present is True
    assert mem.l3_total_kib == 393216          # 384 MiB
    assert mem.coherence_protocol == "directory_mesi"


def test_cpu_block_noc_is_infinity_fabric(epyc_cpu_block):
    """AMD's CCDs hang off the IOD via Infinity Fabric IFOP links."""
    assert epyc_cpu_block.noc.topology == CPUNoCTopology.IO_DIE_PLUS_CCD
    # unit_count = CCD count per OnDieFabric base field semantics
    assert epyc_cpu_block.noc.unit_count == 12


# ---------------------------------------------------------------------------
# Power + market
# ---------------------------------------------------------------------------

def test_power_envelope(epyc):
    """360W default TDP, configurable 320-400W (cTDP)."""
    assert epyc.power.tdp_watts == 360.0
    assert epyc.power.max_power_watts == 400.0
    assert epyc.power.min_power_watts == 320.0
    assert epyc.power.default_thermal_profile == "360W-default"
    assert len(epyc.power.thermal_profiles) == 1
    prof = epyc.power.thermal_profiles[0]
    assert prof.tdp_watts == 360.0
    assert prof.clock_mhz == 3550.0     # all-core sustained boost


def test_market(epyc):
    assert epyc.market.launch_date == "2022-11-10"
    assert epyc.market.launch_msrp_usd == 11805.0
    assert epyc.market.product_family == "EPYC"
    assert epyc.market.target_market == "datacenter"


# ---------------------------------------------------------------------------
# Performance roll-up
# ---------------------------------------------------------------------------

def test_performance_rollup(epyc):
    """Chip-level peak at 3.55 GHz all-core boost (KPUTheoreticalPerformance shape)."""
    assert epyc.performance.fp32_tflops == pytest.approx(5.45)
    assert epyc.performance.bf16_tflops == pytest.approx(10.91)
    assert epyc.performance.int8_tops == pytest.approx(21.82)


# ---------------------------------------------------------------------------
# Silicon bin -- total Mtx must sum to transistors_billion * 1000
# ---------------------------------------------------------------------------

def test_silicon_bin_sums(epyc_compute_die):
    """78840 Mtx (CCDs) + 0 (cache informational) + 12000 (IOD) = 90840 Mtx."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    # transistors_billion = 90.84 -> 90840 Mtx
    expected_mtx = epyc_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)
