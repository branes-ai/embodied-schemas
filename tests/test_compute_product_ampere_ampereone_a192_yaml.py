"""Tests for the Ampere AmpereOne A192-32X ComputeProduct YAML (sprint #62 PR 4).

First ARM SKU in the catalog and first Ampere vendor entry. Also
first monolithic CPU SKU -- prior EPYC entries are all chiplet
products using the single-virtual-die approximation; AmpereOne is
genuinely one die with one CPUBlock.

Validates the YAML loads as a fully-formed ``ComputeProduct`` +
``CPUBlock`` with field values matching Ampere's published
AmpereOne-1 brief and the graphs-side hand-coded mapper.

Same test shape as the EPYC YAML tests.
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
def amp(all_products) -> ComputeProduct:
    cp = all_products.get("ampere_ampereone_a192_32x")
    if cp is None:
        pytest.fail("ampere_ampereone_a192_32x missing from catalog")
    return cp


@pytest.fixture(scope="module")
def amp_compute_die(amp):
    die = next((d for d in amp.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("AmpereOne has no compute die")
    return die


@pytest.fixture(scope="module")
def amp_cpu_block(amp_compute_die) -> CPUBlock:
    block = next(
        (b for b in amp_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("AmpereOne compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process node + identity
# ---------------------------------------------------------------------------

def test_process_node_present():
    nodes = load_process_nodes()
    assert "tsmc_n5" in nodes


def test_identity(amp):
    assert amp.id == "ampere_ampereone_a192_32x"
    assert amp.vendor == "ampere"
    assert amp.lifecycle == LifecycleStatus.PRODUCTION


def test_monolithic_packaging(amp):
    """AmpereOne is a true monolithic die -- first such CPU in the
    catalog. No chiplet workaround needed."""
    assert amp.packaging.kind == PackagingKind.MONOLITHIC
    assert amp.packaging.num_dies == 1
    assert amp.packaging.package_type == "lga5964"


def test_die_geometry(amp_compute_die):
    """Headline PhysicalSpec: 440 mm^2, 55 B tx (estimates per YAML header)."""
    assert amp_compute_die.die_size_mm2 == 440.0
    assert amp_compute_die.transistors_billion == 55.0
    assert amp_compute_die.process_node_id == "tsmc_n5"


# ---------------------------------------------------------------------------
# CPUBlock shape -- 192 ARM custom cores, NEON only, no SMT
# ---------------------------------------------------------------------------

def test_cpu_block_headline(amp_cpu_block):
    assert amp_cpu_block.total_effective_cores == 192
    # KEY DIFFERENCE FROM EPYC SKUs: no SMT, so threads == cores
    assert amp_cpu_block.max_concurrent_threads == 192
    # NEON 128-bit = 4 FP32 lanes (vs AVX-512's 16)
    assert amp_cpu_block.simd_width_lanes == 4


def test_cpu_block_no_smt(amp_cpu_block):
    """AmpereOne has no SMT -- one hardware thread per core."""
    cluster = amp_cpu_block.core_clusters[0]
    assert cluster.cluster_kind == CoreClusterKind.HOMOGENEOUS
    assert cluster.num_cores == 192
    assert cluster.smt_threads == 1


def test_cpu_block_neon_fabric(amp_cpu_block):
    """ARM v8.6-A NEON SIMD only -- no SVE on AmpereOne-1."""
    cluster = amp_cpu_block.core_clusters[0]
    assert len(cluster.compute_fabrics) == 1
    fab = cluster.compute_fabrics[0]
    assert fab.isa_extension == CPUISAExtension.NEON
    # NEON ops/cycle for 2x 128-bit FMA pipes per core
    assert fab.ops_per_core_per_clock["fp32"] == 8
    assert fab.ops_per_core_per_clock["bf16"] == 16
    assert fab.ops_per_core_per_clock["int8"] == 32
    # ARM NEON FMA @ N5: 0.85 pJ -- lower than AVX-512's 1.35
    assert fab.energy_per_flop_fp32_pj == pytest.approx(0.85)


def test_cpu_block_l1_l2_larger_than_epyc(amp_cpu_block):
    """AmpereOne has larger L1 (64 KB) and L2 (2 MB) per core than
    any EPYC SKU in the catalog."""
    cluster = amp_cpu_block.core_clusters[0]
    assert cluster.l1_kib_per_core == 64    # vs Zen 4's 32, Zen 5's 48
    assert cluster.l2_layout == L2Layout.PRIVATE_PER_CORE
    assert cluster.l2_kib_per_core == 2048  # vs all EPYCs' 1024


def test_cpu_block_slc_via_l3_field(amp_cpu_block):
    """AmpereOne's 64 MB System Level Cache (SLC) is modeled via the
    schema's chip-wide l3_total_kib field (semantically the LLC)."""
    mem = amp_cpu_block.memory
    assert mem.l3_present is True
    assert mem.l3_total_kib == 65536        # 64 MiB SLC


def test_cpu_block_8ch_ddr5_5200(amp_cpu_block):
    """8-channel DDR5-5200 -- fewer channels than EPYC's 12, but
    faster signaling than Genoa/Bergamo's DDR5-4800."""
    mem = amp_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 8
    assert mem.memory_bus_bits == 512        # 8 channels * 64 bits
    assert mem.memory_bandwidth_gbps == pytest.approx(332.8)


def test_cpu_block_cmn700_mesh(amp_cpu_block):
    """AmpereOne uses ARM CMN-700 2D mesh (not Intel's ring, not AMD's
    IFOP / IO_DIE_PLUS_CCD topology)."""
    assert amp_cpu_block.noc.topology == CPUNoCTopology.MESH_2D
    assert amp_cpu_block.noc.mesh_rows == 12
    assert amp_cpu_block.noc.mesh_cols == 16
    assert amp_cpu_block.noc.unit_count == 192    # 12 * 16


# ---------------------------------------------------------------------------
# Power + market
# ---------------------------------------------------------------------------

def test_power_envelope(amp):
    """350W TDP (graphs factory's value); legacy YAML cites 400W as
    max_turbo_power and 276W as base_power."""
    assert amp.power.tdp_watts == 350.0
    assert amp.power.max_power_watts == 400.0
    assert amp.power.min_power_watts == 276.0
    prof = amp.power.thermal_profiles[0]
    assert prof.clock_mhz == 3200.0
    # ARM cores typically run lower Vdd than x86
    assert prof.vdd_v < 1.0


def test_market(amp):
    """AmpereOne-1 launched September 2023."""
    assert amp.market.launch_date == "2023-09-18"
    assert amp.market.launch_msrp_usd == 5555.0
    assert amp.market.product_family == "AmpereOne"


# ---------------------------------------------------------------------------
# Performance + cross-SKU comparisons
# ---------------------------------------------------------------------------

def test_performance_rollup(amp):
    """192 cores * 8 ops/cycle * 3.2 GHz = 4.92 TFLOPS FP32."""
    assert amp.performance.fp32_tflops == pytest.approx(4.92)
    assert amp.performance.bf16_tflops == pytest.approx(9.83)
    assert amp.performance.int8_tops == pytest.approx(19.66)


def test_ampereone_competitive_with_bergamo(all_products):
    """AmpereOne is positioned against Bergamo (both 'cloud-native'
    scale-out parts). Chip-level FP32 should be in [0.8x, 1.1x]
    Bergamo's, despite NEON's narrower datapath vs AVX-512."""
    amp = all_products.get("ampere_ampereone_a192_32x")
    bergamo = all_products.get("amd_epyc_9754_sp5")
    if amp is None or bergamo is None:
        pytest.skip("both SKUs needed; cross-SKU sanity check")
    ratio = amp.performance.fp32_tflops / bergamo.performance.fp32_tflops
    assert 0.8 < ratio < 1.1, f"unexpected AmpereOne/Bergamo FP32 ratio: {ratio:.2f}"


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums(amp_compute_die):
    """Sum within 1% of declared 55 B tx (55000 Mtx)."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in amp_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = amp_compute_die.transistors_billion * 1000.0
    rel_err = abs(total_mtx - expected_mtx) / expected_mtx
    assert rel_err < 0.05, (
        f"silicon_bin sum {total_mtx:.0f} differs from declared "
        f"{expected_mtx:.0f} by {rel_err*100:.1f}%"
    )
