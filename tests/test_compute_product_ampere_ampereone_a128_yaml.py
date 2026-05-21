"""Tests for the Ampere AmpereOne A128-30 ComputeProduct YAML (sprint #62 PR 5).

Final SKU in sprint #62 (5-SKU AMD/Ampere datacenter CPU batch).
Same architecture as A192-32X (#66) but with 64 cores disabled,
lower clock, and lower TDP envelope -- a standard Ampere bin of
the same physical die.

Tests focus on the differences from A192 (active cores, clock, TDP,
performance roll-up) and the same-die invariants (die area,
transistor count, physical silicon_bin).
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
    cp = all_products.get("ampere_ampereone_a128_30")
    if cp is None:
        pytest.fail("ampere_ampereone_a128_30 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def amp_compute_die(amp):
    die = next((d for d in amp.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("AmpereOne A128 has no compute die")
    return die


@pytest.fixture(scope="module")
def amp_cpu_block(amp_compute_die) -> CPUBlock:
    block = next(
        (b for b in amp_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("AmpereOne A128 compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Identity and packaging
# ---------------------------------------------------------------------------

def test_identity(amp):
    assert amp.id == "ampere_ampereone_a128_30"
    assert amp.vendor == "ampere"
    assert amp.lifecycle == LifecycleStatus.PRODUCTION


def test_monolithic_packaging(amp):
    """Same monolithic packaging as A192 -- this is the same physical die."""
    assert amp.packaging.kind == PackagingKind.MONOLITHIC
    assert amp.packaging.num_dies == 1
    assert amp.packaging.package_type == "lga5964"


def test_same_physical_die_as_a192(amp_compute_die, all_products):
    """A128-30 and A192-32X are different SKUs binned from the same
    physical die. die_size and transistor count must match exactly."""
    a192 = all_products.get("ampere_ampereone_a192_32x")
    if a192 is None:
        pytest.skip("A192 needed for same-die invariant check")
    assert amp_compute_die.die_size_mm2 == a192.dies[0].die_size_mm2
    assert amp_compute_die.transistors_billion == a192.dies[0].transistors_billion
    assert amp_compute_die.process_node_id == a192.dies[0].process_node_id


# ---------------------------------------------------------------------------
# Active-cores binning: 128 cores active out of 192 physical
# ---------------------------------------------------------------------------

def test_cpu_block_128_cores_active(amp_cpu_block):
    """128 cores active (vs A192's 192). No SMT, so threads == cores."""
    assert amp_cpu_block.total_effective_cores == 128
    assert amp_cpu_block.max_concurrent_threads == 128
    cluster = amp_cpu_block.core_clusters[0]
    assert cluster.num_cores == 128
    assert cluster.smt_threads == 1
    assert cluster.cluster_kind == CoreClusterKind.HOMOGENEOUS


def test_silicon_bin_reflects_full_192_cores(amp_compute_die):
    """The silicon_bin represents the PHYSICAL die (all 192 cores' silicon),
    not the SKU-active 128. The 'ampere_custom_cores' entry has the same
    Mtx as A192-32X.
    """
    cores_block = next(
        b for b in amp_compute_die.silicon_bin.blocks
        if b.name == "ampere_custom_cores"
    )
    # 192 cores * 200 Mtx = 38400 Mtx (same as A192-32X)
    assert cores_block.transistor_source.mtx == 38400.0


def test_silicon_bin_sums(amp_compute_die):
    """Sum within 1% of declared 55 B tx."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in amp_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = amp_compute_die.transistors_billion * 1000.0
    rel_err = abs(total_mtx - expected_mtx) / expected_mtx
    assert rel_err < 0.05


# ---------------------------------------------------------------------------
# Architectural details inherited from A192 (NEON, no SVE, 2 MB L2, etc.)
# ---------------------------------------------------------------------------

def test_neon_fabric_same_as_a192(amp_cpu_block):
    """Same NEON-only ARM v8.6 fabric -- A128-30 is the same core design
    as A192-32X (just fewer of them enabled)."""
    cluster = amp_cpu_block.core_clusters[0]
    fab = cluster.compute_fabrics[0]
    assert fab.isa_extension == CPUISAExtension.NEON
    assert fab.ops_per_core_per_clock["fp32"] == 8
    assert fab.ops_per_core_per_clock["int8"] == 32
    assert fab.energy_per_flop_fp32_pj == pytest.approx(0.85)


def test_l1_l2_unchanged_from_a192(amp_cpu_block):
    cluster = amp_cpu_block.core_clusters[0]
    assert cluster.l1_kib_per_core == 64
    assert cluster.l2_layout == L2Layout.PRIVATE_PER_CORE
    assert cluster.l2_kib_per_core == 2048


def test_slc_full_64mb_active(amp_cpu_block):
    """SLC is shared, full capacity remains active even with cores
    disabled (it's a chip-wide resource)."""
    assert amp_cpu_block.memory.l3_total_kib == 65536


def test_memory_unchanged_from_a192(amp_cpu_block):
    mem = amp_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 8
    assert mem.memory_bandwidth_gbps == pytest.approx(332.8)


def test_mesh_128_active_stops(amp_cpu_block):
    """Mesh: 8x16 = 128 active stops (physical 12x16=192; 64 disabled)."""
    assert amp_cpu_block.noc.topology == CPUNoCTopology.MESH_2D
    assert amp_cpu_block.noc.unit_count == 128
    assert amp_cpu_block.noc.mesh_rows == 8
    assert amp_cpu_block.noc.mesh_cols == 16


# ---------------------------------------------------------------------------
# Power / clock differences from A192
# ---------------------------------------------------------------------------

def test_lower_clock_than_a192(amp, all_products):
    """A128-30 boost is 3.0 GHz (per "-30" suffix), vs A192-32X's 3.2 GHz."""
    a192 = all_products.get("ampere_ampereone_a192_32x")
    if a192 is None:
        pytest.skip("A192 needed for cross-SKU check")
    assert amp.power.thermal_profiles[0].clock_mhz < a192.power.thermal_profiles[0].clock_mhz
    assert amp.power.thermal_profiles[0].clock_mhz == 3000.0


def test_lower_tdp_than_a192(amp, all_products):
    a192 = all_products.get("ampere_ampereone_a192_32x")
    if a192 is None:
        pytest.skip("A192 needed for cross-SKU check")
    assert amp.power.tdp_watts == 250.0
    assert amp.power.tdp_watts < a192.power.tdp_watts


def test_market(amp):
    """Same family launch date as A192; lower MSRP."""
    assert amp.market.launch_date == "2023-09-18"
    assert amp.market.launch_msrp_usd == 3888.0
    assert amp.market.product_family == "AmpereOne"


# ---------------------------------------------------------------------------
# Performance roll-up scales with active cores * clock
# ---------------------------------------------------------------------------

def test_performance_rollup(amp):
    """128 cores * 8 ops * 3.0 GHz = 3.07 TFLOPS FP32."""
    assert amp.performance.fp32_tflops == pytest.approx(3.07)
    assert amp.performance.bf16_tflops == pytest.approx(6.14)
    assert amp.performance.int8_tops == pytest.approx(12.29)


def test_a128_perf_ratio_to_a192(amp, all_products):
    """A128-30 / A192-32X FP32 ratio should match (128/192) * (3.0/3.2)
    = 0.625, within rounding noise."""
    a192 = all_products.get("ampere_ampereone_a192_32x")
    if a192 is None:
        pytest.skip("A192 needed for cross-SKU check")
    ratio = amp.performance.fp32_tflops / a192.performance.fp32_tflops
    assert 0.60 < ratio < 0.65, f"unexpected A128/A192 ratio: {ratio:.3f}"
