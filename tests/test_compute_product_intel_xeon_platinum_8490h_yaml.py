"""Tests for the Intel Xeon Platinum 8490H ComputeProduct YAML (sprint #68 PR 1).

First Xeon SKU and first datacenter Intel entry. Also the first
catalog SKU using Intel AMX (Advanced Matrix Extensions) -- a
distinct compute fabric alongside AVX-512 on the same core.

Same test shape as the AMD EPYC YAML tests.
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
def xeon(all_products) -> ComputeProduct:
    cp = all_products.get("intel_xeon_platinum_8490h")
    if cp is None:
        pytest.fail("intel_xeon_platinum_8490h missing from catalog")
    return cp


@pytest.fixture(scope="module")
def xeon_compute_die(xeon):
    die = next((d for d in xeon.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Xeon 8490H has no compute die")
    return die


@pytest.fixture(scope="module")
def xeon_cpu_block(xeon_compute_die) -> CPUBlock:
    block = next(
        (b for b in xeon_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("Xeon 8490H compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process node (intel_7 already in catalog from #24)
# ---------------------------------------------------------------------------

def test_process_node_present():
    nodes = load_process_nodes()
    assert "intel_7" in nodes


# ---------------------------------------------------------------------------
# Identity and packaging
# ---------------------------------------------------------------------------

def test_identity(xeon):
    assert xeon.id == "intel_xeon_platinum_8490h"
    assert xeon.vendor == "intel"
    assert xeon.lifecycle == LifecycleStatus.PRODUCTION


def test_packaging_is_xcc_chiplet(xeon):
    """8490H is the XCC (eXtreme Core Count) variant -- 4 tiles
    joined by Intel EMIB."""
    assert xeon.packaging.kind == PackagingKind.CHIPLET
    assert xeon.packaging.num_dies == 4    # XCC = 4 tiles
    assert xeon.packaging.package_type == "lga4677"


def test_die_geometry(xeon_compute_die):
    """4 tiles * ~400 mm^2 = ~1600 mm^2; 4 * 14.4 B tx = ~57.6 B."""
    assert xeon_compute_die.die_size_mm2 == 1600.0
    assert xeon_compute_die.transistors_billion == pytest.approx(57.6)
    assert xeon_compute_die.process_node_id == "intel_7"


# ---------------------------------------------------------------------------
# CPUBlock shape -- single homogeneous server-cluster
# ---------------------------------------------------------------------------

def test_cpu_block_headline(xeon_cpu_block):
    assert xeon_cpu_block.total_effective_cores == 60
    assert xeon_cpu_block.max_concurrent_threads == 120  # SMT=2
    # Full AVX-512 datapath (16 FP32 lanes, not double-pumped)
    assert xeon_cpu_block.simd_width_lanes == 16


def test_cpu_block_homogeneous_60_core(xeon_cpu_block):
    """60-core homogeneous server cluster (no hybrid layout)."""
    assert len(xeon_cpu_block.core_clusters) == 1
    cluster = xeon_cpu_block.core_clusters[0]
    assert cluster.cluster_kind == CoreClusterKind.HOMOGENEOUS
    assert cluster.num_cores == 60
    assert cluster.smt_threads == 2
    assert cluster.l2_layout == L2Layout.PRIVATE_PER_CORE
    # Sapphire Rapids enlarged L1D (48 KB) and L2 (2 MB) per core
    assert cluster.l1_kib_per_core == 48
    assert cluster.l2_kib_per_core == 2048


# ---------------------------------------------------------------------------
# Multi-fabric: AVX-512 + AMX on the same cluster (FIRST in catalog)
# ---------------------------------------------------------------------------

def test_cluster_has_avx512_and_amx_fabrics(xeon_cpu_block):
    """8490H is the first catalog SKU with multi-fabric (AVX-512 +
    AMX). Sapphire Rapids cores can issue both fabrics independently."""
    cluster = xeon_cpu_block.core_clusters[0]
    assert len(cluster.compute_fabrics) == 2
    fabric_isas = {f.isa_extension for f in cluster.compute_fabrics}
    assert fabric_isas == {
        CPUISAExtension.AVX512_BF16,
        CPUISAExtension.AMX_BF16,
    }


def test_avx512_fabric_full_width(xeon_cpu_block):
    """Intel server AVX-512 is full-width (2 pipes * 16 lanes = 32 FP32
    ops/cycle), unlike Zen 4's double-pumped 256-bit datapath."""
    cluster = xeon_cpu_block.core_clusters[0]
    avx = next(
        f for f in cluster.compute_fabrics
        if f.isa_extension == CPUISAExtension.AVX512_BF16
    )
    assert avx.ops_per_core_per_clock["fp32"] == 32   # 2x EPYC 9654's 16 (double-pumped)
    assert avx.ops_per_core_per_clock["bf16"] == 64
    assert avx.ops_per_core_per_clock["int8"] == 128
    assert avx.energy_per_flop_fp32_pj == pytest.approx(1.5)


def test_amx_fabric_massive_matmul_throughput(xeon_cpu_block):
    """AMX is the AI uplift fabric. 2 tiles per core: each tile does
    16x16x32 BF16 = 1024 ops or 16x16x64 INT8 = 2048 ops per cycle."""
    cluster = xeon_cpu_block.core_clusters[0]
    amx = next(
        f for f in cluster.compute_fabrics
        if f.isa_extension == CPUISAExtension.AMX_BF16
    )
    assert amx.ops_per_core_per_clock["bf16"] == 2048   # 2 tiles * 1024
    assert amx.ops_per_core_per_clock["int8"] == 4096   # 2 tiles * 2048
    # AMX is bf16/int8 only -- no FP32/FP64 on tile path
    assert "fp32" not in amx.ops_per_core_per_clock
    assert "fp64" not in amx.ops_per_core_per_clock


# ---------------------------------------------------------------------------
# Memory + NoC
# ---------------------------------------------------------------------------

def test_cpu_block_8ch_ddr5_4800(xeon_cpu_block):
    """8-channel DDR5-4800 (vs EPYC's 12-channel)."""
    mem = xeon_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 8
    assert mem.memory_bus_bits == 512                # 8 * 64
    assert mem.memory_bandwidth_gbps == pytest.approx(307.2)


def test_cpu_block_l3_distributed_across_60_cores(xeon_cpu_block):
    """112.5 MB L3 distributed across the mesh (1.875 MB per core)."""
    assert xeon_cpu_block.memory.l3_total_kib == 115200
    # 115200 / 1024 == 112.5 MB; / 60 cores == 1.875 MB/core
    assert xeon_cpu_block.memory.coherence_protocol == "snoopy_mesi"


def test_cpu_block_mesh_2d_topology(xeon_cpu_block):
    """Sapphire Rapids uses Intel's server mesh-2D NoC (not the AMD
    IO_DIE_PLUS_CCD topology, not the ARM CMN-700 mesh)."""
    assert xeon_cpu_block.noc.topology == CPUNoCTopology.MESH_2D
    assert xeon_cpu_block.noc.unit_count == 60


# ---------------------------------------------------------------------------
# Power + market
# ---------------------------------------------------------------------------

def test_power_envelope(xeon):
    """350W TDP (Sapphire Rapids server class)."""
    assert xeon.power.tdp_watts == 350.0
    assert xeon.power.max_power_watts == 420.0
    prof = xeon.power.thermal_profiles[0]
    assert prof.clock_mhz == 2900.0


def test_market(xeon):
    """Sapphire Rapids launched January 2023."""
    assert xeon.market.launch_date == "2023-01-10"
    assert xeon.market.launch_msrp_usd == 17000.0
    assert xeon.market.product_family == "Xeon Platinum"


# ---------------------------------------------------------------------------
# Performance roll-up
# ---------------------------------------------------------------------------

def test_amx_bf16_dominates_chip_perf(xeon):
    """AMX delivers the headline AI throughput.
    60 cores * 2048 BF16 ops/cycle * 2.9 GHz = 356.4 TFLOPS BF16."""
    assert xeon.performance.bf16_tflops == pytest.approx(356.4)
    # AVX-512 path: 60 * 32 * 2.9 = 5.57 TFLOPS FP32
    assert xeon.performance.fp32_tflops == pytest.approx(5.57)
    # AMX is ~22x AVX-512 BF16 throughput
    # AVX-512 BF16 would be: 60 * 64 * 2.9 = 11.14 TFLOPS;
    # AMX BF16 = 356.4 -> 356.4 / 11.14 = ~32x advantage
    # (Higher than the typical ~22x due to AMX tile-level reuse.)


def test_int8_amx_path(xeon):
    """AMX INT8 path: 60 cores * 4096 ops/cycle * 2.9 GHz = 712.9 TOPS."""
    assert xeon.performance.int8_tops == pytest.approx(712.9)


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation
# ---------------------------------------------------------------------------

def test_silicon_bin_sums(xeon_compute_die):
    """4 tiles * 14400 Mtx each = 57600 Mtx = 57.6 B tx."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in xeon_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = xeon_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)
