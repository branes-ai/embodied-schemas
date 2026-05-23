"""Tests for the AMD EPYC 9965 (Turin Dense) ComputeProduct YAML.

Originally authored in sprint #62 PR 3 (#65) with a single-virtual-die
representation. **Rewritten for the v13 multi-die representation in
sprint #245 PR 5** -- the THIRD SKU to use the IOBlock kind, and the
FIRST to introduce a NEW IOD silicon family (Turin IOD, distinct from
the Genoa IOD shared between EPYC 9654 / 9754).

The 12 Zen 5c CCDs are now aggregated into one compute die (TSMC N4P
placeholder) and the Turin IOD is a separate die (TSMC N6) with its
own IOBlock. The 12 IFOP links between CCDs and IOD are modeled via
Die.interconnects[] on the compute die.

Headline PhysicalSpec sums are preserved: 876 + 400 = 1276 mm^2,
132 + 13 = 145 B tx (matches the prior single-virtual-die
representation that the downstream PhysicalSpec loader already
handles via die-level summation).

**Distinct-IOD invariant**: Turin IOD is silicon-level distinct from
Genoa IOD. Cross-SKU tests pin the deltas explicitly: DDR5-6000 vs
DDR5-4800, CXL 2.0 vs CXL 1.1, ~13 B vs ~12 B transistors.
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    CoreClusterKind,
    CPUBlock,
    CPUISAExtension,
    CPUNoCTopology,
    DieRole,
    InterconnectLevel,
    IOBlock,
    IOFabricTopology,
    L2Layout,
    LifecycleStatus,
    PackagingKind,
    PCIeGen,
    TopologyKind,
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
    """Pick the compute die by role."""
    die = next((d for d in epyc.dies if d.die_role == DieRole.COMPUTE), None)
    if die is None:
        pytest.fail("EPYC 9965 has no compute die")
    return die


@pytest.fixture(scope="module")
def epyc_io_die(epyc):
    """Pick the IO die by role (new in v13 multi-die rewrite)."""
    die = next((d for d in epyc.dies if d.die_role == DieRole.IO), None)
    if die is None:
        pytest.fail("EPYC 9965 has no IO die (v13 multi-die representation)")
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


@pytest.fixture(scope="module")
def epyc_io_block(epyc_io_die) -> IOBlock:
    block = next(
        (b for b in epyc_io_die.blocks if isinstance(b, IOBlock)),
        None,
    )
    if block is None:
        pytest.fail("EPYC 9965 IO die has no IOBlock (v13 multi-die representation)")
    return block


# ---------------------------------------------------------------------------
# Process nodes: tsmc_n4p (compute placeholder; N3E is the more-accurate
# node per AMD Oct 2024 but not yet in catalog) + tsmc_n6 (Turin IOD).
# ---------------------------------------------------------------------------

def test_process_nodes_present():
    nodes = load_process_nodes()
    assert "tsmc_n4p" in nodes  # Zen 5c CCDs (placeholder; N3E pending)
    assert "tsmc_n6" in nodes   # Turin IOD


# ---------------------------------------------------------------------------
# Top-level identity and packaging
# ---------------------------------------------------------------------------

def test_identity(epyc):
    assert epyc.id == "amd_epyc_9965_sp5"
    assert epyc.vendor == "amd"
    assert epyc.lifecycle == LifecycleStatus.PRODUCTION


def test_packaging_still_13_dies(epyc):
    """v13 invariant: packaging.num_dies still reflects the PHYSICAL
    chiplet count (12 CCDs + 1 IOD). The dies[] aggregation (2 entries)
    is the schema's modeling convention."""
    assert epyc.packaging.kind == PackagingKind.CHIPLET
    assert epyc.packaging.num_dies == 13       # same chiplet count as 9654
    assert epyc.packaging.package_type == "sp5"


# ---------------------------------------------------------------------------
# Multi-die structure (v13 rewrite)
# ---------------------------------------------------------------------------

def test_dies_aggregation_is_two_entries(epyc):
    """v13: dies[] aggregates the 12 CCDs into 1 compute die + 1 IOD die."""
    assert len(epyc.dies) == 2
    die_roles = sorted(d.die_role.value for d in epyc.dies)
    assert die_roles == ["compute", "io"]


def test_compute_die_geometry(epyc_compute_die):
    """Compute die: 12 Zen 5c CCDs aggregated. 12 * 73 = 876 mm^2;
    12 * 11 B = 132 B tx; process node tsmc_n4p (N3E pending)."""
    assert epyc_compute_die.die_size_mm2 == 876.0
    assert epyc_compute_die.transistors_billion == 132.0
    assert epyc_compute_die.process_node_id == "tsmc_n4p"


def test_io_die_geometry(epyc_io_die):
    """IO die: NEW Turin IOD (distinct from Genoa IOD). 400 mm^2,
    ~13 B tx, process node tsmc_n6."""
    assert epyc_io_die.die_id == "turin_iod"   # distinct from Genoa's "genoa_iod"
    assert epyc_io_die.die_size_mm2 == 400.0
    assert epyc_io_die.transistors_billion == 13.0
    assert epyc_io_die.process_node_id == "tsmc_n6"
    assert epyc_io_die.die_role == DieRole.IO


def test_die_sums_preserve_prior_headline_values(epyc):
    """v13 multi-die rewrite preserves headline PhysicalSpec values
    (which the downstream graphs loader sums across dies[]):
    876 + 400 = 1276 mm^2; 132 + 13 = 145 B tx."""
    total_area = sum(d.die_size_mm2 for d in epyc.dies)
    total_tx = sum(d.transistors_billion for d in epyc.dies)
    assert total_area == pytest.approx(1276.0)
    assert total_tx == pytest.approx(145.0)


# ---------------------------------------------------------------------------
# IFOP interconnects (v13 new modeling)
# ---------------------------------------------------------------------------

def test_compute_die_has_ifop_interconnects(epyc_compute_die):
    """v13 new: compute die's interconnects[] models the 12 IFOP links
    to the Turin IOD (same SerDes as Genoa, same per-link bandwidth)."""
    assert len(epyc_compute_die.interconnects) == 1
    ifop = epyc_compute_die.interconnects[0]
    assert ifop.interconnect_id == "ifop_compute_to_iod"
    assert ifop.level == InterconnectLevel.DIE_TO_DIE
    assert ifop.topology == TopologyKind.POINT_TO_POINT
    assert ifop.num_links == 12   # one IFOP per CCD (12 CCDs in Turin Dense)
    assert ifop.coherent is True
    assert ifop.per_link_bandwidth_gbps == pytest.approx(36.0)


def test_io_die_interconnects_empty(epyc_io_die):
    """IO die's interconnects[] is empty (IFOPs modeled from compute-die
    side to avoid double-counting). Matches Genoa/Bergamo convention."""
    assert epyc_io_die.interconnects == []


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
    """Same ops/clock convention as Zen 4 YAMLs; Zen 5's full-width
    FMA datapath sustains these per cycle vs Zen 4's amortization."""
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
# IOBlock shape (v13 NEW) -- Turin IOD with DDR5-6000 + CXL 2.0
# ---------------------------------------------------------------------------

def test_io_block_kind_dispatches(epyc_io_block):
    """v13: discriminator dispatches to IOBlock."""
    assert epyc_io_block.kind == "io"


def test_io_block_memory_subsystem_ddr5_6000(epyc_io_block):
    """IOMemorySubsystem on the Turin IOD: 12-channel DDR5-6000 with ECC.
    576 GB/s peak bandwidth (vs Genoa IOD's 460.8 GB/s)."""
    mem = epyc_io_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 12
    assert mem.memory_bus_bits == 768
    assert mem.memory_bandwidth_gbps == pytest.approx(576.0)
    assert mem.ecc_supported is True


def test_io_block_coherence_fabric_is_infinity_fabric(epyc_io_block):
    """INFINITY_FABRIC topology (same family as Genoa, wider routers)."""
    fabric = epyc_io_block.coherence_fabric
    assert fabric.topology == IOFabricTopology.INFINITY_FABRIC
    assert fabric.unit_count == 12   # one stop per CCD
    assert fabric.bisection_bandwidth_gbps == 2048.0


def test_io_block_pcie_surface_cxl_2_0(epyc_io_block):
    """128 PCIe Gen5 lanes + CXL 2.0 (vs Genoa IOD's CXL 1.1).
    CXL 2.0 uplift adds memory pooling semantics."""
    assert epyc_io_block.pcie_lanes == 128
    assert epyc_io_block.pcie_generation == PCIeGen.PCIE_5
    assert epyc_io_block.cxl_supported is True
    assert epyc_io_block.cxl_version == "2.0"


def test_io_block_inter_socket_links(epyc_io_block):
    """4x AMD G-link inter-socket links (used in 2P configs).
    Same SerDes signaling as Genoa."""
    assert len(epyc_io_block.inter_socket_links) == 1
    glink = epyc_io_block.inter_socket_links[0]
    assert glink.name == "AMD G-link"
    assert glink.link_count == 4
    assert glink.bandwidth_per_link_gbps == 250.0


def test_io_block_security_and_management(epyc_io_block):
    """AMD PSP + PMC + boot ROM populated on the Turin IOD.
    Idle power is ~5W higher than Genoa IOD (DDR5-6000 PHY refresh)."""
    assert epyc_io_block.security_processor_kind == "AMD PSP"
    assert epyc_io_block.power_management_controller is True
    assert epyc_io_block.boot_rom_present is True
    assert epyc_io_block.idle_power_watts == 45.0   # vs Genoa IOD's 40.0


# ---------------------------------------------------------------------------
# Cross-SKU distinct-IOD invariants (Turin IOD vs Genoa IOD)
# ---------------------------------------------------------------------------

def test_io_die_distinct_from_genoa_iod(all_products):
    """Turin IOD is silicon-level distinct from Genoa IOD. Pin the
    deltas explicitly so accidental re-unification (e.g., copy-paste
    drift) is caught.

    Same family (Infinity Fabric, ARM PSP, 12-channel DDR5, 128 PCIe
    Gen5, 4x G-link) but distinct die_id, transistor count, and
    memory/CXL specs."""
    turin = all_products.get("amd_epyc_9965_sp5")
    genoa = all_products.get("amd_epyc_9654_sp5")
    if turin is None or genoa is None:
        pytest.skip("both SKUs needed for distinct-IOD invariant check")

    turin_iod = next(d for d in turin.dies if d.die_role == DieRole.IO)
    genoa_iod = next(d for d in genoa.dies if d.die_role == DieRole.IO)

    # Distinct die_id (different physical silicon).
    assert turin_iod.die_id != genoa_iod.die_id
    assert turin_iod.die_id == "turin_iod"
    assert genoa_iod.die_id == "genoa_iod"

    # Distinct transistor counts (~13 B vs ~12 B).
    assert turin_iod.transistors_billion > genoa_iod.transistors_billion

    # Same process node family (both N6).
    assert turin_iod.process_node_id == genoa_iod.process_node_id == "tsmc_n6"


def test_io_block_distinct_from_genoa_io_block(all_products):
    """Turin IOBlock has distinct values from Genoa IOBlock on the
    fields that changed at the silicon level: memory bandwidth,
    CXL version, idle power."""
    turin = all_products.get("amd_epyc_9965_sp5")
    genoa = all_products.get("amd_epyc_9654_sp5")
    if turin is None or genoa is None:
        pytest.skip("both SKUs needed for distinct-IOBlock invariant check")

    def io_block_of(cp):
        iod = next(d for d in cp.dies if d.die_role == DieRole.IO)
        return next(b for b in iod.blocks if isinstance(b, IOBlock))

    iob_turin = io_block_of(turin)
    iob_genoa = io_block_of(genoa)

    # Memory: Turin DDR5-6000 > Genoa DDR5-4800
    assert iob_turin.memory.memory_bandwidth_gbps > iob_genoa.memory.memory_bandwidth_gbps
    assert iob_turin.memory.memory_bandwidth_gbps == pytest.approx(576.0)
    assert iob_genoa.memory.memory_bandwidth_gbps == pytest.approx(460.8)

    # CXL spec uplift: 2.0 vs 1.1
    assert iob_turin.cxl_version == "2.0"
    assert iob_genoa.cxl_version == "1.1"

    # Idle power: Turin IOD slightly higher (DDR5-6000 PHY refresh)
    assert iob_turin.idle_power_watts > iob_genoa.idle_power_watts

    # Same family: same coherence topology, same PSP kind, same PCIe gen
    assert iob_turin.coherence_fabric.topology == iob_genoa.coherence_fabric.topology
    assert iob_turin.security_processor_kind == iob_genoa.security_processor_kind
    assert iob_turin.pcie_generation == iob_genoa.pcie_generation
    assert iob_turin.pcie_lanes == iob_genoa.pcie_lanes


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
# Silicon-bin reconciliation (per-die after v13)
# ---------------------------------------------------------------------------

def test_compute_die_silicon_bin_sums_to_compute_tx(epyc_compute_die):
    """Compute die's silicon_bin Mtx sum must match compute die's
    transistors_billion (132 B = 132,000 Mtx)."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = epyc_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)


def test_io_die_silicon_bin_sums_to_io_tx(epyc_io_die):
    """IO die's silicon_bin decomposes the Turin IOD into 6 sub-blocks
    summing to 13 B tx."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_io_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = epyc_io_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)


def test_io_die_uses_canonical_block_names(epyc_io_die):
    """Turin IOD must use the canonical Genoa-class IOD silicon_bin
    decomposition (iod_imc_*, iod_pcie_*, iod_glink_*, iod_infinity_fabric,
    iod_psp_security, iod_pmc_*) established by 9654 in sprint #245 PR 3.
    Future IODs (Intel UPI-class, future AMD revisions) should follow."""
    names = {b.name for b in epyc_io_die.silicon_bin.blocks}
    expected = {
        "iod_imc_ddr5_phy",
        "iod_pcie_gen5_phy",
        "iod_glink_phy",
        "iod_infinity_fabric",
        "iod_psp_security",
        "iod_pmc_smu_misc",
    }
    assert names == expected
