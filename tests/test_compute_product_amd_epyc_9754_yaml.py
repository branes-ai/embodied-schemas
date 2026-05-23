"""Tests for the AMD EPYC 9754 (Bergamo) ComputeProduct YAML.

Originally authored in sprint #62 PR 2 (#64) with a single-virtual-die
representation. **Rewritten for the v13 multi-die representation in
sprint #245 PR 4** -- the SECOND SKU to use the IOBlock kind (joins
EPYC 9654 from PR 3). The 8 Zen 4c CCDs are now aggregated into one
compute die (TSMC N5) and the Genoa IOD is a separate die (TSMC N6)
with its own IOBlock. The 8 IFOP links between CCDs and IOD are
modeled via Die.interconnects[] on the compute die.

Headline PhysicalSpec sums are preserved: 581.6 + 397 = 978.6 mm^2,
77.6 + 12 = 89.6 B tx (matches the prior single-virtual-die
representation that the downstream PhysicalSpec loader already
handles via die-level summation).

**Shared-IOD invariant**: Bergamo reuses the Genoa IOD silicon
unchanged. The ``genoa_iod`` die in this YAML is byte-identical to
the one in ``amd_epyc_9654_sp5.yaml``. Cross-SKU tests assert this
explicitly.
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
    cp = all_products.get("amd_epyc_9754_sp5")
    if cp is None:
        pytest.fail("amd_epyc_9754_sp5 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def epyc_compute_die(epyc):
    """Pick the compute die by role."""
    die = next((d for d in epyc.dies if d.die_role == DieRole.COMPUTE), None)
    if die is None:
        pytest.fail("EPYC 9754 has no compute die")
    return die


@pytest.fixture(scope="module")
def epyc_io_die(epyc):
    """Pick the IO die by role (new in v13 multi-die rewrite)."""
    die = next((d for d in epyc.dies if d.die_role == DieRole.IO), None)
    if die is None:
        pytest.fail("EPYC 9754 has no IO die (v13 multi-die representation)")
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


@pytest.fixture(scope="module")
def epyc_io_block(epyc_io_die) -> IOBlock:
    block = next(
        (b for b in epyc_io_die.blocks if isinstance(b, IOBlock)),
        None,
    )
    if block is None:
        pytest.fail("EPYC 9754 IO die has no IOBlock (v13 multi-die representation)")
    return block


# ---------------------------------------------------------------------------
# Process nodes: both tsmc_n5 (CCDs) and tsmc_n6 (IOD) now visible at the Die level
# ---------------------------------------------------------------------------

def test_process_nodes_present():
    """v13: both N5 and N6 process nodes resolve at the die level."""
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


def test_packaging_still_9_dies(epyc):
    """v13 invariant: packaging.num_dies still reflects the PHYSICAL
    chiplet count (8 CCDs + 1 IOD). The dies[] aggregation (2 entries)
    is the schema's modeling convention, distinct from physical reality."""
    assert epyc.packaging.kind == PackagingKind.CHIPLET
    assert epyc.packaging.num_dies == 9   # one fewer than 9654 (8+1 vs 12+1)
    assert epyc.packaging.package_type == "sp5"  # same SP5 socket as 9654


# ---------------------------------------------------------------------------
# Multi-die structure (v13 rewrite)
# ---------------------------------------------------------------------------

def test_dies_aggregation_is_two_entries(epyc):
    """v13: dies[] aggregates the 8 CCDs into 1 compute die + 1 IOD die."""
    assert len(epyc.dies) == 2
    die_roles = sorted(d.die_role.value for d in epyc.dies)
    assert die_roles == ["compute", "io"]


def test_compute_die_geometry(epyc_compute_die):
    """Compute die: 8 Zen 4c CCDs aggregated. 8 * 72.7 = 581.6 mm^2;
    8 * 9.7 B = 77.6 B tx; process node tsmc_n5."""
    assert epyc_compute_die.die_size_mm2 == pytest.approx(581.6)
    assert epyc_compute_die.transistors_billion == pytest.approx(77.6)
    assert epyc_compute_die.process_node_id == "tsmc_n5"


def test_io_die_geometry(epyc_io_die):
    """IO die: Genoa IOD (REUSED unchanged from 9654). 397 mm^2,
    ~12 B tx, process node tsmc_n6 (previously hidden in YAML
    comments; now visible at die level)."""
    assert epyc_io_die.die_id == "genoa_iod"   # same die_id as 9654
    assert epyc_io_die.die_size_mm2 == 397.0
    assert epyc_io_die.transistors_billion == 12.0
    assert epyc_io_die.process_node_id == "tsmc_n6"
    assert epyc_io_die.die_role == DieRole.IO


def test_die_sums_preserve_prior_headline_values(epyc):
    """v13 multi-die rewrite preserves headline PhysicalSpec values
    (which the downstream graphs loader sums across dies[]):
    581.6 + 397 = 978.6 mm^2; 77.6 + 12 = 89.6 B tx."""
    total_area = sum(d.die_size_mm2 for d in epyc.dies)
    total_tx = sum(d.transistors_billion for d in epyc.dies)
    assert total_area == pytest.approx(978.6)
    assert total_tx == pytest.approx(89.6)


# ---------------------------------------------------------------------------
# IFOP interconnects (v13 new modeling)
# ---------------------------------------------------------------------------

def test_compute_die_has_ifop_interconnects(epyc_compute_die):
    """v13 new: compute die's interconnects[] models the 8 IFOP links
    to the IOD (vs 12 on Genoa -- one IFOP per CCD, CCD count drops)."""
    assert len(epyc_compute_die.interconnects) == 1
    ifop = epyc_compute_die.interconnects[0]
    assert ifop.interconnect_id == "ifop_compute_to_iod"
    assert ifop.level == InterconnectLevel.DIE_TO_DIE
    assert ifop.topology == TopologyKind.POINT_TO_POINT
    assert ifop.num_links == 8   # one IFOP per CCD (vs 12 on Genoa)
    assert ifop.coherent is True
    assert ifop.per_link_bandwidth_gbps == pytest.approx(36.0)


def test_io_die_interconnects_empty(epyc_io_die):
    """IO die's interconnects[] is empty (IFOPs modeled from compute-die
    side to avoid double-counting). Matches 9654's convention."""
    assert epyc_io_die.interconnects == []


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
# IOBlock shape (v13 NEW) -- byte-identical to 9654 (shared physical IOD)
# ---------------------------------------------------------------------------

def test_io_block_kind_dispatches(epyc_io_block):
    """v13: discriminator dispatches to IOBlock."""
    assert epyc_io_block.kind == "io"


def test_io_block_memory_subsystem(epyc_io_block):
    """IOMemorySubsystem on the Genoa IOD: 12-channel DDR5-4800 with ECC.
    Same as 9654 (shared IOD)."""
    mem = epyc_io_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 12
    assert mem.memory_bus_bits == 768
    assert mem.memory_bandwidth_gbps == pytest.approx(460.8)
    assert mem.ecc_supported is True


def test_io_block_coherence_fabric_is_infinity_fabric(epyc_io_block):
    """Same INFINITY_FABRIC topology as 9654 (shared IOD silicon)."""
    fabric = epyc_io_block.coherence_fabric
    assert fabric.topology == IOFabricTopology.INFINITY_FABRIC
    assert fabric.unit_count == 12   # IOD-internal fabric stops; 9754 only populates 8
    assert fabric.bisection_bandwidth_gbps == 2048.0


def test_io_block_pcie_surface(epyc_io_block):
    """128 PCIe Gen5 lanes + CXL 1.1 (Genoa IOD's PCIe surface)."""
    assert epyc_io_block.pcie_lanes == 128
    assert epyc_io_block.pcie_generation == PCIeGen.PCIE_5
    assert epyc_io_block.cxl_supported is True
    assert epyc_io_block.cxl_version == "1.1"


def test_io_block_inter_socket_links(epyc_io_block):
    """4x AMD G-link inter-socket links (used in 2P configs). Same as 9654."""
    assert len(epyc_io_block.inter_socket_links) == 1
    glink = epyc_io_block.inter_socket_links[0]
    assert glink.name == "AMD G-link"
    assert glink.link_count == 4
    assert glink.bandwidth_per_link_gbps == 250.0


def test_io_block_security_and_management(epyc_io_block):
    """AMD PSP + PMC + boot ROM all populated on the Genoa IOD. Same as 9654."""
    assert epyc_io_block.security_processor_kind == "AMD PSP"
    assert epyc_io_block.power_management_controller is True
    assert epyc_io_block.boot_rom_present is True
    assert epyc_io_block.idle_power_watts == 40.0


# ---------------------------------------------------------------------------
# Cross-SKU shared-IOD invariant (Bergamo reuses Genoa IOD silicon)
# ---------------------------------------------------------------------------

def test_io_die_matches_9654_genoa_iod(all_products):
    """The Genoa IOD is REUSED unchanged in Bergamo. The genoa_iod
    die in 9754 must match the genoa_iod die in 9654 across all
    silicon-level fields (die_id, area, transistor count, process
    node, silicon_bin sum). This invariant pins the shared-IOD
    physical-reality assumption."""
    nine_seven = all_products.get("amd_epyc_9754_sp5")
    nine_six = all_products.get("amd_epyc_9654_sp5")
    if nine_seven is None or nine_six is None:
        pytest.skip("both SKUs needed for shared-IOD invariant check")

    iod_9754 = next(d for d in nine_seven.dies if d.die_role == DieRole.IO)
    iod_9654 = next(d for d in nine_six.dies if d.die_role == DieRole.IO)

    assert iod_9754.die_id == iod_9654.die_id == "genoa_iod"
    assert iod_9754.die_size_mm2 == iod_9654.die_size_mm2
    assert iod_9754.transistors_billion == iod_9654.transistors_billion
    assert iod_9754.process_node_id == iod_9654.process_node_id

    # silicon_bin block names must match across both SKUs (same physical
    # silicon, same canonical decomposition).
    names_9754 = sorted(b.name for b in iod_9754.silicon_bin.blocks)
    names_9654 = sorted(b.name for b in iod_9654.silicon_bin.blocks)
    assert names_9754 == names_9654


def test_io_block_matches_9654_io_block(all_products):
    """The IOBlock fields on Bergamo's IOD must be byte-identical to
    Genoa's (same physical silicon, same feature surface)."""
    nine_seven = all_products.get("amd_epyc_9754_sp5")
    nine_six = all_products.get("amd_epyc_9654_sp5")
    if nine_seven is None or nine_six is None:
        pytest.skip("both SKUs needed for shared-IOBlock invariant check")

    def io_block_of(cp):
        iod = next(d for d in cp.dies if d.die_role == DieRole.IO)
        return next(b for b in iod.blocks if isinstance(b, IOBlock))

    iob_9754 = io_block_of(nine_seven)
    iob_9654 = io_block_of(nine_six)

    assert iob_9754.pcie_lanes == iob_9654.pcie_lanes
    assert iob_9754.pcie_generation == iob_9654.pcie_generation
    assert iob_9754.cxl_supported == iob_9654.cxl_supported
    assert iob_9754.cxl_version == iob_9654.cxl_version
    assert iob_9754.security_processor_kind == iob_9654.security_processor_kind
    assert iob_9754.power_management_controller == iob_9654.power_management_controller
    assert iob_9754.boot_rom_present == iob_9654.boot_rom_present
    assert iob_9754.idle_power_watts == iob_9654.idle_power_watts
    assert iob_9754.coherence_fabric.topology == iob_9654.coherence_fabric.topology
    assert iob_9754.memory.memory_bandwidth_gbps == iob_9654.memory.memory_bandwidth_gbps
    assert iob_9754.memory.memory_controllers == iob_9654.memory.memory_controllers


def test_die_size_smaller_than_9654(all_products):
    """Cross-SKU sanity: Bergamo's package is ~17% smaller than Genoa's
    despite +33% cores (8 CCDs vs 12, sharing the same IOD). Compares
    chip-level die area sums across all dies."""
    nine_seven = all_products.get("amd_epyc_9754_sp5")
    nine_six = all_products.get("amd_epyc_9654_sp5")
    if nine_seven is None or nine_six is None:
        pytest.skip("both SKUs needed; this is a cross-SKU sanity check")
    bergamo_area = sum(d.die_size_mm2 for d in nine_seven.dies)
    genoa_area = sum(d.die_size_mm2 for d in nine_six.dies)
    assert bergamo_area < genoa_area


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
# Silicon-bin reconciliation (per-die after v13)
# ---------------------------------------------------------------------------

def test_compute_die_silicon_bin_sums_to_compute_tx(epyc_compute_die):
    """Compute die's silicon_bin Mtx sum must match compute die's
    transistors_billion (77.6 B = 77,600 Mtx)."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = epyc_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)


def test_io_die_silicon_bin_sums_to_io_tx(epyc_io_die):
    """IO die's silicon_bin decomposes the Genoa IOD into 6 sub-blocks
    summing to 12 B tx (shared with 9654)."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_io_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = epyc_io_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)


def test_io_die_uses_canonical_block_names(epyc_io_die):
    """Bergamo's IOD must use the canonical Genoa IOD silicon_bin
    decomposition (iod_imc_*, iod_pcie_*, iod_glink_*, iod_infinity_fabric,
    iod_psp_security, iod_pmc_*) established by 9654 in sprint #245 PR 3."""
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
