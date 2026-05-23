"""Tests for the AMD EPYC 9654 ComputeProduct YAML.

Originally authored in sprint #62 PR 1 (#63) with a single-virtual-die
representation (no IOBlock schema kind existed yet). **Rewritten for
the v13 multi-die representation in sprint #245 PR 3** -- the FIRST
SKU to use the new IOBlock kind. The 12 Zen 4 CCDs are now aggregated
into one compute die (TSMC N5) and the Genoa IOD is a separate die
(TSMC N6) with its own IOBlock. The 12 IFOP links between CCDs and
IOD are modeled via Die.interconnects[] on the compute die.

Headline PhysicalSpec sums are preserved: 792 + 397 = 1189 mm^2,
78.84 + 12 = 90.84 B tx (matches the prior single-virtual-die
representation that the downstream PhysicalSpec loader already
handles via die-level summation).
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
    cp = all_products.get("amd_epyc_9654_sp5")
    if cp is None:
        pytest.fail("amd_epyc_9654_sp5 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def epyc_compute_die(epyc):
    """Pick the compute die by role."""
    die = next((d for d in epyc.dies if d.die_role == DieRole.COMPUTE), None)
    if die is None:
        pytest.fail("EPYC 9654 has no compute die")
    return die


@pytest.fixture(scope="module")
def epyc_io_die(epyc):
    """Pick the IO die by role (new in v13 multi-die rewrite)."""
    die = next((d for d in epyc.dies if d.die_role == DieRole.IO), None)
    if die is None:
        pytest.fail("EPYC 9654 has no IO die (v13 multi-die representation)")
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


@pytest.fixture(scope="module")
def epyc_io_block(epyc_io_die) -> IOBlock:
    block = next(
        (b for b in epyc_io_die.blocks if isinstance(b, IOBlock)),
        None,
    )
    if block is None:
        pytest.fail("EPYC 9654 IO die has no IOBlock (v13 multi-die representation)")
    return block


# ---------------------------------------------------------------------------
# Process nodes: both tsmc_n5 (CCDs) and tsmc_n6 (IOD) now visible at the Die level
# ---------------------------------------------------------------------------

def test_process_nodes_present():
    """v13: both N5 and N6 process nodes resolve at the die level, no
    longer just in YAML comments."""
    nodes = load_process_nodes()
    assert "tsmc_n5" in nodes  # CCD process
    assert "tsmc_n6" in nodes  # IOD process (NOW visible in dies[])


# ---------------------------------------------------------------------------
# Top-level identity and packaging
# ---------------------------------------------------------------------------

def test_identity(epyc):
    assert epyc.id == "amd_epyc_9654_sp5"
    assert epyc.vendor == "amd"
    assert epyc.lifecycle == LifecycleStatus.PRODUCTION


def test_packaging_still_13_dies(epyc):
    """v13 invariant: packaging.num_dies still reflects the PHYSICAL
    chiplet count (12 CCDs + 1 IOD). The dies[] aggregation (2 entries)
    is the schema's modeling convention, distinct from physical reality."""
    assert epyc.packaging.kind == PackagingKind.CHIPLET
    assert epyc.packaging.num_dies == 13
    assert epyc.packaging.package_type == "sp5"


# ---------------------------------------------------------------------------
# Multi-die structure (v13 rewrite)
# ---------------------------------------------------------------------------

def test_dies_aggregation_is_two_entries(epyc):
    """v13: dies[] aggregates the 12 CCDs into 1 compute die + 1 IOD
    die. packaging.num_dies (13) still reflects physical chiplet count."""
    assert len(epyc.dies) == 2
    die_roles = sorted(d.die_role.value for d in epyc.dies)
    assert die_roles == ["compute", "io"]


def test_compute_die_geometry(epyc_compute_die):
    """Compute die: 12 Zen 4 CCDs aggregated. 12 * 66 = 792 mm^2;
    12 * 6.57 B = 78.84 B tx; process node tsmc_n5."""
    assert epyc_compute_die.die_size_mm2 == 792.0
    assert epyc_compute_die.transistors_billion == pytest.approx(78.84)
    assert epyc_compute_die.process_node_id == "tsmc_n5"


def test_io_die_geometry(epyc_io_die):
    """IO die: Genoa IOD. 397 mm^2, ~12 B tx, process node tsmc_n6
    (previously hidden in YAML comments; now visible at die level)."""
    assert epyc_io_die.die_size_mm2 == 397.0
    assert epyc_io_die.transistors_billion == 12.0
    assert epyc_io_die.process_node_id == "tsmc_n6"
    assert epyc_io_die.die_role == DieRole.IO


def test_die_sums_preserve_prior_headline_values(epyc):
    """v13 multi-die rewrite preserves headline PhysicalSpec values
    (which the downstream graphs loader sums across dies[]):
    792 + 397 = 1189 mm^2; 78.84 + 12 = 90.84 B tx."""
    total_area = sum(d.die_size_mm2 for d in epyc.dies)
    total_tx = sum(d.transistors_billion for d in epyc.dies)
    assert total_area == pytest.approx(1189.0)
    assert total_tx == pytest.approx(90.84)


# ---------------------------------------------------------------------------
# IFOP interconnects (v13 new modeling)
# ---------------------------------------------------------------------------

def test_compute_die_has_ifop_interconnects(epyc_compute_die):
    """v13 new: compute die's interconnects[] models the 12 IFOP links
    to the IOD. Previously empty (single-virtual-die representation had
    no cross-die links to model)."""
    assert len(epyc_compute_die.interconnects) == 1
    ifop = epyc_compute_die.interconnects[0]
    assert ifop.interconnect_id == "ifop_compute_to_iod"
    assert ifop.level == InterconnectLevel.DIE_TO_DIE
    assert ifop.topology == TopologyKind.POINT_TO_POINT
    assert ifop.num_links == 12  # one IFOP per CCD
    assert ifop.coherent is True
    assert ifop.per_link_bandwidth_gbps == pytest.approx(36.0)


def test_io_die_interconnects_empty(epyc_io_die):
    """IO die's interconnects[] is empty (the IFOPs are modeled from
    the compute-die side to avoid double-counting). Schema-design
    choice; either side could carry the entries."""
    assert epyc_io_die.interconnects == []


# ---------------------------------------------------------------------------
# CPUBlock shape (unchanged from prior YAML; same single homogeneous
# cluster of 96 Zen 4 cores on the compute die)
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
    """The CPUBlock's NoC still uses IO_DIE_PLUS_CCD topology (the
    AMD-style coherence fabric description from the CPU's perspective)."""
    assert epyc_cpu_block.noc.topology == CPUNoCTopology.IO_DIE_PLUS_CCD
    assert epyc_cpu_block.noc.unit_count == 12


# ---------------------------------------------------------------------------
# IOBlock shape (v13 NEW)
# ---------------------------------------------------------------------------

def test_io_block_first_in_catalog(epyc_io_block):
    """v13: EPYC 9654 is the FIRST SKU in the catalog to use IOBlock.
    Verify the block kind discriminator dispatches correctly."""
    assert epyc_io_block.kind == "io"


def test_io_block_memory_subsystem(epyc_io_block):
    """IOMemorySubsystem on the Genoa IOD: 12-channel DDR5-4800 with ECC."""
    mem = epyc_io_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_controllers == 12
    assert mem.memory_bus_bits == 768
    assert mem.memory_bandwidth_gbps == pytest.approx(460.8)
    assert mem.ecc_supported is True


def test_io_block_coherence_fabric_is_infinity_fabric(epyc_io_block):
    """First use of IOFabricTopology.INFINITY_FABRIC in the catalog."""
    fabric = epyc_io_block.coherence_fabric
    assert fabric.topology == IOFabricTopology.INFINITY_FABRIC
    assert fabric.unit_count == 12   # one stop per CCD
    assert fabric.bisection_bandwidth_gbps == 2048.0


def test_io_block_pcie_surface(epyc_io_block):
    """128 PCIe Gen5 lanes + CXL 1.1 (Genoa's PCIe surface)."""
    assert epyc_io_block.pcie_lanes == 128
    assert epyc_io_block.pcie_generation == PCIeGen.PCIE_5
    assert epyc_io_block.cxl_supported is True
    assert epyc_io_block.cxl_version == "1.1"


def test_io_block_inter_socket_links(epyc_io_block):
    """4x AMD G-link inter-socket links (used in 2P configs)."""
    assert len(epyc_io_block.inter_socket_links) == 1
    glink = epyc_io_block.inter_socket_links[0]
    assert glink.name == "AMD G-link"
    assert glink.link_count == 4
    assert glink.bandwidth_per_link_gbps == 250.0


def test_io_block_security_and_management(epyc_io_block):
    """AMD PSP + PMC + boot ROM all populated on the Genoa IOD."""
    assert epyc_io_block.security_processor_kind == "AMD PSP"
    assert epyc_io_block.power_management_controller is True
    assert epyc_io_block.boot_rom_present is True
    assert epyc_io_block.idle_power_watts == 40.0


# ---------------------------------------------------------------------------
# Power + market (unchanged from prior YAML)
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
    """Chip-level peak at 3.55 GHz all-core boost."""
    assert epyc.performance.fp32_tflops == pytest.approx(5.45)
    assert epyc.performance.bf16_tflops == pytest.approx(10.91)
    assert epyc.performance.int8_tops == pytest.approx(21.82)


# ---------------------------------------------------------------------------
# Silicon-bin reconciliation (per-die after v13)
# ---------------------------------------------------------------------------

def test_compute_die_silicon_bin_sums_to_compute_tx(epyc_compute_die):
    """Compute die's silicon_bin Mtx sum must match compute die's
    transistors_billion (78.84 B = 78,840 Mtx)."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_compute_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = epyc_compute_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)


def test_io_die_silicon_bin_sums_to_io_tx(epyc_io_die):
    """IO die's silicon_bin decomposes the Genoa IOD into 6 sub-blocks
    (IMC + PCIe + G-link + IF + PSP + PMC) summing to 12 B tx."""
    total_mtx = sum(
        b.transistor_source.mtx
        for b in epyc_io_die.silicon_bin.blocks
        if hasattr(b.transistor_source, "mtx")
    )
    expected_mtx = epyc_io_die.transistors_billion * 1000.0
    assert total_mtx == pytest.approx(expected_mtx)


def test_io_die_silicon_bin_has_canonical_block_names(epyc_io_die):
    """v13 introduces the canonical Genoa IOD silicon_bin decomposition.
    Future IODs (Turin, etc.) should follow the same block-name
    convention (iod_imc_*, iod_pcie_*, iod_glink_*, iod_infinity_fabric,
    iod_psp_security, iod_pmc_*)."""
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
