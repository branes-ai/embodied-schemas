"""Tests for the first CPU SKU YAML: intel_core_i7_12700k.

Catalog gain: 12 KPU + 2 GPU + 1 CPU = 15 products. Validates the
new YAML loads as a fully-formed ``ComputeProduct`` + ``CPUBlock``
with field values matching the Intel ARK spec and the graphs-side
mapper. Also pins the new ``intel_7`` process node (first entry in
``data/process-nodes/intel/``).

Same test shape as the GPU YAML test files
(``test_compute_product_v2_jetson_agx_orin_yaml.py``,
``test_compute_product_v2_jetson_agx_thor_yaml.py``).
"""

import pytest

from embodied_schemas import (
    BlockKind,
    ComputeProduct,
    CoreClusterKind,
    CPUBlock,
    CPUISAExtension,
    CPUNoCTopology,
    GPUBlock,
    KPUBlock,
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
def i7(all_products) -> ComputeProduct:
    cp = all_products.get("intel_core_i7_12700k")
    if cp is None:
        pytest.fail("intel_core_i7_12700k missing from catalog")
    return cp


@pytest.fixture(scope="module")
def i7_compute_die(i7):
    """Pick the compute die by role, not positional index. Same
    pattern Thor adopted after PR #21 CodeRabbit triage."""
    die = next((d for d in i7.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("i7-12700K has no compute die")
    return die


@pytest.fixture(scope="module")
def i7_cpu_block(i7_compute_die) -> CPUBlock:
    """Pick CPUBlock by type. Resilient to future addition of an
    iGPU GPUBlock on the same die."""
    block = next(
        (b for b in i7_compute_die.blocks if isinstance(b, CPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("i7-12700K compute die has no CPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process node: intel_7 added by this PR
# ---------------------------------------------------------------------------

def test_intel_7_process_node_present():
    """Intel 7 is the new foundry directory's first process node."""
    nodes = load_process_nodes()
    assert "intel_7" in nodes
    n = nodes["intel_7"]
    assert n.foundry.value == "intel"
    assert n.node_nm == 10   # Intel 7 == 10ESF; despite the "7" suffix
    # SRAM bitcell is slightly behind TSMC N5 (0.0312 um^2 vs 0.021)
    assert n.densities["sram_hd"].mtx_per_mm2 == 260.0


def test_intel_foundry_directory_exists():
    """First-of-its-kind: intel/ vendor directory under process-nodes/."""
    from pathlib import Path
    from embodied_schemas.loaders import get_data_dir
    intel_dir = get_data_dir() / "process-nodes" / "intel"
    assert intel_dir.is_dir(), f"Expected intel/ process-node directory at {intel_dir}"
    yamls = list(intel_dir.glob("*.yaml"))
    assert len(yamls) >= 1
    assert any(y.name == "intel_7.yaml" for y in yamls)


# ---------------------------------------------------------------------------
# Catalog: i7 sits alongside KPU + GPU SKUs
# ---------------------------------------------------------------------------

def test_catalog_now_includes_one_cpu_sku(all_products):
    """v3 first data PR: catalog gains the first CPU SKU."""
    intel_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "intel")
    assert intel_skus == ["intel_core_i7_12700k"]


def test_catalog_has_15_total_products(all_products):
    """Tight: 12 KPU + 2 GPU + 1 CPU = 15. Tighten so future additions
    fire as deliberate-update reminders."""
    counts_by_kind: dict[str, int] = {}
    for cp in all_products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    # v4 data PR (#26-pending) adds hailo_hailo_8 as the first NPU SKU;
    # the catalog total grew from 15 to 16 then. This assertion
    # tolerates that addition while still pinning the KPU/GPU/CPU counts.
    expected_subset = {"kpu": 12, "gpu": 2, "cpu": 1}
    for kind, count in expected_subset.items():
        assert counts_by_kind.get(kind, 0) == count, (
        f"unexpected catalog composition: {counts_by_kind}"
    )


def test_kpu_and_gpu_skus_unaffected_by_cpu_addition(all_products):
    """Additive guarantee: adding intel/ vendor directory must not
    perturb stillwater/ or nvidia/ loading."""
    stillwater = [s for s, cp in all_products.items() if cp.vendor == "stillwater"]
    nvidia = [s for s, cp in all_products.items() if cp.vendor == "nvidia"]
    assert len(stillwater) == 12
    assert len(nvidia) == 2


# ---------------------------------------------------------------------------
# i7-12700K identity / die / silicon_bin reconciliation
# ---------------------------------------------------------------------------

def test_i7_identity(i7):
    assert i7.id == "intel_core_i7_12700k"
    assert i7.vendor == "intel"
    assert i7.packaging.kind == PackagingKind.MONOLITHIC
    assert i7.packaging.num_dies == 1
    assert i7.lifecycle == LifecycleStatus.PRODUCTION


def test_i7_die_references_intel_7_process(i7_compute_die):
    """Alder Lake die ships on Intel 7 (= 10nm Enhanced SuperFin)."""
    die = i7_compute_die
    assert die.process_node_id == "intel_7"
    assert die.die_size_mm2 == pytest.approx(215.0, rel=0.05)
    assert die.transistors_billion == pytest.approx(22.0, rel=0.1)


def test_i7_silicon_bin_reconciles_with_die_total(i7_compute_die):
    """Sanity: silicon_bin sum within 10% of declared die-level transistor count."""
    die = i7_compute_die
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
# CPUBlock structural fields
# ---------------------------------------------------------------------------

def test_i7_block_is_cpu(i7_cpu_block):
    assert isinstance(i7_cpu_block, CPUBlock)
    assert i7_cpu_block.kind == "cpu"


def test_i7_hybrid_clusters(i7_cpu_block):
    """8 Golden Cove P-cores + 4 Gracemont E-cores in two clusters."""
    block = i7_cpu_block
    assert len(block.core_clusters) == 2
    kinds = {c.cluster_kind for c in block.core_clusters}
    assert kinds == {CoreClusterKind.PERFORMANCE, CoreClusterKind.EFFICIENT}

    p_cluster = next(c for c in block.core_clusters
                     if c.cluster_kind == CoreClusterKind.PERFORMANCE)
    e_cluster = next(c for c in block.core_clusters
                     if c.cluster_kind == CoreClusterKind.EFFICIENT)

    assert p_cluster.num_cores == 8
    assert p_cluster.smt_threads == 2     # P-core SMT
    assert e_cluster.num_cores == 4
    assert e_cluster.smt_threads == 1     # E-core no SMT

    # Effective cores = 8 + int(4 * 0.6) = 10
    assert block.total_effective_cores == 10
    # max_concurrent_threads = 8*2 + 4*1 = 20
    assert block.max_concurrent_threads == 20


def test_i7_l2_layouts_differ_per_cluster(i7_cpu_block):
    """P-cores have 1.25 MB private L2 each (PRIVATE_PER_CORE).
    E-cores share 2 MB across the 4-cluster (SHARED_PER_CLUSTER).
    This is the L2Layout enum's whole reason for existing."""
    block = i7_cpu_block
    p = next(c for c in block.core_clusters
             if c.cluster_kind == CoreClusterKind.PERFORMANCE)
    e = next(c for c in block.core_clusters
             if c.cluster_kind == CoreClusterKind.EFFICIENT)

    assert p.l2_layout == L2Layout.PRIVATE_PER_CORE
    assert p.l2_kib_per_core == 1280   # 1.25 MB
    assert p.l2_kib_shared == 0

    assert e.l2_layout == L2Layout.SHARED_PER_CLUSTER
    assert e.l2_kib_per_core == 0
    assert e.l2_kib_shared == 2048     # 2 MB shared across 4 E-cores


def test_i7_avx_vnni_fabric_on_both_clusters(i7_cpu_block):
    """Both P and E clusters carry AVX-VNNI fabrics (the dominant
    path for INT8 matmul on Alder Lake). P-fabric is 2-FMA-pipe;
    E-fabric is 1-FMA-pipe (half the per-core throughput)."""
    for cluster in i7_cpu_block.core_clusters:
        assert len(cluster.compute_fabrics) == 1
        fabric = cluster.compute_fabrics[0]
        assert fabric.isa_extension == CPUISAExtension.AVX_VNNI

    p = next(c for c in i7_cpu_block.core_clusters
             if c.cluster_kind == CoreClusterKind.PERFORMANCE)
    e = next(c for c in i7_cpu_block.core_clusters
             if c.cluster_kind == CoreClusterKind.EFFICIENT)
    assert p.compute_fabrics[0].ops_per_core_per_clock["fp32"] == 16   # 2 FMA pipes
    assert e.compute_fabrics[0].ops_per_core_per_clock["fp32"] == 8    # 1 FMA pipe
    assert p.compute_fabrics[0].ops_per_core_per_clock["int8"] == 64   # AVX-VNNI
    assert e.compute_fabrics[0].ops_per_core_per_clock["int8"] == 32


def test_i7_memory_subsystem(i7_cpu_block):
    """DDR5-4800 dual-channel; 25 MB shared L3 LLC; no L4;
    snoopy_mesi coherence."""
    mem = i7_cpu_block.memory
    assert mem.memory_type == MemoryType.DDR5
    assert mem.memory_size_gb == 64.0
    assert mem.memory_bandwidth_gbps == pytest.approx(76.8)
    assert mem.l3_present is True
    assert mem.l3_total_kib == 25 * 1024
    assert mem.l4_present is False
    assert mem.l4_total_kib == 0
    assert mem.coherence_protocol == "snoopy_mesi"


def test_i7_noc_is_double_ring(i7_cpu_block):
    """Alder Lake uses a double ring bus (data + snoop)."""
    noc = i7_cpu_block.noc
    assert noc.topology == CPUNoCTopology.DOUBLE_RING
    assert noc.stop_count == 12   # 8 P + 4 E ring stops


def test_i7_simd_efficiency_carries_op_kind_map(i7_cpu_block):
    """CPU-specific concept: vectorization friendliness per op kind."""
    eff = i7_cpu_block.simd_efficiency_by_op_kind
    assert eff["elementwise"] == 0.95
    assert eff["matrix"] == 0.80
    assert eff["default"] == 0.70


def test_i7_round_trips_through_serialize(i7):
    """Full ComputeProduct -> JSON -> ComputeProduct round-trip
    preserves CPUBlock discriminator dispatch."""
    payload = i7.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    rebuilt_block = next(
        b for d in rebuilt.dies
        if d.die_role.value == "compute"
        for b in d.blocks
        if isinstance(b, CPUBlock)
    )
    assert rebuilt_block.total_effective_cores == 10
    assert rebuilt_block.max_concurrent_threads == 20
