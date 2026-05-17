"""Tests for the first CGRA SKU YAML: stanford_plasticine_v2.

Catalog gain: 12 KPU + 2 GPU + 1 CPU + 3 NPU + 1 CGRA = 19 ComputeProducts.
First SKU to exercise:
  - ``BlockKind.CGRA`` and the full ``CGRABlock`` schema (landed in #34)
  - ``CGRAFabricKind.PCU_SPATIAL_DATAFLOW``
  - ``CGRAMemorySubsystem.has_host_dram=True`` path (DDR4 via host bus)
  - ``CGRABlock.reconfig_overhead_cycles`` (the defining CGRA Achilles heel)
  - The ``stanford`` vendor directory
  - ``LifecycleStatus.ENGINEERING_SAMPLE`` (research SKU)

Same shape as ``test_compute_product_v4_coral_edge_tpu_yaml.py`` and
its Hailo siblings.
"""

import pytest

from embodied_schemas import (
    CGRABlock,
    CGRAFabricKind,
    CGRANoCTopology,
    ComputeProduct,
    LifecycleStatus,
    PackagingKind,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import DataConfidence


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def plasticine(all_products) -> ComputeProduct:
    cp = all_products.get("stanford_plasticine_v2")
    if cp is None:
        pytest.fail("stanford_plasticine_v2 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def plasticine_compute_die(plasticine):
    die = next((d for d in plasticine.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Plasticine has no compute die")
    return die


@pytest.fixture(scope="module")
def plasticine_cgra_block(plasticine_compute_die) -> CGRABlock:
    block = next(
        (b for b in plasticine_compute_die.blocks if isinstance(b, CGRABlock)),
        None,
    )
    if block is None:
        pytest.fail("Plasticine compute die has no CGRABlock")
    return block


# ---------------------------------------------------------------------------
# Catalog: Plasticine v2 is the first CGRA SKU (first Stanford vendor)
# ---------------------------------------------------------------------------

def test_catalog_includes_plasticine(all_products):
    stanford_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "stanford")
    assert stanford_skus == ["stanford_plasticine_v2"]


def test_catalog_contains_plasticine(all_products):
    """At-least-one-CGRA check (subset semantics). The DPU follow-up
    grew the catalog past the original 19-product mark; the DPU PR's
    own contract test pins the new total."""
    counts_by_kind: dict[str, int] = {}
    for cp in all_products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    # CGRA count is the focus of this PR; pin >= 1 without locking
    # the total catalog size.
    assert counts_by_kind.get("cgra", 0) >= 1


def test_other_vendors_unaffected_by_plasticine_addition(all_products):
    """Additive guarantee: adding stanford/ vendor directory must not
    perturb stillwater/, nvidia/, intel/, hailo/ loading. google/
    loosened to >= 1 since the TPU v4 follow-up adds a 2nd google SKU."""
    stillwater = [s for s, cp in all_products.items() if cp.vendor == "stillwater"]
    nvidia = [s for s, cp in all_products.items() if cp.vendor == "nvidia"]
    intel = [s for s, cp in all_products.items() if cp.vendor == "intel"]
    hailo = [s for s, cp in all_products.items() if cp.vendor == "hailo"]
    google = [s for s, cp in all_products.items() if cp.vendor == "google"]
    assert len(stillwater) == 12
    assert len(nvidia) == 2
    assert len(intel) == 1
    assert len(hailo) == 2
    assert len(google) >= 1


# ---------------------------------------------------------------------------
# Plasticine identity / die / silicon_bin
# ---------------------------------------------------------------------------

def test_plasticine_identity(plasticine):
    assert plasticine.id == "stanford_plasticine_v2"
    assert plasticine.vendor == "stanford"
    assert plasticine.packaging.kind == PackagingKind.MONOLITHIC
    assert plasticine.packaging.package_type == "bare_die"
    # Engineering sample -- academic prototype, not commercial production
    assert plasticine.lifecycle == LifecycleStatus.ENGINEERING_SAMPLE


def test_plasticine_die_references_gf_28nm(plasticine_compute_die):
    """Plasticine v2 ships on GF 28nm SLP (same process as Coral; landed
    in embodied-schemas#32)."""
    die = plasticine_compute_die
    assert die.process_node_id == "gf_28nm"
    assert die.die_role.value == "compute"
    assert die.die_size_mm2 == pytest.approx(45.0, rel=0.05)
    assert die.transistors_billion == pytest.approx(0.52, rel=0.1)


def test_plasticine_silicon_bin_reconciles_with_die_total(plasticine_compute_die):
    die = plasticine_compute_die
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


def test_plasticine_silicon_bin_has_cgra_count_refs(plasticine_compute_die):
    """The CGRA paper exercise specified per-block count_refs:
    pcu_array, pmu_sram, shared_sram, noc_routers, config_fabric,
    host_phy, control_logic. config_fabric is CGRA-unique (no NPU has it)."""
    names = {b.name for b in plasticine_compute_die.silicon_bin.blocks}
    assert "pcu_array" in names
    assert "pmu_sram" in names
    assert "config_fabric" in names, "CGRA-specific config_fabric block missing"


# ---------------------------------------------------------------------------
# CGRABlock structural fields
# ---------------------------------------------------------------------------

def test_plasticine_block_is_cgra(plasticine_cgra_block):
    assert isinstance(plasticine_cgra_block, CGRABlock)
    assert plasticine_cgra_block.kind == "cgra"


def test_plasticine_pcu_hierarchy(plasticine_cgra_block):
    """32 PCUs with 8 MACs each -- Plasticine v2 design point."""
    block = plasticine_cgra_block
    assert block.num_pcus == 32
    assert block.macs_per_pcu == 8
    assert block.max_concurrent_models == 1


def test_plasticine_reconfig_overhead(plasticine_cgra_block):
    """1000 cycles for full-fabric reconfiguration -- the Achilles
    heel of CGRAs vs fixed-function NPUs."""
    block = plasticine_cgra_block
    assert block.reconfig_overhead_cycles == 1000
    # Plasticine v2 is whole-fabric only
    assert block.supports_partial_reconfig is False


def test_plasticine_pcu_spatial_dataflow_fabric(plasticine_cgra_block):
    """First SKU in catalog with NPUDataflowKind / CGRAFabricKind
    PCU_SPATIAL_DATAFLOW. 320 INT8 + 80 FP16 ops per PCU per clock."""
    block = plasticine_cgra_block
    assert len(block.compute_fabrics) == 1
    fabric = block.compute_fabrics[0]
    assert fabric.fabric_kind == CGRAFabricKind.PCU_SPATIAL_DATAFLOW
    assert fabric.ops_per_unit_per_clock["int8"] == 320
    assert fabric.ops_per_unit_per_clock["fp16"] == 80


def test_plasticine_multi_precision_includes_fp(plasticine_cgra_block):
    """CGRAs support FP via emulation (unlike INT-only NPUs); the
    energy_scaling dict carries the FP overhead so downstream estimates
    penalize FP workloads."""
    block = plasticine_cgra_block
    assert set(block.multi_precision_alu) == {"int8", "fp16", "fp32"}
    scaling = block.compute_fabrics[0].energy_scaling
    # FP entries are CGRA-distinctive (NPU fabrics have INT-only scaling)
    assert scaling.get("fp16", 0) > 1.0   # FP16 is more expensive than INT8 baseline
    assert scaling.get("fp32", 0) > 1.0


# ---------------------------------------------------------------------------
# Memory: first SKU with host_dram_* populated (vs NPU's external_dram_*)
# ---------------------------------------------------------------------------

def test_plasticine_memory_has_host_ddr4(plasticine_cgra_block):
    """First SKU to populate has_host_dram=True. Plasticine reaches
    DDR4 via the host bus (architecturally like Coral, schema-distinct
    from NPU's chip-attached external_dram)."""
    mem = plasticine_cgra_block.memory
    assert mem.has_host_dram is True
    assert mem.host_dram_type == MemoryType.DDR4
    assert mem.host_dram_size_gb == pytest.approx(4.0, rel=0.01)
    assert mem.host_dram_bandwidth_gbps == pytest.approx(12.8, rel=0.1)


def test_plasticine_on_chip_sram(plasticine_cgra_block):
    """64 KiB PMU per PCU + 2 MiB shared L2 = 4 MiB total on-chip SRAM."""
    mem = plasticine_cgra_block.memory
    assert mem.pmu_kib_per_pcu == 64
    assert mem.shared_sram_kib == 2 * 1024   # 2 MiB shared L2
    assert mem.coherence_protocol == "none"


# ---------------------------------------------------------------------------
# NoC: 4x8 mesh (the first MESH_2D under CGRA)
# ---------------------------------------------------------------------------

def test_plasticine_noc_is_mesh_2d_4x8_low_confidence(plasticine_cgra_block):
    noc = plasticine_cgra_block.noc
    assert noc.topology == CGRANoCTopology.MESH_2D
    assert noc.unit_count == 32
    assert noc.mesh_rows == 4
    assert noc.mesh_cols == 8
    assert noc.confidence == DataConfidence.THEORETICAL


# ---------------------------------------------------------------------------
# Power / performance / market
# ---------------------------------------------------------------------------

def test_plasticine_single_thermal_profile(plasticine):
    """15W envelope, passive air cooling. No DVFS on the academic prototype."""
    power = plasticine.power
    assert power.tdp_watts == 15.0
    assert power.default_thermal_profile == "default"
    profile_names = {p.name for p in power.thermal_profiles}
    assert profile_names == {"default"}
    assert power.thermal_profiles[0].cooling_solution_id == "passive_heatsink_large"
    assert power.thermal_profiles[0].clock_mhz == pytest.approx(1000.0)


def test_plasticine_chip_level_performance_matches_paper(plasticine):
    """10.24 TOPS INT8 (32 * 320 * 1 GHz). 1.28 TFLOPS emulated FP32.
    No INT4, no BF16."""
    perf = plasticine.performance
    assert perf.int8_tops == pytest.approx(10.24)
    assert perf.fp32_tflops == pytest.approx(1.28)
    assert perf.int4_tops == 0.0
    assert perf.bf16_tflops == 0.0


def test_plasticine_is_research_sku(plasticine):
    """Academic prototype -- not commercially available."""
    market = plasticine.market
    assert market.target_market == "research"
    assert market.is_available is False
    assert market.product_family == "Plasticine"


def test_plasticine_round_trips_through_serialize(plasticine):
    """Full ComputeProduct -> JSON -> ComputeProduct round-trip
    preserves CGRABlock discriminator dispatch and reconfig fields."""
    payload = plasticine.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    rebuilt_block = next(
        b for d in rebuilt.dies
        if d.die_role.value == "compute"
        for b in d.blocks
        if isinstance(b, CGRABlock)
    )
    assert rebuilt_block.num_pcus == 32
    assert rebuilt_block.macs_per_pcu == 8
    assert rebuilt_block.reconfig_overhead_cycles == 1000
    assert rebuilt_block.memory.has_host_dram is True
    assert rebuilt_block.compute_fabrics[0].fabric_kind == CGRAFabricKind.PCU_SPATIAL_DATAFLOW
