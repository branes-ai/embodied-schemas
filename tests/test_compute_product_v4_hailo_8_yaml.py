"""Tests for the first NPU SKU YAML: hailo_hailo_8.

Catalog gain: 12 KPU + 2 GPU + 1 CPU + 1 NPU = 16 products. Validates
the new YAML loads as a fully-formed ``ComputeProduct`` + ``NPUBlock``
with field values matching the Hailo-8 datasheet and the graphs-side
mapper.

Same test shape as the GPU and CPU data PR test files
(``test_compute_product_v2_jetson_agx_orin_yaml.py``,
``test_compute_product_v3_intel_i7_12700k_yaml.py``).
"""

import pytest

from embodied_schemas import (
    BlockKind,
    ComputeProduct,
    CPUBlock,
    GPUBlock,
    KPUBlock,
    LifecycleStatus,
    NPUBlock,
    NPUDataflowKind,
    NPUNoCTopology,
    NPUSramLayout,
    PackagingKind,
)
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import DataConfidence


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def hailo8(all_products) -> ComputeProduct:
    cp = all_products.get("hailo_hailo_8")
    if cp is None:
        pytest.fail("hailo_hailo_8 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def hailo8_compute_die(hailo8):
    """Pick the compute die by role, not positional index."""
    die = next((d for d in hailo8.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Hailo-8 has no compute die")
    return die


@pytest.fixture(scope="module")
def hailo8_npu_block(hailo8_compute_die) -> NPUBlock:
    """Pick NPUBlock by type. Resilient to future addition of co-
    located blocks on the same die."""
    block = next(
        (b for b in hailo8_compute_die.blocks if isinstance(b, NPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("Hailo-8 compute die has no NPUBlock")
    return block


# ---------------------------------------------------------------------------
# Catalog: Hailo-8 is the first NPU SKU
# ---------------------------------------------------------------------------

def test_catalog_includes_hailo_8(all_products):
    """Hailo-8 must remain present after subsequent hailo/ additions
    (Hailo-10H joined this directory in the follow-up data PR)."""
    hailo_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "hailo")
    assert "hailo_hailo_8" in hailo_skus


def test_other_vendors_unaffected_by_hailo_addition(all_products):
    """Additive guarantee: adding the hailo/ vendor directory must not
    perturb stillwater/, nvidia/, or intel/ loading. The Hailo-10H YAML
    follow-up keeps both hailo SKUs accounted for but should not change
    the other vendor counts either."""
    stillwater = [s for s, cp in all_products.items() if cp.vendor == "stillwater"]
    nvidia = [s for s, cp in all_products.items() if cp.vendor == "nvidia"]
    intel = [s for s, cp in all_products.items() if cp.vendor == "intel"]
    assert len(stillwater) == 12
    assert len(nvidia) == 2
    assert len(intel) == 2  # i7-12700k + Xeon 8490H (sprint #68 PR 1)


# ---------------------------------------------------------------------------
# Hailo-8 identity / die / silicon_bin
# ---------------------------------------------------------------------------

def test_hailo8_identity(hailo8):
    assert hailo8.id == "hailo_hailo_8"
    assert hailo8.vendor == "hailo"
    assert hailo8.packaging.kind == PackagingKind.MONOLITHIC
    assert hailo8.packaging.package_type == "m_dot_2"
    assert hailo8.lifecycle == LifecycleStatus.PRODUCTION


def test_hailo8_die_references_tsmc_n16(hailo8_compute_die):
    """Hailo-8 ships on TSMC N16. The catalog already has tsmc/n16.yaml
    from the KPU sprint -- no new process-node YAML for this PR."""
    die = hailo8_compute_die
    assert die.process_node_id == "tsmc_n16"
    assert die.die_role.value == "compute"
    assert die.die_size_mm2 == pytest.approx(30.0, rel=0.05)
    assert die.transistors_billion == pytest.approx(2.0, rel=0.1)


def test_hailo8_silicon_bin_reconciles_with_die_total(hailo8_compute_die):
    die = hailo8_compute_die
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
# NPUBlock structural fields
# ---------------------------------------------------------------------------

def test_hailo8_block_is_npu(hailo8_npu_block):
    assert isinstance(hailo8_npu_block, NPUBlock)
    assert hailo8_npu_block.kind == "npu"


def test_hailo8_dataflow_hierarchy(hailo8_npu_block):
    """32 dataflow units, scalar (lanes_per_unit=1)."""
    block = hailo8_npu_block
    assert block.num_dataflow_units == 32
    assert block.lanes_per_unit == 1
    assert block.max_concurrent_models == 1   # single compiled model
    # Higher default occupancy than GPU (0.3) / CPU (0.4)
    assert block.min_occupancy >= 0.8


def test_hailo8_structure_driven_dataflow_fabric(hailo8_npu_block):
    """Single fabric: structure-driven dataflow. 500 INT8 ops/unit/clock,
    1000 INT4 (2x packed)."""
    block = hailo8_npu_block
    assert len(block.compute_fabrics) == 1
    fabric = block.compute_fabrics[0]
    assert fabric.dataflow_kind == NPUDataflowKind.STRUCTURE_DRIVEN
    assert fabric.ops_per_unit_per_clock["int8"] == 500
    assert fabric.ops_per_unit_per_clock["int4"] == 1000


def test_hailo8_no_fp_precisions(hailo8_npu_block):
    """Hailo-8 is INT8/INT4 only -- no FP support. multi_precision_alu
    should not advertise any FP precisions."""
    block = hailo8_npu_block
    assert set(block.multi_precision_alu) == {"int8", "int4"}
    fabric_precisions = set(block.compute_fabrics[0].ops_per_unit_per_clock)
    assert fabric_precisions == {"int8", "int4"}


def test_hailo8_memory_is_sram_only(hailo8_npu_block):
    """Hailo-8 has NO external DRAM. All memory is on-chip SRAM."""
    mem = hailo8_npu_block.memory
    assert mem.has_external_dram is False
    assert mem.external_dram_type is None
    assert mem.external_dram_size_gb is None
    assert mem.external_dram_bandwidth_gbps is None
    # On-chip SRAM hierarchy
    assert mem.sram_kib_per_unit == 512   # 512 KiB per dataflow unit
    assert mem.shared_sram_kib == 8 * 1024   # 8 MiB shared
    assert mem.shared_sram_layout == NPUSramLayout.SHARED
    assert mem.coherence_protocol == "none"


def test_hailo8_noc_is_mesh_2d_low_confidence(hailo8_npu_block):
    """8x4 mesh of 32 dataflow units, low confidence (Hailo doesn't
    publish NoC details)."""
    noc = hailo8_npu_block.noc
    assert noc.topology == NPUNoCTopology.MESH_2D
    assert noc.unit_count == 32
    assert noc.mesh_rows == 8
    assert noc.mesh_cols == 4
    assert noc.confidence == DataConfidence.THEORETICAL


def test_hailo8_single_thermal_profile(hailo8):
    """Hailo-8 has a single fixed-frequency operating point (2.5W)."""
    power = hailo8.power
    assert power.tdp_watts == 2.5
    assert power.default_thermal_profile == "2.5W"
    profile_names = {p.name for p in power.thermal_profiles}
    assert profile_names == {"2.5W"}
    # Passive cooling
    assert power.thermal_profiles[0].cooling_solution_id == "passive_heatsink_small"


def test_hailo8_chip_level_performance_matches_marketing(hailo8):
    """26 TOPS INT8 (marketed), 52 TOPS INT4 (2x). No FP."""
    perf = hailo8.performance
    assert perf.int8_tops == 26.0
    assert perf.int4_tops == 52.0
    assert perf.fp32_tflops == 0.0   # not supported
    assert perf.bf16_tflops == 0.0   # not supported


def test_hailo8_round_trips_through_serialize(hailo8):
    """Full ComputeProduct -> JSON -> ComputeProduct round-trip
    preserves NPUBlock discriminator dispatch."""
    payload = hailo8.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    rebuilt_block = next(
        b for d in rebuilt.dies
        if d.die_role.value == "compute"
        for b in d.blocks
        if isinstance(b, NPUBlock)
    )
    assert rebuilt_block.num_dataflow_units == 32
    assert rebuilt_block.memory.has_external_dram is False
