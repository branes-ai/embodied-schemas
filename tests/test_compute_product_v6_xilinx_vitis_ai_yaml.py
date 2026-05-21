"""Tests for the first DPU SKU YAML: xilinx_vitis_ai_b4096.

Catalog gain: 12 KPU + 2 GPU + 1 CPU + 3 NPU + 1 CGRA + 1 DPU = 20
ComputeProducts. First SKU to exercise:
  - ``BlockKind.DPU`` and the full ``DPUBlock`` schema (landed in #36)
  - ``DPUFabricKind.AIE_ML_V1``
  - ``DPUNoCTopology.AIE_MESH``
  - ``DPUBlock.is_statically_reconfigurable=True`` + non-None
    ``bitstream_load_time_ms``
  - ``DPUComputeFabric.fpga_fabric_overhead_factor=1.25`` (the
    FPGA-vs-ASIC energy penalty)
  - The ``xilinx`` vendor directory
  - First production-lifecycle FPGA-based SKU

Same shape as ``test_compute_product_v5_plasticine_v2_yaml.py`` and
its NPU siblings.
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    DPUBlock,
    DPUFabricKind,
    DPUNoCTopology,
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
def vitis_ai(all_products) -> ComputeProduct:
    cp = all_products.get("xilinx_vitis_ai_b4096")
    if cp is None:
        pytest.fail("xilinx_vitis_ai_b4096 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def vitis_ai_compute_die(vitis_ai):
    die = next((d for d in vitis_ai.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Vitis AI has no compute die")
    return die


@pytest.fixture(scope="module")
def vitis_ai_dpu_block(vitis_ai_compute_die) -> DPUBlock:
    block = next(
        (b for b in vitis_ai_compute_die.blocks if isinstance(b, DPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("Vitis AI compute die has no DPUBlock")
    return block


# ---------------------------------------------------------------------------
# Catalog: Vitis AI is the first DPU SKU (first Xilinx vendor)
# ---------------------------------------------------------------------------

def test_catalog_includes_vitis_ai(all_products):
    xilinx_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "xilinx")
    assert xilinx_skus == ["xilinx_vitis_ai_b4096"]


def test_catalog_has_at_least_one_dpu(all_products):
    """At-least-one-DPU check (subset semantics). The TPU v4 follow-up
    grew the catalog past the original 20-product mark; the TPU PR's
    own contract test pins the new total."""
    counts_by_kind: dict[str, int] = {}
    for cp in all_products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    # DPU count is the focus of this PR; pin >= 1 without locking total
    assert counts_by_kind.get("dpu", 0) >= 1


def test_other_vendors_unaffected_by_xilinx_addition(all_products):
    """Additive guarantee: adding xilinx/ vendor directory must not
    perturb stillwater/, nvidia/, intel/, hailo/, stanford/ loading.
    google/ loosened to >= 1 since the TPU v4 follow-up adds a 2nd
    google SKU."""
    counts_by_vendor: dict[str, int] = {}
    for cp in all_products.values():
        counts_by_vendor[cp.vendor] = counts_by_vendor.get(cp.vendor, 0) + 1
    assert counts_by_vendor.get("stillwater") == 12
    assert counts_by_vendor.get("nvidia") == 2
    assert counts_by_vendor.get("intel") == 4  # i7-12700k + 3 Xeons: 8490H + 8592+ + 6980P (sprint #68 PR 1-3)
    assert counts_by_vendor.get("hailo") == 2
    assert counts_by_vendor.get("google") >= 1
    assert counts_by_vendor.get("stanford") == 1
    assert counts_by_vendor.get("xilinx") == 1


# ---------------------------------------------------------------------------
# Vitis AI identity / die / silicon_bin
# ---------------------------------------------------------------------------

def test_vitis_ai_identity(vitis_ai):
    assert vitis_ai.id == "xilinx_vitis_ai_b4096"
    assert vitis_ai.vendor == "xilinx"
    assert vitis_ai.packaging.kind == PackagingKind.MONOLITHIC
    assert vitis_ai.packaging.package_type == "bga"
    # Production lifecycle (commercial Versal VE2302)
    assert vitis_ai.lifecycle == LifecycleStatus.PRODUCTION


def test_vitis_ai_die_references_tsmc_n16(vitis_ai_compute_die):
    """Versal VE2302 ships on TSMC N16. The catalog already has tsmc_n16
    from the KPU sprint -- no new process-node YAML needed."""
    die = vitis_ai_compute_die
    assert die.process_node_id == "tsmc_n16"
    assert die.die_role.value == "compute"
    # VE2302 full SoC die ~600 mm^2 (large because Versal includes ARM
    # + FPGA fabric + AIE + I/O PHYs)
    assert die.die_size_mm2 == pytest.approx(600.0, rel=0.1)
    assert die.transistors_billion == pytest.approx(3.7, rel=0.15)


def test_vitis_ai_silicon_bin_reconciles_with_die_total(vitis_ai_compute_die):
    die = vitis_ai_compute_die
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


def test_vitis_ai_silicon_bin_has_dpu_count_refs(vitis_ai_compute_die):
    """The DPU paper exercise specified per-block count_refs:
    aie_tile_array, tile_scratchpad_sram, shared_sram, aie_noc,
    fpga_fabric_overhead, ddr4_phy, etc. fpga_fabric_overhead is
    DPU-unique (no other architecture has FPGA glue logic in the bin)."""
    names = {b.name for b in vitis_ai_compute_die.silicon_bin.blocks}
    assert "aie_tile_array" in names
    assert "fpga_fabric_overhead" in names, \
        "DPU-specific fpga_fabric_overhead block missing"
    assert "ddr4_phy" in names


# ---------------------------------------------------------------------------
# DPUBlock structural fields
# ---------------------------------------------------------------------------

def test_vitis_ai_block_is_dpu(vitis_ai_dpu_block):
    assert isinstance(vitis_ai_dpu_block, DPUBlock)
    assert vitis_ai_dpu_block.kind == "dpu"


def test_vitis_ai_aie_tile_hierarchy(vitis_ai_dpu_block):
    """B4096 config: 64 AIE-ML v1 tiles, 64 MACs each = 4096 MACs total.
    AIE-ML v1 has wide-vector datapaths (SIMD lanes per tile)."""
    block = vitis_ai_dpu_block
    assert block.num_aie_tiles == 64
    assert block.macs_per_tile == 64
    assert block.simd_lanes_per_tile == 8


def test_vitis_ai_fpga_reconfiguration_model(vitis_ai_dpu_block):
    """First SKU exercising the static FPGA reconfiguration model:
    is_statically_reconfigurable=True + bitstream_load_time_ms populated
    (deployment-only cost, not runtime like CGRA's per-cycle reconfig)."""
    block = vitis_ai_dpu_block
    assert block.is_statically_reconfigurable is True
    # ~2 seconds bitstream load (Xilinx-typical for VE2302)
    assert block.bitstream_load_time_ms == pytest.approx(2000.0)


def test_vitis_ai_aie_ml_v1_fabric(vitis_ai_dpu_block):
    """First SKU in catalog with DPUFabricKind.AIE_ML_V1. 128 INT8 +
    32 FP16 ops per AIE tile per clock."""
    block = vitis_ai_dpu_block
    assert len(block.compute_fabrics) == 1
    fabric = block.compute_fabrics[0]
    assert fabric.fabric_kind == DPUFabricKind.AIE_ML_V1
    assert fabric.ops_per_unit_per_clock["int8"] == 128
    assert fabric.ops_per_unit_per_clock["fp16"] == 32


def test_vitis_ai_fpga_fabric_overhead_factor(vitis_ai_dpu_block):
    """The defining DPU characteristic: ~25% FPGA-vs-ASIC penalty.
    First SKU to exercise fpga_fabric_overhead_factor > 1.0."""
    block = vitis_ai_dpu_block
    fabric = block.compute_fabrics[0]
    assert fabric.fpga_fabric_overhead_factor == pytest.approx(1.25, rel=0.01)


def test_vitis_ai_multi_precision_includes_native_fp16(vitis_ai_dpu_block):
    """DPUs distinguish from INT-only NPUs by including native FP16
    (AIE-ML hardware support) plus emulated FP32. CGRA's Plasticine
    emulates BOTH FP16 and FP32; Vitis AI's FP16 is native."""
    block = vitis_ai_dpu_block
    assert set(block.multi_precision_alu) == {"int8", "fp16", "fp32"}
    scaling = block.compute_fabrics[0].energy_scaling
    # FP16 is native AIE-ML; scaling is 2.5x INT8 (vs CGRA's 3.3x for emulated)
    assert scaling.get("fp16", 0) == pytest.approx(2.5, rel=0.01)
    # FP32 is emulated; scaling is 5x INT8 (lower than CGRA's 6.7x because
    # AIE-ML has better building blocks for emulation)
    assert scaling.get("fp32", 0) == pytest.approx(5.0, rel=0.01)


# ---------------------------------------------------------------------------
# Memory: first DPU with populated external_dram_* (chip-attached DDR4)
# ---------------------------------------------------------------------------

def test_vitis_ai_memory_has_chip_attached_ddr4(vitis_ai_dpu_block):
    """First DPU SKU to populate has_external_dram=True. Versal VE2302
    has on-die DDR4 controllers (NPU-style chip-attached, distinct from
    CGRA Plasticine's host-bus DDR4)."""
    mem = vitis_ai_dpu_block.memory
    assert mem.has_external_dram is True
    assert mem.external_dram_type == MemoryType.DDR4
    assert mem.external_dram_size_gb == pytest.approx(8.0, rel=0.01)
    assert mem.external_dram_bandwidth_gbps == pytest.approx(50.0, rel=0.1)


def test_vitis_ai_on_chip_sram(vitis_ai_dpu_block):
    """64 KiB scratchpad per AIE tile + 4 MiB shared L2 = 8 MiB total
    on-chip SRAM."""
    mem = vitis_ai_dpu_block.memory
    assert mem.scratchpad_kib_per_tile == 64
    assert mem.shared_sram_kib == 4 * 1024
    assert mem.coherence_protocol == "none"


# ---------------------------------------------------------------------------
# NoC: 8x8 AIE_MESH for 64 tiles
# ---------------------------------------------------------------------------

def test_vitis_ai_noc_is_aie_mesh_8x8(vitis_ai_dpu_block):
    """First SKU in catalog with DPUNoCTopology.AIE_MESH."""
    noc = vitis_ai_dpu_block.noc
    assert noc.topology == DPUNoCTopology.AIE_MESH
    assert noc.unit_count == 64
    assert noc.mesh_rows == 8
    assert noc.mesh_cols == 8
    assert noc.confidence == DataConfidence.THEORETICAL


# ---------------------------------------------------------------------------
# Scheduler: DPU-specific multi-model concurrency + wave quantization
# ---------------------------------------------------------------------------

def test_vitis_ai_supports_multi_model_concurrency(vitis_ai_dpu_block):
    """DPUs partition AIE tiles to run multiple compiled models
    simultaneously. Vitis AI default: 4. Distinguishes from NPU/CGRA
    (1 each)."""
    block = vitis_ai_dpu_block
    assert block.max_concurrent_models == 4


def test_vitis_ai_wave_quantization_is_pair(vitis_ai_dpu_block):
    """AIE tiles ship and configure in pairs per Xilinx documentation."""
    block = vitis_ai_dpu_block
    assert block.wave_quantization == 2


# ---------------------------------------------------------------------------
# Power / performance / market
# ---------------------------------------------------------------------------

def test_vitis_ai_single_thermal_profile(vitis_ai):
    """20W envelope, active-fan cooling. No DVFS on edge VE2302."""
    power = vitis_ai.power
    assert power.tdp_watts == 20.0
    assert power.default_thermal_profile == "default"
    profile_names = {p.name for p in power.thermal_profiles}
    assert profile_names == {"default"}
    assert power.thermal_profiles[0].cooling_solution_id == "active_fan"
    assert power.thermal_profiles[0].clock_mhz == pytest.approx(1250.0)


def test_vitis_ai_chip_level_performance_matches_marketing(vitis_ai):
    """10.24 TOPS INT8 theoretical (64 tiles * 128 ops/clk * 1.25 GHz).
    2.56 TFLOPS FP16 native. 0.32 TFLOPS FP32 emulated. No INT4
    (added in AIE-ML v2)."""
    perf = vitis_ai.performance
    assert perf.int8_tops == pytest.approx(10.24)
    assert perf.fp32_tflops == pytest.approx(0.32, rel=0.05)
    assert perf.int4_tops == 0.0   # not supported on AIE-ML v1
    assert perf.bf16_tflops == 0.0


def test_vitis_ai_production_lifecycle(vitis_ai):
    """Versal VE2302 has been in production since 2022."""
    market = vitis_ai.market
    assert market.target_market == "edge"
    assert market.is_available is True
    assert market.product_family == "Vitis AI"


def test_vitis_ai_round_trips_through_serialize(vitis_ai):
    """Full ComputeProduct -> JSON -> ComputeProduct round-trip
    preserves DPUBlock discriminator + reconfig + FPGA-overhead fields."""
    payload = vitis_ai.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    rebuilt_block = next(
        b for d in rebuilt.dies
        if d.die_role.value == "compute"
        for b in d.blocks
        if isinstance(b, DPUBlock)
    )
    assert rebuilt_block.num_aie_tiles == 64
    assert rebuilt_block.macs_per_tile == 64
    assert rebuilt_block.is_statically_reconfigurable is True
    assert rebuilt_block.bitstream_load_time_ms == pytest.approx(2000.0)
    assert rebuilt_block.memory.has_external_dram is True
    assert rebuilt_block.compute_fabrics[0].fabric_kind == DPUFabricKind.AIE_ML_V1
    assert rebuilt_block.compute_fabrics[0].fpga_fabric_overhead_factor == pytest.approx(1.25)
