"""Tests for the fourth DSP SKU YAML: ti_tda4vm.

Catalog gain: brings DSP count from 3 to 4. **First SoC-integrated DSP
SKU** (vs the 3 IP cores landed before). Exercises several DSPBlock
schema features for the first time:

  - ``deployment_kind=SOC_INTEGRATED``
  - ``external_dram_bandwidth_kind=measured`` (vs IP cores' ``typical``)
  - ``vliw_issue_width=8`` (C7x is TI's VLIW DSP)
  - **Multi-profile DVFS**: 2 ``DSPThermalProfile`` entries (10W / 20W)
  - ``DSPFabricKind.VLIW_SCALAR`` (first use; C7x is VLIW)

Opens batch 2 of the DSP follow-up sprint (graphs#223): TI TDA4 family.
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    DramAttachment,
    DSPBlock,
    DSPDeploymentKind,
    DSPFabricKind,
    LifecycleStatus,
    PackagingKind,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def tda4vm(all_products) -> ComputeProduct:
    cp = all_products.get("ti_tda4vm")
    if cp is None:
        pytest.fail("ti_tda4vm missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tda4vm_block(tda4vm) -> DSPBlock:
    block = tda4vm.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: TI TDA4VM lives in new ti/ vendor directory
# ---------------------------------------------------------------------------

def test_catalog_includes_ti_tda4vm(all_products):
    """At this SKU's PR baseline ti/ had 1 entry; subset semantics so
    further TI TDA4 follow-ups (graphs#223 batch 2) don't regress this."""
    ti_skus = {s for s, cp in all_products.items() if cp.vendor == "ti"}
    assert "ti_tda4vm" in ti_skus


def test_catalog_dsp_count_at_least_four(all_products):
    """Cadence + Synopsys + CEVA + TDA4VM >= 4 DSP SKUs."""
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count >= 4


# ---------------------------------------------------------------------------
# Identity / packaging
# ---------------------------------------------------------------------------

def test_tda4vm_identity(tda4vm):
    assert tda4vm.id == "ti_tda4vm"
    assert tda4vm.name == "Texas Instruments TDA4VM (Jacinto 7)"
    assert tda4vm.vendor == "ti"


def test_tda4vm_packaging(tda4vm):
    """SoC-integrated DSP uses real packaging (fcbga), not 'ip_core'."""
    assert tda4vm.packaging.kind == PackagingKind.MONOLITHIC
    assert tda4vm.packaging.package_type == "fcbga"


def test_tda4vm_process_node_is_16nm(tda4vm):
    """Per TI's official docs, TDA4VM is 16nm FinFET. The graphs
    hand-coded factory's process_node_nm=28 was a bug; YAML uses
    the corrected tsmc_n16."""
    assert tda4vm.dies[0].process_node_id == "tsmc_n16"


# ---------------------------------------------------------------------------
# First-SoC-integrated DSP features
# ---------------------------------------------------------------------------

def test_tda4vm_deployment_kind_soc_integrated(tda4vm_block):
    """**First SoC-integrated DSP SKU**. The 3 IP cores before were
    STANDALONE_IP."""
    assert tda4vm_block.deployment_kind == DSPDeploymentKind.SOC_INTEGRATED


def test_tda4vm_external_dram_bandwidth_kind_measured(tda4vm_block):
    """**First SoC DSP with measured external_dram_bandwidth_kind**
    (vs IP cores' typical-integration). 60 GB/s LPDDR4x dual-channel."""
    mem = tda4vm_block.memory
    assert mem.external_dram_bandwidth_kind == "measured"
    assert mem.external_dram_type == MemoryType.LPDDR4X
    assert mem.external_dram_bandwidth_gbps == 60.0
    assert mem.dram_attachment == DramAttachment.CHIP_ATTACHED


def test_tda4vm_vliw_issue_width_is_8(tda4vm_block):
    """**First VLIW DSP SKU**: C7x is TI's 8-wide VLIW DSP architecture.
    Informational field (mappers don't use it for analytical roofline)."""
    assert tda4vm_block.vliw_issue_width == 8


# ---------------------------------------------------------------------------
# Multi-fabric: C7x DSP + MMA (C7x first)
# ---------------------------------------------------------------------------

def test_tda4vm_has_two_fabrics(tda4vm_block):
    """C7x DSP (VLIW) + MMA (tensor accelerator)."""
    assert len(tda4vm_block.compute_fabrics) == 2


def test_tda4vm_c7x_fabric_is_vliw_scalar(tda4vm_block):
    """**First DSPFabricKind.VLIW_SCALAR use**: C7x is VLIW (not pure SIMD)."""
    c7x = tda4vm_block.compute_fabrics[0]
    assert c7x.fabric_kind == DSPFabricKind.VLIW_SCALAR
    assert c7x.num_units == 8                # 8 C7x cores
    assert "fp32" in c7x.ops_per_unit_per_clock
    assert "fp16" in c7x.ops_per_unit_per_clock
    # C7x doesn't carry INT8 ops directly (MMA does)
    assert "int8" not in c7x.ops_per_unit_per_clock


def test_tda4vm_mma_fabric_is_tensor_matrix(tda4vm_block):
    """MMA is a tensor matrix accelerator (single unit)."""
    mma = tda4vm_block.compute_fabrics[1]
    assert mma.fabric_kind == DSPFabricKind.TENSOR_MATRIX
    assert mma.num_units == 1                # Single MMA unit
    # 8000 ops/cycle/unit = 8 TOPS @ 1.0 GHz
    assert mma.ops_per_unit_per_clock["int8"] == 8000


def test_tda4vm_int8_peak_matches_marketed(tda4vm_block):
    """8 TOPS INT8 (TI-marketed, MMA path)."""
    int8_peak = tda4vm_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"]
    assert int8_peak == pytest.approx(8e12, rel=0.01)


def test_tda4vm_fp32_peak_matches_marketed(tda4vm_block):
    """80 GFLOPS FP32 (TI-marketed, C7x path)."""
    fp32_peak = tda4vm_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["fp32"]
    assert fp32_peak == pytest.approx(80e9, rel=0.01)


# ---------------------------------------------------------------------------
# Multi-profile DVFS (first DSP test of len > 1 thermal_profiles)
# ---------------------------------------------------------------------------

def test_tda4vm_multi_profile_dvfs(tda4vm_block):
    """**First DSP SKU with 2 thermal profiles**: 10W front-camera
    + 20W full-ADAS-system. Exercises DSPBlock.thermal_profiles
    list with len > 1 for the first time on DSP."""
    assert len(tda4vm_block.thermal_profiles) == 2
    names = {p.name for p in tda4vm_block.thermal_profiles}
    assert names == {"10W", "20W"}


def test_tda4vm_default_thermal_profile_is_10w(tda4vm_block):
    """Front-camera ADAS is the most common automotive deployment."""
    assert tda4vm_block.default_thermal_profile_name == "10W"


def test_tda4vm_10w_profile_uses_passive_cooling(tda4vm_block):
    """10W is automotive passive cooling."""
    p = next(p for p in tda4vm_block.thermal_profiles if p.name == "10W")
    assert p.tdp_watts == 10.0
    assert p.cooling_solution_id == "passive_fanless"
    assert p.clock_mhz == 850.0
    assert p.dvfs_enabled is True


def test_tda4vm_20w_profile_uses_active_cooling(tda4vm_block):
    """20W is automotive active-fan cooling."""
    p = next(p for p in tda4vm_block.thermal_profiles if p.name == "20W")
    assert p.tdp_watts == 20.0
    assert p.cooling_solution_id == "active_fan"
    assert p.clock_mhz == 950.0
    assert p.dvfs_enabled is True


# ---------------------------------------------------------------------------
# Memory hierarchy
# ---------------------------------------------------------------------------

def test_tda4vm_memory_l1_l2_match_hand_coded(tda4vm_block):
    """48 KB L1D per C7x core + 8 MB MSMC SRAM."""
    mem = tda4vm_block.memory
    assert mem.l1_size_bytes_per_unit == 48 * 1024
    assert mem.l2_size_bytes_total == 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Chip-level power
# ---------------------------------------------------------------------------

def test_tda4vm_chip_power_envelope(tda4vm):
    """Default 10W; max 22W (covers 20W profile + boost headroom)."""
    assert tda4vm.power.tdp_watts == 10.0
    assert tda4vm.power.default_thermal_profile == "10W"
    assert tda4vm.power.max_power_watts >= 20.0
