"""Tests for the sixth DSP SKU YAML: ti_tda4vh.

Third TI TDA4 family member; **first DSP SKU with LPDDR5**. Same
schema patterns as TDA4VM/AL (multi-fabric C7x + MMA, multi-profile
DVFS, SoC-integrated, measured bandwidth) but at 4x compute scale
(32 TOPS INT8, 4x MMAv2 units, 4x C7x cluster).
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    DramAttachment,
    DSPBlock,
    DSPDeploymentKind,
    DSPFabricKind,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def tda4vh(all_products) -> ComputeProduct:
    cp = all_products.get("ti_tda4vh")
    if cp is None:
        pytest.fail("ti_tda4vh missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tda4vh_block(tda4vh) -> DSPBlock:
    block = tda4vh.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: third TI TDA4 family member
# ---------------------------------------------------------------------------

def test_ti_vendor_has_three_skus(all_products):
    ti_skus = set(s for s, cp in all_products.items() if cp.vendor == "ti")
    assert {"ti_tda4vm", "ti_tda4al", "ti_tda4vh"}.issubset(ti_skus)


def test_catalog_dsp_count_at_least_six(all_products):
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count >= 6


# ---------------------------------------------------------------------------
# Scale: 4x TDA4VM (32 TOPS, 4 MMAv2 units, 32 C7x cores)
# ---------------------------------------------------------------------------

def test_tda4vh_int8_peak_is_32_tops(tda4vh_block):
    """**4x TDA4VM** (4x MMAv2 units): 32 TOPS INT8."""
    int8_peak = tda4vh_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"]
    assert int8_peak == pytest.approx(32e12, rel=0.01)


def test_tda4vh_fp32_peak_is_320_gflops(tda4vh_block):
    """**4x TDA4VM** (4x C7x cluster): 320 GFLOPS FP32."""
    fp32_peak = tda4vh_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["fp32"]
    assert fp32_peak == pytest.approx(320e9, rel=0.01)


def test_tda4vh_has_32_c7x_cores(tda4vh_block):
    """4x TDA4VM's 8 cores."""
    c7x = tda4vh_block.compute_fabrics[0]
    assert c7x.fabric_kind == DSPFabricKind.VLIW_SCALAR
    assert c7x.num_units == 32


def test_tda4vh_has_4_mma_units(tda4vh_block):
    """4x MMAv2 accelerators (vs TDA4VM's 1)."""
    mma = tda4vh_block.compute_fabrics[1]
    assert mma.fabric_kind == DSPFabricKind.TENSOR_MATRIX
    assert mma.num_units == 4


# ---------------------------------------------------------------------------
# Memory: first LPDDR5 DSP SKU
# ---------------------------------------------------------------------------

def test_tda4vh_uses_lpddr5(tda4vh_block):
    """**First DSP SKU with LPDDR5** (vs LPDDR4x in TDA4VM/AL).
    100 GB/s @ 6400 MT/s."""
    mem = tda4vh_block.memory
    assert mem.external_dram_type == MemoryType.LPDDR5
    assert mem.external_dram_bandwidth_gbps == 100.0
    assert mem.external_dram_bandwidth_kind == "measured"
    assert mem.dram_attachment == DramAttachment.CHIP_ATTACHED


def test_tda4vh_l2_is_16_mib(tda4vh_block):
    """16 MiB MSMC SRAM (2x TDA4VM's 8 MiB) for multi-accelerator load."""
    assert tda4vh_block.memory.l2_size_bytes_total == 16 * 1024 * 1024


def test_tda4vh_main_memory_is_16_gb(tda4vh_block):
    """Up to 16 GiB LPDDR5 (2x TDA4VM)."""
    cp_main = tda4vh_block.memory.external_dram_size_gb
    assert cp_main == 16.0


# ---------------------------------------------------------------------------
# Thermal: 20W + 35W (higher than TDA4VM's 10W + 20W)
# ---------------------------------------------------------------------------

def test_tda4vh_thermal_profiles_are_20w_and_35w(tda4vh_block):
    """Higher thermal envelope than TDA4VM (10W + 20W) or TDA4AL (10W + 18W)."""
    assert len(tda4vh_block.thermal_profiles) == 2
    names = {p.name for p in tda4vh_block.thermal_profiles}
    assert names == {"20W", "35W"}


def test_tda4vh_default_profile_is_20w(tda4vh_block):
    """20W is the default for multi-camera L2+ ADAS."""
    assert tda4vh_block.default_thermal_profile_name == "20W"


def test_tda4vh_35w_profile_for_full_l3_l4_autonomy(tda4vh_block):
    p = next(p for p in tda4vh_block.thermal_profiles if p.name == "35W")
    assert p.tdp_watts == 35.0
    assert p.clock_mhz == 950.0


# ---------------------------------------------------------------------------
# Scheduler attributes: multi-accelerator deltas
# ---------------------------------------------------------------------------

def test_tda4vh_lower_min_occupancy(tda4vh_block):
    """0.60 (vs TDA4VM/AL's 0.70) -- multi-accelerator coordination cost."""
    assert tda4vh_block.min_occupancy == 0.60


def test_tda4vh_more_concurrent_kernels(tda4vh_block):
    """4x accelerators allow 8 concurrent kernels (vs TDA4VM/AL's 4)."""
    assert tda4vh_block.max_concurrent_kernels == 8


def test_tda4vh_wave_quantization_is_8(tda4vh_block):
    """8 wave_quantization aligns with 4-MMAv2 grouping."""
    assert tda4vh_block.wave_quantization == 8


# ---------------------------------------------------------------------------
# Common features (already validated by TDA4VM)
# ---------------------------------------------------------------------------

def test_tda4vh_deployment_kind_soc_integrated(tda4vh_block):
    assert tda4vh_block.deployment_kind == DSPDeploymentKind.SOC_INTEGRATED


def test_tda4vh_vliw_issue_width_is_8(tda4vh_block):
    assert tda4vh_block.vliw_issue_width == 8


def test_tda4vh_process_node_is_16nm(tda4vh):
    """16nm FinFET (corrected from hand-coded's buggy 28nm)."""
    assert tda4vh.dies[0].process_node_id == "tsmc_n16"


def test_tda4vh_model_tier_high(tda4vh):
    """High-end ADAS (vs TDA4VM/AL's mid)."""
    assert tda4vh.market.model_tier == "high"


def test_tda4vh_chip_power_envelope(tda4vh):
    """Default 20W; max 38W (covers 35W + boost headroom)."""
    assert tda4vh.power.tdp_watts == 20.0
    assert tda4vh.power.default_thermal_profile == "20W"
    assert tda4vh.power.max_power_watts >= 35.0
