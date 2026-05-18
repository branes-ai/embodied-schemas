"""Tests for the seventh DSP SKU YAML: ti_tda4vl. Closes batch 2.

Entry-level Jacinto 7: half the compute of TDA4VM (4 TOPS INT8,
4 C7x cores). Lower thermal envelope (7W + 12W; both passive).
Cost-sensitive ADAS deployments.
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
def tda4vl(all_products) -> ComputeProduct:
    cp = all_products.get("ti_tda4vl")
    if cp is None:
        pytest.fail("ti_tda4vl missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tda4vl_block(tda4vl) -> DSPBlock:
    block = tda4vl.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: closes TI TDA4 family (4 SKUs)
# ---------------------------------------------------------------------------

def test_ti_vendor_has_all_four_tda4_skus(all_products):
    """**Closes TI TDA4 family** (graphs#223 batch 2 complete)."""
    ti_skus = {s for s, cp in all_products.items() if cp.vendor == "ti"}
    assert ti_skus == {"ti_tda4vm", "ti_tda4al", "ti_tda4vh", "ti_tda4vl"}


def test_catalog_dsp_count_at_least_seven(all_products):
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count >= 7


# ---------------------------------------------------------------------------
# Scale: half of TDA4VM (4 TOPS, 4 C7x cores, 1 MMAv2 unit)
# ---------------------------------------------------------------------------

def test_tda4vl_int8_peak_is_4_tops(tda4vl_block):
    """Half of TDA4VM's 8 TOPS."""
    int8_peak = tda4vl_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"]
    assert int8_peak == pytest.approx(4e12, rel=0.01)


def test_tda4vl_fp32_peak_is_40_gflops(tda4vl_block):
    """Half of TDA4VM's 80 GFLOPS."""
    fp32_peak = tda4vl_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["fp32"]
    assert fp32_peak == pytest.approx(40e9, rel=0.01)


def test_tda4vl_has_4_c7x_cores(tda4vl_block):
    """Half of TDA4VM's 8 cores."""
    c7x = tda4vl_block.compute_fabrics[0]
    assert c7x.fabric_kind == DSPFabricKind.VLIW_SCALAR
    assert c7x.num_units == 4


def test_tda4vl_mma_v2_at_half_capacity(tda4vl_block):
    """1 MMAv2 unit at 4000 ops/cycle (half of TDA4AL's 8000).
    This is the lowest-capacity tensor accelerator in the catalog."""
    mma = tda4vl_block.compute_fabrics[1]
    assert mma.fabric_kind == DSPFabricKind.TENSOR_MATRIX
    assert mma.num_units == 1
    assert mma.ops_per_unit_per_clock["int8"] == 4000


# ---------------------------------------------------------------------------
# Thermal: 7W + 12W (lowest of all TDA4 variants; both passive)
# ---------------------------------------------------------------------------

def test_tda4vl_thermal_profiles_are_7w_and_12w(tda4vl_block):
    """Lowest thermal envelope in the TDA4 family."""
    assert len(tda4vl_block.thermal_profiles) == 2
    names = {p.name for p in tda4vl_block.thermal_profiles}
    assert names == {"7W", "12W"}


def test_tda4vl_default_profile_is_7w(tda4vl_block):
    """7W for entry-level single-camera ADAS."""
    assert tda4vl_block.default_thermal_profile_name == "7W"


def test_tda4vl_both_profiles_use_passive_cooling(tda4vl_block):
    """Entry-level form factor: even 12W stays passive (vs TDA4VM/AL
    where the upper profile uses active fan)."""
    for p in tda4vl_block.thermal_profiles:
        assert p.cooling_solution_id == "passive_fanless"


# ---------------------------------------------------------------------------
# Memory: smaller (4 GiB vs TDA4VM's 8 GiB) but same LPDDR4x bandwidth
# ---------------------------------------------------------------------------

def test_tda4vl_main_memory_is_4_gb(tda4vl_block):
    """Cost-sensitive: 4 GiB (half of TDA4VM's 8 GiB)."""
    assert tda4vl_block.memory.external_dram_size_gb == 4.0


def test_tda4vl_bandwidth_matches_tda4vm(tda4vl_block):
    """Same 60 GB/s LPDDR4x (cost optimization preserves the bandwidth)."""
    assert tda4vl_block.memory.external_dram_bandwidth_gbps == 60.0


# ---------------------------------------------------------------------------
# Common features (already validated by other TDA4 SKUs)
# ---------------------------------------------------------------------------

def test_tda4vl_deployment_kind_soc_integrated(tda4vl_block):
    assert tda4vl_block.deployment_kind == DSPDeploymentKind.SOC_INTEGRATED


def test_tda4vl_uses_lpddr4x(tda4vl_block):
    mem = tda4vl_block.memory
    assert mem.external_dram_type == MemoryType.LPDDR4X
    assert mem.dram_attachment == DramAttachment.CHIP_ATTACHED
    assert mem.external_dram_bandwidth_kind == "measured"


def test_tda4vl_vliw_issue_width_is_8(tda4vl_block):
    assert tda4vl_block.vliw_issue_width == 8


def test_tda4vl_process_node_is_16nm(tda4vl):
    assert tda4vl.dies[0].process_node_id == "tsmc_n16"


def test_tda4vl_model_tier_entry(tda4vl):
    """**Entry-level** (vs VM/AL=mid, VH=high)."""
    assert tda4vl.market.model_tier == "entry"
