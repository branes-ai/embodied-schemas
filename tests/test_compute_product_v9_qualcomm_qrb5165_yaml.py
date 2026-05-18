"""Tests for the ninth DSP SKU YAML: qualcomm_qrb5165.

Second Qualcomm DSP. Robotics platform based on Snapdragon 865 (7nm).
Single thermal profile (vs SA8775P's 3-profile DVFS). Smaller scale
(15 TOPS vs SA8775P's 32). Same Hexagon DSP architecture family.
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
def qrb5165(all_products) -> ComputeProduct:
    cp = all_products.get("qualcomm_qrb5165")
    if cp is None:
        pytest.fail("qualcomm_qrb5165 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def qrb5165_block(qrb5165) -> DSPBlock:
    block = qrb5165.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: second Qualcomm DSP
# ---------------------------------------------------------------------------

def test_qualcomm_vendor_has_at_least_two_skus(all_products):
    qualcomm_skus = {s for s, cp in all_products.items() if cp.vendor == "qualcomm"}
    assert {"qualcomm_sa8775p", "qualcomm_qrb5165"}.issubset(qualcomm_skus)


def test_catalog_dsp_count_at_least_nine(all_products):
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count >= 9


# ---------------------------------------------------------------------------
# QRB5165-specific: 7nm, 15 TOPS, single profile, HTA+HVX
# ---------------------------------------------------------------------------

def test_qrb5165_process_node_is_tsmc_n7(qrb5165):
    """7nm TSMC (Snapdragon 865 generation; vs SA8775P's 5nm)."""
    assert qrb5165.dies[0].process_node_id == "tsmc_n7"


def test_qrb5165_int8_peak_is_15_tops(qrb5165_block):
    """Qualcomm-marketed 15 TOPS INT8 (vs SA8775P's 32)."""
    int8_peak = qrb5165_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"]
    assert int8_peak == pytest.approx(15e12, rel=0.01)


def test_qrb5165_has_two_fabrics(qrb5165_block):
    """HTA tensor + HVX vector multi-fabric (Hexagon 698 architecture)."""
    assert len(qrb5165_block.compute_fabrics) == 2


def test_qrb5165_hta_fabric_first(qrb5165_block):
    """HTA is fabric[0] -- primary INT8 path (12 of 15 TOPS)."""
    hta = qrb5165_block.compute_fabrics[0]
    assert hta.fabric_kind == DSPFabricKind.TENSOR_MATRIX
    assert hta.num_units == 1
    assert hta.ops_per_unit_per_clock["int8"] == 9024


def test_qrb5165_hvx_fabric_second(qrb5165_block):
    """HVX is fabric[1] -- 4x 1024-bit SIMD units."""
    hvx = qrb5165_block.compute_fabrics[1]
    assert hvx.fabric_kind == DSPFabricKind.VECTOR_SIMD
    assert hvx.num_units == 4


def test_qrb5165_single_thermal_profile(qrb5165_block):
    """7W passive single profile (vs SA8775P's 3-profile DVFS)."""
    assert len(qrb5165_block.thermal_profiles) == 1
    p = qrb5165_block.thermal_profiles[0]
    assert p.name == "7W"
    assert p.tdp_watts == 7.0
    assert p.cooling_solution_id == "passive_fanless"


def test_qrb5165_default_thermal_profile(qrb5165_block):
    assert qrb5165_block.default_thermal_profile_name == "7W"


# ---------------------------------------------------------------------------
# Memory: LPDDR5 quad-channel (smaller than SA8775P)
# ---------------------------------------------------------------------------

def test_qrb5165_lpddr5_quad_channel(qrb5165_block):
    """44 GB/s LPDDR5 (vs SA8775P's 90 GB/s)."""
    mem = qrb5165_block.memory
    assert mem.external_dram_type == MemoryType.LPDDR5
    assert mem.external_dram_bandwidth_gbps == 44.0
    assert mem.external_dram_bandwidth_kind == "measured"
    assert mem.dram_attachment == DramAttachment.CHIP_ATTACHED
    assert mem.external_dram_size_gb == 16.0


# ---------------------------------------------------------------------------
# Common features
# ---------------------------------------------------------------------------

def test_qrb5165_deployment_kind_soc_integrated(qrb5165_block):
    assert qrb5165_block.deployment_kind == DSPDeploymentKind.SOC_INTEGRATED


def test_qrb5165_vliw_issue_width_is_4(qrb5165_block):
    """Hexagon DSP is 4-issue VLIW (same as SA8775P)."""
    assert qrb5165_block.vliw_issue_width == 4


def test_qrb5165_model_tier_mid(qrb5165):
    """Mid-range edge AI (vs SA8775P=high automotive)."""
    assert qrb5165.market.model_tier == "mid"


def test_qrb5165_robotics_product_family(qrb5165):
    assert qrb5165.market.product_family == "Robotics-Platform"
