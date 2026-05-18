"""Tests for the tenth (and last DSP batch follow-up) SKU YAML: qualcomm_qcs6490.

**Closes DSP batch #223** -- all 9 follow-up DSP SKUs YAML-backed
after this SKU's cleanup PR lands. Third Qualcomm DSP. Entry-level
edge AI on TSMC N6 (**first 6nm SKU in the catalog**; tsmc_n6 process
node added in this PR).

12 TOPS INT8 / 24 TOPS INT4. 3-profile DVFS (5W battery / 10W standard
/ 15W max). HTA tensor + HVX vector multi-fabric (16 HVX units vs
QRB5165's 4).
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
from embodied_schemas.loaders import load_compute_products, load_process_nodes


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def qcs6490(all_products) -> ComputeProduct:
    cp = all_products.get("qualcomm_qcs6490")
    if cp is None:
        pytest.fail("qualcomm_qcs6490 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def qcs6490_block(qcs6490) -> DSPBlock:
    block = qcs6490.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: closes DSP batch (10 DSP SKUs total = 1 reference + 9 follow-ups)
# ---------------------------------------------------------------------------

def test_qualcomm_vendor_has_all_three_skus(all_products):
    qualcomm_skus = {s for s, cp in all_products.items() if cp.vendor == "qualcomm"}
    assert qualcomm_skus == {"qualcomm_sa8775p", "qualcomm_qrb5165", "qualcomm_qcs6490"}


def test_catalog_dsp_count_at_least_ten(all_products):
    """**Closes DSP batch #223** -- 10 DSP SKUs (Cadence reference +
    9 follow-ups: Synopsys EV7x, CEVA NPM11, TDA4VM/AL/VH/VL,
    SA8775P, QRB5165, QCS6490)."""
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count >= 10


# ---------------------------------------------------------------------------
# First 6nm SKU in the catalog
# ---------------------------------------------------------------------------

def test_tsmc_n6_process_node_loads():
    """**First 6nm SKU**: tsmc_n6 process node added in this PR."""
    nodes = load_process_nodes()
    n6 = nodes.get("tsmc_n6")
    assert n6 is not None
    assert n6.node_nm == 6
    assert n6.foundry == "tsmc"


def test_qcs6490_uses_tsmc_n6(qcs6490):
    """**First 6nm SKU** in the compute_products catalog."""
    assert qcs6490.dies[0].process_node_id == "tsmc_n6"


# ---------------------------------------------------------------------------
# QCS6490-specific: 12 TOPS / 24 TOPS INT4 / 3-profile DVFS
# ---------------------------------------------------------------------------

def test_qcs6490_int8_peak_is_12_tops(qcs6490_block):
    """Qualcomm-marketed 12 TOPS INT8 (entry-level vs SA8775P's 32)."""
    int8_peak = qcs6490_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"]
    assert int8_peak == pytest.approx(12e12, rel=0.01)


def test_qcs6490_int4_is_2x_int8(qcs6490_block):
    """24 TOPS INT4 (2x INT8 throughput)."""
    int4_peak = qcs6490_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int4"]
    assert int4_peak == pytest.approx(24e12, rel=0.01)


def test_qcs6490_has_three_thermal_profiles(qcs6490_block):
    """3-profile DVFS: 5W (battery) + 10W (default) + 15W (max).
    Same profile count as SA8775P but in a smaller envelope."""
    assert len(qcs6490_block.thermal_profiles) == 3
    names = {p.name for p in qcs6490_block.thermal_profiles}
    assert names == {"5W", "10W", "15W"}


def test_qcs6490_default_profile_is_10w(qcs6490_block):
    """10W is the standard edge AI deployment."""
    assert qcs6490_block.default_thermal_profile_name == "10W"


def test_qcs6490_5w_is_battery_passive(qcs6490_block):
    p = next(p for p in qcs6490_block.thermal_profiles if p.name == "5W")
    assert p.tdp_watts == 5.0
    assert p.cooling_solution_id == "passive_fanless"


def test_qcs6490_15w_is_active_fan(qcs6490_block):
    p = next(p for p in qcs6490_block.thermal_profiles if p.name == "15W")
    assert p.tdp_watts == 15.0
    assert p.cooling_solution_id == "active_fan"


# ---------------------------------------------------------------------------
# Multi-fabric: HTA + HVX (16 units, scaled up from QRB5165's 4)
# ---------------------------------------------------------------------------

def test_qcs6490_has_two_fabrics(qcs6490_block):
    assert len(qcs6490_block.compute_fabrics) == 2


def test_qcs6490_hta_fabric_first(qcs6490_block):
    """HTA is fabric[0] -- primary INT8 path (~7.5 of 12 TOPS)."""
    hta = qcs6490_block.compute_fabrics[0]
    assert hta.fabric_kind == DSPFabricKind.TENSOR_MATRIX
    assert hta.num_units == 1
    assert hta.ops_per_unit_per_clock["int8"] == 5000


def test_qcs6490_hvx_has_16_units(qcs6490_block):
    """**16 HVX units** -- scaled up from QRB5165's 4 (more vector
    parallelism for the entry-level edge AI workload mix)."""
    hvx = qcs6490_block.compute_fabrics[1]
    assert hvx.fabric_kind == DSPFabricKind.VECTOR_SIMD
    assert hvx.num_units == 16


# ---------------------------------------------------------------------------
# Memory: LPDDR4X (cheaper than QRB5165's LPDDR5)
# ---------------------------------------------------------------------------

def test_qcs6490_uses_lpddr4x(qcs6490_block):
    """LPDDR4X (vs QRB5165's LPDDR5) -- cheaper memory for entry-level."""
    mem = qcs6490_block.memory
    assert mem.external_dram_type == MemoryType.LPDDR4X
    assert mem.external_dram_bandwidth_gbps == 40.0
    assert mem.external_dram_bandwidth_kind == "measured"
    assert mem.dram_attachment == DramAttachment.CHIP_ATTACHED
    assert mem.external_dram_size_gb == 8.0


# ---------------------------------------------------------------------------
# Common features
# ---------------------------------------------------------------------------

def test_qcs6490_deployment_kind_soc_integrated(qcs6490_block):
    assert qcs6490_block.deployment_kind == DSPDeploymentKind.SOC_INTEGRATED


def test_qcs6490_vliw_issue_width_is_4(qcs6490_block):
    """Hexagon DSP is 4-issue VLIW (same as SA8775P and QRB5165)."""
    assert qcs6490_block.vliw_issue_width == 4


def test_qcs6490_model_tier_entry(qcs6490):
    """Entry-level edge AI (vs QRB5165=mid, SA8775P=high)."""
    assert qcs6490.market.model_tier == "entry"


def test_qcs6490_wave_quantization_is_2(qcs6490_block):
    """Smaller wave grouping than SA8775P (4) or QRB5165 (4)."""
    assert qcs6490_block.wave_quantization == 2
