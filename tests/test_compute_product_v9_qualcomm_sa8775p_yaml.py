"""Tests for the eighth DSP SKU YAML: qualcomm_sa8775p.

**Opens batch 3 of #223**: Qualcomm family. Most complex DSP YAML to
date. Exercises several DSPBlock schema features for the first time:

  - **First 3-profile DVFS DSP**: 20W passive + 30W active + 45W max
  - **First TSMC N5 (5nm)** DSP SKU
  - **First Hexagon DSP** in the catalog (vs C7x for TI)
  - INT4 + INT8 + INT16 + FP16 mixed precision on both fabrics

Multi-fabric: HMX tensor (PRIMARY; 30 TOPS) + HVX vector (2 TOPS) =
32 TOPS marketed INT8. Tensor-first ordering means loader picks HMX
for chip-level surfaces (compute_units / energy_per_flop_fp32).
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
def sa8775p(all_products) -> ComputeProduct:
    cp = all_products.get("qualcomm_sa8775p")
    if cp is None:
        pytest.fail("qualcomm_sa8775p missing from catalog")
    return cp


@pytest.fixture(scope="module")
def sa8775p_block(sa8775p) -> DSPBlock:
    block = sa8775p.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: opens Qualcomm batch
# ---------------------------------------------------------------------------

def test_catalog_includes_qualcomm_sa8775p(all_products):
    """**First Qualcomm DSP SKU** -- opens batch 3 of graphs#223."""
    qualcomm_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "qualcomm")
    assert "qualcomm_sa8775p" in qualcomm_skus


def test_catalog_dsp_count_at_least_eight(all_products):
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count >= 8


# ---------------------------------------------------------------------------
# First TSMC N5 DSP
# ---------------------------------------------------------------------------

def test_sa8775p_process_node_is_tsmc_n5(sa8775p):
    """**First DSP SKU on 5nm**. All prior DSPs are 16nm; TDA4 family
    is 16nm; IP cores are 16nm. SA8775P advances to TSMC N5."""
    assert sa8775p.dies[0].process_node_id == "tsmc_n5"


# ---------------------------------------------------------------------------
# First 3-profile DVFS DSP
# ---------------------------------------------------------------------------

def test_sa8775p_has_three_thermal_profiles(sa8775p_block):
    """**First 3-profile DVFS DSP**: 20W passive cockpit + 30W active
    ADAS + 45W max L3-autonomous. Prior DSPs maxed at 2 profiles
    (TDA4VM/AL/VH/VL); IP cores had 1."""
    assert len(sa8775p_block.thermal_profiles) == 3
    names = {p.name for p in sa8775p_block.thermal_profiles}
    assert names == {"20W", "30W", "45W"}


def test_sa8775p_default_profile_is_30w(sa8775p_block):
    """30W is the most common automotive deployment (ADAS + cockpit)."""
    assert sa8775p_block.default_thermal_profile_name == "30W"


def test_sa8775p_20w_passive(sa8775p_block):
    """20W is passive cooling for cockpit-compute deployments."""
    p = next(p for p in sa8775p_block.thermal_profiles if p.name == "20W")
    assert p.tdp_watts == 20.0
    assert p.cooling_solution_id == "passive_fanless"
    assert p.clock_mhz == 1600.0   # 67% of 2.4 GHz boost


def test_sa8775p_30w_active(sa8775p_block):
    """30W is active fan for ADAS + cockpit."""
    p = next(p for p in sa8775p_block.thermal_profiles if p.name == "30W")
    assert p.tdp_watts == 30.0
    assert p.cooling_solution_id == "active_fan"
    assert p.clock_mhz == 2000.0


def test_sa8775p_45w_max_performance(sa8775p_block):
    """45W is max-performance for L3 autonomous driving."""
    p = next(p for p in sa8775p_block.thermal_profiles if p.name == "45W")
    assert p.tdp_watts == 45.0
    assert p.clock_mhz == 2300.0


# ---------------------------------------------------------------------------
# Multi-fabric: HMX tensor + HVX vector (HMX first / primary)
# ---------------------------------------------------------------------------

def test_sa8775p_has_two_fabrics(sa8775p_block):
    assert len(sa8775p_block.compute_fabrics) == 2


def test_sa8775p_hmx_fabric_is_primary(sa8775p_block):
    """HMX (Hexagon Matrix eXtensions) is fabric[0] -- the canonical
    INT8 path (30 of 32 TOPS). Loader picks HMX for chip-level surfaces."""
    hmx = sa8775p_block.compute_fabrics[0]
    assert hmx.fabric_kind == DSPFabricKind.TENSOR_MATRIX
    assert hmx.num_units == 2                      # Dual HMX units
    assert hmx.ops_per_unit_per_clock["int8"] == 7500


def test_sa8775p_hvx_fabric_is_secondary(sa8775p_block):
    """HVX (Hexagon Vector eXtensions) is fabric[1] -- 1024-bit SIMD
    for activations/pre/post-processing."""
    hvx = sa8775p_block.compute_fabrics[1]
    assert hvx.fabric_kind == DSPFabricKind.VECTOR_SIMD
    assert hvx.num_units == 4                       # Quad HVX units
    assert hvx.ops_per_unit_per_clock["int8"] == 256


# ---------------------------------------------------------------------------
# Precision peaks: 32 TOPS INT8 marketed; 64 TOPS INT4
# ---------------------------------------------------------------------------

def test_sa8775p_int8_peak_matches_marketed(sa8775p_block):
    """32 TOPS INT8 (Qualcomm-marketed); HMX dominates."""
    assert sa8775p_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"] == pytest.approx(32e12, rel=0.01)


def test_sa8775p_int4_is_2x_int8(sa8775p_block):
    """64 TOPS INT4 (2x INT8 throughput)."""
    assert sa8775p_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int4"] == pytest.approx(64e12, rel=0.01)


def test_sa8775p_fp16_is_8_tflops(sa8775p_block):
    """8 TFLOPS FP16 (half of INT16)."""
    assert sa8775p_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["fp16"] == pytest.approx(8e12, rel=0.01)


# ---------------------------------------------------------------------------
# Memory: LPDDR5 automotive-grade
# ---------------------------------------------------------------------------

def test_sa8775p_uses_lpddr5(sa8775p_block):
    mem = sa8775p_block.memory
    assert mem.external_dram_type == MemoryType.LPDDR5
    assert mem.external_dram_bandwidth_gbps == 90.0
    assert mem.external_dram_bandwidth_kind == "measured"
    assert mem.dram_attachment == DramAttachment.CHIP_ATTACHED
    assert mem.external_dram_size_gb == 16.0


# ---------------------------------------------------------------------------
# Common features (already validated by other DSP SKUs)
# ---------------------------------------------------------------------------

def test_sa8775p_deployment_kind_soc_integrated(sa8775p_block):
    assert sa8775p_block.deployment_kind == DSPDeploymentKind.SOC_INTEGRATED


def test_sa8775p_vliw_issue_width_is_4(sa8775p_block):
    """**Hexagon DSP is 4-issue VLIW** (vs TI C7x's 8-issue).
    First DSP with vliw_issue_width != 8."""
    assert sa8775p_block.vliw_issue_width == 4


def test_sa8775p_multi_task_scheduling(sa8775p_block):
    """Automotive multi-task: ADAS + cockpit concurrent. 16 kernels."""
    assert sa8775p_block.max_concurrent_kernels == 16
    assert sa8775p_block.min_occupancy == 0.5    # Lower than TI's 0.70 (more flexible)


def test_sa8775p_model_tier_high(sa8775p):
    """High-end ADAS L2+/L3."""
    assert sa8775p.market.model_tier == "high"


def test_sa8775p_chip_power_envelope(sa8775p):
    """Default 30W; max 50W (covers 45W profile + boost)."""
    assert sa8775p.power.tdp_watts == 30.0
    assert sa8775p.power.default_thermal_profile == "30W"
    assert sa8775p.power.max_power_watts >= 45.0
