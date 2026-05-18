"""Tests for the fifth DSP SKU YAML: ti_tda4al.

Catalog gain: brings DSP count from 4 to 5+. Second TI TDA4 family
member (after TDA4VM). Same C7x + MMA architecture but with **MMAv2**
and a lower thermal ceiling (18W vs TDA4VM's 20W).

Shorter test file than TDA4VM since the schema features were
exercised first there. Focuses on TDA4AL-specific deltas: MMAv2
fabric, 18W thermal profile, ti/ vendor grouping.
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
def tda4al(all_products) -> ComputeProduct:
    cp = all_products.get("ti_tda4al")
    if cp is None:
        pytest.fail("ti_tda4al missing from catalog")
    return cp


@pytest.fixture(scope="module")
def tda4al_block(tda4al) -> DSPBlock:
    block = tda4al.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: second TI TDA4 family member
# ---------------------------------------------------------------------------

def test_ti_vendor_has_at_least_two_skus(all_products):
    """At this SKU's PR baseline ti/ had 2 entries; subset semantics
    so further TI TDA4 follow-ups (graphs#223) don't regress this."""
    ti_skus = {s for s, cp in all_products.items() if cp.vendor == "ti"}
    assert {"ti_tda4al", "ti_tda4vm"}.issubset(ti_skus)


def test_catalog_dsp_count_at_least_five(all_products):
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count >= 5


# ---------------------------------------------------------------------------
# TDA4AL-specific: MMAv2 + 18W profile
# ---------------------------------------------------------------------------

def test_tda4al_identity(tda4al):
    assert tda4al.id == "ti_tda4al"
    assert tda4al.vendor == "ti"


def test_tda4al_process_node_is_16nm(tda4al):
    """16nm FinFET per TI docs (same correction as TDA4VM)."""
    assert tda4al.dies[0].process_node_id == "tsmc_n16"


def test_tda4al_deployment_and_dram(tda4al_block):
    assert tda4al_block.deployment_kind == DSPDeploymentKind.SOC_INTEGRATED
    assert tda4al_block.memory.external_dram_bandwidth_kind == "measured"
    assert tda4al_block.memory.external_dram_type == MemoryType.LPDDR4X
    assert tda4al_block.memory.dram_attachment == DramAttachment.CHIP_ATTACHED


def test_tda4al_multi_fabric_c7x_plus_mma(tda4al_block):
    """Same shape as TDA4VM: C7x VLIW + MMA tensor."""
    assert len(tda4al_block.compute_fabrics) == 2
    assert tda4al_block.compute_fabrics[0].fabric_kind == DSPFabricKind.VLIW_SCALAR
    assert tda4al_block.compute_fabrics[1].fabric_kind == DSPFabricKind.TENSOR_MATRIX
    # 8 C7x cores + 1 MMAv2 unit
    assert tda4al_block.compute_fabrics[0].num_units == 8
    assert tda4al_block.compute_fabrics[1].num_units == 1


def test_tda4al_vliw_issue_width_is_8(tda4al_block):
    """Same C7x VLIW (8-wide) as TDA4VM."""
    assert tda4al_block.vliw_issue_width == 8


def test_tda4al_int8_peak_matches_marketed(tda4al_block):
    """8 TOPS INT8 (same marketed peak as TDA4VM; MMAv2 path)."""
    assert tda4al_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"] == pytest.approx(8e12, rel=0.01)


# ---------------------------------------------------------------------------
# Multi-profile DVFS: 10W passive + 18W active (vs TDA4VM's 10W + 20W)
# ---------------------------------------------------------------------------

def test_tda4al_thermal_profiles_are_10w_and_18w(tda4al_block):
    """**TDA4AL delta**: 18W max profile (vs TDA4VM's 20W) -- better
    power efficiency from MMAv2."""
    assert len(tda4al_block.thermal_profiles) == 2
    names = {p.name for p in tda4al_block.thermal_profiles}
    assert names == {"10W", "18W"}


def test_tda4al_default_profile_is_10w(tda4al_block):
    assert tda4al_block.default_thermal_profile_name == "10W"


def test_tda4al_18w_profile_has_higher_sustained_clock_than_tda4vm(tda4al_block):
    """TDA4AL @ 18W: 980 MHz sustained (vs TDA4VM @ 20W: 950 MHz)."""
    p = next(p for p in tda4al_block.thermal_profiles if p.name == "18W")
    assert p.tdp_watts == 18.0
    assert p.clock_mhz == 980.0
    assert p.cooling_solution_id == "active_fan"


def test_tda4al_10w_profile_uses_passive_cooling(tda4al_block):
    p = next(p for p in tda4al_block.thermal_profiles if p.name == "10W")
    assert p.tdp_watts == 10.0
    assert p.cooling_solution_id == "passive_fanless"


# ---------------------------------------------------------------------------
# Chip-level power
# ---------------------------------------------------------------------------

def test_tda4al_chip_power_envelope(tda4al):
    """Default 10W; max 20W (covers 18W profile + boost headroom)."""
    assert tda4al.power.tdp_watts == 10.0
    assert tda4al.power.default_thermal_profile == "10W"
    assert tda4al.power.max_power_watts >= 18.0
