"""Contract test for the GF 28nm process node YAML.

Precursor for the Coral Edge TPU follow-up tracked at
branes-ai/graphs#192 (Coral half). The Coral SKU YAML will reference
this node via ``process_node_id: gf_28nm``; this test pins the entry's
identity, the bulk-planar topology, and the key density / energy /
leakage values so the SKU YAML can be authored against a stable base.

Same shape as the per-node existence/value tests added in earlier
sprint data PRs (e.g. ``test_compute_product_v2_jetson_agx_thor_yaml``
for ``tsmc_n4p``).
"""

import pytest

from embodied_schemas.loaders import load_process_nodes
from embodied_schemas.process_node import (
    CircuitClass,
    TransistorTopology,
)


@pytest.fixture(scope="module")
def gf_28nm():
    nodes = load_process_nodes()
    node = nodes.get("gf_28nm")
    if node is None:
        pytest.fail("gf_28nm missing from process node catalog")
    return node


# ---------------------------------------------------------------------------
# Identity / topology
# ---------------------------------------------------------------------------

def test_gf_28nm_identity(gf_28nm):
    assert gf_28nm.id == "gf_28nm"
    assert gf_28nm.foundry.value == "globalfoundries"
    assert gf_28nm.node_name == "28SLP"
    assert gf_28nm.node_nm == 28


def test_gf_28nm_is_bulk_planar(gf_28nm):
    """28nm predates FinFET (12LP, 16FF+) and is structurally bulk
    planar HKMG. First entry in the catalog exercising the
    ``BULK_PLANAR`` enum value."""
    assert gf_28nm.transistor_topology == TransistorTopology.BULK_PLANAR


def test_gf_28nm_no_body_bias(gf_28nm):
    """Bulk planar has no body terminal accessible to the designer
    (unlike FD-SOI 12FDX which has +/-3V body bias range)."""
    assert gf_28nm.body_bias_supported is False


def test_gf_28nm_vdd_is_slp_nominal(gf_28nm):
    """28SLP nominal is 0.85 V (vs 1.0 V for 28HP)."""
    assert gf_28nm.nominal_vdd_v == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Densities -- the key field set for downstream Mtx <-> area conversion
# ---------------------------------------------------------------------------

def test_gf_28nm_density_libraries_present(gf_28nm):
    """The catalog needs at minimum hp / balanced / sram_hd for the
    Coral SKU silicon_bin reconciliation; lp_logic and others are
    optional for completeness."""
    keys = {k.value if hasattr(k, "value") else str(k)
            for k in gf_28nm.densities}
    required = {"hp_logic", "balanced_logic", "sram_hd", "io"}
    assert required.issubset(keys), (
        f"missing required density libraries: {required - keys}"
    )


def test_gf_28nm_balanced_logic_density(gf_28nm):
    """28nm planar density is ~2.5x lower than 12LP FinFET (which is
    30 Mtx/mm^2 for balanced). Pin 12 Mtx/mm^2 so a future revision
    that drifts the density fires a deliberate-update reminder."""
    balanced = gf_28nm.densities[CircuitClass.BALANCED_LOGIC]
    assert balanced.mtx_per_mm2 == pytest.approx(12.0)


def test_gf_28nm_sram_hd_density(gf_28nm):
    """28nm 6T HD-SRAM bitcell is ~0.124 um^2; with periphery
    overhead the entry-level density is ~28 Mtx/mm^2 (vs 12LP at 65)."""
    sram_hd = gf_28nm.densities[CircuitClass.SRAM_HD]
    assert sram_hd.mtx_per_mm2 == pytest.approx(28.0)


# ---------------------------------------------------------------------------
# Energy: confirms the Coral 4 TOPS / 2 W envelope is plausible at the
# raw energy_per_op_pj value (with systolic-array reuse advantage applied
# at the SKU level).
# ---------------------------------------------------------------------------

def test_gf_28nm_balanced_logic_int8_energy(gf_28nm):
    """0.60 pJ per INT8 op at 28nm planar balanced logic. Coral Edge
    TPU at 4 TOPS / 2 W = ~0.5 pJ/op including systolic reuse; raw
    0.60 pJ with ~20% systolic reuse advantage matches."""
    assert gf_28nm.energy_per_op_pj["balanced_logic:int8"] == pytest.approx(0.60)


def test_gf_28nm_balanced_logic_fp32_energy(gf_28nm):
    """~5.5 pJ FP32 -- 28nm planar penalty vs ~2.7 pJ at 16FF+."""
    assert gf_28nm.energy_per_op_pj["balanced_logic:fp32"] == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Leakage -- SLP is the low-leakage 28nm variant
# ---------------------------------------------------------------------------

def test_gf_28nm_lp_logic_leakage_is_low(gf_28nm):
    """28SLP lp_logic leakage at 0.0035 W/mm^2 -- the SLP variant's
    selling point is low leakage for always-on edge / battery flows."""
    assert gf_28nm.leakage_w_per_mm2[CircuitClass.LP_LOGIC] == pytest.approx(0.0035)


# ---------------------------------------------------------------------------
# Cooling -- 28nm at edge TDPs (2-5 W) is passive-friendly
# ---------------------------------------------------------------------------

def test_gf_28nm_supports_passive_cooling(gf_28nm):
    """Coral Edge TPU is 2.0 W passive; the node must declare
    compatibility with passive heatsink cooling."""
    cooling = set(gf_28nm.cooling_compatible)
    assert "passive_heatsink_small" in cooling
    assert "passive_fanless" in cooling
