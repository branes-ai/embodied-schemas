"""Tests for the SWaP-C² estimators in ``scripts/swapc2_estimators.py``
(RFC 0001 R1.2, phase S2).

1. silicon_cost_v1: gross dies per wafer, Murphy yield, cost per good die.
2. cooling_sizing_v1: per-W volume / mass / cost from the volumetric
   thermal-resistance model, and the fixed fan / pump parts.
3. The YAML writer keeps comments and unowned fields, and the catalog is in
   sync with the estimator (``cooling --check``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import swapc2_estimators as est  # noqa: E402

from embodied_schemas.loaders import load_cooling_solutions, load_process_nodes  # noqa: E402
from embodied_schemas.swapc2 import ValueBasis  # noqa: E402

# ---------------------------------------------------------------------------
# 1. silicon_cost_v1
# ---------------------------------------------------------------------------


class TestSiliconCost:
    def test_gross_dies_per_wafer(self):
        # pi*150^2/100 - pi*300/sqrt(200) = 706.86 - 66.64 = 640.2
        assert est.gross_dies_per_wafer(100.0) == 640

    def test_bigger_dies_fewer_per_wafer(self):
        assert est.gross_dies_per_wafer(800.0) < est.gross_dies_per_wafer(100.0)

    def test_smaller_wafer(self):
        assert est.gross_dies_per_wafer(100.0, 200.0) < est.gross_dies_per_wafer(100.0, 300.0)

    @pytest.mark.parametrize("bad", [(0.0, 300.0), (100.0, 0.0)])
    def test_dies_per_wafer_rejects_non_positive(self, bad):
        with pytest.raises(ValueError):
            est.gross_dies_per_wafer(*bad)

    def test_murphy_defect_free(self):
        assert est.murphy_yield(100.0, 0.0) == 1.0

    def test_murphy_at_ad_one(self):
        # A = 1 cm^2, D0 = 1 / cm^2: ((1 - e^-1) / 1)^2
        assert est.murphy_yield(100.0, 1.0) == pytest.approx((1 - math.exp(-1)) ** 2)

    def test_murphy_falls_with_area(self):
        assert est.murphy_yield(600.0, 0.1) < est.murphy_yield(100.0, 0.1)

    def test_die_cost(self):
        node = load_process_nodes()["tsmc_n7"].model_copy(
            update={"wafer_cost_usd": 10000.0, "defect_density_per_cm2": 0.1}
        )
        c = est.die_cost(100.0, node)
        y = ((1 - math.exp(-0.1)) / 0.1) ** 2
        assert c.gross_dies == 640
        assert c.yield_fraction == pytest.approx(y)
        assert c.cost_usd.value == pytest.approx(10000.0 / (640 * y))
        assert c.cost_usd.basis == ValueBasis.ESTIMATED
        assert est.SILICON_COST in c.cost_usd.source

    def test_die_cost_needs_inputs(self):
        node = load_process_nodes()["tsmc_n7"].model_copy(
            update={"wafer_cost_usd": None, "defect_density_per_cm2": None}
        )
        with pytest.raises(ValueError, match="required"):
            est.die_cost(100.0, node)


# ---------------------------------------------------------------------------
# 2. cooling_sizing_v1
# ---------------------------------------------------------------------------


def P(value):
    return est.Param(value, "test")


class TestCoolingSizing:
    def test_volume_and_mass_per_watt(self):
        s = est.CoolingSizing(
            r_vol=P(600.0), delta_t=P(40.0), solid_fraction=P(0.2), density=P(2.7)
        )
        fields = est.size_cooling(s)
        assert fields["volume_cm3_per_w"] == pytest.approx(15.0)
        assert fields["mass_g_per_w"] == pytest.approx(15.0 * 0.2 * 2.7)
        assert "cost_usd_per_w" not in fields and "weight_g" not in fields

    def test_fixed_parts_and_cost(self):
        s = est.CoolingSizing(
            r_vol=P(100.0),
            delta_t=P(50.0),
            solid_fraction=P(0.25),
            density=P(2.7),
            base_mass_g=P(20.0),
            base_cost_usd=P(4.0),
            parasitic_power_w=P(0.6),
            cost_per_cm3=P(0.05),
        )
        fields = est.size_cooling(s)
        assert fields["weight_g"] == 20.0
        assert fields["cost_usd"] == 4.0
        assert fields["parasitic_power_w"] == 0.6
        assert fields["cost_usd_per_w"] == pytest.approx(2.0 * 0.05)

    def test_sizing_source_names_model_and_params(self):
        s = est.CoolingSizing(
            r_vol=est.Param(600.0, "vendor note"),
            delta_t=P(40.0),
            solid_fraction=P(0.2),
            density=P(2.7),
        )
        src = est.sizing_source(s)
        assert src.startswith(est.COOLING_SIZING)
        assert "r_vol=600 [vendor note]" in src


# ---------------------------------------------------------------------------
# 3. YAML writer and catalog sync
# ---------------------------------------------------------------------------

YAML = """# A comment that must survive.
id: x
weight_g: 400.0
cost_usd: 25.0
source: "ref"   # trailing comment
confidence: theoretical
"""


class TestRender:
    def test_replaces_in_place_and_inserts_before_source(self):
        out = est.render(
            YAML, {"weight_g": 20.0, "mass_g_per_w": 1.5, "basis": "estimated"}, est.COOLING_OWNED
        )
        assert out.startswith("# A comment that must survive.\nid: x\nweight_g: 20.0\n")
        assert 'mass_g_per_w: 1.5\nbasis: "estimated"\nsource: "ref"   # trailing comment' in out

    def test_removes_owned_fields_not_produced(self):
        out = est.render(YAML, {"weight_g": 20.0}, est.COOLING_OWNED)
        assert "cost_usd" not in out

    def test_idempotent(self):
        fields = {"weight_g": 20.0, "mass_g_per_w": 1.5}
        once = est.render(YAML, fields, est.COOLING_OWNED)
        assert est.render(once, fields, est.COOLING_OWNED) == once


def test_catalog_matches_estimator():
    """Every sized cooling YAML holds exactly what cooling_sizing_v1 writes."""
    assert est.run_cooling(write=False) == 0


def test_process_nodes_match_source_db():
    """Every node's silicon-cost inputs are exactly what the source DB gives."""
    assert est.run_nodes(write=False) == 0


def test_no_hand_entered_wafer_costs():
    """A node carries silicon-cost inputs only through NODE_INPUTS."""
    for node in load_process_nodes().values():
        if node.wafer_cost_usd is not None or node.defect_density_per_cm2 is not None:
            assert node.id in est.NODE_INPUTS, node.id


def test_render_replaces_block_scalars():
    text = 'id: x\nwafer_cost_source: >-\n  line one\n  line two\nsource: "s"\n'
    out = est.render(text, {"wafer_cost_source": "new"}, est.NODE_OWNED)
    assert out == 'id: x\nwafer_cost_source: "new"\nsource: "s"\n'


def test_sized_entries_validate_and_are_estimated():
    cooling = load_cooling_solutions()
    for entry_id in est.SIZED_ENTRIES:
        entry = cooling[entry_id]
        assert entry.basis == ValueBasis.ESTIMATED
        assert entry.sizing_source.startswith(est.COOLING_SIZING)
        assert entry.mass_g_per_w is not None and entry.volume_cm3_per_w is not None


def test_delta_t_matches_entry_limits():
    """Each sized entry's dT is its own junction_c_max - ambient_c_max."""
    cooling = load_cooling_solutions()
    for entry_id in est.SIZED_ENTRIES:
        sizing = est.cooling_sizing(entry_id)
        entry = cooling[entry_id]
        assert sizing.delta_t.value == entry.junction_c_max - entry.ambient_c_max, entry_id


def test_every_param_is_sourced():
    for entry_id in est.SIZED_ENTRIES:
        sizing = est.cooling_sizing(entry_id)
        for name in est.CoolingSizing.__dataclass_fields__:
            param = getattr(sizing, name)
            assert param is None or param.source.strip(), (entry_id, name)
