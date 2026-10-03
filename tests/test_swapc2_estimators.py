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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import swapc2_estimators as est  # noqa: E402

from embodied_schemas.loaders import load_cooling_solutions, load_process_nodes  # noqa: E402
from embodied_schemas.process_node import DataConfidence  # noqa: E402
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


def test_sizing_needs_an_air_rating():
    spreader = load_cooling_solutions()["smarc_heat_spreader_82x50"]
    with pytest.raises(ValueError, match="no ambient_c_max"):
        est.delta_t(spreader)


# ---------------------------------------------------------------------------
# Review hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value", [("delta_t", 0.0), ("delta_t", -5.0), ("r_vol", 0.0), ("solid_fraction", 1.5)]
)
def test_size_cooling_rejects_bad_inputs(field, value):
    good = dict(r_vol=P(600.0), delta_t=P(40.0), solid_fraction=P(0.2), density=P(2.7))
    good[field] = P(value)
    with pytest.raises(ValueError, match=field):
        est.size_cooling(est.CoolingSizing(**good))


def test_render_keeps_inline_comment_on_owned_field():
    text = "id: x\nweight_g: 400.0  # mounting hardware included\nsource: s\n"
    out = est.render(text, {"weight_g": 20.0}, est.COOLING_OWNED)
    assert "weight_g: 20.0  # mounting hardware included\n" in out


def test_write_fails_on_an_invalid_rewrite(tmp_path, monkeypatch):
    import shutil

    cool = tmp_path / "cooling-solutions"
    shutil.copytree(est.COOLING_DIR, cool)
    monkeypatch.setattr(est, "COOLING_DIR", cool)
    monkeypatch.setattr(est, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(est, "expected_cooling_fields", lambda _id: {"weight_g": -1.0})
    with pytest.raises(Exception, match="weight_g"):
        est.run_cooling(write=True)


# ---------------------------------------------------------------------------
# products: die cost on the compute-product catalog (S3a)
# ---------------------------------------------------------------------------


def test_products_match_silicon_cost():
    """Every eligible product's die_cost_usd is exactly what silicon_cost_v1 gives."""
    assert est.run_products(write=False) == 0


def test_die_cost_eligibility():
    from embodied_schemas.loaders import load_compute_products

    nodes = load_process_nodes()
    products = load_compute_products()
    eligible = {p for p, cp in products.items() if est.die_cost_eligible(cp, nodes)}
    assert "kpu_t64_32x32_lp5x4_7nm_tsmc_hpc" in eligible
    assert "amd_epyc_9654_sp5" not in eligible  # aggregated chiplets (13 dies as 2)
    assert "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp" not in eligible  # N16 has no sourced D0
    assert "seco_som_smarc_qcs6490" not in eligible  # a module has no dies of its own
    for pid in eligible:
        cost = products[pid].swapc2.cost.die_cost_usd
        assert cost.basis == ValueBasis.ESTIMATED and est.SILICON_COST in cost.source


def test_die_cost_value_matches_model():
    from embodied_schemas.loaders import load_compute_products

    p = load_compute_products()["kpu_t512_32x32_lp5x32_7nm_tsmc_hpc"]
    node = load_process_nodes()[p.dies[0].process_node_id]
    expected = est.die_cost(p.dies[0].die_size_mm2, node).cost_usd.value
    assert p.swapc2.cost.die_cost_usd.value == pytest.approx(expected, abs=0.005)


def test_die_cost_is_not_a_unit_cost():
    from embodied_schemas import resolve_swapc2
    from embodied_schemas.loaders import load_compute_products

    p = load_compute_products()["google_tpu_edge_pro"]
    r = resolve_swapc2(p, cooling=load_cooling_solutions())
    assert r.die_cost_usd is not None
    assert r.unit_cost_1_usd is None  # no stated price; die cost does not stand in


def test_render_emits_mapping_block():
    text = "id: x\nconfidence: theoretical\n"
    out = est.render(
        text,
        {"swapc2": {"cost": {"die_cost_usd": {"value": 1.5}}}},
        ("swapc2",),
        anchor="confidence",
    )
    assert out == (
        "id: x\nswapc2:\n  cost:\n    die_cost_usd:\n      value: 1.5\n" "confidence: theoretical\n"
    )


# ---------------------------------------------------------------------------
# PR #106 review: anchor-less insert, targeted nested edit, no PDK overlay
# ---------------------------------------------------------------------------


def test_render_appends_when_anchor_absent():
    out = est.render("id: x\nname: y", {"basis": "estimated"}, est.COOLING_OWNED)
    assert out == 'id: x\nname: y\nbasis: "estimated"\n'


DIE = {"value": 2.5, "basis": "estimated"}

EXISTING = """id: p
swapc2:
  # the module's own envelope
  size:
    dimensions_mm: {length_mm: 1.0}   # keep me

  cost:
    unit_price_1_usd: {value: 9.0}  # list price
    die_cost_usd:
      value: 1.0
      basis: estimated

  power:
    input_voltage_v: [5.0, 5.0]
confidence: theoretical
"""


class TestSetNested:
    def test_replaces_only_the_leaf(self):
        out = est.set_nested(EXISTING, est.DIE_COST_PATH, DIE, anchor="confidence")
        assert "      value: 2.5\n" in out and "value: 1.0" not in out
        for kept in (
            "  # the module's own envelope\n",
            "{length_mm: 1.0}   # keep me\n",
            "{value: 9.0}  # list price\n",
            "  power:\n",
            "input_voltage_v: [5.0, 5.0]\n",
        ):
            assert kept in out
        # The blank lines inside swapc2 survive and nothing stale is left behind.
        assert out.count("\n\n") == 2
        assert yaml.safe_load(out)["swapc2"]["cost"] == {
            "unit_price_1_usd": {"value": 9.0},
            "die_cost_usd": DIE,
        }

    def test_idempotent(self):
        once = est.set_nested(EXISTING, est.DIE_COST_PATH, DIE, anchor="confidence")
        assert est.set_nested(once, est.DIE_COST_PATH, DIE, anchor="confidence") == once

    def test_adds_missing_leaf_inside_existing_parent(self):
        text = "swapc2:\n  cost:\n    unit_price_1_usd: {value: 9.0}  # keep\nconfidence: x\n"
        out = est.set_nested(text, est.DIE_COST_PATH, DIE, anchor="confidence")
        assert "{value: 9.0}  # keep\n    die_cost_usd:\n" in out
        assert yaml.safe_load(out)["swapc2"]["cost"]["die_cost_usd"] == DIE

    def test_adds_missing_branch_inside_existing_swapc2(self):
        text = "swapc2:\n  power:\n    input_voltage_v: [5.0, 5.0]\nconfidence: x\n"
        out = est.set_nested(text, est.DIE_COST_PATH, DIE, anchor="confidence")
        data = yaml.safe_load(out)["swapc2"]
        assert data["power"] == {"input_voltage_v": [5.0, 5.0]}
        assert data["cost"]["die_cost_usd"] == DIE

    def test_creates_block_before_anchor(self):
        out = est.set_nested("id: p\nconfidence: x\n", est.DIE_COST_PATH, DIE, anchor="confidence")
        assert out.startswith("id: p\nswapc2:\n  cost:\n    die_cost_usd:\n")
        assert out.endswith("confidence: x\n")

    def test_appends_without_anchor(self):
        """A product without `confidence:` (it has a default) still gets its cost."""
        out = est.set_nested("id: p\nname: q", est.DIE_COST_PATH, DIE, anchor="confidence")
        assert yaml.safe_load(out) == {
            "id": "p",
            "name": "q",
            "swapc2": {"cost": {"die_cost_usd": DIE}},
        }


def test_products_writer_ignores_private_overlay(tmp_path, monkeypatch):
    """A confidential PDK overlay must not feed public die costs: the value
    the writer writes is the public-node calculation, not the overlay's."""
    import shutil

    from embodied_schemas.loaders import load_compute_products

    pid = "kpu_t64_32x32_lp5x4_7nm_tsmc_hpc"
    public = load_process_nodes(include_overlay=False)["tsmc_n7"]
    overlay = public.model_copy(
        update={
            "confidence": DataConfidence.CALIBRATED,
            "wafer_cost_usd": 1.0,
            "wafer_cost_source": "CONFIDENTIAL PDK",
        }
    )
    pdk = tmp_path / "pdk"
    pdk.mkdir()
    (pdk / "n7.yaml").write_text(yaml.safe_dump(overlay.model_dump(mode="json")))
    monkeypatch.setenv("PROCESS_NODE_DATA_DIR", str(pdk))
    assert load_process_nodes()["tsmc_n7"].wafer_cost_usd == 1.0  # the overlay is live

    # Write into a copy of the catalog whose target product has no die cost yet.
    products_dir = tmp_path / "compute_products"
    shutil.copytree(est.PRODUCTS_DIR, products_dir)
    monkeypatch.setattr(est, "PRODUCTS_DIR", products_dir)
    monkeypatch.setattr(est, "REPO_ROOT", tmp_path)
    target = est._paths_by_id(products_dir)[pid]
    data = yaml.safe_load(target.read_text())
    del data["swapc2"]
    target.write_text(yaml.safe_dump(data, sort_keys=False))
    assert est.run_products(write=True) == 0

    written = yaml.safe_load(target.read_text())["swapc2"]["cost"]["die_cost_usd"]
    die = load_compute_products()[pid].dies[0]
    expected = est.die_cost(die.die_size_mm2, public).cost_usd.value
    assert written["value"] == pytest.approx(round(expected, 2))
    assert written["value"] != pytest.approx(
        round(est.die_cost(die.die_size_mm2, overlay).cost_usd.value, 2)
    )
    assert "CONFIDENTIAL" not in target.read_text()


class TestSetNestedFlowStyle:
    def test_flow_parent_refused(self):
        text = "swapc2:\n  cost: {unit_price_1_usd: {value: 9.0}}\nconfidence: x\n"
        with pytest.raises(ValueError, match="swapc2.cost is not a block mapping"):
            est.set_nested(text, est.DIE_COST_PATH, DIE, anchor="confidence")

    def test_multiline_flow_parent_refused(self):
        text = "swapc2:\n  cost:\n    {unit_price_1_usd: {value: 9.0},\n     x: 1}\nconfidence: x\n"
        with pytest.raises(ValueError, match="swapc2.cost is not a block mapping"):
            est.set_nested(text, est.DIE_COST_PATH, DIE, anchor="confidence")

    def test_flow_leaf_is_replaced(self):
        """A flow-style leaf under block parents is fine: the whole entry is replaced."""
        text = "swapc2:\n  cost:\n    die_cost_usd: {value: 1.0}  # old\nconfidence: x\n"
        out = est.set_nested(text, est.DIE_COST_PATH, DIE, anchor="confidence")
        assert yaml.safe_load(out)["swapc2"]["cost"]["die_cost_usd"] == DIE

    @pytest.mark.parametrize(
        "text",
        ["swapc2:\nconfidence: x\n", "swapc2:\n  cost:\nconfidence: x\n"],
        ids=["empty_swapc2", "empty_cost"],
    )
    def test_empty_parent_is_a_mapping(self, text):
        out = est.set_nested(text, est.DIE_COST_PATH, DIE, anchor="confidence")
        assert yaml.safe_load(out)["swapc2"]["cost"]["die_cost_usd"] == DIE

    def test_result_is_verified(self, monkeypatch):
        """If the line edit ever goes wrong, the re-parse catches it before a write."""
        monkeypatch.setattr(est, "_set_nested_lines", lambda *a: "swapc2: [unbalanced\n")
        with pytest.raises(ValueError, match="invalid YAML"):
            est.set_nested("id: p\n", est.DIE_COST_PATH, DIE)
        monkeypatch.setattr(est, "_set_nested_lines", lambda *a: "id: changed\n")
        with pytest.raises(ValueError, match="changed more than the target"):
            est.set_nested("id: p\n", est.DIE_COST_PATH, DIE)


def test_price_fit_skips_heatsinks_without_full_envelope():
    """A priced heatsink missing a dimension is left out, not an IndexError."""
    from embodied_schemas.sources import Observation, SourceDB

    db = est.source_db()
    partial = [
        Observation(
            category="heatsink",
            subject="partial_sink",
            quantity=q,
            value=v,
            unit=u,
            as_of="2026",
            basis="datasheet",
            source_id="lee_1995_select_heat_sink",
            quote="test",
            variant=var,
            conditions=cond,
        )
        for q, v, u, var, cond in [
            ("length", 30.0, "mm", None, {}),
            ("width", 30.0, "mm", None, {}),
            ("mass", 20.0, "g", None, {}),
            ("unit_price", 1000.0, "usd", "qty_1", {"quantity": 1}),
        ]
    ]
    extended = SourceDB(list(db.documents.values()), list(db.observations.values()) + partial)
    assert "partial_sink" not in est.heatsink_subjects(extended)
    assert est.sink_price_fit(extended) == est.sink_price_fit(db)  # its $1000 is ignored
