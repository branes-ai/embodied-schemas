"""Hailo M.2 modules (RFC 0001 S3b): the module YAMLs hold exactly the
figures recorded in the source database (``data/sources/observations/
m2_modules.yaml``), and nothing unsourced.

1. Structure: ``module`` products containing the existing Hailo chips.
2. Every SWaP-C² value equals its source-DB figure: dimensions, mass, price,
   and the thermal-profile powers.
3. What no source gives stays unset (10H mass, 1K prices).
4. SWaP-C² resolves through the sized passive heatsink.
"""

import pytest

from embodied_schemas import ProductKind, ValueBasis, aggregate_peak, resolve_swapc2
from embodied_schemas.hardware import FormFactor
from embodied_schemas.loaders import load_compute_products, load_cooling_solutions
from embodied_schemas.sources import load_source_db

H8 = "hailo_8_m2_2242_m"
H10 = "hailo_10h_m2_2280_8gb"


@pytest.fixture(scope="module")
def products():
    return load_compute_products()


@pytest.fixture(scope="module")
def db():
    return load_source_db()


class TestStructure:
    @pytest.mark.parametrize("module,chip", [(H8, "hailo_hailo_8"), (H10, "hailo_hailo_10h")])
    def test_module_contains_its_chip(self, products, module, chip):
        m = products[module]
        assert m.kind is ProductKind.MODULE and m.dies == []
        assert [r.id for r in m.contains] == [chip]
        assert m.packaging.form_factor is FormFactor.M2

    def test_peak_reaches_the_chip(self, products):
        assert aggregate_peak(products[H8], products).sum["int8"] == pytest.approx(26e12)
        assert aggregate_peak(products[H10], products).sum["int8"] == pytest.approx(20e12)


class TestHeldToSourceDB:
    def test_hailo_8_dimensions(self, products, db):
        d = products[H8].swapc2.size.dimensions_mm
        for q in ("length", "width", "height"):
            assert getattr(d, f"{q}_mm") == db.value(f"{H8}.{q}@hailo_8_m2_key_m_datasheet"), q

    def test_hailo_10h_dimensions(self, products, db):
        d = products[H10].swapc2.size.dimensions_mm
        src = "hailo_10h_m2_key_m_datasheet"
        assert d.length_mm == db.value(f"{H10}.length@{src}")
        assert d.width_mm == db.value(f"{H10}.width@{src}")
        expected_h = (
            db.value(f"{H10}.height.top_components@{src}")
            + db.value("m2_card_pcb.height@hackaday_2022_m2_for_hackers")
            + db.value(f"{H10}.height.bottom_components@{src}")
        )
        assert d.height_mm == pytest.approx(expected_h)
        assert d.basis == ValueBasis.DERIVED

    def test_mass_and_prices(self, products, db):
        h8 = products[H8].swapc2
        assert h8.weight.mass_g.value == db.value(f"{H8}.mass@waveshare_hailo_8")
        assert h8.cost.unit_price_1_usd.value == db.value(
            f"{H8}.unit_price.qty_1@waveshare_hailo_8"
        )
        h10 = products[H10].swapc2
        assert h10.cost.unit_price_1_usd.value == db.value(
            f"{H10}.unit_price.qty_1@upshop_hailo_10h_8g_m2"
        )

    def test_every_cited_key_exists(self, products, db):
        """Each swapc2 source string names only source-DB keys that exist."""
        import re

        for mid in (H8, H10):
            text = products[mid].model_dump_json()
            keys = set(re.findall(r"[a-z0-9_.]+@[a-z0-9_]+", text))
            assert keys, mid
            for key in keys:
                if key.startswith("."):  # '.width@doc' shorthand after a full key
                    continue
                assert key in db.observations, key

    def test_profile_powers(self, products, db):
        src = "hailo_8_m2_key_m_datasheet"
        h8 = {p.name: p.tdp_watts for p in products[H8].power.thermal_profiles}
        assert set(h8.values()) == {
            db.value(f"{H8}.power.{v}@{src}") for v in ("typ_mobilenet_ssd", "typ_resnet50", "tdp")
        }
        assert products[H8].power.max_power_watts == db.value(f"{H8}.power.max@{src}")
        h10 = {p.tdp_watts for p in products[H10].power.thermal_profiles}
        src10 = "hailo_10h_m2_key_m_datasheet"
        assert h10 == {db.value(f"{H10}.power.{v}@{src10}") for v in ("typ", "max")}


class TestUnsourcedStaysUnset:
    def test_no_1k_prices(self, products):
        for mid in (H8, H10):
            assert products[mid].swapc2.cost.unit_price_1k_usd is None

    def test_hailo_10h_has_no_mass(self, products):
        assert products[H10].swapc2.weight is None


class TestResolution:
    def test_mass_scales_with_profile(self, products):
        cooling = load_cooling_solutions()
        low = resolve_swapc2(products[H8], "2.4W", cooling, products)
        tdp = resolve_swapc2(products[H8], "8.65W", cooling, products)
        assert low.mass_g.value < tdp.mass_g.value
        assert low.mass_g.basis == ValueBasis.ESTIMATED  # the sized heatsink is an estimate

    def test_hailo_10h_mass_unresolved(self, products):
        r = resolve_swapc2(products[H10], cooling=load_cooling_solutions(), products=products)
        assert r.mass_g is None and r.power_w.value == pytest.approx(8.25)
