"""The legacy HardwareEntry Hailo M.2 files hold the source-DB figures.

They were wrong on size, mass, TDP, price and PCIe width (corrected
2026-10-03, RFC 0001 S3). Until load_hardware() becomes a shim over the
unified catalog, these tests keep them equal to the datasheet figures in
``data/sources/observations/m2_modules.yaml`` -- and keep unsourced figures
(module mass for the 8L and 10H) unset.
"""

import pytest

from embodied_schemas.loaders import load_hardware
from embodied_schemas.sources import load_source_db

# legacy id -> (source-DB subject, datasheet document, price key, mass key or None)
LEGACY = {
    "hailo_8_m2": (
        "hailo_8_m2_2242_m",
        "hailo_8_m2_key_m_datasheet",
        "hailo_8_m2_2242_m.unit_price.qty_1@waveshare_hailo_8",
        "hailo_8_m2_2242_m.mass@waveshare_hailo_8",
    ),
    "hailo_8l_m2": (
        "hailo_8l_m2_2280_bm",
        "hailo_8l_m2_key_bm_datasheet",
        "hailo_8l_m2_2280_bm.unit_price.qty_1@upshop_hailo_8_m2",
        None,
    ),
    "hailo_10h_m2": (
        "hailo_10h_m2_2280_8gb",
        "hailo_10h_m2_key_m_datasheet",
        "hailo_10h_m2_2280_8gb.unit_price.qty_1@upshop_hailo_10h_8g_m2",
        None,
    ),
}


@pytest.fixture(scope="module")
def hw():
    return load_hardware()


@pytest.fixture(scope="module")
def db():
    return load_source_db()


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_footprint_matches_datasheet(hw, db, legacy_id):
    subject, doc, _, _ = LEGACY[legacy_id]
    width, length, _ = hw[legacy_id].physical.dimensions_mm
    assert width == db.value(f"{subject}.width@{doc}")
    assert length == db.value(f"{subject}.length@{doc}")


@pytest.mark.parametrize("legacy_id", ["hailo_8_m2", "hailo_8l_m2"])
def test_height_matches_datasheet(hw, db, legacy_id):
    subject, doc, _, _ = LEGACY[legacy_id]
    assert hw[legacy_id].physical.dimensions_mm[2] == db.value(f"{subject}.height@{doc}")


def test_hailo_10h_height_is_derived(hw, db):
    subject, doc, _, _ = LEGACY["hailo_10h_m2"]
    expected = (
        db.value(f"{subject}.height.top_components@{doc}")
        + db.value("m2_card_pcb.height@hackaday_2022_m2_for_hackers")
        + db.value(f"{subject}.height.bottom_components@{doc}")
    )
    assert hw["hailo_10h_m2"].physical.dimensions_mm[2] == pytest.approx(expected)


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_mass_is_sourced_or_unset(hw, db, legacy_id):
    _, _, _, mass_key = LEGACY[legacy_id]
    mass = hw[legacy_id].physical.weight_grams
    assert mass == (db.value(mass_key) if mass_key else None)


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_price_matches_listing(hw, db, legacy_id):
    _, _, price_key, _ = LEGACY[legacy_id]
    assert hw[legacy_id].cost_usd == db.value(price_key)


def test_power_matches_datasheet(hw, db):
    h8 = hw["hailo_8_m2"].power
    s, d = "hailo_8_m2_2242_m", "hailo_8_m2_key_m_datasheet"
    assert h8.tdp_watts == db.value(f"{s}.power.tdp@{d}")
    assert h8.typical_power_watts == db.value(f"{s}.power.typ_resnet50@{d}")
    assert max(m.power_watts for m in h8.power_modes) == db.value(f"{s}.power.max@{d}")

    h8l = hw["hailo_8l_m2"].power
    s, d = "hailo_8l_m2_2280_bm", "hailo_8l_m2_key_bm_datasheet"
    assert h8l.tdp_watts == db.value(f"{s}.power.tdp@{d}")
    assert h8l.typical_power_watts == db.value(f"{s}.power.typ_resnet50@{d}")

    h10 = hw["hailo_10h_m2"].power
    s, d = "hailo_10h_m2_2280_8gb", "hailo_10h_m2_key_m_datasheet"
    assert h10.tdp_watts == db.value(f"{s}.power.max@{d}")  # no TDP stated
    assert h10.typical_power_watts == db.value(f"{s}.power.typ@{d}")
    assert min(m.power_watts for m in h10.power_modes) == db.value(f"{s}.power.typ_qwen2@{d}")


def test_unified_modules_agree_with_legacy(hw):
    """The legacy entries and the unified module products state the same figures."""
    from embodied_schemas.loaders import load_compute_products

    cps = load_compute_products()
    for legacy_id, module_id in (
        ("hailo_8_m2", "hailo_8_m2_2242_m"),
        ("hailo_10h_m2", "hailo_10h_m2_2280_8gb"),
    ):
        legacy, module = hw[legacy_id], cps[module_id]
        dims = module.swapc2.size.dimensions_mm
        assert legacy.physical.dimensions_mm == [dims.width_mm, dims.length_mm, dims.height_mm]
        assert legacy.cost_usd == module.swapc2.cost.unit_price_1_usd.value
        assert legacy.power.tdp_watts == module.power.tdp_watts


# legacy id -> {power-mode name: source-DB power variant}
MODES = {
    "hailo_8_m2": {"Typical": "typ_resnet50", "Light": "typ_mobilenet_ssd", "Peak": "max"},
    "hailo_8l_m2": {"Typical": "typ_resnet50", "Light": "typ_mobilenet_ssd", "Peak": "max"},
    "hailo_10h_m2": {"Typical": "typ", "Light": "typ_qwen2", "Peak": "max"},
}


@pytest.mark.parametrize("legacy_id", MODES)
def test_every_power_mode_matches_datasheet(hw, db, legacy_id):
    """Each named legacy power mode equals its source-DB figure, and the modes
    are exactly those (no unsourced extras)."""
    subject, doc, _, _ = LEGACY[legacy_id]
    modes = {m.name: m.power_watts for m in hw[legacy_id].power.power_modes}
    assert set(modes) == set(MODES[legacy_id])
    for name, variant in MODES[legacy_id].items():
        assert modes[name] == db.value(f"{subject}.power.{variant}@{doc}"), (legacy_id, name)


@pytest.mark.parametrize(
    "legacy_id,module_id",
    [("hailo_8_m2", "hailo_8_m2_2242_m"), ("hailo_10h_m2", "hailo_10h_m2_2280_8gb")],
)
def test_power_modes_agree_with_unified_module(hw, legacy_id, module_id):
    """Peak is the module's max_power_watts; every other mode is one of its profiles."""
    from embodied_schemas.loaders import load_compute_products

    module = load_compute_products()[module_id]
    modes = {m.name: m.power_watts for m in hw[legacy_id].power.power_modes}
    assert modes.pop("Peak") == module.power.max_power_watts
    profiles = {p.tdp_watts for p in module.power.thermal_profiles}
    assert set(modes.values()) <= profiles
