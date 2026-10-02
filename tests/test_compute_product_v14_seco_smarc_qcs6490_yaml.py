"""SECO SOM-SMARC-QCS6490: the first ``kind: module`` product and the SWaP-C²
test case (RFC 0001 D6 / D8 / R1, phase S1).

Pins the modeling choices recorded in the YAML:
1. The module contains the QCS6490 SoC and adds no dies of its own.
2. Sourced facts: 82 x 50 mm footprint, 5 V input, industrial range, LPDDR5.
3. SWaP-C² resolves power and envelope at every profile, through the SMARC
   standard heat spreader. Mass and unit cost stay unresolved: SECO
   publishes neither, and the catalog does not guess them.
4. D4 aggregation reaches the SoC's DSP block through ``contains``.
"""

import pytest

from embodied_schemas import (
    ProductKind,
    ValueBasis,
    aggregate_peak,
    resolve_swapc2,
)
from embodied_schemas.hardware import FormFactor
from embodied_schemas.loaders import (
    load_capability_tiers,
    load_compute_products,
    load_cooling_solutions,
)

MODULE_ID = "seco_som_smarc_qcs6490"
SPREADER_ID = "smarc_heat_spreader_82x50"


@pytest.fixture(scope="module")
def products():
    return load_compute_products()


@pytest.fixture(scope="module")
def cooling():
    return load_cooling_solutions()


@pytest.fixture(scope="module")
def module(products):
    return products[MODULE_ID]


class TestStructure:
    def test_is_a_module_containing_the_soc(self, module, products):
        assert module.kind is ProductKind.MODULE
        assert module.dies == []
        assert [r.id for r in module.contains] == ["qualcomm_qcs6490"]
        assert products["qualcomm_qcs6490"].kind is ProductKind.CHIP

    def test_smarc_form_factor(self, module):
        assert module.packaging.form_factor is FormFactor.SMARC

    def test_sourced_facts(self, module):
        dims = module.swapc2.size.dimensions_mm
        assert (dims.length_mm, dims.width_mm) == (82.0, 50.0)
        # 1.3 + 1.2 + 3.0 mm: the SMARC 2.1.1 height maxima.
        assert dims.height_mm == pytest.approx(5.5)
        assert module.swapc2.power.input_voltage_v == (5.0, 5.0)
        assert module.environmental.operating_temp_c == [-30.0, 85.0]
        assert module.memory.memory_type == "LPDDR5"

    def test_unpublished_facts_are_not_guessed(self, module):
        assert module.swapc2.weight is None
        assert module.swapc2.cost is None

    def test_profiles_mirror_the_soc(self, module, products):
        soc = products["qualcomm_qcs6490"]
        assert [p.tdp_watts for p in module.power.thermal_profiles] == [
            p.tdp_watts for p in soc.power.thermal_profiles
        ]
        assert {p.cooling_solution_id for p in module.power.thermal_profiles} == {SPREADER_ID}


class TestSWaPC2:
    @pytest.mark.parametrize("profile,watts", [("5W", 5.0), ("10W", 10.0), ("15W", 15.0)])
    def test_power_and_envelope_resolve(self, module, products, cooling, profile, watts):
        r = resolve_swapc2(module, profile, cooling, products)
        assert r.power_w.value == pytest.approx(watts)
        # Spreader plate (82 x 42 x 3 mm) on the top face, inside the footprint.
        env = r.envelope
        assert (env.length_mm, env.width_mm, env.height_mm) == pytest.approx((82, 50, 8.5))
        assert r.envelope_cm3.value == pytest.approx(82 * 50 * 8.5 / 1000)
        assert r.envelope.basis == ValueBasis.DERIVED
        assert r.warnings == []

    def test_mass_and_cost_unresolved(self, module, products, cooling):
        r = resolve_swapc2(module, "10W", cooling, products)
        assert r.mass_g is None
        assert r.unit_cost_1_usd is None and r.unit_cost_1k_usd is None
        assert any(u.startswith("mass_g") for u in r.unresolved)
        assert any(u.startswith("unit_price_1_usd") for u in r.unresolved)

    def test_fits_micro_autonomy_volume(self, module, products, cooling):
        tier = load_capability_tiers()["micro_autonomy"]
        r = resolve_swapc2(module, "10W", cooling, products)
        assert r.envelope_cm3.value <= tier.form_factor.max_compute_volume_cm3


class TestPeakAggregation:
    def test_reaches_the_soc_block(self, module, products):
        agg = aggregate_peak(module, products)
        assert agg.not_expanded == []
        assert agg.blocks == [f"{MODULE_ID}>qualcomm_qcs6490/qcs6490_die/dsp[0]"]
        assert agg.sum["int8"] == pytest.approx(module.performance.int8_tops * 1e12)

    def test_property_does_not_expand(self, module):
        assert module.performance_by_aggregation.not_expanded == ["qualcomm_qcs6490"]
