"""Tests for v14 ComputeProduct additions (RFC 0001 rev 2-4, phase S1).

Validates that:
1. SWaP-C² fact types (``SourcedValue``, ``SourcedDimensions``, ``SWaPC2Spec``)
   enforce their ranges and ``extra: forbid``.
2. Levels of integration (D6/D8): chip-level kinds need dies; a module /
   board / system needs dies or ``contains``; ``check_contains_references``
   catches unknown ids and cycles.
3. D4 peak aggregation: sum / min / max per precision over the blocks that
   support it, multiplicity from ``contains``, IO blocks skipped, headline
   fallback for single-block products.
4. ``resolve_swapc2`` (D7, D9): input-rail power, mass, envelope, unit cost,
   the D5 roll-up over ``contains``, and its unresolved / warning reporting.
5. ``CoolingSolutionEntry`` sizing helpers.
"""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    ComputeProduct,
    CostSpec,
    DSPBlock,
    InputPowerSpec,
    ProductKind,
    ProductRef,
    SizeSpec,
    SourcedDimensions,
    SourcedValue,
    SWaPC2Spec,
    ValueBasis,
    WeightSpec,
    aggregate_peak,
    check_contains_references,
    resolve_swapc2,
)
from embodied_schemas.compute_block_common import TheoreticalPerformance
from embodied_schemas.cooling_solution import CoolingMechanism, CoolingSolutionEntry
from embodied_schemas.loaders import load_compute_products, load_cooling_solutions
from embodied_schemas.process_node import DataConfidence

CAL = DataConfidence.CALIBRATED
THEO = DataConfidence.THEORETICAL


def sv(value, basis=ValueBasis.DATASHEET, confidence=CAL, source="test"):
    return SourcedValue(value=value, basis=basis, confidence=confidence, source=source)


@pytest.fixture(scope="module")
def catalog():
    return load_compute_products()


@pytest.fixture(scope="module")
def cooling():
    return load_cooling_solutions()


@pytest.fixture
def chip(catalog):
    """A real single-DSP-block chip to derive test products from."""
    return catalog["qualcomm_qcs6490"]


def make_module(chip, **overrides):
    """A module containing ``chip``, with no dies of its own."""
    data = chip.model_dump()
    data.update(
        id="test_module",
        name="Test module",
        kind="module",
        dies=[],
        contains=[{"id": chip.id}],
        packaging={"kind": "board", "num_dies": 1, "form_factor": "smarc"},
    )
    data.update(overrides)
    return ComputeProduct.model_validate(data)


def with_dsp_peaks(chip, peaks):
    """``chip`` with one DSP block per entry of ``peaks`` (ops/s by precision)."""
    block = chip.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    blocks = [
        block.model_copy(
            update={
                "theoretical_performance": TheoreticalPerformance(peak_ops_per_sec_by_precision=p)
            }
        )
        for p in peaks
    ]
    die = chip.dies[0].model_copy(update={"blocks": blocks})
    return chip.model_copy(update={"dies": [die]})


# ---------------------------------------------------------------------------
# 1. Fact types
# ---------------------------------------------------------------------------


class TestFactTypes:
    def test_sourced_value_requires_source(self):
        with pytest.raises(ValidationError):
            SourcedValue(value=1.0, basis="datasheet", confidence="calibrated", source="")

    def test_sourced_value_forbids_extra(self):
        with pytest.raises(ValidationError):
            SourcedValue(
                value=1.0, basis="datasheet", confidence="calibrated", source="x", unit="g"
            )

    def test_dimensions_volume_and_footprint(self):
        d = SourcedDimensions(
            length_mm=82,
            width_mm=50,
            height_mm=5,
            basis="datasheet",
            confidence="calibrated",
            source="SMARC 2.1",
        )
        assert d.footprint_mm2 == 4100
        assert d.volume_cm3 == pytest.approx(20.5)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_mass_must_be_positive(self, bad):
        with pytest.raises(ValidationError, match="mass_g"):
            WeightSpec(mass_g=sv(bad))

    @pytest.mark.parametrize("bad", [0.0, 1.2])
    def test_conversion_efficiency_range(self, bad):
        with pytest.raises(ValidationError, match="conversion_efficiency"):
            InputPowerSpec(conversion_efficiency=sv(bad))

    def test_input_voltage_order(self):
        with pytest.raises(ValidationError, match="input_voltage_v"):
            InputPowerSpec(input_voltage_v=(12.0, 5.0))

    def test_price_non_negative(self):
        with pytest.raises(ValidationError, match="unit_price_1k_usd"):
            CostSpec(unit_price_1k_usd=sv(-1.0))

    def test_volume_derived_from_dimensions(self):
        spec = SWaPC2Spec(
            size=SizeSpec(
                dimensions_mm=SourcedDimensions(
                    length_mm=10,
                    width_mm=10,
                    height_mm=10,
                    basis="datasheet",
                    confidence="calibrated",
                    source="x",
                )
            )
        )
        assert spec.volume_cm3.value == pytest.approx(1.0)
        assert spec.volume_cm3.basis == ValueBasis.DERIVED

    def test_stated_volume_wins(self):
        spec = SWaPC2Spec(size=SizeSpec(volume_cm3=sv(3.0)))
        assert spec.volume_cm3.value == 3.0


# ---------------------------------------------------------------------------
# 2. Levels of integration
# ---------------------------------------------------------------------------


class TestLevels:
    def test_module_kind_exists(self):
        assert ProductKind("module") is ProductKind.MODULE
        assert not ProductKind.MODULE.is_silicon
        assert ProductKind.CHIP.is_silicon

    def test_chip_needs_dies(self, chip):
        data = chip.model_dump()
        data["dies"] = []
        with pytest.raises(ValidationError, match="needs at least one die"):
            ComputeProduct.model_validate(data)

    def test_module_with_contains_only(self, chip):
        m = make_module(chip)
        assert m.dies == [] and m.contains[0].id == chip.id

    def test_module_needs_dies_or_contains(self, chip):
        with pytest.raises(ValidationError, match="needs dies or contains"):
            make_module(chip, contains=[])

    def test_self_containment_rejected(self, chip):
        with pytest.raises(ValidationError, match="contains itself"):
            make_module(chip, contains=[{"id": "test_module"}])

    def test_product_ref_count_positive(self):
        with pytest.raises(ValidationError):
            ProductRef(id="x", count=0)

    def test_catalog_contains_references_clean(self, catalog):
        assert check_contains_references(catalog) == []

    def test_unknown_reference_reported(self, chip):
        m = make_module(chip, contains=[{"id": "nope"}])
        errors = check_contains_references({m.id: m})
        assert errors == ["test_module: contains unknown product 'nope'"]

    def test_cycle_reported(self, chip):
        a = make_module(chip, id="a", contains=[{"id": "b"}])
        b = make_module(chip, id="b", contains=[{"id": "a"}])
        errors = check_contains_references({"a": a, "b": b})
        assert any("cycle" in e for e in errors)


# ---------------------------------------------------------------------------
# 3. D4 peak aggregation
# ---------------------------------------------------------------------------


class TestPeakAggregation:
    def test_single_block_sum_min_max_equal(self, chip):
        agg = chip.performance_by_aggregation
        assert agg.sum == agg.min == agg.max
        assert agg.sum["int8"] == pytest.approx(12e12)

    def test_three_forms_over_blocks(self, chip):
        p = with_dsp_peaks(
            chip,
            [
                {"int8": 10e12, "fp16": 2e12},
                {"int8": 4e12, "fp16": 0.0},  # zero = unsupported
                {"int8": 6e12},
            ],
        )
        agg = aggregate_peak(p)
        assert agg.sum["int8"] == pytest.approx(20e12)
        assert agg.min["int8"] == pytest.approx(4e12)
        assert agg.max["int8"] == pytest.approx(10e12)
        assert agg.block_count["int8"] == 3
        # fp16: only the first block supports it; the zero is not a min of 0.
        assert agg.min["fp16"] == agg.max["fp16"] == agg.sum["fp16"] == pytest.approx(2e12)
        assert agg.block_count["fp16"] == 1
        assert agg.headline_fallback == []

    def test_block_without_peak_is_reported_not_zero(self, chip):
        p = with_dsp_peaks(chip, [{"int8": 10e12}, {"int8": 5e12}])
        die = p.dies[0]
        no_peak = die.blocks[1].model_copy(update={"theoretical_performance": None})
        # DSPBlock requires theoretical_performance; emulate a block kind that
        # leaves it out by bypassing validation.
        die = die.model_copy(update={"blocks": [die.blocks[0], no_peak]})
        p = p.model_copy(update={"dies": [die]})
        agg = aggregate_peak(p)
        assert agg.sum["int8"] == pytest.approx(10e12)
        assert len(agg.blocks_without_peak) == 1

    def test_module_expands_contains_with_count(self, chip, catalog):
        m = make_module(chip, contains=[{"id": chip.id, "count": 2}])
        assert aggregate_peak(m).not_expanded == [chip.id]
        agg = aggregate_peak(m, catalog)
        assert agg.sum["int8"] == pytest.approx(24e12)
        assert agg.max["int8"] == pytest.approx(12e12)
        assert agg.block_count["int8"] == 2

    def test_headline_fallback(self, catalog):
        agg = aggregate_peak(catalog["hailo_hailo_8"])
        assert agg.headline_fallback == ["hailo_hailo_8"]
        assert agg.sum["int8"] == pytest.approx(26e12)

    def test_io_blocks_skipped(self, catalog):
        agg = aggregate_peak(catalog["amd_epyc_9654_sp5"])
        assert all("/io[" not in b for b in agg.blocks + agg.blocks_without_peak)

    def test_kpu_peak_matches_headline(self, catalog):
        p = catalog["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
        agg = p.performance_by_aggregation
        assert agg.headline_fallback == []
        assert agg.sum["int8"] == pytest.approx(p.performance.int8_tops * 1e12, rel=1e-3)

    def test_aggregation_is_not_serialized(self, chip):
        assert "performance_by_aggregation" not in chip.model_dump()


# ---------------------------------------------------------------------------
# 4. resolve_swapc2
# ---------------------------------------------------------------------------

DIMS = SourcedDimensions(
    length_mm=82,
    width_mm=50,
    height_mm=6,
    basis="datasheet",
    confidence="calibrated",
    source="datasheet",
)


def full_spec(**power):
    return SWaPC2Spec(
        size=SizeSpec(dimensions_mm=DIMS),
        weight=WeightSpec(mass_g=sv(30.0)),
        power=InputPowerSpec(**power) if power else None,
        cost=CostSpec(unit_price_1_usd=sv(200.0), unit_price_1k_usd=sv(150.0)),
    )


def fixed_cooling(**kw):
    base = dict(
        id="test_sink",
        name="Test sink",
        cooling_mechanism="passive_heatsink_small",
        max_power_density_w_per_mm2=1.0,
        max_total_w=20.0,
        ambient_c_max=50.0,
        junction_c_max=105.0,
        weight_g=20.0,
        cost_usd=5.0,
        volume_cm3=41.0,
        source="test",
        confidence="calibrated",
        last_updated="2026-10-02",
        basis="datasheet",
    )
    base.update(kw)
    return CoolingSolutionEntry(**base)


def bind(product, cooling_id):
    profiles = [
        p.model_copy(update={"cooling_solution_id": cooling_id})
        for p in product.power.thermal_profiles
    ]
    return product.model_copy(
        update={"power": product.power.model_copy(update={"thermal_profiles": profiles})}
    )


class TestResolve:
    def test_full_resolution(self, chip):
        p = bind(chip.model_copy(update={"swapc2": full_spec()}), "test_sink")
        r = resolve_swapc2(p, "10W", {"test_sink": fixed_cooling()})
        assert r.unresolved == [] and r.warnings == []
        assert r.power_w.value == pytest.approx(10.0)
        assert r.mass_g.value == pytest.approx(50.0)
        # 41 cm^3 over the 82 x 50 mm footprint is 10 mm of height.
        assert r.envelope.height_mm == pytest.approx(16.0)
        assert r.envelope_cm3.value == pytest.approx(82 * 50 * 16 / 1000)
        assert r.unit_cost_1_usd.value == pytest.approx(205.0)
        assert r.unit_cost_1k_usd.value == pytest.approx(155.0)
        assert r.mass_g.basis == ValueBasis.DERIVED
        # The profile TDP carries the product's (theoretical) confidence.
        assert r.power_w.confidence == THEO

    def test_power_boundary(self, chip):
        p = bind(
            chip.model_copy(update={"swapc2": full_spec(conversion_efficiency=sv(0.8))}),
            "fan",
        )
        fan = fixed_cooling(id="fan", cooling_mechanism="active_fan", parasitic_power_w=1.0)
        r = resolve_swapc2(p, "10W", {"fan": fan})
        # 10 W / 0.8 + 1 W: the fan is already input-rail power.
        assert r.power_w.value == pytest.approx(13.5)

    def test_active_cooling_without_parasitic_warns(self, chip):
        p = bind(chip.model_copy(update={"swapc2": full_spec()}), "fan")
        fan = fixed_cooling(id="fan", cooling_mechanism="active_fan")
        r = resolve_swapc2(p, "10W", {"fan": fan})
        assert any("parasitic_power_w" in w for w in r.warnings)

    def test_stated_cooling_dimensions(self, chip):
        p = bind(chip.model_copy(update={"swapc2": full_spec()}), "sink")
        sink = fixed_cooling(id="sink", volume_cm3=None, dimensions_mm=(90.0, 50.0, 12.0))
        r = resolve_swapc2(p, "10W", {"sink": sink})
        assert (r.envelope.length_mm, r.envelope.width_mm, r.envelope.height_mm) == (90, 50, 18)

    def test_sized_cooling_scales_with_watts(self, chip):
        p = bind(chip.model_copy(update={"swapc2": full_spec()}), "sized")
        sized = fixed_cooling(id="sized", weight_g=10.0, mass_g_per_w=4.0, basis="estimated")
        r5 = resolve_swapc2(p, "5W", {"sized": sized})
        r15 = resolve_swapc2(p, "15W", {"sized": sized})
        assert r5.mass_g.value == pytest.approx(30 + 10 + 20)
        assert r15.mass_g.value == pytest.approx(30 + 10 + 60)
        assert r15.mass_g.basis == ValueBasis.ESTIMATED

    def test_tdp_over_cooling_capacity_warns(self, chip):
        p = bind(chip.model_copy(update={"swapc2": full_spec()}), "small")
        r = resolve_swapc2(p, "15W", {"small": fixed_cooling(id="small", max_total_w=12.0)})
        assert any("exceeds cooling" in w for w in r.warnings)

    def test_missing_facts_are_unresolved(self, chip, cooling):
        r = resolve_swapc2(chip, "5W", cooling)
        assert r.power_w is not None
        assert r.mass_g is None and r.envelope is None and r.unit_cost_1_usd is None
        assert len(r.unresolved) == 4

    def test_contains_rollup(self, chip, catalog):
        child = chip.model_copy(update={"swapc2": full_spec()})
        module = bind(make_module(chip, contains=[{"id": chip.id, "count": 2}]), "test_sink")
        products = {**catalog, chip.id: child}
        r = resolve_swapc2(module, "10W", {"test_sink": fixed_cooling()}, products)
        assert r.mass_g.value == pytest.approx(2 * 30 + 20)
        assert r.mass_g.basis == ValueBasis.ESTIMATED
        assert r.unit_cost_1k_usd.value == pytest.approx(2 * 150 + 5)
        # Envelope is not additive: the module states no dimensions.
        assert r.envelope is None

    def test_nested_contains_rollup(self, chip, catalog):
        """A board -> module (no mass of its own) -> chip rolls up through the
        intermediate module."""
        child = chip.model_copy(update={"swapc2": full_spec()})
        module = make_module(chip, id="mid_module", contains=[{"id": chip.id, "count": 2}])
        board = bind(
            make_module(chip, id="top_board", kind="board", contains=[{"id": "mid_module"}]),
            "test_sink",
        )
        products = {**catalog, chip.id: child, "mid_module": module}
        r = resolve_swapc2(board, "10W", {"test_sink": fixed_cooling()}, products)
        assert r.mass_g.value == pytest.approx(2 * 30 + 20)
        assert r.unit_cost_1_usd.value == pytest.approx(2 * 200 + 5)
        assert r.mass_g.basis == ValueBasis.ESTIMATED

    def test_nested_rollup_reports_the_missing_leaf(self, chip, catalog):
        module = make_module(chip, id="mid_module")
        board = bind(
            make_module(chip, id="top_board", kind="board", contains=[{"id": "mid_module"}]),
            "test_sink",
        )
        r = resolve_swapc2(
            board, "10W", {"test_sink": fixed_cooling()}, {**catalog, "mid_module": module}
        )
        assert r.mass_g is None
        assert any(f"{chip.id!r} states no mass_g" in u for u in r.unresolved)

    def test_rollup_needs_products(self, chip):
        module = bind(make_module(chip), "test_sink")
        r = resolve_swapc2(module, "10W", {"test_sink": fixed_cooling()})
        assert any("without products" in u for u in r.unresolved)

    def test_unknown_profile(self, chip, cooling):
        with pytest.raises(KeyError, match="no thermal profile"):
            resolve_swapc2(chip, "99W", cooling)

    def test_unknown_cooling(self, chip):
        with pytest.raises(KeyError, match="unknown cooling solution"):
            resolve_swapc2(chip, "5W", {})

    def test_default_profile(self, chip, cooling):
        assert resolve_swapc2(chip, cooling=cooling).thermal_profile == "10W"


# ---------------------------------------------------------------------------
# 5. Cooling sizing helpers
# ---------------------------------------------------------------------------


class TestCoolingSizing:
    def test_fixed_only(self):
        c = fixed_cooling()
        assert c.mass_g_at(100) == 20.0 and c.cost_usd_at(1) == 5.0

    def test_base_plus_per_watt(self):
        c = fixed_cooling(weight_g=10.0, mass_g_per_w=2.0, cost_usd=1.0, cost_usd_per_w=0.5)
        assert c.mass_g_at(10) == pytest.approx(30.0)
        assert c.cost_usd_at(10) == pytest.approx(6.0)

    def test_nothing_stated(self):
        c = fixed_cooling(weight_g=None, cost_usd=None, volume_cm3=None)
        assert c.mass_g_at(5) is None and c.cost_usd_at(5) is None
        assert c.volume_cm3_at(5) is None

    def test_fanless_occupies_nothing(self, cooling):
        assert cooling["passive_fanless"].volume_cm3_at(5) == 0.0

    def test_surface_rating_bounds_ambient(self):
        with pytest.raises(ValidationError, match="exceeds surface_c_max"):
            fixed_cooling(ambient_c_max=90.0, surface_c_max=85.0)
        assert fixed_cooling(ambient_c_max=50.0, surface_c_max=85.0).surface_c_max == 85.0

    def test_smarc_spreader_is_surface_rated(self, cooling):
        spreader = cooling["smarc_heat_spreader_82x50"]
        assert spreader.surface_c_max == 85.0
        assert spreader.ambient_c_max <= spreader.surface_c_max
        assert cooling["active_fan"].surface_c_max is None

    def test_is_active(self, cooling):
        assert cooling["active_fan"].is_active
        assert not cooling["passive_heatsink_small"].is_active
        assert CoolingMechanism.ACTIVE_FAN.value == "active_fan"


# ---------------------------------------------------------------------------
# 6. Additive serialization: unset v14 fields stay out of dumps
# ---------------------------------------------------------------------------


class TestAdditiveDumps:
    V14_PRODUCT_FIELDS = (
        "contains",
        "swapc2",
        "memory",
        "environmental",
        "interfaces",
        "software",
        "product_url",
    )

    def test_existing_product_dump_has_no_v14_keys(self, chip):
        dump = chip.model_dump(mode="json")
        assert not set(self.V14_PRODUCT_FIELDS) & set(dump)
        assert "form_factor" not in dump["packaging"]
        assert "suitable_for" not in dump["market"]
        assert "theoretical_performance" in dump["dies"][0]["blocks"][0]  # DSP: required

    def test_unset_block_peak_not_dumped(self, catalog):
        block = catalog["hailo_hailo_8"].dies[0].blocks[0]
        assert "theoretical_performance" not in block.model_dump()

    def test_set_fields_are_dumped_and_round_trip(self, chip):
        m = make_module(chip, swapc2=full_spec().model_dump())
        dump = m.model_dump(mode="json")
        assert dump["contains"] == [
            {"id": chip.id, "count": 1, "slot": None, "role": None, "notes": ""}
        ]
        assert dump["packaging"]["form_factor"] == "smarc"
        assert ComputeProduct.model_validate(dump) == m

    def test_cooling_default_basis_not_dumped(self, cooling):
        assert "basis" not in cooling["active_fan"].model_dump()
        assert cooling["smarc_heat_spreader_82x50"].model_dump()["basis"] == "datasheet"

    def test_json_schema_keeps_fields(self):
        props = ComputeProduct.model_json_schema(mode="serialization")["properties"]
        assert {"swapc2", "contains", "memory"} <= set(props)
