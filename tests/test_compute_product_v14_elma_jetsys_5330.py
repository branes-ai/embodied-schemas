"""Elma JetSys-5330 rugged system (RFC 0001 S3f).

The first ``system`` product, and the first that contains a catalog module
(the NVIDIA Jetson AGX Orin 64GB). These tests hold the system, its integral
enclosure cooling entry and the corrected legacy HardwareEntry to the Elma
datasheet figures in ``data/sources/observations/rugged_systems.yaml``, and
check the D5 / D4 roll-ups through ``contains``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_jetson_skus as gen  # noqa: E402

from embodied_schemas import ProductKind  # noqa: E402
from embodied_schemas.compute_product import aggregate_peak  # noqa: E402
from embodied_schemas.loaders import (  # noqa: E402
    load_compute_products,
    load_cooling_solutions,
    load_hardware,
)
from embodied_schemas.sources import load_source_db  # noqa: E402
from embodied_schemas.swapc2 import ValueBasis, resolve_swapc2  # noqa: E402

SYSTEM = "elma_jetsys_5330_orin"
SUBJECT = "elma_jetsys_5330"
MODULE = "nvidia_jetson_agx_orin_64gb"
COOLING = "elma_jetsys_5330_enclosure"


@pytest.fixture(scope="module")
def products():
    return load_compute_products()


@pytest.fixture(scope="module")
def system(products):
    return products[SYSTEM]


@pytest.fixture(scope="module")
def db():
    return load_source_db()


def _obs(db, quantity, variant=None):
    (obs,) = [o for o in db.find(quantity, SUBJECT) if o.variant == variant]
    return obs


def _v(db, quantity, variant=None):
    return _obs(db, quantity, variant).value


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_is_a_system_containing_the_agx_orin_module(system, products):
    assert system.kind is ProductKind.SYSTEM and not system.dies
    (ref,) = system.contains
    assert (ref.id, ref.count, ref.role) == (MODULE, 1, "compute")
    assert products[MODULE].kind is ProductKind.MODULE


def test_memory_is_the_contained_modules(system, products):
    assert system.memory == products[MODULE].memory


# ---------------------------------------------------------------------------
# Datasheet facts
# ---------------------------------------------------------------------------


def test_envelope_matches_datasheet(system, db):
    d = system.swapc2.size.dimensions_mm
    assert (d.length_mm, d.width_mm, d.height_mm) == (
        _v(db, "length"),
        _v(db, "width"),
        _v(db, "height"),
    )
    assert d.basis is ValueBasis.DATASHEET


def test_mass_is_the_datasheet_maximum(system, db):
    mass = system.swapc2.weight.mass_g
    assert mass.value == _v(db, "mass", "max")
    assert _v(db, "mass", "min") < mass.value
    assert mass.source == "elma_jetsys_5330.mass.max@elma_jetsys_5330_ds"


def test_input_voltage_matches_datasheet(system, db):
    rng = _obs(db, "input_voltage")
    assert system.swapc2.power.input_voltage_v == (rng.value_min, rng.value_max)


def test_power_matches_datasheet(system, db):
    power = system.power
    assert power.max_power_watts == power.tdp_watts == _v(db, "power", "max")
    assert power.min_power_watts == _v(db, "power", "min")
    (profile,) = power.thermal_profiles
    assert profile.tdp_watts == _v(db, "power", "max")
    assert profile.clock_mhz == _v(db, "frequency", "gpu_max")
    assert profile.cooling_solution_id == COOLING


def test_temperatures_match_datasheet(system, db):
    env = system.environmental
    # "55°C or 71°C operational, depending on the configuration": the floor.
    assert env.operating_temp_c == [
        _v(db, "temperature", "operating_min"),
        _obs(db, "temperature", "operating_max").value_min,
    ]
    assert env.storage_temp_c == [
        _v(db, "temperature", "storage_min"),
        _v(db, "temperature", "storage_max"),
    ]


def test_headline_is_the_module_gpu_at_elmas_clock(system, products, db):
    """The template arithmetic at Elma's 1.3 GHz gives the headline, and its
    FP32 rounds to Elma's '5.3 FP32 TFLOPs'."""
    gpu = products[MODULE].dies[0].blocks[0].model_dump(mode="json")
    peaks = gen.fabric_peaks(gpu, gpu["num_sms"], _v(db, "frequency", "gpu_max"))
    perf = system.performance
    assert (perf.int8_tops, perf.bf16_tflops, perf.fp32_tflops) == pytest.approx(
        (peaks["int8_tops"], peaks["bf16_tflops"], peaks["fp32_tflops"]), abs=0.01
    )
    assert round(perf.fp32_tflops, 1) == _v(db, "fp32_throughput")


def test_no_unit_price_is_stated(system):
    """Elma quotes on request; the system states no price of its own."""
    assert system.swapc2.cost is None


# ---------------------------------------------------------------------------
# Integral enclosure cooling
# ---------------------------------------------------------------------------


def test_enclosure_cooling_adds_nothing_on_top(db):
    cs = load_cooling_solutions()[COOLING]
    watts = _v(db, "power", "max")
    assert cs.mass_g_at(watts) == 0.0
    assert cs.volume_cm3_at(watts) == 0.0
    assert cs.cost_usd_at(watts) == 0.0
    assert not cs.is_active
    assert cs.max_total_w == watts
    assert cs.ambient_c_max == _obs(db, "temperature", "operating_max").value_min


# ---------------------------------------------------------------------------
# Resolution and roll-ups
# ---------------------------------------------------------------------------


def test_resolve_swapc2_is_the_systems_own_figures(system, products, db):
    r = resolve_swapc2(system, products=products)
    assert r.power_w.value == _v(db, "power", "max")
    assert r.mass_g.value == _v(db, "mass", "max")
    env = r.envelope
    assert (env.length_mm, env.width_mm, env.height_mm) == (
        _v(db, "length"),
        _v(db, "width"),
        _v(db, "height"),
    )
    assert not r.warnings


def test_unit_cost_rolls_up_from_the_module_as_an_estimate(system, products):
    """D5: with no system price, the 1K cost is the contained module's 1K
    price -- an estimated lower bound (carrier, enclosure and I/O excluded).
    The module has no quantity-1 price, so that axis stays unresolved."""
    r = resolve_swapc2(system, products=products)
    module_1k = products[MODULE].swapc2.cost.unit_price_1k_usd
    assert r.unit_cost_1k_usd.value == module_1k.value
    assert r.unit_cost_1k_usd.basis is ValueBasis.ESTIMATED
    assert r.unit_cost_1_usd is None
    assert any(u.startswith("unit_price_1_usd:") for u in r.unresolved)


def test_d4_rollup_counts_the_module_at_its_default_profile(system, products):
    """aggregate_peak expands contains with each product's own headline: the
    module counts at its default profile, not at the system's 1.3 GHz."""
    agg = aggregate_peak(system, products)
    module = products[MODULE].performance
    assert agg.block_count["fp32"] == 1
    assert agg.sum["fp32"] == pytest.approx(module.fp32_tflops * 1e12)


# ---------------------------------------------------------------------------
# Legacy HardwareEntry
# ---------------------------------------------------------------------------


def test_legacy_entry_matches_datasheet(db):
    hw = load_hardware()[SYSTEM]
    assert hw.physical.weight_grams == _v(db, "mass", "max")
    assert hw.physical.dimensions_mm == [_v(db, "length"), _v(db, "width"), _v(db, "height")]
    rng = _obs(db, "input_voltage")
    assert hw.power.input_voltage_v == [rng.value_min, rng.value_max]
    assert hw.power.tdp_watts == _v(db, "power", "max")
    assert [m.power_watts for m in hw.power.power_modes] == [
        _v(db, "power", "min"),
        _v(db, "power", "max"),
    ]
    assert all(m.cpu_freq_mhz is None for m in hw.power.power_modes)
    assert hw.environmental.operating_temp_c == [
        _v(db, "temperature", "operating_min"),
        _obs(db, "temperature", "operating_max").value_min,
    ]
    assert hw.capabilities.peak_tflops_fp32 == _v(db, "fp32_throughput")
