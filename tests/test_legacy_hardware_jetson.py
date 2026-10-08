"""The legacy HardwareEntry Jetson files hold the source-DB figures.

They were wrong on mass, size, power modes, peaks and price (several described
developer kits rather than modules; corrected 2026-10-08). Until
load_hardware() becomes a shim over the unified catalog, these tests keep them
equal to the NVIDIA figures in ``data/sources/observations/jetson_modules.yaml``
-- and keep unsourced figures (CPU clocks, most TFLOPS peaks) unset.
"""

import pytest

from embodied_schemas.loaders import load_hardware
from embodied_schemas.sources import load_source_db

# legacy id -> source-DB subject
LEGACY = {
    "nvidia_jetson_agx_orin_64gb": "nvidia_jetson_agx_orin_64gb",
    "nvidia_jetson_agx_orin_32gb": "nvidia_jetson_agx_orin_32gb",
    "nvidia_jetson_orin_nx_16gb": "nvidia_jetson_orin_nx_16gb",
    "nvidia_jetson_orin_nx_8gb": "nvidia_jetson_orin_nx_8gb",
    "nvidia_jetson_orin_nano_8gb": "nvidia_jetson_orin_nano_8gb",
    "nvidia_jetson_agx_thor_128gb": "nvidia_jetson_t5000",
}
THOR = "nvidia_jetson_agx_thor_128gb"


@pytest.fixture(scope="module")
def hw():
    return load_hardware()


@pytest.fixture(scope="module")
def db():
    return load_source_db()


def _nvidia(db, subject, quantity, variant=None, **conditions):
    """Observations from NVIDIA's own documents (resellers and EDOM's withdrawn
    Thor Tensor Core count are corroboration, not the reference)."""
    return [
        o
        for o in db.find(quantity, subject, variant=variant, **conditions)
        if o.source_id.startswith("nvidia_")
    ]


def _v(db, subject, quantity, variant=None, **conditions):
    (obs,) = _nvidia(db, subject, quantity, variant, **conditions)
    return obs.value


def _top(db, subject, quantity, variant=None):
    """The MAXN_SUPER figure where NVIDIA publishes one, else the base figure:
    the entries' top power mode (and tdp) is MAXN_SUPER on those modules."""
    sup = f"{variant}_super" if variant else "super"
    if _nvidia(db, subject, quantity, sup):
        return _v(db, subject, quantity, sup)
    return _v(db, subject, quantity, variant)


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_physical_matches_datasheet(hw, db, legacy_id):
    s = LEGACY[legacy_id]
    phys = hw[legacy_id].physical
    assert phys.weight_grams == _v(db, s, "mass")
    length, width, height = phys.dimensions_mm
    assert (length, width) == (_v(db, s, "length"), _v(db, s, "width"))
    sourced_height = _nvidia(db, s, "height")
    if sourced_height:
        assert height == sourced_height[0].value


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_memory_and_cores_match_datasheet(hw, db, legacy_id):
    s = LEGACY[legacy_id]
    cap = hw[legacy_id].capabilities
    assert cap.memory_gb == _v(db, s, "memory_capacity")
    assert cap.memory_bandwidth_gbps == _top(db, s, "memory_bandwidth")
    assert cap.compute_units == _v(db, s, "cuda_cores")
    tensor = _nvidia(db, s, "tensor_cores")
    assert cap.tensor_cores == (tensor[0].value if tensor else None)


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_peak_int8_is_sparse_headline_at_top_mode(hw, db, legacy_id):
    s = LEGACY[legacy_id]
    variant = "fp8_sparse" if legacy_id == THOR else "int8_sparse"
    assert hw[legacy_id].capabilities.peak_tops_int8 == _top(db, s, "ai_throughput", variant)


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_unsourced_tflops_are_unset(hw, legacy_id):
    cap = hw[legacy_id].capabilities
    assert cap.peak_tflops_fp16 is None and cap.peak_tflops_bf16 is None
    if legacy_id != THOR:
        assert cap.peak_tflops_fp32 is None


def test_thor_fp32_is_nvidia_maxn(hw, db):
    assert hw[THOR].capabilities.peak_tflops_fp32 == _v(
        db, "nvidia_jetson_t5000", "fp32_throughput", "maxn"
    )


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_power_modes_are_nvidias(hw, db, legacy_id):
    """Every NVIDIA power mode appears, then MAXN / MAXN_SUPER at maximum
    module power; GPU clocks are set only where the DB has them."""
    s = LEGACY[legacy_id]
    power = hw[legacy_id].power
    pmax = _v(db, s, "power", "max")
    assert power.tdp_watts == pmax
    modes = sorted(o.value for o in db.find("power", s) if (o.variant or "").startswith("mode_"))
    watts = [m.power_watts for m in power.power_modes]
    assert watts[: len(modes)] == modes
    assert watts[-1] == pmax
    clocks = {o.variant: o.value for o in db.find("frequency", s)}
    for mode in power.power_modes:
        assert mode.cpu_freq_mhz is None
        if mode.gpu_freq_mhz is None:
            continue
        per_mode = clocks.get(f"gpu_{int(mode.power_watts)}w")
        top = clocks.get("gpu_max_super", clocks.get("gpu_max"))
        assert mode.gpu_freq_mhz in (per_mode, top), mode.name


def test_thor_modes_are_70_90_120_maxn(hw):
    modes = [(m.name, m.power_watts, m.gpu_freq_mhz) for m in hw[THOR].power.power_modes]
    assert modes == [
        ("70W", 70.0, 1158),
        ("90W", 90.0, 1259),
        ("120W", 120.0, 1386),
        ("MAXN", 130.0, 1575),
    ]


@pytest.mark.parametrize("legacy_id", LEGACY)
def test_cost_is_current_1ku_price(hw, db, legacy_id):
    s = LEGACY[legacy_id]
    price = _v(db, s, "unit_price", "qty_1000", listing="current")
    assert hw[legacy_id].cost_usd == price


def test_orin_nano_entry_is_the_module(hw):
    assert hw["nvidia_jetson_orin_nano_8gb"].name == "NVIDIA Jetson Orin Nano 8GB"
