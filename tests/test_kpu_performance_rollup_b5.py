"""Phase B5 of the KPU heterogeneous-tile sprint (branes-ai/graphs#268).

Performance roll-up by tile kind:

- ``KPUTheoreticalPerformance`` gains ``peak_ops_per_sec_by_precision``,
  ``by_tile_kind`` (programmable kinds only) and
  ``fixed_function_throughput`` (units/s per function_id).
- The legacy fields must agree with the peak map when it is present.
- ``derive_kpu_performance`` computes every field from the tiles and
  reproduces the graphs generator's legacy numbers for all catalog SKUs.
- ``KPUEntry`` / ``ComputeProduct`` check a declared roll-up against their
  tiles at the default profile's clock.

Fixed-function tiles never add to programmable ops/s or TOPS (plan R13).
Figures in the fixture are illustrative, not catalog data.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    PROGRAMMABLE_TILE_KINDS,
    ComputeProduct,
    KPUArchitecture,
    KPUBlock,
    KPUEntry,
    KPUTheoreticalPerformance,
    KPUTileKind,
    derive_kpu_performance,
    load_compute_products,
    load_kpus,
)

T64_ID = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"
CATALOG = load_compute_products()
KPU_PRODUCTS = {
    sku: cp
    for sku, cp in CATALOG.items()
    if any(isinstance(b, KPUBlock) for d in cp.dies for b in d.blocks)
}
T64 = KPU_PRODUCTS[T64_ID]
LEGACY = ("int8_tops", "bf16_tflops", "fp32_tflops", "int4_tops")
PE, SYS = KPUTileKind.PE_FABRIC, KPUTileKind.SYSTOLIC


def _default_clock(power) -> float:
    name = power.default_thermal_profile
    return next(p.clock_mhz for p in power.thermal_profiles if p.name == name)


def _kpu_tiles(cp) -> list:
    return [t for d in cp.dies for b in d.blocks if isinstance(b, KPUBlock) for t in b.tiles]


def _isp(tile_type: str, num_tiles: int = 1, units_per_clock: float = 2) -> dict:
    return {
        "tile_kind": "fixed_function",
        "tile_type": tile_type,
        "num_tiles": num_tiles,
        "core": {
            "function_id": "isp.raw_to_yuv",
            "contract": {"inputs": ["bayer_raw_frame"], "outputs": ["yuv420_frame"]},
            "numeric_formats": ["uint16"],
            "throughput": {"unit": "pixel", "units_per_clock": units_per_clock},
            "energy": {"pj_per_unit": 30.0, "ref_node_id": "tsmc_n16"},
        },
    }


def _arch(with_ff: bool = True) -> dict:
    """T64's pe_fabric mix + 2 systolic tiles (+ 2 ISP classes, 3 tiles)."""
    base = T64.dies[0].blocks[0].to_architecture().model_dump(mode="json")
    tiles = base["tiles"] + [
        {
            "tile_kind": "systolic",
            "tile_type": "Systolic-INT8",
            "num_tiles": 2,
            "array_rows": 32,
            "array_cols": 32,
            "circuit_class": "balanced_logic",
            "mac": {
                "unit_id": "mac",
                "op": "mac",
                "modes": [
                    {"operand_format": "int8", "accumulate_format": "int32"},
                    {"operand_format": "bf16", "issue_interval_cycles": 2},
                ],
            },
        },
    ]
    if with_ff:
        tiles += [_isp("ISP-A", num_tiles=2), _isp("ISP-B", units_per_clock=4)]
    return {**base, "tiles": tiles, "total_tiles": sum(t["num_tiles"] for t in tiles)}


# ---------------------------------------------------------------------------
# Backward compatibility: the catalog, and the generator's legacy convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", sorted(KPU_PRODUCTS))
def test_catalog_performance_is_legacy_and_derivable(sku):
    cp = KPU_PRODUCTS[sku]
    perf = cp.performance
    assert not perf.declares_rollup
    derived = derive_kpu_performance(_kpu_tiles(cp), _default_clock(cp.power))
    for field in LEGACY:
        assert getattr(derived, field) == getattr(perf, field), field
    # A catalog SKU can opt in: the derived roll-up validates in place.
    again = ComputeProduct.model_validate(
        {**cp.model_dump(mode="json"), "performance": derived.model_dump(mode="json")}
    )
    assert again.performance.declares_rollup


def test_legacy_kpu_entries_are_derivable():
    for sku, entry in load_kpus().items():
        derived = derive_kpu_performance(
            entry.kpu_architecture.tiles, entry.power.default_profile.clock_mhz
        )
        assert derived.model_dump(include=set(LEGACY)) == entry.performance.model_dump(
            include=set(LEGACY)
        ), sku


# ---------------------------------------------------------------------------
# derive_kpu_performance
# ---------------------------------------------------------------------------


def test_derive_splits_by_kind_and_keeps_fixed_function_out_of_tops():
    arch = KPUArchitecture.model_validate(_arch())
    clock = 500.0
    hz = clock * 1e6
    perf = derive_kpu_performance(arch.tiles, clock)

    assert set(perf.by_tile_kind) == {PE, SYS}
    assert set(perf.by_tile_kind) <= PROGRAMMABLE_TILE_KINDS
    # 2 systolic tiles x 32x32 cells x 2 ops (MAC); bf16 issues every 2nd clock.
    assert perf.by_tile_kind[SYS] == {"int8": 2 * 1024 * 2 * hz, "bf16": 2 * 1024 * hz}
    pe_int8 = sum(t.num_tiles * t.ops_per_tile_per_clock["int8"] for t in arch.tiles[:3]) * hz
    assert math.isclose(perf.by_tile_kind[PE]["int8"], pe_int8)
    assert math.isclose(
        perf.peak_ops_per_sec_by_precision["int8"], pe_int8 + 2 * 1024 * 2 * hz
    )
    # Two ISP classes share a function_id: 2 tiles x 2 px/clk + 1 tile x 4 px/clk.
    assert perf.fixed_function_throughput == {"isp.raw_to_yuv": (2 * 2 + 4) * hz}
    assert perf.int8_tops == round(perf.peak_ops_per_sec_by_precision["int8"] / 1e12, 1)

    # Fixed-function tiles change nothing but their own throughput map.
    no_ff = derive_kpu_performance(KPUArchitecture.model_validate(_arch(False)).tiles, clock)
    assert no_ff.fixed_function_throughput is None
    assert no_ff.model_dump(exclude={"fixed_function_throughput"}) == perf.model_dump(
        exclude={"fixed_function_throughput"}
    )


def test_derived_rollup_round_trips():
    perf = derive_kpu_performance(KPUArchitecture.model_validate(_arch()).tiles, 500.0)
    again = KPUTheoreticalPerformance.model_validate(perf.model_dump(mode="json"))
    assert again == perf
    assert list(perf.model_dump().keys())[:4] == list(LEGACY)
    assert set(perf.model_dump(mode="json")["by_tile_kind"]) == {"pe_fabric", "systolic"}


# ---------------------------------------------------------------------------
# KPUTheoreticalPerformance self-validation
# ---------------------------------------------------------------------------


def _perf(**over) -> dict:
    base = derive_kpu_performance(KPUArchitecture.model_validate(_arch()).tiles, 500.0)
    return {**base.model_dump(mode="json"), **over}


def test_rollup_self_consistency():
    # Legacy fields match the peak to their precision: rounded or exact both pass.
    ok = _perf()
    ok["int8_tops"] = ok["peak_ops_per_sec_by_precision"]["int8"] / 1e12
    KPUTheoreticalPerformance.model_validate(ok)

    data = _perf()
    data["by_tile_kind"]["fixed_function"] = {"int8": 1.0}
    with pytest.raises(ValidationError, match=r"programmable kinds only .* \['fixed_function'\]"):
        KPUTheoreticalPerformance.model_validate(data)

    data = _perf()
    data["by_tile_kind"]["systolic"]["int8"] *= 2
    with pytest.raises(ValidationError, match="must be the sum of by_tile_kind"):
        KPUTheoreticalPerformance.model_validate(data)

    data = _perf()
    data["int8_tops"] += 0.2
    with pytest.raises(ValidationError, match=r"int8_tops = .* disagrees with .*\['int8'\]"):
        KPUTheoreticalPerformance.model_validate(data)

    data = _perf(int4_tops=None)
    data["peak_ops_per_sec_by_precision"]["int4"] = 1e12
    data["by_tile_kind"]["pe_fabric"]["int4"] = 1e12
    with pytest.raises(ValidationError, match="int4_tops is None but"):
        KPUTheoreticalPerformance.model_validate(data)

    for bad in (-1.0, float("nan"), float("inf")):
        data = _perf()
        data["fixed_function_throughput"]["isp.raw_to_yuv"] = bad
        with pytest.raises(ValidationError, match="must be finite and >= 0"):
            KPUTheoreticalPerformance.model_validate(data)

    with pytest.raises(ValidationError, match=r"keys \['isp'\] are not function_ids"):
        KPUTheoreticalPerformance.model_validate(_perf(fixed_function_throughput={"isp": 1.0}))


def test_by_tile_kind_alone_needs_no_peak():
    data = _perf(peak_ops_per_sec_by_precision=None, fixed_function_throughput=None)
    assert KPUTheoreticalPerformance.model_validate(data).declares_rollup


# ---------------------------------------------------------------------------
# Cross-checks against the architecture
# ---------------------------------------------------------------------------


def _product(clock_for_perf: float | None = None, **perf_over) -> dict:
    cp = T64.model_dump(mode="json")
    arch = KPUArchitecture.model_validate(_arch())
    cp["dies"][0]["blocks"] = [KPUBlock.from_architecture(arch).model_dump(mode="json")]
    clock = clock_for_perf or _default_clock(T64.power)
    perf = derive_kpu_performance(arch.tiles, clock).model_dump(mode="json")
    cp["performance"] = {**perf, **perf_over}
    return cp


def _entry(clock_for_perf: float | None = None, **perf_over) -> dict:
    entry = load_kpus()[T64_ID]
    arch = KPUArchitecture.model_validate(_arch())
    clock = clock_for_perf or entry.power.default_profile.clock_mhz
    data = entry.model_dump(mode="json")
    data["kpu_architecture"] = arch.model_dump(mode="json")
    perf = derive_kpu_performance(arch.tiles, clock).model_dump(mode="json")
    data["performance"] = {**perf, **perf_over}
    return data


@pytest.mark.parametrize("model, build", [(ComputeProduct, _product), (KPUEntry, _entry)])
def test_declared_rollup_must_match_the_tiles(model, build):
    obj = model.model_validate(build())
    assert obj.performance.fixed_function_throughput == {
        "isp.raw_to_yuv": pytest.approx(8 * _default_clock(obj.power) * 1e6)
    }

    # Derived at the wrong clock: every field disagrees; the peak is reported first.
    derived_msg = "derived from the tiles"
    with pytest.raises(ValidationError, match=f"peak_ops_per_sec_by_precision .* {derived_msg}"):
        model.model_validate(build(clock_for_perf=100.0))

    # Only the per-kind map declared, and wrong.
    data = build(peak_ops_per_sec_by_precision=None, fixed_function_throughput=None)
    data["performance"]["by_tile_kind"]["systolic"]["int8"] /= 2
    with pytest.raises(ValidationError, match=rf"by_tile_kind\['systolic'\] .* {derived_msg}"):
        model.model_validate(data)

    data = build(fixed_function_throughput={"isp.raw_to_yuv": 1.0})
    with pytest.raises(ValidationError, match=f"fixed_function_throughput .* {derived_msg}"):
        model.model_validate(data)


def test_legacy_only_performance_is_not_checked_against_tiles():
    # Legacy numbers that do not match the tiles still load: the graphs
    # tile_mix_consistency validator owns that check, as before.
    data = T64.model_dump(mode="json")
    data["performance"]["int8_tops"] = 1.0
    assert ComputeProduct.model_validate(data).performance.int8_tops == 1.0


def test_products_without_a_kpu_block_are_not_checked_against_tiles():
    cp = next(c for c in CATALOG.values() if c.id not in KPU_PRODUCTS)
    perf = cp.performance.model_dump(mode="json")
    perf["peak_ops_per_sec_by_precision"] = {
        "int8": perf["int8_tops"] * 1e12,
        "bf16": perf["bf16_tflops"] * 1e12,
        "fp32": perf["fp32_tflops"] * 1e12,
        **({"int4": perf["int4_tops"] * 1e12} if perf["int4_tops"] else {}),
    }
    again = ComputeProduct.model_validate({**cp.model_dump(mode="json"), "performance": perf})
    assert again.performance.declares_rollup


def test_unknown_default_profile_is_a_validation_error():
    # Checked by Power itself, so it surfaces as a normal ValidationError
    # rather than a StopIteration from the roll-up check (CodeRabbit on #93).
    for cp in (T64, next(c for c in CATALOG.values() if c.id not in KPU_PRODUCTS)):
        data = cp.model_dump(mode="json")
        data["power"]["default_thermal_profile"] = "9000W"
        with pytest.raises(ValidationError, match="default_thermal_profile='9000W' is not in"):
            ComputeProduct.model_validate(data)
    assert T64.power.default_profile.name == T64.power.default_thermal_profile
