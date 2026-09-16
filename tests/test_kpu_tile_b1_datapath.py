"""Phase B1 of the KPU heterogeneous-tile sprint (branes-ai/graphs#268).

New types: number formats, operators, functional units, PE datapaths, energy
references (``datapath.py``). New optional ``KPUTileSpec`` fields: tile_kind,
tile_class_id, datapath, footprint, local_memory, power_domain_id, placement.

Pins:
  * backward compatibility: every catalog SKU loads unchanged, and tile_kind /
    tile_class_id are filled in;
  * the datapath model can express every catalog tile class exactly (the
    datapath-derived ops equal the declared ops_per_tile_per_clock);
  * validation of formats, operators, energy anchors and cross-references.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.test_kpu_catalog import legacy_kpu_blocks
from embodied_schemas import (
    DEFAULT_OPS_PER_INVOCATION,
    AbsoluteEnergy,
    FunctionalUnit,
    KPUArchitecture,
    KPUBlock,
    KPUTileKind,
    KPUTileSpec,
    NumberFormatFamily,
    OpKind,
    PEDatapath,
    RelativeEnergy,
    UnitMode,
    load_compute_products,
    parse_number_format,
    tile_class_slug,
)
from embodied_schemas.process_node import CircuitClass


# The backward-compatibility contract is about the SKUs that shipped
# before the heterogeneous work; see tests/kpu_catalog.py.
KPU_BLOCKS = legacy_kpu_blocks()


# ---------------------------------------------------------------------------
# Number formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "int4",
        "int8",
        "int16",
        "int32",
        "uint8",
        "fp64",
        "fp32",
        "tf32",
        "fp16",
        "bf16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp4",
    ],
)
def test_legacy_precision_names_are_valid_formats(name):
    parse_number_format(name)


@pytest.mark.parametrize(
    "name,family,bits,extra",
    [
        ("lns16", NumberFormatFamily.LNS, 16, {}),
        ("posit16", NumberFormatFamily.POSIT, 16, {"es": 2}),
        ("posit8_0", NumberFormatFamily.POSIT, 8, {"es": 0}),
        ("fixed8.8", NumberFormatFamily.FIXED, 16, {"integer_bits": 8, "fraction_bits": 8}),
        ("bf16", NumberFormatFamily.FLOAT, 16, {"exponent_bits": 8}),
    ],
)
def test_new_formats_parse(name, family, bits, extra):
    spec = parse_number_format(name)
    assert (spec.family, spec.bits) == (family, bits)
    for k, v in extra.items():
        assert getattr(spec, k) == v


@pytest.mark.parametrize(
    "name",
    ["int7", "fp12", "lns12", "posit12", "posit16_7", "fixed0.8", "fixed40.40", "float32", ""],
)
def test_invalid_formats_are_rejected(name):
    with pytest.raises(ValueError):
        parse_number_format(name)


# ---------------------------------------------------------------------------
# Units and datapaths
# ---------------------------------------------------------------------------


def test_ops_counting_convention():
    assert DEFAULT_OPS_PER_INVOCATION[OpKind.MAC] == 2
    assert DEFAULT_OPS_PER_INVOCATION[OpKind.FMA] == 2
    assert DEFAULT_OPS_PER_INVOCATION[OpKind.LERP] == 3
    assert DEFAULT_OPS_PER_INVOCATION[OpKind.MIN_PLUS] == 2
    assert set(DEFAULT_OPS_PER_INVOCATION) == set(OpKind)


def test_unit_ops_per_clock_by_mode():
    unit = FunctionalUnit(
        unit_id="int_mac",
        op=OpKind.MAC,
        modes=[
            UnitMode(operand_format="int8", accumulate_format="int32"),
            UnitMode(operand_format="int4", lanes=2),
        ],
    )
    assert unit.ops_per_clock() == {"mac:int8": 2, "mac:int4": 4}
    half_rate = FunctionalUnit(
        unit_id="div",
        op=OpKind.DIV,
        modes=[UnitMode(operand_format="fp32", issue_interval_cycles=4)],
    )
    assert half_rate.ops_per_clock() == {"div:fp32": 0.25}
    override = FunctionalUnit(
        unit_id="c",
        op=OpKind.CORDIC,
        ops_per_invocation=6,
        modes=[UnitMode(operand_format="fixed2.14")],
    )
    assert override.ops_per_clock() == {"cordic:fixed2.14": 6}


def test_unit_rejects_duplicate_modes_and_bad_formats():
    with pytest.raises(ValidationError, match="duplicate mode formats"):
        FunctionalUnit(
            unit_id="u",
            op=OpKind.MAC,
            modes=[UnitMode(operand_format="int8"), UnitMode(operand_format="int8")],
        )
    with pytest.raises(ValidationError, match="unknown number format"):
        UnitMode(operand_format="float32")


def test_energy_refs():
    rel = RelativeEnergy(anchor="balanced_logic:int8", ratio=0.9)
    assert rel.kind == "relative"
    with pytest.raises(ValidationError, match="unknown circuit class"):
        RelativeEnergy(anchor="fast_logic:int8", ratio=1.0)
    with pytest.raises(ValidationError, match="must be '<circuit_class>:<format>'"):
        RelativeEnergy(anchor="balanced_logic", ratio=1.0)
    with pytest.raises(ValidationError):
        RelativeEnergy(anchor="balanced_logic:int8", ratio=0)
    ab = AbsoluteEnergy(pj=4.2, ref_node_id="tsmc_n65")
    assert ab.circuit_class == CircuitClass.BALANCED_LOGIC
    # Discriminated by kind inside a mode.
    m = UnitMode.model_validate(
        {
            "operand_format": "lns16",
            "energy": {"kind": "absolute", "pj": 0.2, "ref_node_id": "tsmc_n16"},
        }
    )
    assert isinstance(m.energy, AbsoluteEnergy)


def test_datapath_sums_concurrent_units_and_projects_legacy_keys():
    dp = PEDatapath(
        datapath_id="sgm_pe",
        functional_units=[
            FunctionalUnit(
                unit_id="sad0", op=OpKind.ABS_DIFF, modes=[UnitMode(operand_format="uint8")]
            ),
            FunctionalUnit(
                unit_id="sad1", op=OpKind.ABS_DIFF, modes=[UnitMode(operand_format="uint8")]
            ),
            FunctionalUnit(
                unit_id="mp", op=OpKind.MIN_PLUS, modes=[UnitMode(operand_format="int16")]
            ),
            FunctionalUnit(unit_id="mac", op=OpKind.MAC, modes=[UnitMode(operand_format="int8")]),
        ],
    )
    assert dp.ops_per_pe_per_clock() == {"abs_diff:uint8": 4, "min_plus:int16": 2, "mac:int8": 2}
    # Only MAC / FMA map onto legacy precision keys.
    assert dp.legacy_precision_ops_per_pe() == {"int8": 2}
    with pytest.raises(ValidationError, match="duplicate unit_id"):
        PEDatapath(
            datapath_id="x", functional_units=[dp.functional_units[0], dp.functional_units[0]]
        )


# ---------------------------------------------------------------------------
# Datapaths that reproduce every catalog tile class
# ---------------------------------------------------------------------------


def _unit(uid, op, *modes):
    return FunctionalUnit(unit_id=uid, op=op, modes=[UnitMode(**m) for m in modes])


def _catalog_datapath(tile: KPUTileSpec) -> PEDatapath:
    """The per-PE datapath that each catalog tile class implies.

    32x32 tiles: 1 MAC/PE/clk at INT8 (2 at INT4); BF16 / FP16 FMA at half rate
    (FP32 at quarter rate on BF16-primary). T768 16x8 tiles double the lanes.
    T768's 8x8 weight-stationary Matrix tiles run 64 INT8 MACs per PE per clock.
    """
    key = (tile.tile_type, tile.pe_array_rows, tile.pe_array_cols)
    mac, fma = OpKind.MAC, OpKind.FMA
    table = {
        ("INT8-primary", 32, 32): [
            _unit(
                "int_mac", mac, {"operand_format": "int8"}, {"operand_format": "int4", "lanes": 2}
            ),
            _unit(
                "fp_fma",
                fma,
                {"operand_format": "bf16", "issue_interval_cycles": 2},
                {"operand_format": "fp16", "issue_interval_cycles": 2},
            ),
        ],
        ("BF16-primary", 32, 32): [
            _unit("int_mac", mac, {"operand_format": "int8"}),
            _unit(
                "fp_fma",
                fma,
                {"operand_format": "bf16", "issue_interval_cycles": 2},
                {"operand_format": "fp16", "issue_interval_cycles": 2},
                {"operand_format": "fp32", "issue_interval_cycles": 4},
            ),
        ],
        ("Matrix", 32, 32): [
            _unit("int_mac", mac, {"operand_format": "int8"}),
            _unit(
                "fp_fma",
                fma,
                {"operand_format": "bf16", "issue_interval_cycles": 2},
                {"operand_format": "fp16", "issue_interval_cycles": 2},
            ),
        ],
        ("INT8-primary", 16, 8): [
            _unit(
                "int_mac",
                mac,
                {"operand_format": "int8", "lanes": 2},
                {"operand_format": "int4", "lanes": 4},
            ),
            _unit("fp_fma", fma, {"operand_format": "bf16"}, {"operand_format": "fp16"}),
        ],
        ("BF16-primary", 16, 8): [
            _unit("int_mac", mac, {"operand_format": "int8", "lanes": 2}),
            _unit(
                "fp_fma",
                fma,
                {"operand_format": "bf16"},
                {"operand_format": "fp16"},
                {"operand_format": "fp32", "issue_interval_cycles": 2},
            ),
        ],
        ("Matrix", 8, 8): [
            _unit("int_mac", mac, {"operand_format": "int8", "lanes": 64}),
            _unit(
                "fp_fma",
                fma,
                {"operand_format": "bf16", "lanes": 32},
                {"operand_format": "fp16", "lanes": 32},
            ),
        ],
    }
    return PEDatapath(datapath_id=tile_class_slug(tile.tile_type), functional_units=table[key])


@pytest.mark.parametrize("sku", sorted(KPU_BLOCKS))
def test_every_catalog_tile_class_is_expressible_as_a_datapath(sku):
    for tile in KPU_BLOCKS[sku].tiles:
        dp = _catalog_datapath(tile)
        with_dp = KPUTileSpec.model_validate({**tile.model_dump(), "datapath": dp.model_dump()})
        assert with_dp.ops_per_pe_per_clock() == dp.ops_per_pe_per_clock()


def test_datapath_mismatch_is_reported():
    tile = KPU_BLOCKS["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"].tiles[0]  # INT8-primary
    wrong = PEDatapath(
        datapath_id="w",
        functional_units=[
            _unit("int_mac", OpKind.MAC, {"operand_format": "int8", "lanes": 2}),
        ],
    )
    with pytest.raises(ValidationError) as exc:
        KPUTileSpec.model_validate({**tile.model_dump(), "datapath": wrong.model_dump()})
    msg = str(exc.value)
    assert "int8: declared 2048, datapath gives 1024 PEs x 4 = 4096" in msg
    assert "declared but not produced by the datapath: ['bf16', 'fp16', 'int4']" in msg


def test_datapath_circuit_class_must_match_tile():
    tile = KPU_BLOCKS["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"].tiles[2]  # Matrix, hp_logic
    dp = _catalog_datapath(tile).model_copy(update={"circuit_class": CircuitClass.BALANCED_LOGIC})
    with pytest.raises(
        ValidationError, match="built in balanced_logic but pe_circuit_class is hp_logic"
    ):
        KPUTileSpec.model_validate({**tile.model_dump(), "datapath": dp.model_dump()})


def test_new_format_precision_keys():
    """A custom-format datapath (LNS MAC) declares its format as a precision key."""
    tile = KPUTileSpec(
        tile_type="LNS16-MAC",
        num_tiles=4,
        pe_array_rows=32,
        pe_array_cols=32,
        pe_circuit_class=CircuitClass.BALANCED_LOGIC,
        ops_per_tile_per_clock={"lns16": 2048},
        datapath=PEDatapath(
            datapath_id="lns16_mac",
            functional_units=[
                FunctionalUnit(
                    unit_id="lns_mac",
                    op=OpKind.MAC,
                    modes=[
                        UnitMode(
                            operand_format="lns16",
                            energy=RelativeEnergy(anchor="balanced_logic:int8", ratio=0.9),
                        )
                    ],
                ),
                FunctionalUnit(
                    unit_id="lerp", op=OpKind.LERP, modes=[UnitMode(operand_format="fp16")]
                ),
            ],
        ),
    )
    assert tile.tile_class_id == "lns16_mac"
    assert tile.ops_per_pe_per_clock() == {"mac:lns16": 2, "lerp:fp16": 3}


# ---------------------------------------------------------------------------
# KPUTileSpec / KPUArchitecture B1 fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", sorted(KPU_BLOCKS))
def test_catalog_loads_unchanged_with_defaults(sku):
    block = KPU_BLOCKS[sku]
    for tile in block.tiles:
        assert tile.tile_kind == KPUTileKind.PE_FABRIC
        assert tile.tile_class_id == tile_class_slug(tile.tile_type)
        assert tile.datapath is None and tile.footprint is None and tile.local_memory is None
        assert tile.sites_per_tile == 1
    assert KPUBlock.model_validate(block.model_dump(mode="json")) == block


def test_tile_class_slug():
    assert tile_class_slug("INT8-primary") == "int8_primary"
    assert tile_class_slug("Matrix") == "matrix"
    assert tile_class_slug("  VIO (Navion-class) ") == "vio_navion_class"


def _tile(**over):
    base = dict(
        tile_type="INT8-primary",
        num_tiles=4,
        pe_array_rows=32,
        pe_array_cols=32,
        pe_circuit_class="balanced_logic",
        ops_per_tile_per_clock={"int8": 2048},
    )
    return {**base, **over}


def test_explicit_tile_class_id_and_fields():
    t = KPUTileSpec.model_validate(
        _tile(
            tile_class_id="int8_a",
            footprint={"rows": 2, "cols": 2, "absorbs_memory_cells": True},
            local_memory=[{"level": "l1", "kib": 4, "per": "pe"}, {"level": "l2", "kib": 32}],
            power_domain_id="cluster_0",
            placement={"affinity": "io_edge"},
        )
    )
    assert t.tile_class_id == "int8_a" and t.sites_per_tile == 4
    with pytest.raises(ValidationError):
        KPUTileSpec.model_validate(_tile(tile_class_id="Bad-Id"))
    with pytest.raises(ValidationError, match="duplicate local_memory levels"):
        KPUTileSpec.model_validate(
            _tile(local_memory=[{"level": "l2", "kib": 1}, {"level": "l2", "kib": 2}])
        )
    with pytest.raises(ValidationError):
        KPUTileSpec.model_validate(
            _tile(tile_kind="systolic")
        )  # later phases get their own classes


def test_architecture_cross_references():
    block = KPU_BLOCKS["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
    arch = block.to_architecture().model_dump()
    dup = [dict(t) for t in arch["tiles"]]
    dup[1]["tile_class_id"] = dup[0]["tile_class_id"]
    with pytest.raises(ValidationError, match="duplicate tile_class_id"):
        KPUArchitecture.model_validate({**arch, "tiles": dup})

    adj = [dict(t) for t in arch["tiles"]]
    adj[0]["placement"] = {"adjacent_to": ["nope"]}
    with pytest.raises(ValidationError, match="unknown tile_class_id 'nope'"):
        KPUArchitecture.model_validate({**arch, "tiles": adj})
    adj[0]["placement"] = {"adjacent_to": [adj[0]["tile_class_id"]]}
    with pytest.raises(ValidationError, match="lists itself"):
        KPUArchitecture.model_validate({**arch, "tiles": adj})
    adj[0]["placement"] = {"adjacent_to": [adj[1]["tile_class_id"]]}
    assert KPUArchitecture.model_validate({**arch, "tiles": adj}).tiles[
        0
    ].placement.adjacent_to == ["bf16_primary"]


# ---------------------------------------------------------------------------
# Review fixes (CodeRabbit on #88)
# ---------------------------------------------------------------------------


def test_legacy_projection_takes_max_across_alternative_op_kinds():
    """mac:int8 and fma:int8 are different keys (alternative modes): the int8
    peak is their max. Two units on the SAME key are concurrent and add up."""
    alt = PEDatapath(
        datapath_id="alt",
        functional_units=[
            _unit("mac", OpKind.MAC, {"operand_format": "int8"}),  # 2 ops
            _unit("fma", OpKind.FMA, {"operand_format": "int8", "lanes": 2}),  # 4 ops
        ],
    )
    assert alt.ops_per_pe_per_clock() == {"mac:int8": 2, "fma:int8": 4}
    assert alt.legacy_precision_ops_per_pe() == {"int8": 4}

    concurrent = PEDatapath(
        datapath_id="conc",
        functional_units=[
            _unit("mac0", OpKind.MAC, {"operand_format": "int8"}),
            _unit("mac1", OpKind.MAC, {"operand_format": "int8"}),
        ],
    )
    assert concurrent.legacy_precision_ops_per_pe() == {"int8": 4}


def test_absolute_energy_requires_a_reference_node():
    with pytest.raises(ValidationError):
        AbsoluteEnergy(pj=1.0, ref_node_id="")
