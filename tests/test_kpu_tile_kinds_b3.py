"""Phase B3 of the KPU heterogeneous-tile sprint (branes-ai/graphs#268).

Tile kinds: ``KPUTileBase`` + ``KPUTileSpec`` (pe_fabric) + ``SystolicTile`` +
``FixedFunctionTile`` (a ``FunctionCore``), discriminated on ``tile_kind``
(missing = pe_fabric) as ``AnyKPUTile`` in ``KPUArchitectureBase.tiles``.

Pins:
  * backward compatibility: catalog tiles still load as KPUTileSpec, with the
    exact serialized key order they had before the base class;
  * union dispatch (including unknown / reserved kinds);
  * systolic derivations, function-core validation, fixed-function rules;
  * a full heterogeneous architecture round-trips through KPUBlock.

Energy / throughput figures in the examples are illustrative, not catalog data.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from embodied_schemas import (
    AnyKPUTile,
    CoreSiliconBlock,
    FixedFunctionTile,
    FunctionCore,
    FunctionEnergy,
    FunctionThroughput,
    KPUArchitecture,
    KPUBlock,
    KPUTileBase,
    KPUTileKind,
    KPUTileSpec,
    PEFabricTile,
    SystolicKernel,
    SystolicTile,
    WorkUnit,
    load_compute_products,
)

TILE = TypeAdapter(AnyKPUTile)

LEGACY_TILE_KEY_ORDER = [
    "tile_kind",
    "tile_type",
    "tile_class_id",
    "num_tiles",
    "pe_array_rows",
    "pe_array_cols",
    "pe_circuit_class",
    "ops_per_tile_per_clock",
    "schedule_class",
    "pipeline_fill_cycles",
    "pipeline_drain_cycles",
    "notes",
    "datapath",
    "footprint",
    "local_memory",
    "power_domain_id",
    "placement",
    "interconnect",
]


def _kpu_blocks() -> dict[str, KPUBlock]:
    return {
        sku: b
        for sku, cp in load_compute_products().items()
        for d in cp.dies
        for b in d.blocks
        if isinstance(b, KPUBlock)
    }


KPU_BLOCKS = _kpu_blocks()
T64 = KPU_BLOCKS["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]


def _isp_core(**over) -> dict:
    core = {
        "function_id": "isp.raw_to_yuv",
        "contract": {
            "inputs": ["bayer_raw_frame"],
            "outputs": ["yuv420_frame"],
            "config_limits": {"max_width": 1440, "max_height": 1080, "max_fps": 60},
        },
        "numeric_formats": ["uint16", "fixed8.8"],
        "throughput": {"unit": "pixel", "units_per_clock": 2, "fmax_mhz_ref": 600},
        "energy": {
            "pj_per_unit": 30.0,
            "ref_node_id": "tsmc_n16",
            "logic_fraction": 0.6,
            "sram_fraction": 0.4,
        },
        "ops_equivalent_per_unit": {"int16": 200},
        "io": {"input_bytes_per_unit": 2, "output_bytes_per_unit": 1.5},
        "local_memory": [{"level": "line_buffer", "kib": 96}],
        "silicon": [{"name": "isp_pipeline", "circuit_class": "balanced_logic", "mtx": 45}],
    }
    return {**core, **over}


def _vio_core() -> dict:
    return {
        "function_id": "vio.stereo_inertial",
        "contract": {"inputs": ["stereo_gray_frame", "imu_sample"], "outputs": ["pose_6dof"]},
        "numeric_formats": ["fixed16.16", "fp32"],
        "throughput": {"unit": "frame", "cycles_per_unit": 2.0e6},
        "energy": {
            "pj_per_unit": 1.0e8,
            "ref_node_id": "tsmc_n65",
            "logic_fraction": 0.5,
            "sram_fraction": 0.5,
            "source": "Navion-class (MIT, JSSC 2019); illustrative",
        },
        "local_memory": [{"level": "state", "kib": 854}],
        "silicon": [
            {
                "name": "vio_core",
                "circuit_class": "balanced_logic",
                "area_mm2": 20.0,
                "ref_node_id": "tsmc_n65",
            }
        ],
    }


def _systolic(**over) -> dict:
    t = {
        "tile_kind": "systolic",
        "tile_type": "Systolic-INT8",
        "num_tiles": 4,
        "array_rows": 64,
        "array_cols": 64,
        "circuit_class": "balanced_logic",
        "mac": {
            "unit_id": "mac",
            "op": "mac",
            "modes": [
                {"operand_format": "int8", "accumulate_format": "int32"},
                {"operand_format": "bf16", "issue_interval_cycles": 2},
            ],
        },
        "local_memory": [
            {"level": "weight_buffer", "kib": 64},
            {"level": "accumulator", "kib": 32, "circuit_class": "sram_hp"},
        ],
    }
    return {**t, **over}


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", sorted(KPU_BLOCKS))
def test_catalog_tiles_are_pe_fabric_with_unchanged_key_order(sku):
    for tile in KPU_BLOCKS[sku].tiles:
        assert type(tile) is KPUTileSpec
        assert isinstance(tile, KPUTileBase)
        assert list(tile.model_dump().keys()) == LEGACY_TILE_KEY_ORDER
        assert list(tile.model_dump(mode="json").keys()) == LEGACY_TILE_KEY_ORDER


def test_pe_fabric_alias():
    assert PEFabricTile is KPUTileSpec


# ---------------------------------------------------------------------------
# Union dispatch
# ---------------------------------------------------------------------------


def test_union_dispatch():
    legacy = T64.tiles[0].model_dump(mode="json")
    legacy.pop("tile_kind")
    assert type(TILE.validate_python(legacy)) is KPUTileSpec  # missing kind = pe_fabric
    assert type(TILE.validate_python(_systolic())) is SystolicTile
    ff = {"tile_kind": "fixed_function", "tile_type": "ISP", "num_tiles": 1, "core": _isp_core()}
    assert type(TILE.validate_python(ff)) is FixedFunctionTile
    for bad in ("scalar", "io_bridge", "nope"):
        with pytest.raises(ValidationError):
            TILE.validate_python({**ff, "tile_kind": bad})


def test_kind_specific_fields_do_not_leak_across_kinds():
    with pytest.raises(ValidationError):  # pe_fabric fields on a systolic tile
        TILE.validate_python(_systolic(pe_array_rows=32))
    with pytest.raises(ValidationError):  # a systolic tile lacking its array
        TILE.validate_python({k: v for k, v in _systolic().items() if k != "array_rows"})


# ---------------------------------------------------------------------------
# Systolic tiles
# ---------------------------------------------------------------------------


def test_systolic_derivations():
    t = SystolicTile.model_validate(_systolic())
    assert t.tile_kind == KPUTileKind.SYSTOLIC and t.tile_class_id == "systolic_int8"
    assert t.pes_per_tile == 4096 and t.total_pes == 4 * 4096
    # 4096 cells x 1 MAC = 8192 INT8 ops; BF16 at half rate = 4096.
    assert t.ops_per_tile_per_clock == {"int8": 8192, "bf16": 4096}
    assert t.pipeline_fill_cycles == t.pipeline_drain_cycles == 64
    assert t.supported_kernels == [SystolicKernel.GEMM, SystolicKernel.CONV2D]


def test_systolic_rules():
    with pytest.raises(ValidationError, match="no_local_reuse"):
        SystolicTile.model_validate(_systolic(dataflow="no_local_reuse"))
    with pytest.raises(ValidationError, match="must be a mac or fma"):
        SystolicTile.model_validate(
            _systolic(
                mac={"unit_id": "x", "op": "min_plus", "modes": [{"operand_format": "int16"}]}
            )
        )
    with pytest.raises(ValidationError):
        SystolicTile.model_validate(_systolic(supported_kernels=[]))


# ---------------------------------------------------------------------------
# Function cores and fixed-function tiles
# ---------------------------------------------------------------------------


def test_function_core_validation():
    core = FunctionCore.model_validate(_isp_core())
    assert core.units_per_clock == 2
    assert FunctionCore.model_validate(_vio_core()).units_per_clock == pytest.approx(5e-7)
    with pytest.raises(ValidationError):
        FunctionCore.model_validate(_isp_core(function_id="isp"))  # needs a dotted id
    with pytest.raises(ValidationError, match="unknown number format"):
        FunctionCore.model_validate(_isp_core(numeric_formats=["float16"]))
    with pytest.raises(ValidationError):
        FunctionCore.model_validate(_isp_core(ops_equivalent_per_unit={"int7": 1}))
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValidationError, match="finite and >= 0"):
            FunctionCore.model_validate(_isp_core(ops_equivalent_per_unit={"int16": bad}))
    with pytest.raises(ValidationError, match="duplicate local_memory levels"):
        FunctionCore.model_validate(
            _isp_core(
                local_memory=[
                    {"level": "line_buffer", "kib": 1},
                    {"level": "line_buffer", "kib": 2},
                ]
            )
        )


def test_throughput_energy_and_silicon_rules():
    with pytest.raises(ValidationError, match="exactly one of units_per_clock or cycles_per_unit"):
        FunctionThroughput(unit=WorkUnit.PIXEL)
    with pytest.raises(ValidationError, match="exactly one of units_per_clock or cycles_per_unit"):
        FunctionThroughput(unit=WorkUnit.PIXEL, units_per_clock=1, cycles_per_unit=1)
    with pytest.raises(ValidationError, match="exceeds 1"):
        FunctionEnergy(pj_per_unit=1, ref_node_id="tsmc_n16", logic_fraction=0.7, sram_fraction=0.5)
    with pytest.raises(ValidationError):
        FunctionEnergy(pj_per_unit=1, ref_node_id="")
    with pytest.raises(ValidationError, match="exactly one of mtx or area_mm2"):
        CoreSiliconBlock(name="b", circuit_class="balanced_logic")
    with pytest.raises(ValidationError, match="area_mm2 needs ref_node_id"):
        CoreSiliconBlock(name="b", circuit_class="balanced_logic", area_mm2=2.0)


def test_fixed_function_tile():
    t = FixedFunctionTile.model_validate(
        {
            "tile_type": "ISP",
            "num_tiles": 1,
            "core": _isp_core(),
            "placement": {"affinity": "io_edge"},
        }
    )
    assert t.tile_kind == KPUTileKind.FIXED_FUNCTION and t.tile_class_id == "isp"
    assert t.function_id == "isp.raw_to_yuv" and t.units_per_clock == 2
    assert t.ops_per_tile_per_clock == {}  # never adds to programmable TOPS
    with pytest.raises(ValidationError, match="declare memory in core.local_memory"):
        FixedFunctionTile.model_validate(
            {
                "tile_type": "ISP",
                "num_tiles": 1,
                "core": _isp_core(),
                "local_memory": [{"level": "line_buffer", "kib": 1}],
            }
        )


# ---------------------------------------------------------------------------
# A heterogeneous KPU, end to end
# ---------------------------------------------------------------------------


def _heterogeneous_arch() -> dict:
    arch = T64.to_architecture().model_dump(mode="json")
    tiles = arch["tiles"] + [
        _systolic(),
        {
            "tile_kind": "fixed_function",
            "tile_type": "ISP",
            "num_tiles": 1,
            "core": _isp_core(),
            "placement": {"affinity": "io_edge"},
        },
        {
            "tile_kind": "fixed_function",
            "tile_type": "VIO",
            "num_tiles": 1,
            "core": _vio_core(),
            "footprint": {"rows": 2, "cols": 2, "absorbs_memory_cells": True},
            "placement": {"adjacent_to": ["isp"]},
        },
    ]
    noc = {
        **arch["noc"],
        "overlays": [
            {
                "overlay_id": "isp_to_vio",
                "kind": "stream_link",
                "endpoints": ["isp", "vio"],
                "width_bytes": 32,
            },
        ],
    }
    return {**arch, "tiles": tiles, "noc": noc, "total_tiles": sum(t["num_tiles"] for t in tiles)}


def test_heterogeneous_architecture_round_trips():
    arch = KPUArchitecture.model_validate(_heterogeneous_arch())
    kinds = [type(t).__name__ for t in arch.tiles]
    assert kinds == ["KPUTileSpec"] * 3 + ["SystolicTile", "FixedFunctionTile", "FixedFunctionTile"]
    vio = arch.tiles[-1]
    assert vio.sites_per_tile == 4 and vio.total_sites == 4
    assert sum(t.total_sites for t in arch.tiles) == 64 + 4 + 1 + 4

    block = KPUBlock.from_architecture(arch)
    again = KPUBlock.model_validate(block.model_dump(mode="json"))
    assert again == block
    assert again.to_architecture() == arch
    # Every kind keeps identity first and the shared optional fields last.
    for tile in again.tiles:
        keys = list(tile.model_dump().keys())
        assert keys[:4] == ["tile_kind", "tile_type", "tile_class_id", "num_tiles"]


def test_references_span_all_tile_kinds():
    data = _heterogeneous_arch()
    data["tiles"][-1]["placement"] = {"adjacent_to": ["systolic_int8"]}
    assert KPUArchitecture.model_validate(data).tiles[-1].placement.adjacent_to == ["systolic_int8"]

    dup = _heterogeneous_arch()
    dup["tiles"][-1]["tile_class_id"] = "isp"
    with pytest.raises(ValidationError, match="duplicate tile_class_id"):
        KPUArchitecture.model_validate(dup)

    bad = _heterogeneous_arch()
    bad["noc"]["overlays"][0]["endpoints"] = ["isp", "sgm"]
    with pytest.raises(ValidationError, match=r"unknown tile_class_id \['sgm'\]"):
        KPUArchitecture.model_validate(bad)
