"""Phase B2 of the KPU heterogeneous-tile sprint (branes-ai/graphs#268).

Interconnect overlays: ``FabricInterconnect`` / ``FabricOverlay`` inside a tile
(``KPUTileSpec.interconnect``) and ``NoCOverlay`` on the chip NoC
(``KPUNoCSpec.overlays``). Overlays are statically configured links, never
routed networks.

Pins: backward compatibility (catalog unchanged), shape rules per overlay
kind, fit against the PE array / mesh, and tile-class references of NoC
overlays.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.test_kpu_catalog import legacy_kpu_blocks
from embodied_schemas import (
    FabricInterconnect,
    FabricOverlay,
    FabricOverlayKind,
    KPUArchitecture,
    KPUBlock,
    KPUTileSpec,
    NeighborTopology,
    NoCOverlay,
    NoCOverlayKind,
    OverlayScope,
    load_compute_products,
)


# The backward-compatibility contract is about the SKUs that shipped
# before the heterogeneous work; see tests/kpu_catalog.py.
KPU_BLOCKS = legacy_kpu_blocks()
T64 = KPU_BLOCKS["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]


def _ov(**kw):
    base = dict(overlay_id="o", width_bits=32)
    return FabricOverlay(**{**base, **kw})


def _tile(**over):
    base = dict(
        tile_type="FFT-PE",
        num_tiles=4,
        pe_array_rows=32,
        pe_array_cols=32,
        pe_circuit_class="balanced_logic",
        ops_per_tile_per_clock={"fp16": 2048},
    )
    return KPUTileSpec.model_validate({**base, **over})


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", sorted(KPU_BLOCKS))
def test_catalog_loads_unchanged(sku):
    block = KPU_BLOCKS[sku]
    assert block.noc.overlays is None
    assert all(t.interconnect is None for t in block.tiles)


# ---------------------------------------------------------------------------
# PE-level overlays
# ---------------------------------------------------------------------------


def test_fabric_overlay_shape_rules():
    assert _ov(kind="express", instances_per="row", span=4).span == 4
    assert _ov(kind="row_broadcast", instances_per="row").configuration == "static_per_schedule"
    with pytest.raises(ValidationError, match="express needs span >= 2"):
        _ov(kind="express", instances_per="row")
    with pytest.raises(ValidationError, match="express needs span >= 2"):
        _ov(kind="express", instances_per="row", span=1)
    with pytest.raises(ValidationError, match="span does not apply to row_broadcast"):
        _ov(kind="row_broadcast", instances_per="row", span=4)
    with pytest.raises(ValidationError, match="row_broadcast is per row"):
        _ov(kind="row_broadcast", instances_per="col")
    with pytest.raises(ValidationError, match="col_broadcast is per col"):
        _ov(kind="col_broadcast", instances_per="tile")
    with pytest.raises(ValidationError, match="transpose spans the whole tile"):
        _ov(kind="transpose", instances_per="row")
    with pytest.raises(ValidationError):
        _ov(kind="express", instances_per="row", span=4, configuration="routed")


def test_fabric_interconnect_unique_ids_and_defaults():
    fi = FabricInterconnect(link_bits=32)
    assert fi.base == NeighborTopology.NEAREST_NEIGHBOR_4 and fi.overlays == []
    with pytest.raises(ValidationError, match="duplicate fabric overlay_id"):
        FabricInterconnect(
            link_bits=32,
            overlays=[
                _ov(kind="express", instances_per="row", span=2),
                _ov(kind="express", instances_per="col", span=2),
            ],
        )


def test_tile_with_fft_overlays():
    """A 32x32 fabric with the overlays an FFT / reduction workload wants."""
    tile = _tile(
        interconnect={
            "base": "nearest_neighbor_4",
            "link_bits": 32,
            "overlays": [
                {
                    "overlay_id": "row_bcast",
                    "kind": "row_broadcast",
                    "instances_per": "row",
                    "width_bits": 32,
                },
                {
                    "overlay_id": "express4",
                    "kind": "express",
                    "instances_per": "row",
                    "instances": 2,
                    "span": 4,
                    "width_bits": 32,
                },
                {
                    "overlay_id": "fft",
                    "kind": "butterfly",
                    "instances_per": "row",
                    "width_bits": 32,
                },
                {
                    "overlay_id": "xpose",
                    "kind": "transpose",
                    "instances_per": "tile",
                    "width_bits": 16,
                },
                {
                    "overlay_id": "sum",
                    "kind": "reduction_tree",
                    "instances_per": "col",
                    "width_bits": 32,
                },
            ],
        }
    )
    kinds = {o.kind for o in tile.interconnect.overlays}
    assert FabricOverlayKind.BUTTERFLY in kinds and FabricOverlayKind.TRANSPOSE in kinds


@pytest.mark.parametrize(
    "rows,cols,overlay,match",
    [
        (32, 32, {"kind": "express", "instances_per": "row", "span": 32}, "does not fit a 32x32"),
        (16, 8, {"kind": "express", "instances_per": "row", "span": 8}, "max 7 along its axis"),
        (16, 8, {"kind": "express", "instances_per": "col", "span": 15}, None),  # fits: rows=16
        # Tile scope spans both axes: the span must fit the SHORTER one (8 cols).
        (16, 8, {"kind": "express", "instances_per": "tile", "span": 15}, "max 7 along its axis"),
        (16, 8, {"kind": "express", "instances_per": "tile", "span": 8}, "max 7 along its axis"),
        (16, 8, {"kind": "express", "instances_per": "tile", "span": 7}, None),
        (8, 16, {"kind": "segmented_bus", "instances_per": "tile", "span": 8}, "max 7"),
        (16, 8, {"kind": "transpose", "instances_per": "tile"}, "needs a square PE array"),
        (24, 24, {"kind": "butterfly", "instances_per": "row"}, "power-of-two axis"),
        (16, 8, {"kind": "butterfly", "instances_per": "tile"}, None),  # 16 and 8 both pow2
        (16, 12, {"kind": "butterfly", "instances_per": "col"}, None),  # col axis = rows = 16
        (16, 12, {"kind": "butterfly", "instances_per": "tile"}, "power-of-two axis"),
    ],
)
def test_overlays_must_fit_the_pe_array(rows, cols, overlay, match):
    data = dict(
        pe_array_rows=rows,
        pe_array_cols=cols,
        ops_per_tile_per_clock={"fp16": rows * cols * 2},
        interconnect={
            "link_bits": 32,
            "overlays": [{"overlay_id": "o", "width_bits": 32, **overlay}],
        },
    )
    if match is None:
        assert _tile(**data).interconnect is not None
    else:
        with pytest.raises(ValidationError, match=match):
            _tile(**data)


# ---------------------------------------------------------------------------
# NoC overlays
# ---------------------------------------------------------------------------


def test_noc_overlay_shape_rules():
    link = NoCOverlay(
        overlay_id="isp_to_vio", kind="stream_link", endpoints=["a", "b", "c"], width_bytes=32
    )
    assert link.endpoints == ["a", "b", "c"] and link.configuration == "static_per_schedule"
    with pytest.raises(ValidationError, match="stream_link needs >= 2 endpoints"):
        NoCOverlay(overlay_id="s", kind="stream_link", endpoints=["a"], width_bytes=32)
    with pytest.raises(ValidationError, match="multicast_tree needs >= 2 endpoints"):
        NoCOverlay(overlay_id="m", kind="multicast_tree", width_bytes=32)
    with pytest.raises(ValidationError, match="endpoints repeat"):
        NoCOverlay(overlay_id="s", kind="stream_link", endpoints=["a", "a"], width_bytes=32)
    with pytest.raises(ValidationError, match="express_channel needs span >= 2"):
        NoCOverlay(overlay_id="e", kind="express_channel", width_bytes=32)
    with pytest.raises(ValidationError, match="span does not apply to stream_link"):
        NoCOverlay(overlay_id="s", kind="stream_link", endpoints=["a", "b"], width_bytes=32, span=3)
    assert (
        NoCOverlay(overlay_id="e", kind="express_channel", span=4, width_bytes=32).endpoints == []
    )


def _arch_with_noc_overlays(overlays):
    arch = T64.to_architecture().model_dump()
    return {**arch, "noc": {**arch["noc"], "overlays": overlays}}


def test_noc_overlays_reference_tile_classes():
    ok = KPUArchitecture.model_validate(
        _arch_with_noc_overlays(
            [
                {
                    "overlay_id": "int8_to_matrix",
                    "kind": "stream_link",
                    "endpoints": ["int8_primary", "matrix"],
                    "width_bytes": 32,
                },
                {"overlay_id": "express", "kind": "express_channel", "span": 4, "width_bytes": 16},
                {
                    "overlay_id": "bcast",
                    "kind": "multicast_tree",
                    "endpoints": ["bf16_primary", "int8_primary", "matrix"],
                    "width_bytes": 16,
                },
            ]
        )
    )
    kinds = [o.kind for o in ok.noc.overlays]
    assert kinds == [
        NoCOverlayKind.STREAM_LINK,
        NoCOverlayKind.EXPRESS_CHANNEL,
        NoCOverlayKind.MULTICAST_TREE,
    ]

    with pytest.raises(ValidationError, match=r"unknown tile_class_id \['vio'\]"):
        KPUArchitecture.model_validate(
            _arch_with_noc_overlays(
                [
                    {
                        "overlay_id": "s",
                        "kind": "stream_link",
                        "endpoints": ["int8_primary", "vio"],
                        "width_bytes": 32,
                    },
                ]
            )
        )
    with pytest.raises(ValidationError, match="duplicate NoC overlay_id"):
        KPUArchitecture.model_validate(
            _arch_with_noc_overlays(
                [
                    {"overlay_id": "e", "kind": "express_channel", "span": 2, "width_bytes": 16},
                    {"overlay_id": "e", "kind": "express_channel", "span": 3, "width_bytes": 16},
                ]
            )
        )
    with pytest.raises(ValidationError, match="does not fit a 8x8 mesh"):  # T64 mesh is 8x8
        KPUArchitecture.model_validate(
            _arch_with_noc_overlays(
                [
                    {"overlay_id": "e", "kind": "express_channel", "span": 8, "width_bytes": 16},
                ]
            )
        )


def test_overlay_scope_values():
    assert {s.value for s in OverlayScope} == {"row", "col", "tile"}
