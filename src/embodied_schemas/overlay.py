"""Interconnect overlays for spatial fabrics (branes-ai/graphs#268 Phase B2).

A KPU ``pe_fabric`` tile is a nearest-neighbor, systolic-style domain-flow
fabric. Communication patterns that are not nearest-neighbor (broadcast,
reduction, transpose, butterfly exchanges for FFT, ...) are emulated by
*overlays*: extra wires and muxes laid over the base mesh. The same idea
applies between tiles on the chip NoC: express channels that skip hops,
stream links that chain tile classes producer -> consumer (e.g. ISP -> SGM ->
VIO, so a compute segment's intermediate data never goes through L3 or
DRAM), and multicast trees.

Overlays are **statically configured**. The domain-flow schedule is encoded
in the fabric topology and in the per-schedule configuration of these links.
None of these overlay kinds is a packet router with routing tables, and none
implies a schedule sequencer or ROM.

Types:

- ``FabricInterconnect`` / ``FabricOverlay``: inside a tile, at PE level
  (``kpu.KPUTileSpec.interconnect``).
- ``NoCOverlay``: between tiles, on the chip NoC
  (``kpu.KPUNoCSpec.overlays``).

Checks that need the PE-array dimensions or the tile-class ids live in
``kpu.py``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.datapath import AbsoluteEnergy
from embodied_schemas.process_node import CircuitClass

_OVERLAY_ID_RE = r"^[a-z0-9_]+$"


# ---------------------------------------------------------------------------
# PE-level (inside a tile)
# ---------------------------------------------------------------------------


class NeighborTopology(str, Enum):
    """Base PE-to-PE links of a spatial fabric."""

    NEAREST_NEIGHBOR_4 = "nearest_neighbor_4"  # N / S / E / W
    NEAREST_NEIGHBOR_8 = "nearest_neighbor_8"  # plus diagonals


class FabricOverlayKind(str, Enum):
    """Kind of a PE-level overlay."""

    ROW_BROADCAST = "row_broadcast"  # one source drives every PE in a row
    COL_BROADCAST = "col_broadcast"  # one source drives every PE in a column
    EXPRESS = "express"  # skip links of ``span`` PEs along a row / column
    REDUCTION_TREE = "reduction_tree"  # log-depth reduction across a row / column / tile
    TRANSPOSE = "transpose"  # row i <-> column i exchange (square arrays)
    BUTTERFLY = "butterfly"  # FFT-style exchange network (power-of-two axis)
    SEGMENTED_BUS = "segmented_bus"  # a bus split into segments of ``span`` PEs


class OverlayScope(str, Enum):
    """What one overlay instance spans."""

    ROW = "row"
    COL = "col"
    TILE = "tile"


# Kinds that take a ``span`` (in PEs), with the minimum span that makes sense.
_FABRIC_SPAN_MIN = {FabricOverlayKind.EXPRESS: 2, FabricOverlayKind.SEGMENTED_BUS: 2}


class FabricOverlay(BaseModel):
    """One PE-level overlay network inside a tile.

    ``instances`` is the number of copies per ``instances_per`` unit (e.g. 2
    express links per row). ``energy`` is the energy per *bit* moved across
    one instance, given at a reference node (``AbsoluteEnergy``).
    """

    overlay_id: str = Field(..., pattern=_OVERLAY_ID_RE)
    kind: FabricOverlayKind
    instances_per: OverlayScope
    instances: int = Field(1, ge=1)
    width_bits: int = Field(..., gt=0)
    span: int | None = Field(
        None, description="PEs spanned by one link / segment (express, segmented_bus)"
    )
    configuration: Literal["static_per_schedule"] = Field(
        "static_per_schedule",
        description="Configured per schedule; never a routed / table-driven network",
    )
    circuit_class: CircuitClass = CircuitClass.BALANCED_LOGIC
    mtx_per_instance: float | None = Field(None, ge=0, description="Transistors per instance (M)")
    energy: AbsoluteEnergy | None = Field(None, description="Energy per bit per traversal")
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_shape(self) -> "FabricOverlay":
        if self.kind in _FABRIC_SPAN_MIN:
            lo = _FABRIC_SPAN_MIN[self.kind]
            if self.span is None or self.span < lo:
                raise ValueError(
                    f"overlay {self.overlay_id!r}: {self.kind.value} needs span >= {lo}"
                )
        elif self.span is not None:
            raise ValueError(
                f"overlay {self.overlay_id!r}: span does not apply to {self.kind.value}"
            )
        if self.kind == FabricOverlayKind.ROW_BROADCAST and self.instances_per != OverlayScope.ROW:
            raise ValueError(f"overlay {self.overlay_id!r}: row_broadcast is per row")
        if self.kind == FabricOverlayKind.COL_BROADCAST and self.instances_per != OverlayScope.COL:
            raise ValueError(f"overlay {self.overlay_id!r}: col_broadcast is per col")
        if self.kind == FabricOverlayKind.TRANSPOSE and self.instances_per != OverlayScope.TILE:
            raise ValueError(f"overlay {self.overlay_id!r}: transpose spans the whole tile")
        return self


class FabricInterconnect(BaseModel):
    """PE-to-PE interconnect of a spatial tile: base links plus overlays."""

    base: NeighborTopology = NeighborTopology.NEAREST_NEIGHBOR_4
    link_bits: int = Field(..., gt=0, description="Width of a nearest-neighbor link")
    overlays: list[FabricOverlay] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _unique_overlay_ids(self) -> "FabricInterconnect":
        ids = [o.overlay_id for o in self.overlays]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate fabric overlay_id {dup}")
        return self


# ---------------------------------------------------------------------------
# Tile-level (chip NoC)
# ---------------------------------------------------------------------------


class NoCOverlayKind(str, Enum):
    """Kind of a tile-level overlay on the chip NoC."""

    EXPRESS_CHANNEL = "express_channel"  # skip links of ``span`` mesh hops
    STREAM_LINK = "stream_link"  # ordered producer -> consumer chain of tile classes
    MULTICAST_TREE = "multicast_tree"  # first endpoint feeds all the others


class NoCOverlay(BaseModel):
    """One overlay on the chip NoC.

    ``endpoints`` are tile_class_ids:

    - **stream_link:** the ordered chain producer -> ... -> consumer.
    - **multicast_tree:** the source first, then the receivers.
    - **express_channel:** optional (empty means the channel is general
      purpose).

    ``energy`` is the energy per *byte* per traversal at a reference node.
    """

    overlay_id: str = Field(..., pattern=_OVERLAY_ID_RE)
    kind: NoCOverlayKind
    endpoints: list[str] = Field(default_factory=list)
    width_bytes: int = Field(..., gt=0)
    instances: int = Field(1, ge=1, description="Parallel physical links / lanes")
    span: int | None = Field(None, description="Mesh hops skipped per link (express_channel)")
    configuration: Literal["static_per_schedule"] = "static_per_schedule"
    circuit_class: CircuitClass = CircuitClass.HP_LOGIC
    mtx_per_instance: float | None = Field(None, ge=0, description="Transistors per instance (M)")
    energy: AbsoluteEnergy | None = Field(None, description="Energy per byte per traversal")
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_shape(self) -> "NoCOverlay":
        oid, kind = self.overlay_id, self.kind
        if kind == NoCOverlayKind.EXPRESS_CHANNEL:
            if self.span is None or self.span < 2:
                raise ValueError(f"NoC overlay {oid!r}: express_channel needs span >= 2")
        elif self.span is not None:
            raise ValueError(f"NoC overlay {oid!r}: span does not apply to {kind.value}")
        if kind in (NoCOverlayKind.STREAM_LINK, NoCOverlayKind.MULTICAST_TREE):
            if len(self.endpoints) < 2:
                raise ValueError(f"NoC overlay {oid!r}: {kind.value} needs >= 2 endpoints")
        if len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError(f"NoC overlay {oid!r}: endpoints repeat a tile class")
        return self
