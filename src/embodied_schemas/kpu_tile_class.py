"""KPU tile-class library entries (branes-ai/graphs#268 Phase B6).

A tile class is a reusable tile definition (a PE datapath and interconnect,
a systolic array, or a fixed-function core) that SKUs instantiate. Entries
live in ``data/kpu-tile-classes/<id>.yaml`` and load with
``loaders.load_kpu_tile_classes()``. Stillwater-internal, RTL-characterized
entries go in a private overlay directory named by ``KPU_TILE_DATA_DIR``,
following the ``PROCESS_NODE_DATA_DIR`` precedent.

**SKUs are self-contained.** A SKU stores the fully resolved tile, with
``tile_class_ref`` naming the library entry it came from, so validators and
estimators never need the library. ``KPUTileClassEntry.instantiate`` makes
that resolved copy.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from embodied_schemas.kpu import _TILE_CLASS_ID_RE, AnyKPUTile, KPUTileKind
from embodied_schemas.process_node import DataConfidence

_TILE = TypeAdapter(AnyKPUTile)


class KPUTileClassEntry(BaseModel):
    """One tile class in the library.

    ``tile`` is the template of one tile. Its ``tile_class_id`` is the entry
    ``id``, and its ``num_tiles`` is 1: both are filled in when omitted. SKU
    fields such as ``power_domain_id`` and ``placement`` are normally left
    unset and supplied when the class is instantiated.
    """

    id: str = Field(..., pattern=_TILE_CLASS_ID_RE)
    name: str = Field(..., description="Human-readable name")
    description: str = ""
    tile: AnyKPUTile
    ref_node_id: str | None = Field(
        None,
        description="ProcessNodeEntry id the entry's absolute figures (energy, "
        "area) were taken at; None when every figure is node-relative",
    )
    confidence: DataConfidence = DataConfidence.THEORETICAL
    sources: list[str] = Field(..., min_length=1, description="Citations for the figures")
    last_updated: str = Field(..., description="YYYY-MM-DD")
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _fill_template_identity(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("tile"), dict):
            tile = dict(data["tile"])
            tile.setdefault("num_tiles", 1)
            if "id" in data:
                tile.setdefault("tile_class_id", data["id"])
            data = {**data, "tile": tile}
        return data

    @model_validator(mode="after")
    def _check_template(self) -> "KPUTileClassEntry":
        if any(not s.strip() for s in self.sources):
            raise ValueError(f"tile class {self.id!r}: sources must not contain blank citations")
        t = self.tile
        if t.tile_class_id != self.id:
            raise ValueError(
                f"tile class {self.id!r}: tile.tile_class_id is {t.tile_class_id!r}; "
                f"a library template's id is the entry id"
            )
        if t.num_tiles != 1:
            raise ValueError(
                f"tile class {self.id!r}: a library template describes one tile "
                f"(num_tiles 1), got {t.num_tiles}; set the count when instantiating"
            )
        if t.tile_class_ref is not None:
            raise ValueError(
                f"tile class {self.id!r}: a library template is itself the "
                f"reference; tile_class_ref must be unset"
            )
        return self

    @property
    def tile_kind(self) -> KPUTileKind:
        return self.tile.tile_kind

    def instantiate(self, num_tiles: int, **overrides: Any) -> AnyKPUTile:
        """A resolved SKU tile of this class.

        ``overrides`` replace top-level tile fields (for example
        ``power_domain_id``, ``placement``, ``tile_type``, or a different
        ``tile_class_id`` when a SKU uses the class twice). The result carries
        ``tile_class_ref = self.id`` and is fully validated.
        """
        data = self.tile.model_dump(mode="json", exclude_none=True)
        data.update(overrides)
        data["num_tiles"] = num_tiles
        data["tile_class_ref"] = self.id
        return _TILE.validate_python(data)
