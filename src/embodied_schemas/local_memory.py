"""On-chip memories local to a compute unit (a KPU tile class, a function core).

Architecture-neutral. Used by ``kpu.KPUTileBase.local_memory`` and by
``function_core.FunctionCore.local_memory``. These types were introduced in
``kpu.py`` in Phase B1 (branes-ai/graphs#268) and moved here in Phase B3, so
that non-KPU blocks can use them too. They are still importable from
``embodied_schemas.kpu``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from embodied_schemas.process_node import CircuitClass


class LocalMemoryLevel(str, Enum):
    """Kind of a local memory."""

    L1 = "l1"
    L2 = "l2"
    LINE_BUFFER = "line_buffer"
    WEIGHT_BUFFER = "weight_buffer"
    ACCUMULATOR = "accumulator"
    STATE = "state"  # persistent working state of a function core (e.g. a VIO map)


class LocalMemoryScope(str, Enum):
    """Whether a ``LocalMemory.kib`` figure is per PE or per tile / core."""

    PE = "pe"  # kib is per PE
    TILE = "tile"  # kib is per tile (or per function core)


class LocalMemory(BaseModel):
    """One local memory (per PE, or per tile / core)."""

    level: LocalMemoryLevel
    kib: float = Field(..., gt=0)
    per: LocalMemoryScope = LocalMemoryScope.TILE
    circuit_class: CircuitClass = CircuitClass.SRAM_HD

    model_config = {"extra": "forbid"}


def duplicate_levels(memories: list[LocalMemory] | None) -> list[str]:
    """Levels declared more than once (sorted), for validators."""
    if not memories:
        return []
    levels = [m.level for m in memories]
    return sorted({lv.value for lv in levels if levels.count(lv) > 1})
