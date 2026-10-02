"""Keep additive schema changes invisible in serialized output.

A new optional field defaults to ``None`` (or ``[]``), and Pydantic would
still emit it from ``model_dump()``. Downstream consumers compare dumps
against golden snapshots (graphs' KPU goldens, for one), so every additive
field would show up there as a diff even on entries that never set it.

``omit_if_default`` drops the listed fields from a dump while they hold
their default. Entries that do not use a new field serialize exactly as
before the field existed, and loading the dump back restores the default.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, SerializerFunctionWrapHandler


def omit_if_default(
    model: BaseModel, handler: SerializerFunctionWrapHandler, fields: Iterable[str]
) -> Any:
    """Serialize ``model`` with ``handler``, leaving out each of ``fields``
    whose value equals its declared default.

    Use from a ``@model_serializer(mode="wrap")`` method. Leave that method
    unannotated: an annotated return type replaces the model's
    serialization-mode JSON schema.
    """
    data = handler(model)
    if not isinstance(data, dict):
        return data
    declared = type(model).model_fields
    for name in fields:
        if name in data and getattr(model, name) == declared[name].get_default(
            call_default_factory=True
        ):
            del data[name]
    return data
