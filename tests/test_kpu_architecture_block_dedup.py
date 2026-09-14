"""KPUArchitecture / KPUBlock share one field set (graphs#268, Phase A1).

``KPUArchitecture`` (architect-facing topology) and ``KPUBlock`` (the
``Die.blocks`` member) used to declare the same five fields twice, and
consumers copied them field by field in five places. Both now subclass
``KPUArchitectureBase`` and convert with ``KPUBlock.from_architecture`` /
``KPUBlock.to_architecture``.

These tests pin:
  * the field-set contract (block = architecture + ``kind``),
  * that neither type is an instance of the other (explicit conversion only),
  * lossless round-trip conversion across the whole KPU catalog,
  * the serialized key order of both types (the catalog YAMLs and the graphs
    generator's YAML output lead with ``kind``),
  * discriminated-union dispatch and ``extra="forbid"`` still hold.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    BlockKind,
    ComputeProduct,
    KPUArchitecture,
    KPUArchitectureBase,
    KPUBlock,
    load_compute_products,
)

LEGACY_BLOCK_KEY_ORDER = [
    "kind", "total_tiles", "multi_precision_alu", "tiles", "noc", "memory",
]
LEGACY_ARCH_KEY_ORDER = [
    "total_tiles", "tiles", "noc", "memory", "multi_precision_alu",
]


def _kpu_blocks() -> dict[str, KPUBlock]:
    out = {}
    for sku_id, cp in load_compute_products().items():
        for die in cp.dies:
            for block in die.blocks:
                if isinstance(block, KPUBlock):
                    out[sku_id] = block
    return out


KPU_BLOCKS = _kpu_blocks()


def test_catalog_has_kpu_blocks():
    assert len(KPU_BLOCKS) >= 12


def test_block_fields_are_architecture_fields_plus_kind():
    base = set(KPUArchitectureBase.model_fields)
    assert set(KPUArchitecture.model_fields) == base
    assert set(KPUBlock.model_fields) == base | {"kind"}


def test_shared_field_definitions_are_identical():
    for name, info in KPUArchitectureBase.model_fields.items():
        for cls in (KPUArchitecture, KPUBlock):
            other = cls.model_fields[name]
            assert other.annotation == info.annotation, (cls.__name__, name)
            assert other.description == info.description, (cls.__name__, name)
            assert other.is_required() == info.is_required(), (cls.__name__, name)


def test_neither_type_is_an_instance_of_the_other():
    assert not issubclass(KPUBlock, KPUArchitecture)
    assert not issubclass(KPUArchitecture, KPUBlock)


@pytest.mark.parametrize("sku_id", sorted(KPU_BLOCKS))
def test_round_trip_block_architecture_block(sku_id):
    block = KPU_BLOCKS[sku_id]
    arch = block.to_architecture()
    assert type(arch) is KPUArchitecture
    assert arch.architecture_fields() == block.architecture_fields()
    back = KPUBlock.from_architecture(arch)
    assert type(back) is KPUBlock
    assert back == block
    assert back.kind == BlockKind.KPU


@pytest.mark.parametrize("sku_id", sorted(KPU_BLOCKS))
def test_serialized_key_order_is_preserved(sku_id):
    block = KPU_BLOCKS[sku_id]
    assert list(block.model_dump().keys()) == LEGACY_BLOCK_KEY_ORDER
    assert list(block.model_dump(mode="json").keys()) == LEGACY_BLOCK_KEY_ORDER
    assert list(block.to_architecture().model_dump().keys()) == LEGACY_ARCH_KEY_ORDER


def test_serializer_honours_dump_options():
    block = next(iter(KPU_BLOCKS.values()))
    assert list(block.model_dump(exclude={"noc"}).keys()) == [
        k for k in LEGACY_BLOCK_KEY_ORDER if k != "noc"
    ]
    assert list(block.model_dump(include={"memory", "kind"}).keys()) == ["kind", "memory"]
    # JSON dump round-trips through validation.
    assert KPUBlock.model_validate_json(block.model_dump_json()) == block


def test_discriminated_union_still_dispatches_to_kpu_block():
    cp = next(
        cp for cp in load_compute_products().values()
        if any(isinstance(b, KPUBlock) for d in cp.dies for b in d.blocks)
    )
    again = ComputeProduct.model_validate(cp.model_dump(mode="json"))
    assert any(isinstance(b, KPUBlock) for d in again.dies for b in d.blocks)
    assert again == cp


def test_extra_fields_still_forbidden():
    block = next(iter(KPU_BLOCKS.values()))
    data = block.model_dump()
    with pytest.raises(ValidationError):
        KPUBlock.model_validate({**data, "bogus": 1})
    arch_data = block.to_architecture().model_dump()
    with pytest.raises(ValidationError):
        KPUArchitecture.model_validate({**arch_data, "kind": "kpu"})
