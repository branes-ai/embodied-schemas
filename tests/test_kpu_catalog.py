"""The KPU SKUs that shipped before the heterogeneous-tile work, and the
backward-compatibility contract attached to them.

Phases B1-B6 (branes-ai/graphs#268) added schema capability -- tile kinds,
datapath descriptions, NoC overlays, checkerboards, power domains, a
by-kind performance roll-up -- while changing no catalog *data*. A family
of tests pins that: every catalog KPU tile is a ``KPUTileSpec`` with the
legacy key order, no overlays, no checkerboard, and a performance
roll-up derivable the legacy way.

Those tests were parametrized over "every KPU SKU in the catalog", which
was the same set only while the catalog held nothing else. It now also
holds ``kpu_h64_auto1``, the heterogeneous reference design, which is
supposed to have overlays and a checkerboard -- so parametrizing over the
whole catalog would either fail or, worse, have to be weakened until it
proved nothing about the SKUs it was written for.

``LEGACY_KPU_SKU_IDS`` is therefore explicit rather than derived. Deriving
it (say, "every SKU with one tile kind") would make the contract vacuous:
a legacy SKU that accidentally grew a systolic tile would silently drop
out of its own regression test instead of failing it.

Named ``test_*`` so pytest collects the guard at the bottom: ``python_files
= ["test_*.py"]`` means a plain ``kpu_catalog.py`` would be importable but
never run, which is exactly the kind of guard that looks present and is
not.
"""

from __future__ import annotations

from embodied_schemas import load_compute_products
from embodied_schemas.compute_product import KPUBlock
from embodied_schemas.kpu import KPUTileKind

#: The uniform KPU SKUs that predate the heterogeneous work and keep their
#: original shape. Frozen on purpose: adding a SKU here is a decision, not a
#: side effect.
LEGACY_KPU_SKU_IDS = (
    "kpu_t64_32x32_lp5x4_12nm_gf_fdx",
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t64_32x32_lp5x4_7nm_tsmc_hpc",
    "kpu_t128_32x32_lp5x8_12nm_gf_fdx",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_7nm_tsmc_hpc",
    "kpu_t256_32x32_lp5x16_12nm_gf_fdx",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_7nm_tsmc_hpc",
    "kpu_t512_32x32_lp5x32_12nm_gf_fdx",
    "kpu_t512_32x32_lp5x32_7nm_tsmc_hpc",
)

#: SKUs that predate the heterogeneous work and were then deliberately moved
#: onto a new tile kind: the T768, whose Matrix class became ``systolic``
#: (graphs#268 D8). They left the legacy contract on purpose -- its tiles are
#: no longer all ``KPUTileSpec`` -- so they get their own list rather than
#: silently dropping out of it.
MIGRATED_KPU_SKU_IDS = (
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
)

#: The heterogeneous reference design (graphs#268 E1), at both its nodes.
HETEROGENEOUS_KPU_SKU_IDS = (
    "kpu_h64_auto1_lp5x4_16nm_tsmc_ffp",
    "kpu_h64_auto1_lp5x4_7nm_tsmc_hpc",
)


def kpu_blocks() -> dict[str, KPUBlock]:
    """Every KPU block in the catalog, keyed by SKU id."""
    return {
        sku: block
        for sku, cp in load_compute_products().items()
        for die in cp.dies
        for block in die.blocks
        if isinstance(block, KPUBlock)
    }


#: Every SKU that shipped before the heterogeneous work, migrated or not: the
#: scope of the contracts that survive a tile-kind migration (block round
#: trip, no overlays, the implicit mesh and its cluster partition, the
#: legacy performance figures).
SHIPPED_KPU_SKU_IDS = LEGACY_KPU_SKU_IDS + MIGRATED_KPU_SKU_IDS


def legacy_kpu_blocks() -> dict[str, KPUBlock]:
    """The blocks of ``LEGACY_KPU_SKU_IDS`` only."""
    blocks = kpu_blocks()
    return {sku: blocks[sku] for sku in LEGACY_KPU_SKU_IDS}


def shipped_kpu_blocks() -> dict[str, KPUBlock]:
    """The blocks of ``SHIPPED_KPU_SKU_IDS``."""
    blocks = kpu_blocks()
    return {sku: blocks[sku] for sku in SHIPPED_KPU_SKU_IDS}


def test_the_id_lists_cover_the_catalog_and_do_not_overlap():
    """A new KPU SKU must be classified deliberately, not fall through.

    If this fails, someone added a SKU without deciding whether the legacy
    backward-compatibility contract applies to it.
    """
    catalog = set(kpu_blocks())
    lists = (
        set(LEGACY_KPU_SKU_IDS), set(MIGRATED_KPU_SKU_IDS), set(HETEROGENEOUS_KPU_SKU_IDS)
    )
    assert sum(len(ids) for ids in lists) == len(set().union(*lists)), "a SKU is in two lists"
    classified = set().union(*lists)
    assert classified == catalog, (
        f"unclassified KPU SKUs: {sorted(catalog - classified)}; "
        f"missing from the catalog: {sorted(classified - catalog)}"
    )


def test_the_lists_say_what_they_claim():
    """The classification is not just a label. Without this the lists could
    drift from the data and every contract scoped to them would quietly stop
    testing what it says."""
    blocks = kpu_blocks()

    def kinds(sku):
        return {t.tile_kind for t in blocks[sku].tiles}

    for sku in LEGACY_KPU_SKU_IDS:
        assert kinds(sku) == {KPUTileKind.PE_FABRIC}, f"{sku} is no longer uniform"
        assert blocks[sku].checkerboard is None
    for sku in MIGRATED_KPU_SKU_IDS:
        # Still the implicit mesh it shipped with, now with another kind.
        assert len(kinds(sku)) > 1, f"{sku} is not migrated: {kinds(sku)}"
        assert blocks[sku].checkerboard is None
    for sku in HETEROGENEOUS_KPU_SKU_IDS:
        assert len(kinds(sku)) > 1, f"{sku} is no longer heterogeneous"
        assert blocks[sku].checkerboard is not None
