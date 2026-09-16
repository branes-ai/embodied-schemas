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

#: The twelve uniform KPU SKUs that predate the heterogeneous work. Frozen
#: on purpose: adding a SKU here is a decision, not a side effect.
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


def legacy_kpu_blocks() -> dict[str, KPUBlock]:
    """The blocks of ``LEGACY_KPU_SKU_IDS`` only."""
    blocks = kpu_blocks()
    return {sku: blocks[sku] for sku in LEGACY_KPU_SKU_IDS}


def test_the_two_id_lists_cover_the_catalog_and_do_not_overlap():
    """A new KPU SKU must be classified deliberately, not fall through.

    If this fails, someone added a SKU without deciding whether the legacy
    backward-compatibility contract applies to it.
    """
    catalog = set(kpu_blocks())
    legacy, hetero = set(LEGACY_KPU_SKU_IDS), set(HETEROGENEOUS_KPU_SKU_IDS)
    assert not legacy & hetero
    assert legacy | hetero == catalog, (
        f"unclassified KPU SKUs: {sorted(catalog - legacy - hetero)}; "
        f"missing from the catalog: {sorted((legacy | hetero) - catalog)}"
    )
