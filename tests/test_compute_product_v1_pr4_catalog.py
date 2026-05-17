"""Tests for PR #4: full ComputeProduct catalog (12 KPU SKUs).

Verifies that the ``data/compute_products/`` catalog now mirrors the
legacy ``data/kpus/`` catalog 1:1 -- every legacy KPU SKU has a
content-equivalent ComputeProduct YAML, and the loaders return matched
id sets.

PR #2 tested the single t256_16nm_tsmc_ffp SKU. This file tests the
catalog as a whole: all 12 SKUs load, every legacy SKU has a
ComputeProduct counterpart, and content is preserved field-by-field
for each one.

Once PR #4 lands, consumers can choose to drop ``load_kpus()`` entirely
in favor of ``load_compute_products()`` plus a downstream
``ComputeProduct -> KPUEntry`` adapter (or migrate to the unified
ComputeProduct interface directly). The legacy directory stays in place
for now as a parallel source of truth; future PRs delete it.
"""

from __future__ import annotations

import pytest

from embodied_schemas import (
    BlockKind,
    ComputeProduct,
    DieRole,
    LifecycleStatus,
    PackagingKind,
    ProductKind,
    load_compute_products,
    load_kpus,
)


EXPECTED_SKU_IDS = {
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
}


@pytest.fixture(scope="module")
def cps():
    return load_compute_products()


@pytest.fixture(scope="module")
def kpus():
    return load_kpus()


# ---------------------------------------------------------------------------
# Catalog completeness
# ---------------------------------------------------------------------------

def test_catalog_has_all_12_kpu_skus(cps):
    """Every expected KPU SKU has a ComputeProduct YAML. (Originally
    asserted "no extra SKUs", but v2 added a Jetson AGX Orin GPU SKU
    so the assertion now scopes to the KPU subset.)"""
    kpu_ids = {sku for sku, cp in cps.items() if cp.vendor == "stillwater"}
    missing = EXPECTED_SKU_IDS - kpu_ids
    extra = kpu_ids - EXPECTED_SKU_IDS
    assert not missing, f"missing from compute_products: {sorted(missing)}"
    assert not extra, f"unexpected stillwater SKUs in compute_products: {sorted(extra)}"


def test_compute_products_kpu_subset_matches_legacy_kpus(cps, kpus):
    """The KPU subset of compute_products has exactly the same SKU ids
    as the legacy kpus catalog (1:1 parity, PR #4's main proof, with
    the v2 catalog now scoped to the KPU subset since GPU SKUs joined)."""
    cps_kpu_ids = {sku for sku, cp in cps.items() if cp.vendor == "stillwater"}
    assert cps_kpu_ids == set(kpus.keys()), (
        f"id-set mismatch: "
        f"only in cps (KPU subset): {sorted(cps_kpu_ids - set(kpus))}, "
        f"only in kpus: {sorted(set(kpus) - cps_kpu_ids)}"
    )


def test_all_entries_are_compute_product_instances(cps):
    """Loader returns Pydantic instances, not dicts."""
    for sku_id, cp in cps.items():
        assert isinstance(cp, ComputeProduct), (
            f"cps[{sku_id!r}] is {type(cp).__name__}, not ComputeProduct"
        )


# ---------------------------------------------------------------------------
# Content equivalence (parametrized over all 12 SKUs)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", sorted(EXPECTED_SKU_IDS))
def test_compute_product_content_matches_legacy_kpu_entry(sku_id, cps, kpus):
    """For every SKU, the ComputeProduct YAML carries the same data as
    the legacy KPUEntry YAML. Field-by-field check across identity,
    per-die structure, KPUBlock content, roll-ups, market, and notes."""
    cp = cps[sku_id]
    kpu = kpus[sku_id]

    # Identity
    assert cp.id == kpu.id
    assert cp.name == kpu.name
    assert cp.vendor == kpu.vendor
    assert cp.kind == ProductKind.CHIP
    assert cp.packaging.kind == PackagingKind.MONOLITHIC
    assert cp.packaging.num_dies == kpu.die.num_dies

    # Per-die structure (chiplet caveat -- monolithic is one Die)
    assert len(cp.dies) == 1
    die = cp.dies[0]
    assert die.die_id == "kpu_compute"
    assert die.die_role == DieRole.COMPUTE
    assert die.process_node_id == kpu.process_node_id
    assert die.die_size_mm2 == kpu.die.die_size_mm2
    assert die.transistors_billion == kpu.die.transistors_billion
    assert die.silicon_bin == kpu.silicon_bin
    assert die.clocks == kpu.clocks
    assert die.interconnects == []  # monolithic: no inter-die links

    # KPUBlock content matches KPUArchitecture
    assert len(die.blocks) == 1
    block = die.blocks[0]
    assert block.kind == BlockKind.KPU
    assert block.total_tiles == kpu.kpu_architecture.total_tiles
    assert block.multi_precision_alu == kpu.kpu_architecture.multi_precision_alu
    assert block.tiles == kpu.kpu_architecture.tiles
    assert block.noc == kpu.kpu_architecture.noc
    assert block.memory == kpu.kpu_architecture.memory

    # Roll-ups
    assert cp.performance == kpu.performance
    assert cp.power.tdp_watts == kpu.power.tdp_watts
    assert cp.power.max_power_watts == kpu.power.max_power_watts
    assert cp.power.min_power_watts == kpu.power.min_power_watts
    assert cp.power.idle_power_watts == kpu.power.idle_power_watts
    assert cp.power.default_thermal_profile == kpu.power.default_thermal_profile
    assert cp.power.thermal_profiles == kpu.power.thermal_profiles

    # Market
    assert cp.market.target_market == kpu.market.target_market
    assert cp.market.product_family == kpu.market.product_family
    assert cp.market.model_tier == kpu.market.model_tier
    assert cp.market.is_available == kpu.market.is_available
    assert cp.market.launch_date == kpu.market.launch_date
    assert cp.market.launch_msrp_usd == kpu.market.launch_msrp_usd

    # Lifecycle derived from is_discontinued
    expected_lifecycle = (
        LifecycleStatus.EOL if kpu.market.is_discontinued
        else LifecycleStatus.PRODUCTION
    )
    assert cp.lifecycle == expected_lifecycle

    # Provenance
    assert cp.notes == kpu.notes
    assert cp.last_updated == kpu.last_updated


# ---------------------------------------------------------------------------
# Sweep-axis coverage (sanity-check that the catalog covers what PR #14
# established: tile-count x process-node combinations)
# ---------------------------------------------------------------------------

def test_tile_count_coverage(cps):
    """The KPU subset of the catalog covers tile counts {64, 128, 256, 512, 768}.
    GPU SKUs don't have tile_count -- scoping the assertion to KPUs."""
    tile_counts = {
        cp.dies[0].blocks[0].total_tiles
        for cp in cps.values()
        if cp.dies[0].blocks[0].kind == BlockKind.KPU
    }
    assert tile_counts == {64, 128, 256, 512, 768}, (
        f"unexpected tile_count set: {sorted(tile_counts)}"
    )


def test_process_node_coverage(cps):
    """The catalog covers the expected process nodes. KPU SKUs cover
    {tsmc_n16, gf_12fdx, tsmc_n7}; v2 added samsung_8lpp via Jetson
    AGX Orin. Asserts each KPU node is present and that the GPU one
    is too if any GPU SKUs exist."""
    nodes = {cp.dies[0].process_node_id for cp in cps.values()}
    expected_kpu_nodes = {"tsmc_n16", "gf_12fdx", "tsmc_n7"}
    assert expected_kpu_nodes.issubset(nodes), (
        f"missing expected KPU process nodes; got: {sorted(nodes)}"
    )
    if any(cp.vendor == "nvidia" for cp in cps.values()):
        assert "samsung_8lpp" in nodes


def test_model_tier_coverage(cps):
    """Every SKU has a model_tier from the agreed vocabulary. The
    ``research`` tier was added with Plasticine v2 (first academic
    prototype in the catalog; v5 CGRA sprint)."""
    expected_tiers = {"entry", "mid", "high", "enthusiast", "datacenter", "research"}
    tiers = {cp.market.model_tier for cp in cps.values()}
    assert tiers.issubset(expected_tiers), (
        f"unexpected model_tier(s): {tiers - expected_tiers}"
    )
