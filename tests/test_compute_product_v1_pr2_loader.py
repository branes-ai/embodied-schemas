"""Tests for PR #2: first ComputeProduct YAML + loader.

Verifies:
  1. ``load_compute_products()`` finds and parses the new YAML at
     ``data/compute_products/stillwater/kpu_t256_..._16nm_tsmc_ffp.yaml``
  2. The loaded ``ComputeProduct`` is content-equivalent to the legacy
     ``KPUEntry`` for the same SKU id (proves no data was lost in the
     conversion -- the YAML at the new path holds the same information
     as the YAML at the legacy path)
  3. ``load_compute_products()`` is graceful when the directory is empty
     or missing (returns ``{}``)

This is the additive proof that PR #2 doesn't break anything: both
catalogs coexist, the same SKU appears in both, and the content matches.
"""

from __future__ import annotations

import pytest

from embodied_schemas import (
    BlockKind,
    DieRole,
    LifecycleStatus,
    PackagingKind,
    ProductKind,
    load_compute_products,
    load_kpus,
)


T256_ID = "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"


@pytest.fixture(scope="module")
def cps():
    return load_compute_products()


@pytest.fixture(scope="module")
def kpus():
    return load_kpus()


# ---------------------------------------------------------------------------
# 1. Loader finds + parses the new YAML
# ---------------------------------------------------------------------------

def test_loader_finds_t256(cps):
    """The new YAML is discovered and parses cleanly."""
    assert T256_ID in cps, (
        f"{T256_ID} not in compute_products catalog. "
        f"Available: {sorted(cps)}"
    )


def test_loader_returns_compute_product_instance(cps):
    """Loader returns ComputeProduct (Pydantic) instances, not dicts."""
    cp = cps[T256_ID]
    # Spot-check the v1 spine fields
    assert cp.id == T256_ID
    assert cp.vendor == "stillwater"
    assert cp.kind == ProductKind.CHIP
    assert cp.packaging.kind == PackagingKind.MONOLITHIC
    assert cp.packaging.num_dies == 1


def test_loader_die_structure(cps):
    """Per-die structure (chiplet caveat) -- monolithic is one Die."""
    cp = cps[T256_ID]
    assert len(cp.dies) == 1

    die = cp.dies[0]
    assert die.die_id == "kpu_compute"
    assert die.die_role == DieRole.COMPUTE
    assert die.process_node_id == "tsmc_n16"
    assert die.die_size_mm2 > 0
    assert die.transistors_billion > 0
    assert len(die.blocks) == 1
    assert die.blocks[0].kind == BlockKind.KPU
    # Monolithic v1: no inter-die interconnects
    assert die.interconnects == []


# ---------------------------------------------------------------------------
# 2. Content-equivalence with legacy KPUEntry for the same SKU
# ---------------------------------------------------------------------------

def test_content_equivalent_to_legacy_kpu_entry(cps, kpus):
    """The new ComputeProduct YAML carries the same data as the legacy
    KPUEntry YAML for the same SKU id. This is the additive proof that
    the migration doesn't lose information."""
    cp = cps[T256_ID]
    kpu = kpus[T256_ID]

    # Identity
    assert cp.id == kpu.id
    assert cp.name == kpu.name
    assert cp.vendor == kpu.vendor

    # Per-die fields map to KPUEntry's chip-level fields (monolithic case)
    die = cp.dies[0]
    assert die.process_node_id == kpu.process_node_id
    assert die.die_size_mm2 == kpu.die.die_size_mm2
    assert die.transistors_billion == kpu.die.transistors_billion
    assert die.silicon_bin == kpu.silicon_bin
    assert die.clocks == kpu.clocks

    # KPUBlock content matches KPUArchitecture
    block = die.blocks[0]
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
# 3. Loader is graceful on missing / empty directory
# ---------------------------------------------------------------------------

def test_loader_graceful_on_missing_directory(tmp_path):
    """Loader returns {} when data/compute_products/ does not exist.
    Lets the catalog load on checkouts that predate the directory."""
    # tmp_path has no compute_products/ subdirectory
    result = load_compute_products(data_dir=tmp_path)
    assert result == {}


def test_loader_graceful_on_empty_directory(tmp_path):
    """Loader returns {} when data/compute_products/ exists but is empty."""
    (tmp_path / "compute_products").mkdir()
    result = load_compute_products(data_dir=tmp_path)
    assert result == {}
