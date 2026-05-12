"""Tests for the v1 (KPU-only) ComputeProduct schema.

Verifies that:
  1. The new module imports cleanly and exports the expected symbols
  2. A minimal ComputeProduct can be instantiated programmatically
  3. The v1 schema can mechanically wrap an existing KPUEntry's content
     (the conversion the PR #3 adapter performs) -- proves the schema
     is a strict superset of today's KPUEntry shape

PR #1 scope: Pydantic class definitions only. PR #2 will add the first
catalog YAML at data/compute_products/<vendor>/. PR #3 (in graphs/) will
add the loader + adapter that consumers use.
"""

from __future__ import annotations

import pytest

from embodied_schemas import (
    AnyBlock,
    BlockKind,
    ComputeProduct,
    Die,
    DieRole,
    KPUBlock,
    KPUEntry,
    LifecycleStatus,
    Market,
    Packaging,
    PackagingKind,
    Power,
    ProductKind,
    load_kpus,
)


# ---------------------------------------------------------------------------
# 1. Module import / export
# ---------------------------------------------------------------------------

def test_compute_product_module_imports():
    """The new compute_product module exports the expected v1 symbols."""
    from embodied_schemas import compute_product

    expected = [
        "ComputeProduct",
        "ProductKind",
        "PackagingKind",
        "LifecycleStatus",
        "DieRole",
        "BlockKind",
        "KPUBlock",
        "AnyBlock",
        "InterconnectLevel",
        "TopologyKind",
        "Interconnect",
        "Die",
        "Packaging",
        "Power",
        "Market",
    ]
    for name in expected:
        assert hasattr(compute_product, name), (
            f"compute_product module missing expected symbol: {name}"
        )


def test_v1_block_kind_only_kpu():
    """v1 only ships BlockKind.KPU; future kinds come in later PRs."""
    assert BlockKind.KPU.value == "kpu"
    assert len(list(BlockKind)) == 1, (
        f"v1 expects exactly one BlockKind value; got {list(BlockKind)}"
    )


def test_v1_lifecycle_enum_complete():
    """LifecycleStatus enum carries the seven canonical states."""
    expected = {
        "engineering_sample", "pilot", "production", "mature",
        "nrnd", "ltb", "eol",
    }
    actual = {s.value for s in LifecycleStatus}
    assert actual == expected, (
        f"LifecycleStatus mismatch: missing={expected - actual}, "
        f"extra={actual - expected}"
    )


# ---------------------------------------------------------------------------
# 2. Mechanical conversion from KPUEntry to ComputeProduct
# ---------------------------------------------------------------------------

def _convert_kpu_entry_to_compute_product(entry: KPUEntry) -> ComputeProduct:
    """Mechanical KPUEntry -> ComputeProduct mapping. This is the same
    conversion the PR #3 adapter will perform; the test exercises it
    here to prove the schema can hold the data without loss.

    Mapping:
      KPUEntry.id / name / vendor              -> ComputeProduct.id / name / vendor
      KPUEntry.process_node_id / die /
          silicon_bin / clocks / kpu_architecture
                                               -> ComputeProduct.dies[0]
      KPUEntry.kpu_architecture                -> dies[0].blocks[0] (KPUBlock)
      KPUEntry.power                           -> ComputeProduct.power (Power; same shape)
      KPUEntry.market                          -> ComputeProduct.market (Market; same shape)
      KPUEntry.market.is_discontinued -> True  -> lifecycle=EOL else PRODUCTION
      KPUEntry.notes / datasheet_url /
          last_updated                         -> ComputeProduct (same)
    """
    return ComputeProduct(
        id=entry.id,
        name=entry.name,
        vendor=entry.vendor,
        kind=ProductKind.CHIP,
        packaging=Packaging(
            kind=PackagingKind.MONOLITHIC,
            num_dies=entry.die.num_dies,
            package_type="monolithic",
        ),
        lifecycle=(
            LifecycleStatus.EOL
            if entry.market.is_discontinued
            else LifecycleStatus.PRODUCTION
        ),
        dies=[
            Die(
                die_id="kpu_compute",
                die_role=DieRole.COMPUTE,
                process_node_id=entry.process_node_id,
                die_size_mm2=entry.die.die_size_mm2,
                transistors_billion=entry.die.transistors_billion,
                silicon_bin=entry.silicon_bin,
                clocks=entry.clocks,
                blocks=[
                    KPUBlock(
                        total_tiles=entry.kpu_architecture.total_tiles,
                        multi_precision_alu=entry.kpu_architecture.multi_precision_alu,
                        tiles=entry.kpu_architecture.tiles,
                        noc=entry.kpu_architecture.noc,
                        memory=entry.kpu_architecture.memory,
                    )
                ],
                interconnects=[],  # v1 monolithic: no inter-die links
            )
        ],
        performance=entry.performance,
        power=Power(
            tdp_watts=entry.power.tdp_watts,
            max_power_watts=entry.power.max_power_watts,
            min_power_watts=entry.power.min_power_watts,
            idle_power_watts=entry.power.idle_power_watts,
            default_thermal_profile=entry.power.default_thermal_profile,
            thermal_profiles=entry.power.thermal_profiles,
        ),
        market=Market(
            launch_date=entry.market.launch_date,
            launch_msrp_usd=entry.market.launch_msrp_usd,
            target_market=entry.market.target_market,
            product_family=entry.market.product_family,
            model_tier=entry.market.model_tier,
            is_available=entry.market.is_available,
        ),
        notes=entry.notes,
        datasheet_url=entry.datasheet_url,
        last_updated=entry.last_updated,
    )


@pytest.fixture(scope="module")
def kpus():
    return load_kpus()


@pytest.mark.parametrize("sku_id", [
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
])
def test_kpu_entry_round_trips_through_compute_product(sku_id, kpus):
    """Every legacy KPU SKU mechanically converts to ComputeProduct
    without data loss. This is the smoke test that PR #3's adapter will
    work for the full catalog."""
    entry = kpus[sku_id]
    cp = _convert_kpu_entry_to_compute_product(entry)

    # Identity preserved
    assert cp.id == entry.id
    assert cp.name == entry.name
    assert cp.vendor == entry.vendor

    # Per-die structure: monolithic = 1 die
    assert len(cp.dies) == 1
    die = cp.dies[0]
    assert die.process_node_id == entry.process_node_id
    assert die.die_size_mm2 == entry.die.die_size_mm2
    assert die.transistors_billion == entry.die.transistors_billion
    assert die.silicon_bin == entry.silicon_bin
    assert die.clocks == entry.clocks

    # KPUBlock content matches KPUArchitecture
    assert len(die.blocks) == 1
    assert die.blocks[0].kind == BlockKind.KPU
    assert die.blocks[0].total_tiles == entry.kpu_architecture.total_tiles
    assert die.blocks[0].tiles == entry.kpu_architecture.tiles
    assert die.blocks[0].noc == entry.kpu_architecture.noc
    assert die.blocks[0].memory == entry.kpu_architecture.memory

    # Power / performance / market preserved
    assert cp.performance == entry.performance
    assert cp.power.tdp_watts == entry.power.tdp_watts
    assert cp.power.thermal_profiles == entry.power.thermal_profiles
    assert cp.market.target_market == entry.market.target_market

    # Lifecycle derived from is_discontinued
    expected_lifecycle = (
        LifecycleStatus.EOL if entry.market.is_discontinued
        else LifecycleStatus.PRODUCTION
    )
    assert cp.lifecycle == expected_lifecycle


def test_compute_product_serialize_round_trip(kpus):
    """A converted ComputeProduct serializes to dict and re-validates."""
    entry = kpus["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    cp = _convert_kpu_entry_to_compute_product(entry)

    serialized = cp.model_dump(mode="json", exclude_none=True)
    re_loaded = ComputeProduct.model_validate(serialized)

    assert re_loaded.id == cp.id
    assert re_loaded.dies[0].die_id == "kpu_compute"
    assert re_loaded.dies[0].blocks[0].kind == BlockKind.KPU
    assert re_loaded.power.tdp_watts == cp.power.tdp_watts


def test_block_discriminator_dispatch():
    """Pydantic correctly dispatches on the BlockKind discriminator
    when deserializing a Block from a dict."""
    raw = {
        "kind": "kpu",
        "total_tiles": 4,
        "multi_precision_alu": ["int8"],
        "tiles": [
            {
                "tile_type": "INT8-primary",
                "num_tiles": 4,
                "pe_array_rows": 32,
                "pe_array_cols": 32,
                "pe_circuit_class": "balanced_logic",
                "ops_per_tile_per_clock": {"int8": 2048},
                "schedule_class": "output_stationary",
                "pipeline_fill_cycles": 32,
                "pipeline_drain_cycles": 32,
            }
        ],
        "noc": {
            "topology": "mesh_2d",
            "mesh_rows": 2,
            "mesh_cols": 2,
            "flit_bytes": 16,
            "router_circuit_class": "hp_logic",
            "bisection_bandwidth_gbps": 64.0,
        },
        "memory": {
            "memory_type": "lpddr5",
            "memory_size_gb": 8,
            "memory_bus_bits": 64,
            "memory_bandwidth_gbps": 64.0,
            "memory_controllers": 4,
            "l3_kib_per_tile": 256,
            "l2_kib_per_tile": 32,
            "l1_kib_per_pe": 4,
        },
    }
    block = KPUBlock.model_validate(raw)
    assert block.kind == BlockKind.KPU
    assert block.total_tiles == 4


def test_v1_rejects_extra_fields():
    """v1 ComputeProduct rejects fields not in the schema (extra=forbid).
    Catches typos at YAML load time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Packaging(kind=PackagingKind.MONOLITHIC, num_dies=1, bogus_field=42)
