"""One-shot generator: legacy KPUEntry YAML -> ComputeProduct YAML.

Used for PR #4 of the v1 ComputeProduct migration POC. Reads every
SKU in ``src/embodied_schemas/data/kpus/<vendor>/`` via ``load_kpus()``,
mechanically converts it to a ``ComputeProduct``, and writes the result
to ``src/embodied_schemas/data/compute_products/<vendor>/<id>.yaml``.

Idempotent: re-running regenerates each YAML from the legacy source of
truth. The conversion logic is identical to the
``kpu_entry_to_compute_product`` adapter in
``graphs/src/graphs/hardware/compute_product_loader.py`` (PR #3), kept
duplicated here so embodied-schemas does not gain a graphs dependency.

Run:
    python scripts/generate_compute_product_yamls.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

from embodied_schemas import (
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


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "src" / "embodied_schemas" / "data"
OUT_DIR = DATA_ROOT / "compute_products"


def kpu_entry_to_compute_product(entry: KPUEntry) -> ComputeProduct:
    """Mechanical KPUEntry -> ComputeProduct adapter.

    Monolithic KPU products (the only KPU shape today) collapse to a
    single ``Die`` with ``die_id="kpu_compute"`` and ``die_role=COMPUTE``.
    Intra-die NoC stays in ``KPUBlock.noc`` per the v1 schema, so
    ``Die.interconnects`` is empty. Lifecycle derives from
    ``KPUEntry.market.is_discontinued``.
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
                interconnects=[],
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


def write_yaml(cp: ComputeProduct, out_path: Path) -> None:
    """Dump a ComputeProduct to YAML using Pydantic's json-mode dump
    so enum values serialize as their string values (matching the
    hand-curated t256 YAML in PR #16)."""
    data = cp.model_dump(mode="json", exclude_none=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
            width=88,
        )


def main() -> None:
    kpus = load_kpus()
    print(f"Found {len(kpus)} legacy KPU SKUs")

    written = 0
    skipped = 0
    for sku_id, entry in sorted(kpus.items()):
        cp = kpu_entry_to_compute_product(entry)
        out_path = OUT_DIR / cp.vendor / f"{cp.id}.yaml"
        write_yaml(cp, out_path)
        rel = out_path.relative_to(REPO_ROOT)
        if out_path.exists():
            print(f"  wrote {rel}")
            written += 1
        else:
            print(f"  skipped {rel}")
            skipped += 1

    print(f"\nDone: {written} written, {skipped} skipped")


if __name__ == "__main__":
    main()
