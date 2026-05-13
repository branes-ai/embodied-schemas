"""YAML data loaders with Pydantic validation.

Provides utilities for loading and validating YAML data files
from the data catalog.
"""

import os
from pathlib import Path
from typing import TypeVar, Type
import yaml
from pydantic import BaseModel

from embodied_schemas.hardware import HardwareEntry, ChipEntry
from embodied_schemas.models import ModelEntry
from embodied_schemas.sensors import SensorEntry
from embodied_schemas.usecases import UseCaseEntry
from embodied_schemas.benchmarks import BenchmarkResult
from embodied_schemas.gpu import GPUEntry, GPUArchitectureSummary
from embodied_schemas.cpu import CPUEntry, CPUArchitectureSummary
from embodied_schemas.npu import NPUEntry
from embodied_schemas.operators import OperatorEntry
from embodied_schemas.architectures import SoftwareArchitecture
from embodied_schemas.mission import CapabilityTierEntry, MissionProfileEntry, BatteryEntry
from embodied_schemas.process_node import ProcessNodeEntry
from embodied_schemas.cooling_solution import CoolingSolutionEntry
from embodied_schemas.kpu import (
    KPUArchitecture,
    KPUDieSpec,
    KPUEntry,
    KPUMarket,
    KPUPowerSpec,
)
from embodied_schemas.compute_product import (
    ComputeProduct,
    KPUBlock,
    LifecycleStatus,
    PackagingKind,
)


T = TypeVar("T", bound=BaseModel)


def get_data_dir() -> Path:
    """Get the path to the data directory."""
    return Path(__file__).parent / "data"


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_validate(path: Path, model_class: Type[T]) -> T:
    """Load a YAML file and validate against a Pydantic model.

    Args:
        path: Path to the YAML file
        model_class: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the YAML is malformed
        pydantic.ValidationError: If validation fails
    """
    data = load_yaml(path)
    return model_class.model_validate(data)


def load_all_from_directory(
    directory: Path,
    model_class: Type[T],
    recursive: bool = True,
) -> dict[str, T]:
    """Load all YAML files from a directory.

    Args:
        directory: Directory to scan
        model_class: Pydantic model class to validate against
        recursive: Whether to scan subdirectories

    Returns:
        Dictionary mapping IDs to validated model instances
    """
    results: dict[str, T] = {}

    if not directory.exists():
        return results

    pattern = "**/*.yaml" if recursive else "*.yaml"
    for yaml_path in directory.glob(pattern):
        # Skip schema files
        if yaml_path.name.startswith("_"):
            continue

        try:
            entry = load_and_validate(yaml_path, model_class)
            results[entry.id] = entry
        except Exception as e:
            # Log warning but continue loading other files
            print(f"Warning: Failed to load {yaml_path}: {e}")

    return results


def load_hardware(data_dir: Path | None = None) -> dict[str, HardwareEntry]:
    """Load all hardware entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping hardware IDs to HardwareEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "hardware", HardwareEntry)


def load_chips(data_dir: Path | None = None) -> dict[str, ChipEntry]:
    """Load all chip/SoC entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping chip IDs to ChipEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "chips", ChipEntry)


def load_models(data_dir: Path | None = None) -> dict[str, ModelEntry]:
    """Load all model entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping model IDs to ModelEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "models", ModelEntry)


def load_sensors(data_dir: Path | None = None) -> dict[str, SensorEntry]:
    """Load all sensor entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping sensor IDs to SensorEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "sensors", SensorEntry)


def load_usecases(data_dir: Path | None = None) -> dict[str, UseCaseEntry]:
    """Load all use case entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping use case IDs to UseCaseEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "usecases", UseCaseEntry)


def load_benchmarks(data_dir: Path | None = None) -> dict[str, BenchmarkResult]:
    """Load all benchmark results from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping benchmark IDs to BenchmarkResult instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "benchmarks", BenchmarkResult)


def load_gpus(data_dir: Path | None = None) -> dict[str, GPUEntry]:
    """Load all GPU entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping GPU IDs to GPUEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "gpus", GPUEntry)


def load_gpu_architectures(data_dir: Path | None = None) -> dict[str, GPUArchitectureSummary]:
    """Load all GPU architecture summaries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping architecture IDs to GPUArchitectureSummary instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "gpu_architectures", GPUArchitectureSummary)


def load_cpus(data_dir: Path | None = None) -> dict[str, CPUEntry]:
    """Load all CPU entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping CPU IDs to CPUEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "cpus", CPUEntry)


def load_cpu_architectures(data_dir: Path | None = None) -> dict[str, CPUArchitectureSummary]:
    """Load all CPU architecture summaries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping architecture IDs to CPUArchitectureSummary instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "cpu_architectures", CPUArchitectureSummary)


def load_npus(data_dir: Path | None = None) -> dict[str, NPUEntry]:
    """Load all NPU/AI accelerator entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping NPU IDs to NPUEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "npus", NPUEntry)


_DATA_CONFIDENCE_RANK = {
    "calibrated": 3,
    "interpolated": 2,
    "theoretical": 1,
    "unknown": 0,
}


def load_process_nodes(data_dir: Path | None = None) -> dict[str, ProcessNodeEntry]:
    """Load all process-node entries from the catalog.

    Process nodes describe silicon fabrication: foundry, node name, transistor
    topology, per-library densities and energies. Used by the SKU generator
    and validator framework to do per-circuit-class area / power math.

    Resolution order (Phase 7 of the KPU SKU plan):

    1. **Public catalog** -- ``data_dir or get_data_dir()`` provides the
       baseline THEORETICAL entries shipped in this package
       (``data/process-nodes/<foundry>/<id>.yaml``).
    2. **Optional private overlay** -- if the environment variable
       ``PROCESS_NODE_DATA_DIR`` is set, additional ProcessNode YAMLs at
       that path are merged into the result. PDK-derived data is often
       confidential; the overlay lets CALIBRATED entries live outside
       this public package without forking the loader.
    3. **Confidence-based collision resolution** -- when the same id
       appears in both the public catalog and the overlay, the entry
       with HIGHER ``confidence`` wins (CALIBRATED > INTERPOLATED >
       THEORETICAL > UNKNOWN). A CALIBRATED PDK overlay always
       supersedes the public THEORETICAL estimate; a stale overlay
       does not silently downgrade calibrated public data.

    Args:
        data_dir: Optional path to data directory. Defaults to package data
            via ``get_data_dir()``. The ``PROCESS_NODE_DATA_DIR`` env var
            still applies on top.
    """
    base_dir = data_dir or get_data_dir()
    result = load_all_from_directory(
        base_dir / "process-nodes", ProcessNodeEntry
    )

    overlay_path = os.environ.get("PROCESS_NODE_DATA_DIR")
    if overlay_path:
        overlay_dir = Path(overlay_path)
        if overlay_dir.is_dir():
            overlay = load_all_from_directory(overlay_dir, ProcessNodeEntry)
            for node_id, overlay_entry in overlay.items():
                existing = result.get(node_id)
                if existing is None:
                    result[node_id] = overlay_entry
                    continue
                # Both present -- higher confidence wins.
                ov_rank = _DATA_CONFIDENCE_RANK.get(
                    overlay_entry.confidence.value, 0
                )
                ex_rank = _DATA_CONFIDENCE_RANK.get(
                    existing.confidence.value, 0
                )
                if ov_rank > ex_rank:
                    result[node_id] = overlay_entry
                # ex_rank >= ov_rank: keep existing
        else:
            print(
                f"Warning: PROCESS_NODE_DATA_DIR={overlay_path!r} is not "
                f"a directory; skipping overlay."
            )
    return result


def load_cooling_solutions(
    data_dir: Path | None = None,
) -> dict[str, CoolingSolutionEntry]:
    """Load all cooling-solution entries from the catalog.

    Cooling solutions describe thermal removal: type, max power density
    (W/mm^2), max total W, junction temperature ceiling. Peer of
    ProcessNode -- the thermal-hotspot validator and EM validator both
    consume cooling-solution data alongside process-node data.
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "cooling-solutions", CoolingSolutionEntry)


def _compute_product_to_kpu_entry(
    cp: ComputeProduct,
    process_node: ProcessNodeEntry,
) -> KPUEntry:
    """Reverse-adapt a v1 monolithic-KPU ComputeProduct to a KPUEntry.

    Used by the ``load_kpus()`` backward-compat shim. The legacy
    ``KPUDieSpec`` has redundant copies of foundry / process_name /
    process_nm that the unified schema offloaded to ProcessNode; the
    caller passes the resolved node so we can fill those in.

    Raises ValueError for shapes that KPUEntry can't represent
    (multi-die, non-KPUBlock, mismatched process node).
    """
    if len(cp.dies) != 1:
        raise ValueError(
            f"_compute_product_to_kpu_entry: KPUEntry can only represent "
            f"one die, got {len(cp.dies)} for {cp.id!r}"
        )
    die = cp.dies[0]
    if len(die.blocks) != 1:
        raise ValueError(
            f"_compute_product_to_kpu_entry: KPUEntry can only represent "
            f"one KPUBlock per die, got {len(die.blocks)} for {cp.id!r}"
        )
    block = die.blocks[0]
    if not isinstance(block, KPUBlock):
        raise ValueError(
            f"_compute_product_to_kpu_entry: only KPUBlock is supported, "
            f"got {type(block).__name__} for {cp.id!r}"
        )
    if process_node.id != die.process_node_id:
        raise ValueError(
            f"_compute_product_to_kpu_entry: process_node.id "
            f"{process_node.id!r} does not match die.process_node_id "
            f"{die.process_node_id!r} for {cp.id!r}"
        )

    return KPUEntry(
        id=cp.id,
        name=cp.name,
        vendor=cp.vendor,
        process_node_id=die.process_node_id,
        die=KPUDieSpec(
            architecture="KPU Tile",
            foundry=process_node.foundry,
            process_nm=process_node.node_nm,
            process_name=process_node.node_name,
            transistors_billion=die.transistors_billion,
            die_size_mm2=die.die_size_mm2,
            is_chiplet=cp.packaging.kind != PackagingKind.MONOLITHIC,
            num_dies=cp.packaging.num_dies,
        ),
        kpu_architecture=KPUArchitecture(
            total_tiles=block.total_tiles,
            multi_precision_alu=block.multi_precision_alu,
            tiles=block.tiles,
            noc=block.noc,
            memory=block.memory,
        ),
        silicon_bin=die.silicon_bin,
        clocks=die.clocks,
        performance=cp.performance,
        power=KPUPowerSpec(
            tdp_watts=cp.power.tdp_watts,
            max_power_watts=cp.power.max_power_watts,
            min_power_watts=cp.power.min_power_watts,
            idle_power_watts=cp.power.idle_power_watts,
            default_thermal_profile=cp.power.default_thermal_profile,
            thermal_profiles=cp.power.thermal_profiles,
        ),
        market=KPUMarket(
            launch_date=cp.market.launch_date,
            launch_msrp_usd=cp.market.launch_msrp_usd,
            target_market=cp.market.target_market,
            product_family=cp.market.product_family,
            model_tier=cp.market.model_tier,
            is_available=cp.market.is_available,
            is_discontinued=(cp.lifecycle == LifecycleStatus.EOL),
        ),
        notes=cp.notes,
        datasheet_url=cp.datasheet_url,
        last_updated=cp.last_updated,
    )


def load_kpus(data_dir: Path | None = None) -> dict[str, KPUEntry]:
    """Backward-compat shim: return every KPU SKU as a ``KPUEntry``.

    The legacy ``data/kpus/<vendor>/<id>.yaml`` catalog was retired in
    favor of the unified ``data/compute_products/<vendor>/<id>.yaml``
    catalog. This function now reads from the new catalog and
    reverse-adapts each ComputeProduct to a KPUEntry on the fly,
    preserving the old API for any straggler still consuming KPUEntry.

    Prefer ``load_compute_products()`` for new code -- it returns the
    canonical schema and avoids the per-call reverse-adapt cost.

    Fallback path: if ``data/compute_products/`` is empty or missing
    (e.g., a caller pinned to a pre-PR-#15 checkout that still ships
    the legacy ``data/kpus/`` catalog), the shim reads directly from
    ``data/kpus/<vendor>/<id>.yaml``. Same-shape KPUEntry instances
    either way.

    Returns an empty dict if neither catalog directory exists.
    """
    data_dir = data_dir or get_data_dir()
    cps = load_compute_products(data_dir=data_dir)
    if not cps:
        # Legacy fallback: caller may still have data/kpus/ populated.
        legacy_dir = data_dir / "kpus"
        if legacy_dir.is_dir():
            return load_all_from_directory(legacy_dir, KPUEntry)
        return {}
    process_nodes = load_process_nodes(data_dir=data_dir)
    out: dict[str, KPUEntry] = {}
    for sku_id, cp in cps.items():
        if not cp.dies:
            continue
        node = process_nodes.get(cp.dies[0].process_node_id)
        if node is None:
            # Skip rather than raise: an unresolvable process_node_id is
            # a catalog inconsistency the caller surfaces via validators,
            # not a load-time error.
            continue
        try:
            out[sku_id] = _compute_product_to_kpu_entry(cp, node)
        except ValueError:
            # Non-KPU compute products (e.g., future GPU blocks) can't
            # be represented as KPUEntry; silently skip them here.
            continue
    return out


def load_compute_products(
    data_dir: Path | None = None,
) -> dict[str, ComputeProduct]:
    """Load all ComputeProduct entries from the catalog.

    Reads YAMLs under ``data/compute_products/<vendor>/<id>.yaml`` and
    returns a dict keyed by id. The unified ComputeProduct schema (v1)
    covers KPU monolithic products today; future block kinds (GPU, CPU,
    NPU, DSP, memory dies) extend the discriminated ``blocks`` union as
    they're added.

    Sibling to ``load_kpus()`` during the parallel-migration phase: a
    SKU may exist in either ``data/kpus/`` (legacy KPUEntry) or
    ``data/compute_products/`` (new ComputeProduct). Once all SKUs are
    migrated, ``load_kpus()`` becomes a thin shim over this loader plus
    an adapter; the legacy directory is removed.

    Returns an empty dict if the ``data/compute_products/`` directory
    does not exist (graceful migration -- the catalog still loads even
    on a checkout that predates the directory).
    """
    data_dir = data_dir or get_data_dir()
    cp_dir = data_dir / "compute_products"
    if not cp_dir.is_dir():
        return {}
    return load_all_from_directory(cp_dir, ComputeProduct)


def load_operators(data_dir: Path | None = None) -> dict[str, OperatorEntry]:
    """Load all operator entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping operator IDs to OperatorEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "operators", OperatorEntry)


def load_architectures(data_dir: Path | None = None) -> dict[str, SoftwareArchitecture]:
    """Load all software architecture entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping architecture IDs to SoftwareArchitecture instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "architectures", SoftwareArchitecture)


def load_capability_tiers(data_dir: Path | None = None) -> dict[str, CapabilityTierEntry]:
    """Load all capability tier entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping tier IDs to CapabilityTierEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "capability-tiers", CapabilityTierEntry)


def load_mission_profiles(data_dir: Path | None = None) -> dict[str, MissionProfileEntry]:
    """Load all mission profile entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping profile IDs to MissionProfileEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "mission-profiles", MissionProfileEntry)


def load_batteries(data_dir: Path | None = None) -> dict[str, BatteryEntry]:
    """Load all battery configuration entries from the catalog.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        Dictionary mapping battery IDs to BatteryEntry instances
    """
    data_dir = data_dir or get_data_dir()
    return load_all_from_directory(data_dir / "batteries", BatteryEntry)


def validate_data_integrity(data_dir: Path | None = None) -> list[str]:
    """Validate all data files and return a list of errors.

    Args:
        data_dir: Optional path to data directory. Defaults to package data.

    Returns:
        List of error messages. Empty list means all data is valid.
    """
    data_dir = data_dir or get_data_dir()
    errors: list[str] = []

    # Define what to validate
    validations = [
        ("hardware", HardwareEntry),
        ("chips", ChipEntry),
        ("models", ModelEntry),
        ("sensors", SensorEntry),
        ("usecases", UseCaseEntry),
        ("gpus", GPUEntry),
        ("gpu_architectures", GPUArchitectureSummary),
        ("cpus", CPUEntry),
        ("operators", OperatorEntry),
        ("architectures", SoftwareArchitecture),
        ("capability-tiers", CapabilityTierEntry),
        ("mission-profiles", MissionProfileEntry),
        ("batteries", BatteryEntry),
        ("process-nodes", ProcessNodeEntry),
        ("cooling-solutions", CoolingSolutionEntry),
        ("kpus", KPUEntry),
        ("compute_products", ComputeProduct),
    ]

    for subdir, model_class in validations:
        directory = data_dir / subdir
        if not directory.exists():
            continue

        for yaml_path in directory.glob("**/*.yaml"):
            if yaml_path.name.startswith("_"):
                continue

            try:
                load_and_validate(yaml_path, model_class)
            except Exception as e:
                errors.append(f"{yaml_path}: {e}")

    return errors
