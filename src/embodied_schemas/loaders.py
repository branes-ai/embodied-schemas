"""YAML data loaders with Pydantic validation.

Provides utilities for loading and validating YAML data files
from the data catalog.
"""

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
