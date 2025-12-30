"""Unified registry for accessing all catalog data.

Provides a single interface for querying hardware, models, sensors,
use cases, and benchmarks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

from embodied_schemas.hardware import HardwareEntry, ChipEntry
from embodied_schemas.models import ModelEntry
from embodied_schemas.sensors import SensorEntry
from embodied_schemas.usecases import UseCaseEntry
from embodied_schemas.benchmarks import BenchmarkResult
from embodied_schemas.gpu import GPUEntry, GPUArchitectureSummary
from embodied_schemas.cpu import CPUEntry
from embodied_schemas.loaders import (
    get_data_dir,
    load_hardware,
    load_chips,
    load_models,
    load_sensors,
    load_usecases,
    load_benchmarks,
    load_gpus,
    load_gpu_architectures,
    load_cpus,
)


@dataclass
class CatalogView:
    """A queryable view over a catalog of entries."""

    _entries: dict[str, Any] = field(default_factory=dict)

    def get(self, id: str) -> Any | None:
        """Get an entry by ID."""
        return self._entries.get(id)

    def __getitem__(self, id: str) -> Any:
        """Get an entry by ID, raising KeyError if not found."""
        return self._entries[id]

    def __contains__(self, id: str) -> bool:
        """Check if an entry exists."""
        return id in self._entries

    def __iter__(self):
        """Iterate over all entries."""
        return iter(self._entries.values())

    def __len__(self) -> int:
        """Get the number of entries."""
        return len(self._entries)

    def keys(self):
        """Get all entry IDs."""
        return self._entries.keys()

    def values(self):
        """Get all entries."""
        return self._entries.values()

    def items(self):
        """Get all (ID, entry) pairs."""
        return self._entries.items()

    def find(self, **kwargs) -> list[Any]:
        """Find entries matching the given criteria.

        Supports:
        - Exact match: find(vendor="NVIDIA")
        - List membership: find(suitable_for="edge")
        - Comparison with _min/_max suffix: find(power_watts_max=15)

        Args:
            **kwargs: Field name to value mappings

        Returns:
            List of matching entries
        """
        results = []

        for entry in self._entries.values():
            if self._matches(entry, kwargs):
                results.append(entry)

        return results

    def find_one(self, **kwargs) -> Any | None:
        """Find the first entry matching the given criteria."""
        matches = self.find(**kwargs)
        return matches[0] if matches else None

    def _matches(self, entry: Any, criteria: dict) -> bool:
        """Check if an entry matches all criteria."""
        for key, value in criteria.items():
            if not self._check_criterion(entry, key, value):
                return False
        return True

    def _check_criterion(self, entry: Any, key: str, value: Any) -> bool:
        """Check a single criterion against an entry."""
        # Handle _min and _max suffixes for comparisons
        if key.endswith("_min"):
            field_path = key[:-4]  # Remove _min
            field_value = self._get_nested_field(entry, field_path)
            if field_value is None:
                return False
            return field_value >= value

        if key.endswith("_max"):
            field_path = key[:-4]  # Remove _max
            field_value = self._get_nested_field(entry, field_path)
            if field_value is None:
                return False
            return field_value <= value

        # Get the field value
        field_value = self._get_nested_field(entry, key)

        if field_value is None:
            return False

        # List membership check
        if isinstance(field_value, list):
            return value in field_value

        # Exact match
        return field_value == value

    def _get_nested_field(self, obj: Any, path: str) -> Any:
        """Get a potentially nested field value using dot notation."""
        parts = path.split(".")
        current = obj

        for part in parts:
            if current is None:
                return None

            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current


@dataclass
class Registry:
    """Unified registry for all catalog data."""

    hardware: CatalogView = field(default_factory=CatalogView)
    chips: CatalogView = field(default_factory=CatalogView)
    models: CatalogView = field(default_factory=CatalogView)
    sensors: CatalogView = field(default_factory=CatalogView)
    usecases: CatalogView = field(default_factory=CatalogView)
    benchmarks: CatalogView = field(default_factory=CatalogView)
    gpus: CatalogView = field(default_factory=CatalogView)
    gpu_architectures: CatalogView = field(default_factory=CatalogView)
    cpus: CatalogView = field(default_factory=CatalogView)

    _data_dir: Path | None = None

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "Registry":
        """Load all catalog data from the data directory.

        Args:
            data_dir: Optional custom data directory. Defaults to package data.

        Returns:
            Populated Registry instance
        """
        data_dir = data_dir or get_data_dir()

        registry = cls(_data_dir=data_dir)
        registry.hardware = CatalogView(_entries=load_hardware(data_dir))
        registry.chips = CatalogView(_entries=load_chips(data_dir))
        registry.models = CatalogView(_entries=load_models(data_dir))
        registry.sensors = CatalogView(_entries=load_sensors(data_dir))
        registry.usecases = CatalogView(_entries=load_usecases(data_dir))
        registry.benchmarks = CatalogView(_entries=load_benchmarks(data_dir))
        registry.gpus = CatalogView(_entries=load_gpus(data_dir))
        registry.gpu_architectures = CatalogView(_entries=load_gpu_architectures(data_dir))
        registry.cpus = CatalogView(_entries=load_cpus(data_dir))

        return registry

    def reload(self) -> None:
        """Reload all data from disk."""
        data_dir = self._data_dir or get_data_dir()
        self.hardware = CatalogView(_entries=load_hardware(data_dir))
        self.chips = CatalogView(_entries=load_chips(data_dir))
        self.models = CatalogView(_entries=load_models(data_dir))
        self.sensors = CatalogView(_entries=load_sensors(data_dir))
        self.usecases = CatalogView(_entries=load_usecases(data_dir))
        self.benchmarks = CatalogView(_entries=load_benchmarks(data_dir))
        self.gpus = CatalogView(_entries=load_gpus(data_dir))
        self.gpu_architectures = CatalogView(_entries=load_gpu_architectures(data_dir))
        self.cpus = CatalogView(_entries=load_cpus(data_dir))

    def get_compatible_hardware(self, model_id: str) -> list[HardwareEntry]:
        """Get hardware compatible with a given model.

        Args:
            model_id: ID of the model

        Returns:
            List of compatible hardware entries
        """
        model = self.models.get(model_id)
        if not model:
            return []

        return [
            self.hardware[hw_id]
            for hw_id in model.compatible_hardware
            if hw_id in self.hardware
        ]

    def get_compatible_models(self, hardware_id: str) -> list[ModelEntry]:
        """Get models compatible with a given hardware.

        Args:
            hardware_id: ID of the hardware

        Returns:
            List of compatible model entries
        """
        return [
            model
            for model in self.models
            if hardware_id in model.compatible_hardware
        ]

    def get_benchmark(
        self,
        model_id: str,
        hardware_id: str,
        variant: str | None = None,
    ) -> BenchmarkResult | None:
        """Get a specific benchmark result.

        Args:
            model_id: ID of the model
            hardware_id: ID of the hardware
            variant: Optional specific variant (fp32, fp16, int8, etc.)

        Returns:
            Matching benchmark result or None
        """
        for benchmark in self.benchmarks:
            if benchmark.model_id != model_id:
                continue
            if benchmark.hardware_id != hardware_id:
                continue
            if variant is not None and benchmark.variant != variant:
                continue
            return benchmark

        return None

    def get_benchmarks_for_model(self, model_id: str) -> list[BenchmarkResult]:
        """Get all benchmarks for a model across all hardware.

        Args:
            model_id: ID of the model

        Returns:
            List of benchmark results
        """
        return [b for b in self.benchmarks if b.model_id == model_id]

    def get_benchmarks_for_hardware(self, hardware_id: str) -> list[BenchmarkResult]:
        """Get all benchmarks for a hardware platform across all models.

        Args:
            hardware_id: ID of the hardware

        Returns:
            List of benchmark results
        """
        return [b for b in self.benchmarks if b.hardware_id == hardware_id]

    def is_compatible(
        self,
        model_id: str,
        hardware_id: str,
        variant: str | None = None,
    ) -> bool:
        """Check if a model is compatible with a hardware platform.

        Args:
            model_id: ID of the model
            hardware_id: ID of the hardware
            variant: Optional specific variant

        Returns:
            True if compatible, False otherwise
        """
        model = self.models.get(model_id)
        if not model:
            return False

        # Check explicit compatibility list
        if hardware_id in model.compatible_hardware:
            return True

        # Check if there's a benchmark (implies compatibility)
        if self.get_benchmark(model_id, hardware_id, variant):
            return True

        return False

    def get_usecase_constraints(self, usecase_id: str) -> dict | None:
        """Get constraints for a use case.

        Args:
            usecase_id: ID of the use case

        Returns:
            Dictionary of constraints or None
        """
        usecase = self.usecases.get(usecase_id)
        if not usecase:
            return None

        return {
            name: {
                "max": c.max_value,
                "min": c.min_value,
                "target": c.target_value,
                "tier": c.tier,
                "criticality": c.criticality.value,
            }
            for name, c in usecase.constraints.items()
        }

    def summary(self) -> dict[str, int]:
        """Get a summary of the registry contents.

        Returns:
            Dictionary with counts of each entity type
        """
        return {
            "hardware": len(self.hardware),
            "chips": len(self.chips),
            "models": len(self.models),
            "sensors": len(self.sensors),
            "usecases": len(self.usecases),
            "benchmarks": len(self.benchmarks),
            "gpus": len(self.gpus),
            "gpu_architectures": len(self.gpu_architectures),
            "cpus": len(self.cpus),
        }

    def get_gpus_by_vendor(self, vendor: str) -> list[GPUEntry]:
        """Get all GPUs from a specific vendor.

        Args:
            vendor: Vendor name (nvidia, amd, intel, etc.)

        Returns:
            List of GPUEntry instances from that vendor
        """
        return self.gpus.find(vendor=vendor)

    def get_gpus_by_architecture(self, architecture: str) -> list[GPUEntry]:
        """Get all GPUs using a specific architecture.

        Args:
            architecture: Architecture name (e.g., 'Ada Lovelace', 'RDNA 3')

        Returns:
            List of GPUEntry instances using that architecture
        """
        return [gpu for gpu in self.gpus if gpu.die.architecture == architecture]

    def get_gpus_by_market(self, target_market: str) -> list[GPUEntry]:
        """Get all GPUs targeting a specific market.

        Args:
            target_market: Target market (consumer_desktop, datacenter, etc.)

        Returns:
            List of GPUEntry instances for that market
        """
        return [gpu for gpu in self.gpus if gpu.market.target_market.value == target_market]

    def get_gpu_for_hardware(self, hardware_id: str) -> GPUEntry | None:
        """Get the GPU entry for a hardware platform with an integrated GPU.

        Args:
            hardware_id: ID of the hardware entry (e.g., nvidia_jetson_orin_nano)

        Returns:
            GPUEntry if the hardware has a linked GPU, None otherwise
        """
        hardware = self.hardware.get(hardware_id)
        if not hardware or not hardware.gpu_id:
            return None
        return self.gpus.get(hardware.gpu_id)

    def get_hardware_with_gpu(self, gpu_id: str) -> list[HardwareEntry]:
        """Get all hardware platforms that use a specific GPU.

        Args:
            gpu_id: ID of the GPU entry

        Returns:
            List of HardwareEntry instances that embed this GPU
        """
        gpu = self.gpus.get(gpu_id)
        if not gpu:
            return []

        # Check both directions of the relationship
        results = []

        # From GPU's embedded_in_hardware_ids list
        for hw_id in gpu.embedded_in_hardware_ids:
            hw = self.hardware.get(hw_id)
            if hw:
                results.append(hw)

        # Also check hardware entries that reference this GPU
        for hw in self.hardware:
            if hw.gpu_id == gpu_id and hw not in results:
                results.append(hw)

        return results

    def get_cpus_by_vendor(self, vendor: str) -> list[CPUEntry]:
        """Get all CPUs from a specific vendor.

        Args:
            vendor: Vendor name (intel, amd, arm, etc.)

        Returns:
            List of CPUEntry instances from that vendor
        """
        return self.cpus.find(vendor=vendor)

    def get_cpus_by_architecture(self, architecture: str) -> list[CPUEntry]:
        """Get all CPUs using a specific microarchitecture.

        Args:
            architecture: Architecture name (e.g., 'sapphire_rapids', 'zen4')

        Returns:
            List of CPUEntry instances using that architecture
        """
        return [cpu for cpu in self.cpus if cpu.architecture.value == architecture]

    def get_cpus_by_market(self, target_market: str) -> list[CPUEntry]:
        """Get all CPUs targeting a specific market.

        Args:
            target_market: Target market (datacenter, workstation, desktop_mainstream, etc.)

        Returns:
            List of CPUEntry instances for that market
        """
        return [cpu for cpu in self.cpus if cpu.market.target_market.value == target_market]

    def get_cpus_by_socket(self, socket: str) -> list[CPUEntry]:
        """Get all CPUs for a specific socket.

        Args:
            socket: Socket type (lga4677, sp5, am5, etc.)

        Returns:
            List of CPUEntry instances for that socket
        """
        return [cpu for cpu in self.cpus
                if cpu.platform and cpu.platform.socket.value == socket]
