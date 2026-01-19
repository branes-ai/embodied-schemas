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
from embodied_schemas.cpu import CPUEntry, CPUArchitectureSummary
from embodied_schemas.npu import NPUEntry
from embodied_schemas.operators import OperatorEntry
from embodied_schemas.architectures import SoftwareArchitecture
from embodied_schemas.mission import (
    CapabilityTierEntry,
    MissionProfileEntry,
    BatteryEntry,
    CapabilityTierName,
)
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
    load_cpu_architectures,
    load_npus,
    load_operators,
    load_architectures,
    load_capability_tiers,
    load_mission_profiles,
    load_batteries,
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
    cpu_architectures: CatalogView = field(default_factory=CatalogView)
    npus: CatalogView = field(default_factory=CatalogView)
    operators: CatalogView = field(default_factory=CatalogView)
    architectures: CatalogView = field(default_factory=CatalogView)
    capability_tiers: CatalogView = field(default_factory=CatalogView)
    mission_profiles: CatalogView = field(default_factory=CatalogView)
    batteries: CatalogView = field(default_factory=CatalogView)

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
        registry.cpu_architectures = CatalogView(_entries=load_cpu_architectures(data_dir))
        registry.npus = CatalogView(_entries=load_npus(data_dir))
        registry.operators = CatalogView(_entries=load_operators(data_dir))
        registry.architectures = CatalogView(_entries=load_architectures(data_dir))
        registry.capability_tiers = CatalogView(_entries=load_capability_tiers(data_dir))
        registry.mission_profiles = CatalogView(_entries=load_mission_profiles(data_dir))
        registry.batteries = CatalogView(_entries=load_batteries(data_dir))

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
        self.cpu_architectures = CatalogView(_entries=load_cpu_architectures(data_dir))
        self.npus = CatalogView(_entries=load_npus(data_dir))
        self.operators = CatalogView(_entries=load_operators(data_dir))
        self.architectures = CatalogView(_entries=load_architectures(data_dir))
        self.capability_tiers = CatalogView(_entries=load_capability_tiers(data_dir))
        self.mission_profiles = CatalogView(_entries=load_mission_profiles(data_dir))
        self.batteries = CatalogView(_entries=load_batteries(data_dir))

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
            "cpu_architectures": len(self.cpu_architectures),
            "npus": len(self.npus),
            "operators": len(self.operators),
            "architectures": len(self.architectures),
            "capability_tiers": len(self.capability_tiers),
            "mission_profiles": len(self.mission_profiles),
            "batteries": len(self.batteries),
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

    def get_architecture_operators(self, arch_id: str) -> list[OperatorEntry]:
        """Get all operator entries referenced by an architecture.

        Args:
            arch_id: ID of the architecture

        Returns:
            List of OperatorEntry instances used in the architecture
        """
        arch = self.architectures.get(arch_id)
        if not arch:
            return []

        operators = []
        for op_instance in arch.operators:
            op_entry = self.operators.get(op_instance.operator_id)
            if op_entry:
                operators.append(op_entry)
        return operators

    def get_architectures_using_operator(self, operator_id: str) -> list[SoftwareArchitecture]:
        """Find all architectures that use a specific operator.

        Args:
            operator_id: ID of the operator

        Returns:
            List of SoftwareArchitecture instances that include this operator
        """
        results = []
        for arch in self.architectures:
            for op_instance in arch.operators:
                if op_instance.operator_id == operator_id:
                    results.append(arch)
                    break  # Only add each architecture once
        return results

    def get_architectures_by_platform(self, platform_type: str) -> list[SoftwareArchitecture]:
        """Get all architectures for a specific platform type.

        Args:
            platform_type: Platform type (drone, vehicle, manipulator, etc.)

        Returns:
            List of SoftwareArchitecture instances for that platform
        """
        return [arch for arch in self.architectures
                if arch.platform_type == platform_type]

    # =========================================================================
    # NPU Query Methods
    # =========================================================================

    def get_npus_by_vendor(self, vendor: str) -> list[NPUEntry]:
        """Get all NPUs from a specific vendor.

        Args:
            vendor: Vendor name (hailo, google, intel, qualcomm, etc.)

        Returns:
            List of NPUEntry instances from that vendor
        """
        return self.npus.find(vendor=vendor)

    def get_npus_by_type(self, npu_type: str) -> list[NPUEntry]:
        """Get all NPUs of a specific type.

        Args:
            npu_type: NPU type (discrete, integrated, datacenter, embedded)

        Returns:
            List of NPUEntry instances of that type
        """
        return [npu for npu in self.npus if npu.npu_type.value == npu_type]

    def get_npus_by_tops_range(
        self, min_tops: float = 0, max_tops: float = float("inf")
    ) -> list[NPUEntry]:
        """Get all NPUs within a TOPS performance range.

        Args:
            min_tops: Minimum INT8 TOPS (inclusive)
            max_tops: Maximum INT8 TOPS (inclusive)

        Returns:
            List of NPUEntry instances within the range
        """
        return [
            npu for npu in self.npus
            if min_tops <= npu.compute.peak_tops_int8 <= max_tops
        ]

    def get_npus_by_efficiency(self, min_tops_per_watt: float) -> list[NPUEntry]:
        """Get all NPUs meeting a minimum efficiency threshold.

        Args:
            min_tops_per_watt: Minimum TOPS/watt efficiency

        Returns:
            List of NPUEntry instances meeting the threshold
        """
        return [
            npu for npu in self.npus
            if npu.efficiency_tops_per_watt is not None
            and npu.efficiency_tops_per_watt >= min_tops_per_watt
        ]

    # =========================================================================
    # CPU Architecture Query Methods
    # =========================================================================

    def get_cpu_architectures_by_vendor(self, vendor: str) -> list[CPUArchitectureSummary]:
        """Get all CPU architectures from a specific vendor.

        Args:
            vendor: Vendor name (intel, amd, arm, etc.)

        Returns:
            List of CPUArchitectureSummary instances from that vendor
        """
        return self.cpu_architectures.find(vendor=vendor)

    def get_cpu_architecture_for_cpu(self, cpu_id: str) -> CPUArchitectureSummary | None:
        """Get the architecture summary for a CPU.

        Args:
            cpu_id: ID of the CPU entry

        Returns:
            CPUArchitectureSummary if found, None otherwise
        """
        cpu = self.cpus.get(cpu_id)
        if not cpu:
            return None

        # Match by architecture enum value
        arch_value = cpu.architecture.value
        for arch in self.cpu_architectures:
            if arch_value in arch.id:
                return arch
        return None

    def get_cpus_for_architecture(self, arch_id: str) -> list[CPUEntry]:
        """Get all CPUs using a specific architecture.

        Args:
            arch_id: ID of the CPU architecture summary

        Returns:
            List of CPUEntry instances using that architecture
        """
        arch = self.cpu_architectures.get(arch_id)
        if not arch:
            return []

        # Extract architecture key from ID (e.g., 'intel_raptor_lake' -> 'raptor_lake')
        arch_key = arch_id.split("_", 1)[-1] if "_" in arch_id else arch_id

        return [
            cpu for cpu in self.cpus
            if cpu.architecture.value == arch_key
        ]

    # =========================================================================
    # Capability Tier Query Methods
    # =========================================================================

    def get_tier_for_power(self, power_w: float) -> CapabilityTierEntry | None:
        """Find the capability tier that contains a given power level.

        Args:
            power_w: Power in watts

        Returns:
            CapabilityTierEntry that contains this power level, or None
        """
        # Sort by power_min_w to iterate in order
        sorted_tiers = sorted(self.capability_tiers.values(), key=lambda t: t.power_min_w)
        for tier in sorted_tiers:
            if tier.contains_power(power_w):
                return tier
        return None

    def get_profiles_for_tier(self, tier_name: str | CapabilityTierName) -> list[MissionProfileEntry]:
        """Get all mission profiles for a capability tier.

        Args:
            tier_name: Tier name (string or enum)

        Returns:
            List of MissionProfileEntry instances for that tier
        """
        if isinstance(tier_name, str):
            try:
                tier_name = CapabilityTierName(tier_name)
            except ValueError:
                return []

        return [
            profile for profile in self.mission_profiles
            if profile.tier == tier_name
        ]

    def get_batteries_for_tier(self, tier_name: str | CapabilityTierName) -> list[BatteryEntry]:
        """Get all batteries suitable for a capability tier.

        Args:
            tier_name: Tier name (string or enum)

        Returns:
            List of BatteryEntry instances for that tier
        """
        if isinstance(tier_name, str):
            try:
                tier_name = CapabilityTierName(tier_name)
            except ValueError:
                return []

        return [
            battery for battery in self.batteries
            if battery.typical_tier == tier_name
        ]

    def find_batteries_for_mission(
        self,
        mission_hours: float,
        average_power_w: float,
        tier: CapabilityTierName | str | None = None,
        max_weight_kg: float | None = None,
        safety_margin: float = 0.9,
    ) -> list[BatteryEntry]:
        """Find batteries that can support a mission.

        Args:
            mission_hours: Required mission duration in hours
            average_power_w: Average power consumption in watts
            tier: Optional capability tier filter
            max_weight_kg: Optional maximum weight constraint
            safety_margin: Usable capacity fraction (default 0.9)

        Returns:
            List of suitable battery configurations, sorted by weight
        """
        required_wh = (mission_hours * average_power_w) / safety_margin

        # Convert tier to enum if needed
        tier_enum: CapabilityTierName | None = None
        if tier is not None:
            if isinstance(tier, str):
                try:
                    tier_enum = CapabilityTierName(tier)
                except ValueError:
                    return []
            else:
                tier_enum = tier

        candidates = []
        for battery in self.batteries:
            # Check capacity
            if battery.capacity_wh < required_wh:
                continue

            # Check tier if specified
            if tier_enum is not None and battery.typical_tier != tier_enum:
                continue

            # Check weight if specified
            if max_weight_kg is not None and battery.weight_kg > max_weight_kg:
                continue

            # Check power capability
            if not battery.can_support_power(average_power_w, continuous=True):
                continue

            candidates.append(battery)

        # Sort by weight (lighter first)
        return sorted(candidates, key=lambda b: b.weight_kg)
