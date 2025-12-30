"""Tests for GPU loader and registry integration."""

import pytest
from embodied_schemas import (
    Registry,
    load_gpus,
    load_gpu_architectures,
    GPUEntry,
    GPUVendor,
)
from embodied_schemas.loaders import validate_data_integrity, get_data_dir


class TestGPULoader:
    """Tests for GPU loading functionality."""

    def test_load_gpus_returns_dict(self):
        """Test that load_gpus returns a dictionary."""
        gpus = load_gpus()
        assert isinstance(gpus, dict)

    def test_load_gpus_has_entries(self):
        """Test that load_gpus returns at least one GPU."""
        gpus = load_gpus()
        assert len(gpus) >= 1

    def test_load_gpus_returns_gpu_entries(self):
        """Test that load_gpus returns GPUEntry instances."""
        gpus = load_gpus()
        for gpu_id, gpu in gpus.items():
            assert isinstance(gpu, GPUEntry)
            assert gpu.id == gpu_id

    def test_load_specific_gpus(self):
        """Test that specific expected GPUs are loaded."""
        gpus = load_gpus()
        # We seeded these GPUs
        expected_gpus = [
            "nvidia_geforce_rtx_4090",
            "nvidia_geforce_rtx_4080",
            "amd_radeon_rx_7900_xtx",
            "intel_arc_a770",
        ]
        for gpu_id in expected_gpus:
            assert gpu_id in gpus, f"Expected GPU {gpu_id} not found"

    def test_gpu_data_validity(self):
        """Test that all GPU YAML files pass validation."""
        errors = validate_data_integrity()
        gpu_errors = [e for e in errors if "gpus" in e]
        assert len(gpu_errors) == 0, f"GPU validation errors: {gpu_errors}"

    def test_load_gpu_architectures_empty(self):
        """Test that load_gpu_architectures works (may be empty)."""
        archs = load_gpu_architectures()
        assert isinstance(archs, dict)


class TestGPURegistry:
    """Tests for GPU registry functionality."""

    @pytest.fixture
    def registry(self):
        """Load the registry once for all tests."""
        return Registry.load()

    def test_registry_has_gpus(self, registry):
        """Test that registry includes GPUs."""
        assert "gpus" in registry.summary()
        assert registry.summary()["gpus"] >= 1

    def test_registry_gpu_query(self, registry):
        """Test querying GPUs from registry."""
        gpu = registry.gpus.get("nvidia_geforce_rtx_4090")
        assert gpu is not None
        assert gpu.name == "NVIDIA GeForce RTX 4090"
        assert gpu.vendor == GPUVendor.NVIDIA

    def test_get_gpus_by_vendor_nvidia(self, registry):
        """Test filtering GPUs by NVIDIA vendor."""
        nvidia_gpus = registry.get_gpus_by_vendor("nvidia")
        assert len(nvidia_gpus) >= 2  # RTX 4090, 4080
        for gpu in nvidia_gpus:
            assert gpu.vendor == GPUVendor.NVIDIA

    def test_get_gpus_by_vendor_amd(self, registry):
        """Test filtering GPUs by AMD vendor."""
        amd_gpus = registry.get_gpus_by_vendor("amd")
        assert len(amd_gpus) >= 1  # RX 7900 XTX
        for gpu in amd_gpus:
            assert gpu.vendor == GPUVendor.AMD

    def test_get_gpus_by_vendor_intel(self, registry):
        """Test filtering GPUs by Intel vendor."""
        intel_gpus = registry.get_gpus_by_vendor("intel")
        assert len(intel_gpus) >= 1  # Arc A770
        for gpu in intel_gpus:
            assert gpu.vendor == GPUVendor.INTEL

    def test_get_gpus_by_architecture(self, registry):
        """Test filtering GPUs by architecture."""
        ada_gpus = registry.get_gpus_by_architecture("Ada Lovelace")
        assert len(ada_gpus) >= 2  # RTX 4090, 4080
        for gpu in ada_gpus:
            assert gpu.die.architecture == "Ada Lovelace"

    def test_get_gpus_by_market(self, registry):
        """Test filtering GPUs by target market."""
        consumer_gpus = registry.get_gpus_by_market("consumer_desktop")
        assert len(consumer_gpus) >= 4  # All seeded GPUs are consumer desktop
        for gpu in consumer_gpus:
            assert gpu.market.target_market.value == "consumer_desktop"


class TestGPUEfficiency:
    """Tests for GPU efficiency calculations."""

    @pytest.fixture
    def registry(self):
        return Registry.load()

    def test_efficiency_metrics_rtx_4090(self, registry):
        """Test efficiency metric calculation for RTX 4090."""
        gpu = registry.gpus.get("nvidia_geforce_rtx_4090")
        assert gpu is not None

        efficiency = gpu.compute_efficiency_metrics()
        # RTX 4090: 82.58 TFLOPS / 450W = 0.1835 TFLOPS/W
        assert efficiency.perf_per_watt_tflops is not None
        assert efficiency.perf_per_watt_tflops > 0.15
        assert efficiency.perf_per_watt_tflops < 0.25

    def test_efficiency_comparison(self, registry):
        """Test comparing efficiency across GPUs."""
        gpus = load_gpus()

        efficiencies = {}
        for gpu_id, gpu in gpus.items():
            eff = gpu.compute_efficiency_metrics()
            if eff.perf_per_watt_tflops:
                efficiencies[gpu_id] = eff.perf_per_watt_tflops

        # All GPUs should have efficiency metrics
        assert len(efficiencies) == len(gpus)

        # L40 should be most efficient NVIDIA GPU (90.5 TFLOPS @ 300W = 0.30 TFLOPS/W)
        nvidia_effs = {k: v for k, v in efficiencies.items() if "nvidia" in k}
        assert max(nvidia_effs.values()) == efficiencies["nvidia_l40"]

        # Consumer GPUs: RTX 4090 should be most efficient
        consumer_nvidia = {k: v for k, v in efficiencies.items()
                         if "nvidia_geforce" in k}
        assert max(consumer_nvidia.values()) == efficiencies["nvidia_geforce_rtx_4090"]


class TestGPUDataIntegrity:
    """Tests for GPU data integrity."""

    @pytest.fixture
    def gpus(self):
        return load_gpus()

    def test_all_gpus_have_required_fields(self, gpus):
        """Test that all GPUs have the required fields populated."""
        for gpu_id, gpu in gpus.items():
            assert gpu.id, f"{gpu_id} missing id"
            assert gpu.name, f"{gpu_id} missing name"
            assert gpu.vendor, f"{gpu_id} missing vendor"
            assert gpu.die, f"{gpu_id} missing die"
            assert gpu.compute, f"{gpu_id} missing compute"
            assert gpu.clocks, f"{gpu_id} missing clocks"
            assert gpu.memory, f"{gpu_id} missing memory"
            assert gpu.performance, f"{gpu_id} missing performance"
            assert gpu.power, f"{gpu_id} missing power"
            assert gpu.market, f"{gpu_id} missing market"
            assert gpu.last_updated, f"{gpu_id} missing last_updated"

    def test_all_gpus_have_valid_performance(self, gpus):
        """Test that all GPUs have valid performance specs."""
        for gpu_id, gpu in gpus.items():
            assert gpu.performance.fp32_tflops > 0, f"{gpu_id} has invalid fp32_tflops"
            assert gpu.performance.pixel_rate_gpixels > 0, f"{gpu_id} has invalid pixel_rate"
            assert gpu.performance.texture_rate_gtexels > 0, f"{gpu_id} has invalid texture_rate"

    def test_all_gpus_have_valid_memory(self, gpus):
        """Test that all GPUs have valid memory specs."""
        for gpu_id, gpu in gpus.items():
            assert gpu.memory.memory_size_gb > 0, f"{gpu_id} has invalid memory_size_gb"
            assert gpu.memory.memory_bandwidth_gbps > 0, f"{gpu_id} has invalid bandwidth"
            assert gpu.memory.memory_bus_bits > 0, f"{gpu_id} has invalid bus_bits"

    def test_all_gpus_have_valid_power(self, gpus):
        """Test that all GPUs have valid power specs."""
        for gpu_id, gpu in gpus.items():
            assert gpu.power.tdp_watts > 0, f"{gpu_id} has invalid tdp_watts"


class TestCrossReference:
    """Tests for GPU-Hardware cross-reference functionality."""

    @pytest.fixture
    def registry(self):
        return Registry.load()

    def test_get_gpu_for_hardware_returns_none_for_discrete(self, registry):
        """Test that discrete GPUs don't have associated hardware entries."""
        # This should return None since RTX 4090 is a discrete GPU
        # and we don't have any HardwareEntry linking to it
        gpu = registry.get_gpu_for_hardware("nonexistent_hardware")
        assert gpu is None

    def test_get_hardware_with_gpu_returns_empty_for_discrete(self, registry):
        """Test that discrete GPUs have no embedded hardware by default."""
        # Discrete GPUs aren't embedded in any hardware
        hardware_list = registry.get_hardware_with_gpu("nvidia_geforce_rtx_4090")
        assert hardware_list == []

    def test_embedded_in_hardware_ids_default(self):
        """Test that GPUs have empty embedded_in_hardware_ids by default."""
        gpus = load_gpus()
        for gpu_id, gpu in gpus.items():
            # Discrete GPUs should have empty list
            assert isinstance(gpu.embedded_in_hardware_ids, list)
