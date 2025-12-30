"""Tests for CPU loader and registry integration."""

import pytest
from embodied_schemas import (
    Registry,
    load_cpus,
    CPUEntry,
    CPUVendor,
)
from embodied_schemas.loaders import validate_data_integrity


class TestCPULoader:
    """Tests for CPU loading functionality."""

    def test_load_cpus_returns_dict(self):
        """Test that load_cpus returns a dictionary."""
        cpus = load_cpus()
        assert isinstance(cpus, dict)

    def test_load_cpus_has_entries(self):
        """Test that load_cpus returns CPUs."""
        cpus = load_cpus()
        assert len(cpus) >= 20  # We seeded 22 CPUs

    def test_load_cpus_returns_cpu_entries(self):
        """Test that load_cpus returns CPUEntry instances."""
        cpus = load_cpus()
        for cpu_id, cpu in cpus.items():
            assert isinstance(cpu, CPUEntry)
            assert cpu.id == cpu_id

    def test_load_datacenter_cpus(self):
        """Test that datacenter CPUs are loaded."""
        cpus = load_cpus()
        expected_cpus = [
            "intel_xeon_6980p_lga4710",
            "intel_xeon_platinum_8592_lga4677",
            "amd_epyc_9654_sp5",
            "amd_epyc_9965_sp5",
        ]
        for cpu_id in expected_cpus:
            assert cpu_id in cpus, f"Expected CPU {cpu_id} not found"

    def test_load_desktop_cpus(self):
        """Test that desktop CPUs are loaded."""
        cpus = load_cpus()
        expected_cpus = [
            "intel_core_i9_14900k_lga1700",
            "intel_core_ultra_9_285k_lga1851",
            "amd_ryzen_9_9950x_am5",
            "amd_ryzen_9_7950x3d_am5",
        ]
        for cpu_id in expected_cpus:
            assert cpu_id in cpus, f"Expected CPU {cpu_id} not found"

    def test_load_mobile_cpus(self):
        """Test that mobile CPUs are loaded."""
        cpus = load_cpus()
        expected_cpus = [
            "intel_core_ultra_7_268v_bga",
            "apple_m4_pro_bga",
            "qualcomm_snapdragon_x_elite_x1e_84_100_bga",
        ]
        for cpu_id in expected_cpus:
            assert cpu_id in cpus, f"Expected CPU {cpu_id} not found"

    def test_cpu_data_validity(self):
        """Test that all CPU YAML files pass validation."""
        errors = validate_data_integrity()
        cpu_errors = [e for e in errors if "cpus" in e]
        assert len(cpu_errors) == 0, f"CPU validation errors: {cpu_errors}"


class TestCPURegistry:
    """Tests for CPU registry functionality."""

    @pytest.fixture
    def registry(self):
        """Load the registry once for all tests."""
        return Registry.load()

    def test_registry_has_cpus(self, registry):
        """Test that registry includes CPUs."""
        assert "cpus" in registry.summary()
        assert registry.summary()["cpus"] >= 20

    def test_registry_cpu_query(self, registry):
        """Test querying CPUs from registry."""
        cpu = registry.cpus.get("intel_xeon_6980p_lga4710")
        assert cpu is not None
        assert cpu.name == "Intel Xeon 6980P"
        assert cpu.vendor == CPUVendor.INTEL

    def test_get_cpus_by_vendor_intel(self, registry):
        """Test filtering CPUs by Intel vendor."""
        intel_cpus = registry.get_cpus_by_vendor("intel")
        assert len(intel_cpus) >= 8  # Multiple Intel CPUs
        for cpu in intel_cpus:
            assert cpu.vendor == CPUVendor.INTEL

    def test_get_cpus_by_vendor_amd(self, registry):
        """Test filtering CPUs by AMD vendor."""
        amd_cpus = registry.get_cpus_by_vendor("amd")
        assert len(amd_cpus) >= 8  # Multiple AMD CPUs
        for cpu in amd_cpus:
            assert cpu.vendor == CPUVendor.AMD

    def test_get_cpus_by_architecture(self, registry):
        """Test filtering CPUs by architecture."""
        zen5_cpus = registry.get_cpus_by_architecture("zen5")
        assert len(zen5_cpus) >= 3  # Ryzen 9000 + EPYC 9005
        for cpu in zen5_cpus:
            assert cpu.architecture.value == "zen5"

    def test_get_cpus_by_market(self, registry):
        """Test filtering CPUs by target market."""
        datacenter_cpus = registry.get_cpus_by_market("datacenter")
        assert len(datacenter_cpus) >= 4  # Xeon + EPYC datacenter
        for cpu in datacenter_cpus:
            assert cpu.market.target_market.value == "datacenter"

    def test_get_cpus_by_socket(self, registry):
        """Test filtering CPUs by socket."""
        am5_cpus = registry.get_cpus_by_socket("am5")
        assert len(am5_cpus) >= 4  # Ryzen 9000 + 7000
        for cpu in am5_cpus:
            assert cpu.platform.socket.value == "am5"


class TestCPUDataIntegrity:
    """Tests for CPU data integrity."""

    @pytest.fixture
    def cpus(self):
        return load_cpus()

    def test_all_cpus_have_required_fields(self, cpus):
        """Test that all CPUs have the required fields populated."""
        for cpu_id, cpu in cpus.items():
            assert cpu.id, f"{cpu_id} missing id"
            assert cpu.name, f"{cpu_id} missing name"
            assert cpu.vendor, f"{cpu_id} missing vendor"
            assert cpu.architecture, f"{cpu_id} missing architecture"
            assert cpu.cores, f"{cpu_id} missing cores"
            assert cpu.clocks, f"{cpu_id} missing clocks"
            assert cpu.cache, f"{cpu_id} missing cache"
            assert cpu.memory, f"{cpu_id} missing memory"
            assert cpu.power, f"{cpu_id} missing power"
            assert cpu.market, f"{cpu_id} missing market"
            assert cpu.last_updated, f"{cpu_id} missing last_updated"

    def test_all_cpus_have_valid_cores(self, cpus):
        """Test that all CPUs have valid core counts."""
        for cpu_id, cpu in cpus.items():
            assert cpu.cores.total_cores > 0, f"{cpu_id} has invalid total_cores"
            assert cpu.cores.total_threads > 0, f"{cpu_id} has invalid total_threads"
            assert cpu.cores.total_threads >= cpu.cores.total_cores, \
                f"{cpu_id} has fewer threads than cores"

    def test_all_cpus_have_valid_clocks(self, cpus):
        """Test that all CPUs have valid clock speeds."""
        for cpu_id, cpu in cpus.items():
            assert cpu.clocks.base_clock_mhz > 0, f"{cpu_id} has invalid base_clock"
            assert cpu.clocks.boost_clock_mhz > 0, f"{cpu_id} has invalid boost_clock"
            assert cpu.clocks.boost_clock_mhz >= cpu.clocks.base_clock_mhz, \
                f"{cpu_id} has boost < base clock"

    def test_all_cpus_have_valid_power(self, cpus):
        """Test that all CPUs have valid power specs."""
        for cpu_id, cpu in cpus.items():
            assert cpu.power.tdp_watts > 0, f"{cpu_id} has invalid tdp_watts"


class TestCPUEfficiency:
    """Tests for CPU efficiency calculations."""

    def test_threads_per_watt_calculation(self):
        """Test threads per watt computed field."""
        cpus = load_cpus()

        for cpu_id, cpu in cpus.items():
            tpw = cpu.threads_per_watt
            assert tpw is not None, f"{cpu_id} missing threads_per_watt"
            assert tpw > 0, f"{cpu_id} has invalid threads_per_watt"

            # Verify calculation
            expected = cpu.cores.total_threads / cpu.power.tdp_watts
            assert abs(tpw - expected) < 0.001, f"{cpu_id} threads_per_watt mismatch"

    def test_epyc_most_efficient_datacenter(self):
        """Test that EPYC 9965 has excellent efficiency."""
        cpus = load_cpus()
        epyc_9965 = cpus.get("amd_epyc_9965_sp5")
        assert epyc_9965 is not None

        # 384 threads / 500W = 0.768 threads per watt
        assert epyc_9965.threads_per_watt > 0.7
        assert epyc_9965.threads_per_watt < 1.0
