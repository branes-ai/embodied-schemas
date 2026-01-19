"""Tests for mission, battery, and capability tier schemas."""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    CapabilityTierEntry,
    CapabilityTierName,
    MissionProfileEntry,
    BatteryEntry,
    BatteryChemistry,
    SubsystemType,
    CoolingType,
    ThermalConstraints,
    FormFactorConstraints,
    PowerAllocation,
    DutyCycle,
    OperatingTemperatureRange,
    Registry,
    load_capability_tiers,
    load_mission_profiles,
    load_batteries,
)


class TestCapabilityTierSchemas:
    """Tests for capability tier schemas."""

    def test_capability_tier_name_enum(self):
        """Test CapabilityTierName enum values."""
        assert CapabilityTierName.WEARABLE_AI.value == "wearable_ai"
        assert CapabilityTierName.MICRO_AUTONOMY.value == "micro_autonomy"
        assert CapabilityTierName.INDUSTRIAL_EDGE.value == "industrial_edge"
        assert CapabilityTierName.EMBODIED_AI.value == "embodied_ai"
        assert CapabilityTierName.AUTOMOTIVE_AI.value == "automotive_ai"

    def test_cooling_type_enum(self):
        """Test CoolingType enum values."""
        assert CoolingType.PASSIVE.value == "passive"
        assert CoolingType.ACTIVE_FAN.value == "active_fan"
        assert CoolingType.LIQUID.value == "liquid"

    def test_thermal_constraints_defaults(self):
        """Test ThermalConstraints with default values."""
        thermal = ThermalConstraints()
        assert thermal.max_ambient_temp_c == 40.0
        assert thermal.max_junction_temp_c == 85.0
        assert thermal.typical_cooling == CoolingType.PASSIVE
        assert thermal.sustained_power_derating == 0.8

    def test_form_factor_constraints(self):
        """Test FormFactorConstraints creation."""
        ff = FormFactorConstraints(
            max_compute_weight_kg=0.1,
            max_compute_volume_cm3=50.0,
            max_total_system_weight_kg=2.0,
            typical_form_factors=["drone", "handheld"],
        )
        assert ff.max_compute_weight_kg == 0.1
        assert "drone" in ff.typical_form_factors

    def test_power_allocation_valid(self):
        """Test PowerAllocation with valid ratios."""
        alloc = PowerAllocation(
            perception=0.35,
            control=0.20,
            movement=0.35,
            overhead=0.10,
        )
        assert alloc.validate_sum()

    def test_power_allocation_invalid_ratio(self):
        """Test PowerAllocation rejects ratios outside 0-1."""
        with pytest.raises(ValidationError):
            PowerAllocation(perception=1.5, control=0.2, movement=0.2, overhead=0.1)

    def test_capability_tier_entry_minimal(self):
        """Test CapabilityTierEntry with minimal fields."""
        entry = CapabilityTierEntry(
            id="micro_autonomy",
            name=CapabilityTierName.MICRO_AUTONOMY,
            display_name="Micro-Autonomy",
            power_min_w=1.0,
            power_max_w=10.0,
            description="Small autonomous systems",
            last_updated="2026-01-18",
        )
        assert entry.id == "micro_autonomy"
        assert entry.name == CapabilityTierName.MICRO_AUTONOMY
        assert entry.power_min_w == 1.0

    def test_capability_tier_entry_full(self):
        """Test CapabilityTierEntry with all fields."""
        entry = CapabilityTierEntry(
            id="micro_autonomy",
            name=CapabilityTierName.MICRO_AUTONOMY,
            display_name="Micro-Autonomy",
            power_min_w=1.0,
            power_max_w=10.0,
            description="Small autonomous systems",
            typical_applications=["Inspection drones", "Handheld scanners"],
            example_platforms=["Jetson Orin Nano", "Hailo-8"],
            thermal=ThermalConstraints(
                max_ambient_temp_c=45.0,
                typical_cooling=CoolingType.PASSIVE,
            ),
            form_factor=FormFactorConstraints(
                max_compute_weight_kg=0.1,
                typical_form_factors=["drone", "handheld"],
            ),
            typical_mission_hours=[0.5, 4.0],
            typical_battery_wh_per_kg=180.0,
            typical_allocation=PowerAllocation(),
            last_updated="2026-01-18",
        )
        assert "Inspection drones" in entry.typical_applications
        assert entry.thermal.max_ambient_temp_c == 45.0

    def test_capability_tier_power_range_str(self):
        """Test power_range_str property."""
        entry = CapabilityTierEntry(
            id="test",
            name=CapabilityTierName.MICRO_AUTONOMY,
            display_name="Test",
            power_min_w=1.0,
            power_max_w=10.0,
            description="Test",
            last_updated="2026-01-18",
        )
        assert entry.power_range_str == "1-10W"

    def test_capability_tier_contains_power(self):
        """Test contains_power method."""
        entry = CapabilityTierEntry(
            id="test",
            name=CapabilityTierName.MICRO_AUTONOMY,
            display_name="Test",
            power_min_w=1.0,
            power_max_w=10.0,
            description="Test",
            last_updated="2026-01-18",
        )
        assert entry.contains_power(5.0)
        assert not entry.contains_power(15.0)
        assert not entry.contains_power(0.5)

    def test_capability_tier_typical_power(self):
        """Test typical_power_w property (geometric mean)."""
        entry = CapabilityTierEntry(
            id="test",
            name=CapabilityTierName.MICRO_AUTONOMY,
            display_name="Test",
            power_min_w=1.0,
            power_max_w=10.0,
            description="Test",
            last_updated="2026-01-18",
        )
        # sqrt(1 * 10) = sqrt(10) ≈ 3.16
        assert entry.typical_power_w == pytest.approx(3.162, rel=0.01)

    def test_capability_tier_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            CapabilityTierEntry(
                id="test",
                name=CapabilityTierName.MICRO_AUTONOMY,
                display_name="Test",
                power_min_w=1.0,
                power_max_w=10.0,
                description="Test",
                last_updated="2026-01-18",
                unknown_field="should_fail",
            )


class TestMissionProfileSchemas:
    """Tests for mission profile schemas."""

    def test_duty_cycle_basic(self):
        """Test DutyCycle creation."""
        duty = DutyCycle(
            active_ratio=0.8,
            peak_ratio=0.3,
            description="Continuous operation",
        )
        assert duty.active_ratio == 0.8
        assert duty.peak_ratio == 0.3

    def test_duty_cycle_effective_ratio(self):
        """Test DutyCycle effective_ratio property."""
        duty = DutyCycle(active_ratio=1.0, peak_ratio=0.0)
        # All active, no peak: effective = 1.0 * 1.0 = 1.0
        assert duty.effective_ratio == 1.0

        duty2 = DutyCycle(active_ratio=1.0, peak_ratio=1.0)
        # All active, all peak: effective = 1.0 * 1.0 * 1.5 = 1.5
        assert duty2.effective_ratio == 1.5

    def test_duty_cycle_validation(self):
        """Test DutyCycle ratio validation."""
        with pytest.raises(ValidationError):
            DutyCycle(active_ratio=1.5, peak_ratio=0.2)

        with pytest.raises(ValidationError):
            DutyCycle(active_ratio=0.5, peak_ratio=-0.1)

    def test_mission_profile_entry_minimal(self):
        """Test MissionProfileEntry with minimal fields."""
        entry = MissionProfileEntry(
            id="drone_inspection",
            name="drone-inspection",
            display_name="Drone Inspection",
            tier=CapabilityTierName.MICRO_AUTONOMY,
            description="Aerial inspection mission",
            typical_duration_hours=0.5,
            perception_duty=DutyCycle(active_ratio=0.9, peak_ratio=0.4),
            control_duty=DutyCycle(active_ratio=0.8, peak_ratio=0.3),
            movement_duty=DutyCycle(active_ratio=0.7, peak_ratio=0.3),
            last_updated="2026-01-18",
        )
        assert entry.id == "drone_inspection"
        assert entry.tier == CapabilityTierName.MICRO_AUTONOMY

    def test_mission_profile_entry_full(self):
        """Test MissionProfileEntry with all fields."""
        entry = MissionProfileEntry(
            id="drone_inspection",
            name="drone-inspection",
            display_name="Drone Inspection",
            tier=CapabilityTierName.MICRO_AUTONOMY,
            description="Aerial inspection mission",
            typical_duration_hours=0.5,
            perception_duty=DutyCycle(active_ratio=0.9, peak_ratio=0.4),
            control_duty=DutyCycle(active_ratio=0.8, peak_ratio=0.3),
            movement_duty=DutyCycle(active_ratio=0.7, peak_ratio=0.3),
            peak_perception_power_w=6.0,
            cruise_perception_power_w=4.0,
            peak_movement_power_w=15.0,
            cruise_movement_power_w=8.0,
            environment="outdoor",
            constraints=["wind resistance", "GPS availability"],
            last_updated="2026-01-18",
        )
        assert entry.peak_perception_power_w == 6.0
        assert "wind resistance" in entry.constraints

    def test_mission_profile_average_power_multiplier(self):
        """Test estimate_average_power_multiplier method."""
        entry = MissionProfileEntry(
            id="test",
            name="test",
            display_name="Test",
            tier=CapabilityTierName.MICRO_AUTONOMY,
            description="Test",
            typical_duration_hours=1.0,
            perception_duty=DutyCycle(active_ratio=1.0, peak_ratio=0.0),
            control_duty=DutyCycle(active_ratio=1.0, peak_ratio=0.0),
            movement_duty=DutyCycle(active_ratio=1.0, peak_ratio=0.0),
            last_updated="2026-01-18",
        )
        # All subsystems at 100% active, 0% peak = multiplier should be 1.0
        # (0.35 * 1.0) + (0.20 * 1.0) + (0.35 * 1.0) + (0.10 * 1.0) = 1.0
        multiplier = entry.estimate_average_power_multiplier()
        assert multiplier == pytest.approx(1.0, rel=0.01)


class TestBatterySchemas:
    """Tests for battery schemas."""

    def test_battery_chemistry_enum(self):
        """Test BatteryChemistry enum values."""
        assert BatteryChemistry.LIPO.value == "lipo"
        assert BatteryChemistry.LI_ION.value == "li_ion"
        assert BatteryChemistry.LIFEPO4.value == "lifepo4"

    def test_operating_temperature_range(self):
        """Test OperatingTemperatureRange creation."""
        temp = OperatingTemperatureRange(min_c=-10.0, max_c=50.0)
        assert temp.min_c == -10.0
        assert temp.max_c == 50.0

    def test_battery_entry_minimal(self):
        """Test BatteryEntry with minimal fields."""
        entry = BatteryEntry(
            id="drone_small",
            name="drone-small",
            chemistry=BatteryChemistry.LIPO,
            capacity_wh=50.0,
            voltage_nominal=14.8,
            weight_kg=0.35,
            volume_cm3=150.0,
            operating_temp=OperatingTemperatureRange(min_c=-10.0, max_c=50.0),
            last_updated="2026-01-18",
        )
        assert entry.id == "drone_small"
        assert entry.chemistry == BatteryChemistry.LIPO
        assert entry.capacity_wh == 50.0

    def test_battery_entry_full(self):
        """Test BatteryEntry with all fields."""
        entry = BatteryEntry(
            id="drone_small",
            name="drone-small",
            chemistry=BatteryChemistry.LIPO,
            capacity_wh=50.0,
            voltage_nominal=14.8,
            weight_kg=0.35,
            volume_cm3=150.0,
            max_discharge_rate_c=25.0,
            peak_discharge_rate_c=50.0,
            cycle_life=200,
            operating_temp=OperatingTemperatureRange(min_c=-10.0, max_c=50.0),
            typical_tier=CapabilityTierName.MICRO_AUTONOMY,
            description="Small drone battery",
            last_updated="2026-01-18",
        )
        assert entry.max_discharge_rate_c == 25.0
        assert entry.typical_tier == CapabilityTierName.MICRO_AUTONOMY

    def test_battery_energy_density_properties(self):
        """Test battery energy density computed properties."""
        entry = BatteryEntry(
            id="test",
            name="test",
            chemistry=BatteryChemistry.LIPO,
            capacity_wh=100.0,
            voltage_nominal=14.8,
            weight_kg=0.5,
            volume_cm3=200.0,
            operating_temp=OperatingTemperatureRange(min_c=0.0, max_c=45.0),
            last_updated="2026-01-18",
        )
        # 100 Wh / 0.5 kg = 200 Wh/kg
        assert entry.energy_density_wh_per_kg == pytest.approx(200.0)
        # 100 Wh / 0.2 L = 500 Wh/L
        assert entry.energy_density_wh_per_l == pytest.approx(500.0)

    def test_battery_power_properties(self):
        """Test battery power computed properties."""
        entry = BatteryEntry(
            id="test",
            name="test",
            chemistry=BatteryChemistry.LIPO,
            capacity_wh=100.0,
            voltage_nominal=14.8,
            weight_kg=0.5,
            volume_cm3=200.0,
            max_discharge_rate_c=2.0,
            peak_discharge_rate_c=5.0,
            operating_temp=OperatingTemperatureRange(min_c=0.0, max_c=45.0),
            last_updated="2026-01-18",
        )
        # 100 Wh * 2C = 200W continuous
        assert entry.max_continuous_power_w == pytest.approx(200.0)
        # 100 Wh * 5C = 500W peak
        assert entry.peak_power_w == pytest.approx(500.0)

    def test_battery_runtime_estimation(self):
        """Test estimate_runtime_hours method."""
        entry = BatteryEntry(
            id="test",
            name="test",
            chemistry=BatteryChemistry.LIPO,
            capacity_wh=100.0,
            voltage_nominal=14.8,
            weight_kg=0.5,
            volume_cm3=200.0,
            operating_temp=OperatingTemperatureRange(min_c=0.0, max_c=45.0),
            last_updated="2026-01-18",
        )
        # 100 Wh * 0.9 (safety) / 50W = 1.8 hours
        runtime = entry.estimate_runtime_hours(50.0, safety_margin=0.9)
        assert runtime == pytest.approx(1.8)

    def test_battery_can_support_power(self):
        """Test can_support_power method."""
        entry = BatteryEntry(
            id="test",
            name="test",
            chemistry=BatteryChemistry.LIPO,
            capacity_wh=100.0,
            voltage_nominal=14.8,
            weight_kg=0.5,
            volume_cm3=200.0,
            max_discharge_rate_c=2.0,
            peak_discharge_rate_c=5.0,
            operating_temp=OperatingTemperatureRange(min_c=0.0, max_c=45.0),
            last_updated="2026-01-18",
        )
        # Continuous: max 200W
        assert entry.can_support_power(150.0, continuous=True)
        assert not entry.can_support_power(250.0, continuous=True)
        # Peak: max 500W
        assert entry.can_support_power(400.0, continuous=False)
        assert not entry.can_support_power(600.0, continuous=False)


class TestDataLoading:
    """Tests for loading YAML data files."""

    def test_load_capability_tiers(self):
        """Test loading capability tier YAML files."""
        tiers = load_capability_tiers()
        assert len(tiers) >= 1
        # Check we have at least micro_autonomy
        assert "micro_autonomy" in tiers
        tier = tiers["micro_autonomy"]
        assert tier.name == CapabilityTierName.MICRO_AUTONOMY

    def test_load_mission_profiles(self):
        """Test loading mission profile YAML files."""
        profiles = load_mission_profiles()
        assert len(profiles) >= 1
        # Check we have at least drone_inspection
        assert "drone_inspection" in profiles
        profile = profiles["drone_inspection"]
        assert profile.tier == CapabilityTierName.MICRO_AUTONOMY

    def test_load_batteries(self):
        """Test loading battery YAML files."""
        batteries = load_batteries()
        assert len(batteries) >= 1
        # Check we have at least drone_small
        assert "drone_small" in batteries
        battery = batteries["drone_small"]
        assert battery.chemistry == BatteryChemistry.LIPO


class TestRegistryIntegration:
    """Tests for registry integration with mission schemas."""

    def test_registry_loads_mission_data(self):
        """Test that Registry.load() includes mission data."""
        registry = Registry.load()
        assert len(registry.capability_tiers) >= 1
        assert len(registry.mission_profiles) >= 1
        assert len(registry.batteries) >= 1

    def test_registry_summary_includes_mission(self):
        """Test that Registry.summary() includes mission counts."""
        registry = Registry.load()
        summary = registry.summary()
        assert "capability_tiers" in summary
        assert "mission_profiles" in summary
        assert "batteries" in summary

    def test_registry_get_tier_for_power(self):
        """Test get_tier_for_power query method."""
        registry = Registry.load()
        tier = registry.get_tier_for_power(5.0)
        assert tier is not None
        assert tier.name == CapabilityTierName.MICRO_AUTONOMY

    def test_registry_get_profiles_for_tier(self):
        """Test get_profiles_for_tier query method."""
        registry = Registry.load()
        profiles = registry.get_profiles_for_tier(CapabilityTierName.MICRO_AUTONOMY)
        assert len(profiles) >= 1
        assert all(p.tier == CapabilityTierName.MICRO_AUTONOMY for p in profiles)

    def test_registry_get_batteries_for_tier(self):
        """Test get_batteries_for_tier query method."""
        registry = Registry.load()
        batteries = registry.get_batteries_for_tier(CapabilityTierName.MICRO_AUTONOMY)
        assert len(batteries) >= 1
        assert all(b.typical_tier == CapabilityTierName.MICRO_AUTONOMY for b in batteries)

    def test_registry_find_batteries_for_mission(self):
        """Test find_batteries_for_mission query method."""
        registry = Registry.load()
        # Find batteries for a 30 minute mission at 5W average power
        batteries = registry.find_batteries_for_mission(
            mission_hours=0.5,
            average_power_w=5.0,
            tier=CapabilityTierName.MICRO_AUTONOMY,
        )
        assert len(batteries) >= 1
        # All should support the power and have enough capacity
        for battery in batteries:
            assert battery.can_support_power(5.0, continuous=True)
            # Required Wh = 0.5 * 5 / 0.9 ≈ 2.78 Wh
            assert battery.capacity_wh >= (0.5 * 5.0) / 0.9
