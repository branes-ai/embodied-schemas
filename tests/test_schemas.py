"""Tests for Pydantic schema models."""

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    HardwareEntry,
    HardwareCapability,
    HardwareType,
    ModelEntry,
    ModelVariant,
    SensorEntry,
    UseCaseEntry,
    BenchmarkResult,
    Verdict,
    Confidence,
)
from embodied_schemas.hardware import (
    ComputeParadigm,
    PhysicalSpec,
    FormFactor,
    EnvironmentalSpec,
    PowerSpec,
    PowerMode,
)
from embodied_schemas.models import (
    ModelType,
    ArchitectureSpec,
    InputSpec,
    OutputSpec,
    MemoryRequirements,
    DataType,
    ModelFormat,
)
from embodied_schemas.sensors import (
    SensorCategory,
    CameraSpec,
    SensorInterface,
    InterfaceType,
    SensorPower,
)
from embodied_schemas.usecases import (
    Constraint,
    ConstraintCriticality,
    SuccessCriterion,
    Operator,
    PlatformSpec,
    PlatformType,
    PerceptionRequirement,
)
from embodied_schemas.benchmarks import (
    LatencyMetrics,
    PowerMetrics,
    BenchmarkConditions,
)
from embodied_schemas.constraints import (
    LatencyTier,
    PowerClass,
    get_latency_tier,
    get_power_class,
)


class TestHardwareSchemas:
    """Tests for hardware-related schemas."""

    def test_hardware_capability_minimal(self):
        """Test HardwareCapability with minimal required fields."""
        cap = HardwareCapability(memory_gb=4.0)
        assert cap.memory_gb == 4.0
        assert cap.peak_tops_int8 is None

    def test_hardware_capability_full(self):
        """Test HardwareCapability with all fields."""
        cap = HardwareCapability(
            peak_tops_int8=100.0,
            peak_tflops_fp16=50.0,
            peak_tflops_fp32=25.0,
            memory_gb=32.0,
            memory_bandwidth_gbps=200.0,
            compute_units=128,
            frameworks=["pytorch", "tensorrt"],
        )
        assert cap.peak_tops_int8 == 100.0
        assert "pytorch" in cap.frameworks

    def test_hardware_entry_minimal(self):
        """Test HardwareEntry with minimal required fields."""
        entry = HardwareEntry(
            id="test_device",
            name="Test Device",
            vendor="Test Vendor",
            model="Test Model",
            hardware_type=HardwareType.GPU,
            compute_paradigm=ComputeParadigm.SIMD,
            capabilities=HardwareCapability(memory_gb=8.0),
            last_updated="2024-12-01",
        )
        assert entry.id == "test_device"
        assert entry.hardware_type == HardwareType.GPU

    def test_hardware_entry_with_physical(self):
        """Test HardwareEntry with physical specs."""
        entry = HardwareEntry(
            id="edge_device",
            name="Edge Device",
            vendor="Edge Vendor",
            model="Edge Model",
            hardware_type=HardwareType.NPU,
            compute_paradigm=ComputeParadigm.DATAFLOW,
            capabilities=HardwareCapability(memory_gb=2.0),
            physical=PhysicalSpec(
                form_factor=FormFactor.SOM,
                weight_grams=50.0,
                dimensions_mm=[50.0, 30.0, 10.0],
            ),
            environmental=EnvironmentalSpec(
                operating_temp_c=[-20, 70],
                ip_rating="IP65",
            ),
            last_updated="2024-12-01",
        )
        assert entry.physical.weight_grams == 50.0
        assert entry.environmental.ip_rating == "IP65"

    def test_hardware_entry_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            HardwareEntry(
                id="test",
                name="Test",
                vendor="Test",
                model="Test",
                hardware_type=HardwareType.CPU,
                compute_paradigm=ComputeParadigm.VON_NEUMANN,
                capabilities=HardwareCapability(memory_gb=4.0),
                last_updated="2024-12-01",
                unknown_field="should_fail",
            )


class TestModelSchemas:
    """Tests for model-related schemas."""

    def test_model_variant(self):
        """Test ModelVariant creation."""
        variant = ModelVariant(
            name="int8-tensorrt",
            dtype=DataType.INT8,
            format=ModelFormat.TENSORRT,
            file_size_mb=10.5,
            accuracy_delta=-0.02,
        )
        assert variant.dtype == DataType.INT8
        assert variant.accuracy_delta == -0.02

    def test_model_entry_minimal(self):
        """Test ModelEntry with minimal required fields."""
        entry = ModelEntry(
            id="yolov8n",
            name="YOLOv8n",
            version="8.0",
            vendor="Ultralytics",
            architecture=ArchitectureSpec(
                type=ModelType.OBJECT_DETECTION,
                params_millions=3.2,
            ),
            input=InputSpec(shape=[1, 3, 640, 640]),
            output=OutputSpec(format="boxes_scores_classes"),
            memory=MemoryRequirements(weights_mb=6.5),
            license="AGPL-3.0",
            last_updated="2024-12-01",
        )
        assert entry.id == "yolov8n"
        assert entry.architecture.type == ModelType.OBJECT_DETECTION


class TestSensorSchemas:
    """Tests for sensor-related schemas."""

    def test_sensor_entry_camera(self):
        """Test SensorEntry for a camera."""
        entry = SensorEntry(
            id="imx477",
            name="Sony IMX477",
            vendor="Sony",
            model="IMX477",
            category=SensorCategory.CAMERA,
            camera_spec=CameraSpec(
                sensor_type="CMOS",
                resolution=[4056, 3040],
                max_fps=60,
            ),
            interface=SensorInterface(
                type=InterfaceType.CSI,
                lanes=2,
            ),
            power=SensorPower(
                voltage_v=3.3,
                power_watts=0.66,
            ),
            last_updated="2024-12-01",
        )
        assert entry.category == SensorCategory.CAMERA
        assert entry.camera_spec.max_fps == 60


class TestUseCaseSchemas:
    """Tests for use case-related schemas."""

    def test_constraint(self):
        """Test Constraint creation."""
        constraint = Constraint(
            max_value=33.3,
            tier="real_time",
            criticality=ConstraintCriticality.SAFETY_CRITICAL,
            unit="ms",
        )
        assert constraint.max_value == 33.3
        assert constraint.criticality == ConstraintCriticality.SAFETY_CRITICAL

    def test_success_criterion(self):
        """Test SuccessCriterion creation."""
        criterion = SuccessCriterion(
            metric="latency_ms",
            target=33.3,
            operator=Operator.LT,
        )
        assert criterion.operator == Operator.LT

    def test_usecase_entry_minimal(self):
        """Test UseCaseEntry with minimal fields."""
        entry = UseCaseEntry(
            id="drone_obstacle",
            name="Drone Obstacle Avoidance",
            category="drone",
            description="Real-time obstacle detection for UAVs",
            platform=PlatformSpec(type=PlatformType.DRONE),
            perception=PerceptionRequirement(
                tasks=["object_detection"],
            ),
            last_updated="2024-12-01",
        )
        assert entry.category == "drone"
        assert "object_detection" in entry.perception.tasks


class TestBenchmarkSchemas:
    """Tests for benchmark-related schemas."""

    def test_latency_metrics(self):
        """Test LatencyMetrics creation."""
        metrics = LatencyMetrics(
            mean_ms=45.2,
            std_ms=3.1,
            p95_ms=51.0,
        )
        assert metrics.mean_ms == 45.2

    def test_benchmark_result(self):
        """Test BenchmarkResult creation."""
        result = BenchmarkResult(
            id="bench_001",
            model_id="yolov8s",
            hardware_id="jetson_nano",
            variant="fp16",
            conditions=BenchmarkConditions(
                batch_size=1,
                warmup_iterations=50,
                test_iterations=1000,
            ),
            latency=LatencyMetrics(mean_ms=45.2),
            verdict=Verdict.FAIL,
            confidence=Confidence.HIGH,
            timestamp="2024-12-01T12:00:00Z",
        )
        assert result.verdict == Verdict.FAIL
        assert result.confidence == Confidence.HIGH


class TestConstraintUtilities:
    """Tests for constraint utility functions."""

    def test_get_latency_tier(self):
        """Test latency tier classification."""
        assert get_latency_tier(5.0) == LatencyTier.ULTRA_REAL_TIME
        assert get_latency_tier(25.0) == LatencyTier.REAL_TIME
        assert get_latency_tier(50.0) == LatencyTier.INTERACTIVE
        assert get_latency_tier(2000.0) == LatencyTier.BATCH

    def test_get_power_class(self):
        """Test power class classification."""
        assert get_power_class(1.0) == PowerClass.ULTRA_LOW_POWER
        assert get_power_class(8.0) == PowerClass.LOW_POWER
        assert get_power_class(25.0) == PowerClass.MEDIUM_POWER
        assert get_power_class(80.0) == PowerClass.HIGH_POWER
        assert get_power_class(300.0) == PowerClass.DATACENTER


class TestGPUSchemas:
    """Tests for GPU-specific schemas."""

    def test_gpu_entry_minimal(self):
        """Test GPUEntry with minimal required fields."""
        from embodied_schemas.gpu import (
            GPUEntry,
            GPUVendor,
            DieSpec,
            ComputeResources,
            ClockSpeeds,
            MemorySpec,
            MemoryType,
            TheoreticalPerformance,
            PowerSpec,
            MarketInfo,
            TargetMarket,
            Foundry,
        )

        entry = GPUEntry(
            id="nvidia_rtx_4090",
            name="NVIDIA GeForce RTX 4090",
            vendor=GPUVendor.NVIDIA,
            die=DieSpec(
                gpu_name="AD102",
                architecture="Ada Lovelace",
                foundry=Foundry.TSMC,
                process_nm=5,
                transistors_billion=76.3,
                die_size_mm2=608.5,
            ),
            compute=ComputeResources(
                shaders=16384,
                cuda_cores=16384,
                tmus=512,
                rops=176,
                tensor_cores=512,
            ),
            clocks=ClockSpeeds(
                base_clock_mhz=2235,
                boost_clock_mhz=2520,
                memory_clock_mhz=1313,
            ),
            memory=MemorySpec(
                memory_size_gb=24.0,
                memory_type=MemoryType.GDDR6X,
                memory_bus_bits=384,
                memory_bandwidth_gbps=1008.0,
            ),
            performance=TheoreticalPerformance(
                fp32_tflops=82.58,
                pixel_rate_gpixels=443.5,
                texture_rate_gtexels=1290.2,
            ),
            power=PowerSpec(
                tdp_watts=450,
            ),
            market=MarketInfo(
                launch_date="2022-10-12",
                target_market=TargetMarket.CONSUMER_DESKTOP,
            ),
            last_updated="2024-12-21",
        )
        assert entry.id == "nvidia_rtx_4090"
        assert entry.vendor == GPUVendor.NVIDIA
        assert entry.die.transistors_billion == 76.3
        assert entry.compute.cuda_cores == 16384
        assert entry.performance.fp32_tflops == 82.58

    def test_gpu_compute_efficiency(self):
        """Test GPUEntry efficiency metric computation."""
        from embodied_schemas.gpu import (
            GPUEntry,
            GPUVendor,
            DieSpec,
            ComputeResources,
            ClockSpeeds,
            MemorySpec,
            MemoryType,
            TheoreticalPerformance,
            PowerSpec,
            MarketInfo,
            TargetMarket,
            Foundry,
        )

        entry = GPUEntry(
            id="test_gpu",
            name="Test GPU",
            vendor=GPUVendor.NVIDIA,
            die=DieSpec(
                gpu_name="TEST100",
                architecture="Test Arch",
                foundry=Foundry.TSMC,
                process_nm=5,
                transistors_billion=50.0,
                die_size_mm2=500.0,
            ),
            compute=ComputeResources(shaders=10000, tmus=256, rops=128),
            clocks=ClockSpeeds(base_clock_mhz=2000, boost_clock_mhz=2500, memory_clock_mhz=1000),
            memory=MemorySpec(
                memory_size_gb=16.0,
                memory_type=MemoryType.GDDR6,
                memory_bus_bits=256,
                memory_bandwidth_gbps=500.0,
            ),
            performance=TheoreticalPerformance(
                fp32_tflops=50.0,
                pixel_rate_gpixels=300.0,
                texture_rate_gtexels=600.0,
            ),
            power=PowerSpec(tdp_watts=250),
            market=MarketInfo(launch_date="2024-01-01", target_market=TargetMarket.CONSUMER_DESKTOP),
            last_updated="2024-12-21",
        )

        efficiency = entry.compute_efficiency_metrics()
        assert efficiency.perf_per_watt_tflops == pytest.approx(0.2, rel=0.01)
        assert efficiency.perf_per_mm2_tflops == pytest.approx(0.1, rel=0.01)
        assert efficiency.bandwidth_per_watt_gbps == pytest.approx(2.0, rel=0.01)

    def test_die_spec_transistor_density(self):
        """Test DieSpec transistor density calculation."""
        from embodied_schemas.gpu import DieSpec, Foundry

        die = DieSpec(
            gpu_name="AD102",
            architecture="Ada Lovelace",
            foundry=Foundry.TSMC,
            process_nm=5,
            transistors_billion=76.3,
            die_size_mm2=608.5,
        )
        # 76.3B transistors / 608.5 mm² = 125.4 MTx/mm²
        assert die.transistor_density_mtx_mm2 == pytest.approx(125.4, rel=0.01)

    def test_gpu_entry_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        from embodied_schemas.gpu import (
            GPUEntry,
            GPUVendor,
            DieSpec,
            ComputeResources,
            ClockSpeeds,
            MemorySpec,
            MemoryType,
            TheoreticalPerformance,
            PowerSpec,
            MarketInfo,
            TargetMarket,
            Foundry,
        )

        with pytest.raises(ValidationError):
            GPUEntry(
                id="test_gpu",
                name="Test GPU",
                vendor=GPUVendor.NVIDIA,
                die=DieSpec(
                    gpu_name="TEST",
                    architecture="Test",
                    foundry=Foundry.TSMC,
                    process_nm=5,
                    transistors_billion=1.0,
                    die_size_mm2=100.0,
                ),
                compute=ComputeResources(shaders=1000, tmus=32, rops=16),
                clocks=ClockSpeeds(base_clock_mhz=1000, boost_clock_mhz=1500, memory_clock_mhz=500),
                memory=MemorySpec(
                    memory_size_gb=4.0,
                    memory_type=MemoryType.GDDR6,
                    memory_bus_bits=128,
                    memory_bandwidth_gbps=100.0,
                ),
                performance=TheoreticalPerformance(
                    fp32_tflops=5.0, pixel_rate_gpixels=20.0, texture_rate_gtexels=40.0
                ),
                power=PowerSpec(tdp_watts=75),
                market=MarketInfo(launch_date="2024-01-01", target_market=TargetMarket.EMBEDDED),
                last_updated="2024-12-21",
                unknown_field="should_fail",  # This should cause validation error
            )
