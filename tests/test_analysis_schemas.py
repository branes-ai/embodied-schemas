"""Tests for graph analysis schema models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from embodied_schemas import (
    Verdict,
    Confidence,
    Bottleneck,
    RooflineResult,
    EnergyResult,
    MemoryResult,
    ConcurrencyResult,
    SubgraphBreakdown,
    GraphAnalysisResult,
    ComparisonResult,
    BatchSweepResult,
)


class TestBottleneckEnum:
    """Tests for Bottleneck enum."""

    def test_bottleneck_values(self):
        """Test Bottleneck enum values."""
        assert Bottleneck.COMPUTE_BOUND.value == "compute-bound"
        assert Bottleneck.MEMORY_BOUND.value == "memory-bound"
        assert Bottleneck.BALANCED.value == "balanced"


class TestRooflineResult:
    """Tests for RooflineResult schema."""

    def test_minimal_roofline(self):
        """Test RooflineResult with required fields only."""
        result = RooflineResult(
            latency_ms=10.5,
            bottleneck=Bottleneck.COMPUTE_BOUND,
            utilization_pct=85.0,
            arithmetic_intensity=150.0,
            peak_flops=312e12,
            peak_bandwidth_gbps=2000.0,
            achieved_flops=265e12,
            achieved_bandwidth_gbps=1700.0,
        )
        assert result.latency_ms == 10.5
        assert result.bottleneck == Bottleneck.COMPUTE_BOUND
        assert result.ridge_point is None

    def test_full_roofline(self):
        """Test RooflineResult with all fields."""
        result = RooflineResult(
            latency_ms=10.5,
            bottleneck=Bottleneck.MEMORY_BOUND,
            utilization_pct=45.0,
            arithmetic_intensity=25.0,
            peak_flops=312e12,
            peak_bandwidth_gbps=2000.0,
            achieved_flops=50e12,
            achieved_bandwidth_gbps=1800.0,
            ridge_point=156.0,
        )
        assert result.ridge_point == 156.0
        assert result.bottleneck == Bottleneck.MEMORY_BOUND

    def test_roofline_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            RooflineResult(
                latency_ms=10.5,
                bottleneck=Bottleneck.BALANCED,
                utilization_pct=50.0,
                arithmetic_intensity=100.0,
                peak_flops=100e12,
                peak_bandwidth_gbps=1000.0,
                achieved_flops=50e12,
                achieved_bandwidth_gbps=500.0,
                unknown_field="should_fail",
            )


class TestEnergyResult:
    """Tests for EnergyResult schema."""

    def test_minimal_energy(self):
        """Test EnergyResult with required fields only."""
        result = EnergyResult(
            total_energy_mj=150.0,
            compute_energy_mj=80.0,
            memory_energy_mj=50.0,
            static_energy_mj=20.0,
            average_power_w=200.0,
        )
        assert result.total_energy_mj == 150.0
        assert result.power_gating_enabled is False
        assert result.power_gating_savings_mj is None

    def test_full_energy(self):
        """Test EnergyResult with all fields."""
        result = EnergyResult(
            total_energy_mj=150.0,
            compute_energy_mj=80.0,
            memory_energy_mj=50.0,
            static_energy_mj=20.0,
            average_power_w=200.0,
            peak_power_w=350.0,
            tdp_w=450.0,
            energy_efficiency_gflops_per_w=1.5,
            power_gating_enabled=True,
            power_gating_savings_mj=15.0,
        )
        assert result.power_gating_enabled is True
        assert result.power_gating_savings_mj == 15.0
        assert result.tdp_w == 450.0

    def test_energy_components_sum(self):
        """Test that energy components can sum to total."""
        result = EnergyResult(
            total_energy_mj=150.0,
            compute_energy_mj=80.0,
            memory_energy_mj=50.0,
            static_energy_mj=20.0,
            average_power_w=200.0,
        )
        component_sum = (
            result.compute_energy_mj
            + result.memory_energy_mj
            + result.static_energy_mj
        )
        assert component_sum == result.total_energy_mj


class TestMemoryResult:
    """Tests for MemoryResult schema."""

    def test_minimal_memory(self):
        """Test MemoryResult with required fields only."""
        result = MemoryResult(
            peak_memory_mb=1024.0,
            weights_mb=400.0,
            activations_mb=600.0,
            fits_in_l2=False,
            fits_in_device_memory=True,
        )
        assert result.peak_memory_mb == 1024.0
        assert result.fits_in_l2 is False
        assert result.workspace_mb is None

    def test_full_memory(self):
        """Test MemoryResult with all fields."""
        result = MemoryResult(
            peak_memory_mb=1024.0,
            weights_mb=400.0,
            activations_mb=500.0,
            workspace_mb=124.0,
            fits_in_l2=False,
            fits_in_device_memory=True,
            l2_cache_mb=50.0,
            device_memory_mb=80000.0,
            memory_utilization_pct=1.28,
        )
        assert result.workspace_mb == 124.0
        assert result.l2_cache_mb == 50.0
        assert result.memory_utilization_pct == pytest.approx(1.28)


class TestConcurrencyResult:
    """Tests for ConcurrencyResult schema."""

    def test_concurrency_all_none(self):
        """Test ConcurrencyResult with all optional fields."""
        result = ConcurrencyResult()
        assert result.data_parallelism is None
        assert result.sm_utilization_pct is None

    def test_concurrency_gpu(self):
        """Test ConcurrencyResult for GPU mapping."""
        result = ConcurrencyResult(
            data_parallelism=64,
            tensor_parallelism=1024,
            sm_utilization_pct=92.5,
        )
        assert result.sm_utilization_pct == 92.5

    def test_concurrency_cpu(self):
        """Test ConcurrencyResult for CPU mapping."""
        result = ConcurrencyResult(
            data_parallelism=32,
            core_utilization_pct=75.0,
            vector_utilization_pct=80.0,
        )
        assert result.core_utilization_pct == 75.0


class TestSubgraphBreakdown:
    """Tests for SubgraphBreakdown schema."""

    def test_subgraph_breakdown(self):
        """Test SubgraphBreakdown creation."""
        breakdown = SubgraphBreakdown(
            subgraph_id="sg_0",
            op_types=["conv2d", "relu", "batch_norm"],
            flops=1000000,
            bytes_transferred=500000,
            latency_ms=0.5,
            energy_mj=0.01,
            bottleneck=Bottleneck.COMPUTE_BOUND,
        )
        assert breakdown.subgraph_id == "sg_0"
        assert "conv2d" in breakdown.op_types
        assert breakdown.bottleneck == Bottleneck.COMPUTE_BOUND


class TestGraphAnalysisResult:
    """Tests for GraphAnalysisResult schema."""

    @pytest.fixture
    def sample_roofline(self):
        """Create a sample RooflineResult."""
        return RooflineResult(
            latency_ms=10.5,
            bottleneck=Bottleneck.COMPUTE_BOUND,
            utilization_pct=85.0,
            arithmetic_intensity=150.0,
            peak_flops=312e12,
            peak_bandwidth_gbps=2000.0,
            achieved_flops=265e12,
            achieved_bandwidth_gbps=1700.0,
        )

    @pytest.fixture
    def sample_energy(self):
        """Create a sample EnergyResult."""
        return EnergyResult(
            total_energy_mj=150.0,
            compute_energy_mj=80.0,
            memory_energy_mj=50.0,
            static_energy_mj=20.0,
            average_power_w=200.0,
        )

    @pytest.fixture
    def sample_memory(self):
        """Create a sample MemoryResult."""
        return MemoryResult(
            peak_memory_mb=1024.0,
            weights_mb=400.0,
            activations_mb=600.0,
            fits_in_l2=False,
            fits_in_device_memory=True,
        )

    def test_minimal_graph_analysis(
        self, sample_roofline, sample_energy, sample_memory
    ):
        """Test GraphAnalysisResult with required fields only."""
        result = GraphAnalysisResult(
            verdict=Verdict.PASS,
            confidence=Confidence.HIGH,
            summary="ResNet-18 meets latency target of 20ms on H100",
            model_id="resnet18",
            hardware_id="h100_sxm5_80gb",
            latency_ms=10.5,
            throughput_fps=95.2,
            energy_per_inference_mj=150.0,
            peak_memory_mb=1024.0,
            roofline=sample_roofline,
            energy=sample_energy,
            memory=sample_memory,
        )
        assert result.verdict == Verdict.PASS
        assert result.confidence == Confidence.HIGH
        assert result.batch_size == 1  # default
        assert result.precision == "fp32"  # default
        assert result.suggestions == []  # default
        assert result.warnings == []  # default

    def test_full_graph_analysis(
        self, sample_roofline, sample_energy, sample_memory
    ):
        """Test GraphAnalysisResult with all fields."""
        result = GraphAnalysisResult(
            verdict=Verdict.FAIL,
            confidence=Confidence.MEDIUM,
            summary="ResNet-50 exceeds latency target of 5ms on Jetson Orin",
            model_id="resnet50",
            hardware_id="jetson_orin_agx_64gb",
            batch_size=4,
            precision="fp16",
            timestamp=datetime(2024, 12, 23, 10, 30, 0),
            analyzer_version="2.0.0",
            latency_ms=8.3,
            throughput_fps=481.9,
            energy_per_inference_mj=45.0,
            peak_memory_mb=512.0,
            roofline=sample_roofline,
            energy=sample_energy,
            memory=sample_memory,
            concurrency=ConcurrencyResult(sm_utilization_pct=88.0),
            subgraphs=[
                SubgraphBreakdown(
                    subgraph_id="sg_0",
                    op_types=["conv2d"],
                    flops=1000000,
                    bytes_transferred=500000,
                    latency_ms=2.0,
                    energy_mj=10.0,
                    bottleneck=Bottleneck.COMPUTE_BOUND,
                )
            ],
            total_subgraphs=5,
            fusion_ratio=0.8,
            constraint_metric="latency",
            constraint_threshold=5.0,
            constraint_actual=8.3,
            constraint_margin_pct=-66.0,
            suggestions=["Consider using INT8 quantization", "Try batch size 1"],
            warnings=["Estimate based on roofline model, not measured"],
        )
        assert result.verdict == Verdict.FAIL
        assert result.batch_size == 4
        assert result.precision == "fp16"
        assert len(result.suggestions) == 2
        assert result.constraint_margin_pct == -66.0

    def test_graph_analysis_rejects_extra_fields(
        self, sample_roofline, sample_energy, sample_memory
    ):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            GraphAnalysisResult(
                verdict=Verdict.PASS,
                confidence=Confidence.HIGH,
                summary="Test",
                model_id="test",
                hardware_id="test",
                latency_ms=10.0,
                throughput_fps=100.0,
                energy_per_inference_mj=10.0,
                peak_memory_mb=100.0,
                roofline=sample_roofline,
                energy=sample_energy,
                memory=sample_memory,
                unknown_field="should_fail",
            )


class TestComparisonResult:
    """Tests for ComparisonResult schema."""

    @pytest.fixture
    def sample_analysis_results(self):
        """Create sample GraphAnalysisResult instances."""
        roofline = RooflineResult(
            latency_ms=10.5,
            bottleneck=Bottleneck.COMPUTE_BOUND,
            utilization_pct=85.0,
            arithmetic_intensity=150.0,
            peak_flops=312e12,
            peak_bandwidth_gbps=2000.0,
            achieved_flops=265e12,
            achieved_bandwidth_gbps=1700.0,
        )
        energy = EnergyResult(
            total_energy_mj=150.0,
            compute_energy_mj=80.0,
            memory_energy_mj=50.0,
            static_energy_mj=20.0,
            average_power_w=200.0,
        )
        memory = MemoryResult(
            peak_memory_mb=1024.0,
            weights_mb=400.0,
            activations_mb=600.0,
            fits_in_l2=False,
            fits_in_device_memory=True,
        )

        h100_result = GraphAnalysisResult(
            verdict=Verdict.PASS,
            confidence=Confidence.HIGH,
            summary="ResNet-18 on H100: 10.5ms",
            model_id="resnet18",
            hardware_id="h100_sxm5_80gb",
            latency_ms=10.5,
            throughput_fps=95.2,
            energy_per_inference_mj=150.0,
            peak_memory_mb=1024.0,
            roofline=roofline,
            energy=energy,
            memory=memory,
        )

        orin_result = GraphAnalysisResult(
            verdict=Verdict.PASS,
            confidence=Confidence.HIGH,
            summary="ResNet-18 on Orin: 25.0ms",
            model_id="resnet18",
            hardware_id="jetson_orin_agx_64gb",
            latency_ms=25.0,
            throughput_fps=40.0,
            energy_per_inference_mj=50.0,
            peak_memory_mb=512.0,
            roofline=roofline,
            energy=energy,
            memory=memory,
        )

        return [h100_result, orin_result]

    def test_comparison_result(self, sample_analysis_results):
        """Test ComparisonResult creation."""
        result = ComparisonResult(
            verdict=Verdict.PASS,
            confidence=Confidence.HIGH,
            summary="H100 is fastest, Orin is most efficient for ResNet-18",
            model_id="resnet18",
            hardware_ids=["h100_sxm5_80gb", "jetson_orin_agx_64gb"],
            fastest_hardware="h100_sxm5_80gb",
            most_efficient_hardware="jetson_orin_agx_64gb",
            results=sample_analysis_results,
            recommended_hardware="jetson_orin_agx_64gb",
            recommendation_reason="Best energy efficiency for edge deployment",
        )
        assert result.fastest_hardware == "h100_sxm5_80gb"
        assert result.most_efficient_hardware == "jetson_orin_agx_64gb"
        assert len(result.results) == 2


class TestBatchSweepResult:
    """Tests for BatchSweepResult schema."""

    @pytest.fixture
    def sample_batch_results(self):
        """Create sample GraphAnalysisResult instances for batch sweep."""
        roofline = RooflineResult(
            latency_ms=10.5,
            bottleneck=Bottleneck.COMPUTE_BOUND,
            utilization_pct=85.0,
            arithmetic_intensity=150.0,
            peak_flops=312e12,
            peak_bandwidth_gbps=2000.0,
            achieved_flops=265e12,
            achieved_bandwidth_gbps=1700.0,
        )
        energy = EnergyResult(
            total_energy_mj=150.0,
            compute_energy_mj=80.0,
            memory_energy_mj=50.0,
            static_energy_mj=20.0,
            average_power_w=200.0,
        )
        memory = MemoryResult(
            peak_memory_mb=1024.0,
            weights_mb=400.0,
            activations_mb=600.0,
            fits_in_l2=False,
            fits_in_device_memory=True,
        )

        results = []
        for batch_size, latency, throughput in [(1, 10.0, 100), (4, 15.0, 267), (16, 40.0, 400)]:
            results.append(
                GraphAnalysisResult(
                    verdict=Verdict.PASS,
                    confidence=Confidence.HIGH,
                    summary=f"Batch {batch_size}: {latency}ms, {throughput} FPS",
                    model_id="resnet18",
                    hardware_id="h100_sxm5_80gb",
                    batch_size=batch_size,
                    latency_ms=latency,
                    throughput_fps=float(throughput),
                    energy_per_inference_mj=150.0 / batch_size,
                    peak_memory_mb=1024.0 * batch_size / 4,
                    roofline=roofline,
                    energy=energy,
                    memory=memory,
                )
            )
        return results

    def test_batch_sweep_result(self, sample_batch_results):
        """Test BatchSweepResult creation."""
        result = BatchSweepResult(
            verdict=Verdict.PASS,
            confidence=Confidence.HIGH,
            summary="Batch 16 achieves maximum throughput of 400 FPS",
            model_id="resnet18",
            hardware_id="h100_sxm5_80gb",
            batch_sizes=[1, 4, 16],
            optimal_throughput_batch=16,
            optimal_latency_batch=1,
            optimal_efficiency_batch=16,
            results=sample_batch_results,
            throughput_scales_linearly=False,
            memory_limited_batch=64,
        )
        assert result.optimal_throughput_batch == 16
        assert result.optimal_latency_batch == 1
        assert len(result.results) == 3
        assert result.throughput_scales_linearly is False


class TestVerdictPattern:
    """Tests demonstrating the verdict-first pattern."""

    def test_pass_verdict_with_margin(self):
        """Test PASS verdict with positive margin."""
        roofline = RooflineResult(
            latency_ms=8.0,
            bottleneck=Bottleneck.COMPUTE_BOUND,
            utilization_pct=85.0,
            arithmetic_intensity=150.0,
            peak_flops=312e12,
            peak_bandwidth_gbps=2000.0,
            achieved_flops=265e12,
            achieved_bandwidth_gbps=1700.0,
        )
        energy = EnergyResult(
            total_energy_mj=150.0,
            compute_energy_mj=80.0,
            memory_energy_mj=50.0,
            static_energy_mj=20.0,
            average_power_w=200.0,
        )
        memory = MemoryResult(
            peak_memory_mb=1024.0,
            weights_mb=400.0,
            activations_mb=600.0,
            fits_in_l2=False,
            fits_in_device_memory=True,
        )

        result = GraphAnalysisResult(
            verdict=Verdict.PASS,
            confidence=Confidence.HIGH,
            summary="ResNet-18 meets 10ms latency target with 20% headroom",
            model_id="resnet18",
            hardware_id="h100_sxm5_80gb",
            latency_ms=8.0,
            throughput_fps=125.0,
            energy_per_inference_mj=150.0,
            peak_memory_mb=1024.0,
            roofline=roofline,
            energy=energy,
            memory=memory,
            constraint_metric="latency",
            constraint_threshold=10.0,
            constraint_actual=8.0,
            constraint_margin_pct=20.0,  # 20% headroom
        )

        assert result.verdict == Verdict.PASS
        assert result.constraint_margin_pct > 0  # Positive = headroom

    def test_fail_verdict_with_suggestion(self):
        """Test FAIL verdict with actionable suggestion."""
        roofline = RooflineResult(
            latency_ms=15.0,
            bottleneck=Bottleneck.MEMORY_BOUND,
            utilization_pct=45.0,
            arithmetic_intensity=25.0,
            peak_flops=312e12,
            peak_bandwidth_gbps=2000.0,
            achieved_flops=50e12,
            achieved_bandwidth_gbps=1800.0,
        )
        energy = EnergyResult(
            total_energy_mj=150.0,
            compute_energy_mj=80.0,
            memory_energy_mj=50.0,
            static_energy_mj=20.0,
            average_power_w=200.0,
        )
        memory = MemoryResult(
            peak_memory_mb=1024.0,
            weights_mb=400.0,
            activations_mb=600.0,
            fits_in_l2=False,
            fits_in_device_memory=True,
        )

        result = GraphAnalysisResult(
            verdict=Verdict.FAIL,
            confidence=Confidence.HIGH,
            summary="ResNet-50 exceeds 10ms latency target by 50%",
            model_id="resnet50",
            hardware_id="jetson_orin_nano_8gb",
            latency_ms=15.0,
            throughput_fps=66.7,
            energy_per_inference_mj=50.0,
            peak_memory_mb=512.0,
            roofline=roofline,
            energy=energy,
            memory=memory,
            constraint_metric="latency",
            constraint_threshold=10.0,
            constraint_actual=15.0,
            constraint_margin_pct=-50.0,  # -50% = 50% over budget
            suggestions=[
                "Consider ResNet-18 for this latency target",
                "Try INT8 quantization to reduce memory bandwidth",
            ],
        )

        assert result.verdict == Verdict.FAIL
        assert result.constraint_margin_pct < 0  # Negative = exceeded
        assert len(result.suggestions) > 0
