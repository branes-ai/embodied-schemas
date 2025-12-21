"""Benchmark result schemas for performance measurements.

Defines data structures for storing and querying benchmark results
from model execution on various hardware platforms.
"""

from enum import Enum
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Verdict for a benchmark or analysis result."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class Confidence(str, Enum):
    """Confidence level in a result."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LatencyMetrics(BaseModel):
    """Latency measurement results."""

    mean_ms: float = Field(..., description="Mean latency in milliseconds")
    std_ms: float | None = Field(None, description="Standard deviation in ms")
    min_ms: float | None = Field(None, description="Minimum latency in ms")
    max_ms: float | None = Field(None, description="Maximum latency in ms")
    p50_ms: float | None = Field(None, description="50th percentile (median) in ms")
    p90_ms: float | None = Field(None, description="90th percentile in ms")
    p95_ms: float | None = Field(None, description="95th percentile in ms")
    p99_ms: float | None = Field(None, description="99th percentile in ms")


class ThroughputMetrics(BaseModel):
    """Throughput measurement results."""

    fps: float = Field(..., description="Frames per second")
    samples_per_second: float | None = Field(None, description="Samples processed per second")
    tokens_per_second: float | None = Field(None, description="Tokens per second (for LLMs)")
    batch_size: int = Field(1, description="Batch size used for measurement")


class PowerMetrics(BaseModel):
    """Power consumption measurements."""

    mean_watts: float = Field(..., description="Mean power consumption in watts")
    peak_watts: float | None = Field(None, description="Peak power consumption in watts")
    idle_watts: float | None = Field(None, description="Idle power consumption in watts")
    energy_per_inference_mj: float | None = Field(
        None, description="Energy per inference in millijoules"
    )
    energy_per_token_mj: float | None = Field(
        None, description="Energy per token in millijoules (for LLMs)"
    )


class MemoryMetrics(BaseModel):
    """Memory usage measurements."""

    model_mb: float = Field(..., description="Model size in memory in MB")
    peak_mb: float | None = Field(None, description="Peak memory usage in MB")
    allocated_mb: float | None = Field(None, description="Allocated memory in MB")
    reserved_mb: float | None = Field(None, description="Reserved memory in MB")
    gpu_utilization_percent: float | None = Field(
        None, description="GPU utilization percentage"
    )


class ThermalMetrics(BaseModel):
    """Thermal measurements."""

    start_temp_c: float | None = Field(None, description="Starting temperature in Celsius")
    end_temp_c: float | None = Field(None, description="Ending temperature in Celsius")
    peak_temp_c: float | None = Field(None, description="Peak temperature in Celsius")
    throttled: bool = Field(False, description="Whether thermal throttling occurred")
    throttle_time_percent: float | None = Field(
        None, description="Percentage of time spent throttled"
    )


class AccuracyMetrics(BaseModel):
    """Accuracy verification results."""

    dataset: str = Field(..., description="Dataset used for verification")
    samples: int | None = Field(None, description="Number of samples evaluated")
    map_50: float | None = Field(None, description="mAP@0.5")
    map_50_95: float | None = Field(None, description="mAP@0.5:0.95")
    expected_map_50: float | None = Field(None, description="Expected mAP@0.5")
    accuracy_delta: float | None = Field(
        None, description="Delta from expected accuracy"
    )
    within_tolerance: bool | None = Field(
        None, description="Whether accuracy is within acceptable tolerance"
    )


class BenchmarkConditions(BaseModel):
    """Conditions under which benchmark was run."""

    power_mode: str | None = Field(None, description="Power mode used: 10W, 15W, MAX-N, etc.")
    batch_size: int = Field(1, description="Batch size")
    input_shape: list[int] | None = Field(None, description="Input shape used")
    warmup_iterations: int = Field(50, description="Number of warmup iterations")
    test_iterations: int = Field(1000, description="Number of test iterations")
    ambient_temp_c: float | None = Field(None, description="Ambient temperature in Celsius")
    cooling: str | None = Field(None, description="Cooling method: passive, active, liquid")
    os_version: str | None = Field(None, description="Operating system version")
    driver_version: str | None = Field(None, description="GPU/NPU driver version")
    runtime_version: str | None = Field(None, description="Inference runtime version")


class Reproducibility(BaseModel):
    """Information for reproducing the benchmark."""

    script: str | None = Field(None, description="Script path or command")
    commit: str | None = Field(None, description="Git commit hash")
    repository: str | None = Field(None, description="Repository URL")
    container_image: str | None = Field(None, description="Container image used")
    environment_file: str | None = Field(None, description="Environment file path")


class BenchmarkResult(BaseModel):
    """Complete benchmark result entry."""

    # Identity
    id: str = Field(..., description="Unique benchmark ID")
    model_id: str = Field(..., description="Model ID from model catalog")
    hardware_id: str = Field(..., description="Hardware ID from hardware catalog")
    variant: str = Field(..., description="Model variant: fp32, fp16, int8, etc.")

    # Conditions
    conditions: BenchmarkConditions = Field(..., description="Benchmark conditions")

    # Results
    latency: LatencyMetrics | None = Field(None, description="Latency measurements")
    throughput: ThroughputMetrics | None = Field(None, description="Throughput measurements")
    power: PowerMetrics | None = Field(None, description="Power measurements")
    memory: MemoryMetrics | None = Field(None, description="Memory measurements")
    thermal: ThermalMetrics | None = Field(None, description="Thermal measurements")
    accuracy: AccuracyMetrics | None = Field(None, description="Accuracy verification")

    # Verdict (for use case matching)
    verdict: Verdict | None = Field(
        None, description="Overall verdict against requirements"
    )
    confidence: Confidence = Field(
        Confidence.HIGH, description="Confidence in the results"
    )
    summary: str | None = Field(
        None, description="One-line summary of the result"
    )
    suggestion: str | None = Field(
        None, description="Suggestion if verdict is not PASS"
    )

    # Reproducibility
    reproducibility: Reproducibility | None = Field(
        None, description="Reproducibility information"
    )

    # Metadata
    timestamp: str = Field(..., description="Benchmark timestamp (ISO 8601)")
    benchmark_version: str = Field("1.0.0", description="Benchmark harness version")
    notes: str | None = Field(None, description="Additional notes")

    model_config = {"extra": "forbid"}


class AnalysisResult(BaseModel):
    """Result from an analysis tool (estimated, not measured).

    Used for tool outputs before actual benchmarking.
    """

    # What was analyzed
    model_id: str = Field(..., description="Model ID")
    hardware_id: str = Field(..., description="Hardware ID")
    variant: str | None = Field(None, description="Model variant if applicable")

    # Verdict
    verdict: Verdict = Field(..., description="Analysis verdict")
    confidence: Confidence = Field(..., description="Confidence in the analysis")
    summary: str = Field(..., description="One-line summary")

    # Metric (the specific thing analyzed)
    metric: str = Field(..., description="Metric name: latency, power, memory, etc.")
    measured: float | None = Field(None, description="Measured or estimated value")
    required: float | None = Field(None, description="Required value (from use case)")
    unit: str = Field(..., description="Unit: ms, watts, mb, etc.")
    margin: str | None = Field(
        None, description="Margin from requirement: +35.7%, -10.2%, etc."
    )

    # Evidence
    evidence: str | None = Field(None, description="How this was determined")

    # Next steps
    suggestion: str | None = Field(
        None, description="Actionable suggestion if not PASS"
    )

    model_config = {"extra": "forbid"}
