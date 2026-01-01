"""Scheduling analysis schemas for embodied AI architectures.

Defines verdict-first Pydantic models for architecture-level analysis,
including end-to-end latency, multi-rate scheduling feasibility,
and resource utilization across heterogeneous systems.
"""

from datetime import datetime
from pydantic import BaseModel, Field

from embodied_schemas.benchmarks import Confidence, Verdict


class OperatorTiming(BaseModel):
    """Timing analysis for a single operator instance."""

    operator_instance_id: str = Field(..., description="Instance ID in architecture")
    operator_id: str = Field(..., description="Reference to OperatorEntry ID")

    # Execution characteristics
    execution_target: str = Field(..., description="Where this runs: cpu, gpu, npu, etc.")
    latency_ms: float = Field(..., description="Execution latency in milliseconds")
    memory_mb: float = Field(..., description="Memory footprint in MB")
    power_w: float | None = Field(None, description="Power consumption in watts")

    # Rate analysis
    target_rate_hz: float | None = Field(None, description="Target execution rate")
    achievable_rate_hz: float | None = Field(
        None, description="Maximum achievable rate on this hardware"
    )
    rate_feasible: bool = Field(True, description="Whether target rate is achievable")

    # Bottleneck info
    is_critical_path: bool = Field(
        False, description="Whether this operator is on the critical path"
    )
    limiting_factor: str | None = Field(
        None, description="What limits this operator: compute, memory, dependency"
    )


class DataTransfer(BaseModel):
    """A data transfer between system components."""

    source: str = Field(..., description="Source component: host, gpu0, npu0, etc.")
    destination: str = Field(..., description="Destination component")
    size_mb: float = Field(..., description="Transfer size in MB")
    latency_ms: float = Field(..., description="Transfer latency in milliseconds")
    bandwidth_gbps: float | None = Field(None, description="Achieved bandwidth in GB/s")


class DataMovementAnalysis(BaseModel):
    """Analysis of data movement in the architecture."""

    # Transfers
    host_to_accel_transfers: list[DataTransfer] = Field(
        default_factory=list, description="Host to accelerator transfers"
    )
    accel_to_host_transfers: list[DataTransfer] = Field(
        default_factory=list, description="Accelerator to host transfers"
    )
    inter_operator_transfers: list[DataTransfer] = Field(
        default_factory=list, description="Between-operator transfers"
    )

    # Aggregate metrics
    total_transfer_mb: float = Field(..., description="Total data transferred in MB")
    total_transfer_latency_ms: float = Field(
        ..., description="Total time spent in transfers"
    )
    bandwidth_bottleneck: str | None = Field(
        None, description="Which link is the bottleneck, if any"
    )


class ExecutionTargetUtilization(BaseModel):
    """Utilization of a specific execution target (CPU, GPU, etc.)."""

    target_id: str = Field(..., description="Execution target: cpu, gpu, npu, etc.")
    hardware_name: str | None = Field(None, description="Hardware name if known")

    utilization_pct: float = Field(..., description="Utilization percentage (0-100)")
    memory_utilization_pct: float | None = Field(
        None, description="Memory utilization percentage"
    )
    power_w: float | None = Field(None, description="Power consumption in watts")

    # Operators on this target
    assigned_operators: list[str] = Field(
        default_factory=list, description="Operator instance IDs on this target"
    )


class RateFeasibility(BaseModel):
    """Feasibility analysis for a specific operator rate."""

    operator_instance_id: str = Field(..., description="Operator instance ID")
    target_rate_hz: float = Field(..., description="Target execution rate in Hz")
    achievable: bool = Field(..., description="Whether rate is achievable")
    actual_rate_hz: float | None = Field(None, description="Actual achievable rate")
    margin_pct: float | None = Field(
        None, description="Margin: positive=headroom, negative=exceeded"
    )
    limiting_factor: str | None = Field(
        None, description="What prevents higher rate: compute, memory, dependency"
    )


class SchedulingAnalysisResult(BaseModel):
    """Verdict-first scheduling feasibility analysis.

    Analyzes whether a multi-rate architecture can meet its timing
    requirements on a specific hardware target.
    """

    # === Verdict ===
    verdict: Verdict = Field(..., description="PASS if all rates are achievable")
    confidence: Confidence = Field(..., description="Confidence in the analysis")
    summary: str = Field(
        ..., description="One-sentence summary of scheduling feasibility"
    )

    # === Metadata ===
    architecture_id: str = Field(..., description="Architecture identifier")
    hardware_id: str = Field(..., description="Hardware platform identifier")
    variant_id: str | None = Field(None, description="Architecture variant if used")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Analysis timestamp"
    )

    # === Per-Target Utilization ===
    target_utilization: list[ExecutionTargetUtilization] = Field(
        default_factory=list, description="Utilization per execution target"
    )

    # === Rate Feasibility ===
    rate_analysis: list[RateFeasibility] = Field(
        default_factory=list, description="Per-operator rate feasibility"
    )
    all_rates_feasible: bool = Field(
        ..., description="Whether all operator rates are achievable"
    )

    # === Aggregate Metrics ===
    total_cpu_utilization_pct: float = Field(
        ..., description="Total CPU utilization percentage"
    )
    total_accelerator_utilization_pct: float | None = Field(
        None, description="Total accelerator utilization percentage"
    )
    worst_case_latency_ms: float = Field(
        ..., description="Worst-case end-to-end latency"
    )

    # === Suggestions ===
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions if scheduling is infeasible"
    )

    model_config = {"extra": "forbid"}


class ArchitectureAnalysisResult(BaseModel):
    """Verdict-first architecture analysis result.

    Comprehensive analysis of a software architecture on a hardware target,
    including end-to-end latency, power, memory, and scheduling.
    """

    # === Verdict ===
    verdict: Verdict = Field(..., description="Overall architecture verdict")
    confidence: Confidence = Field(..., description="Confidence in the analysis")
    summary: str = Field(
        ..., description="One-sentence summary of architecture analysis"
    )

    # === Metadata ===
    architecture_id: str = Field(..., description="Architecture identifier")
    architecture_name: str = Field(..., description="Human-readable architecture name")
    hardware_id: str = Field(..., description="Hardware platform identifier")
    hardware_name: str | None = Field(None, description="Human-readable hardware name")
    variant_id: str | None = Field(None, description="Architecture variant if used")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Analysis timestamp"
    )

    # === Key Metrics ===
    end_to_end_latency_ms: float = Field(
        ..., description="End-to-end latency (critical path)"
    )
    throughput_fps: float = Field(..., description="Achievable throughput in FPS")
    total_power_w: float = Field(..., description="Total power consumption in watts")
    total_memory_mb: float = Field(..., description="Total memory footprint in MB")

    # === Constraint Checking ===
    latency_target_ms: float | None = Field(
        None, description="Target latency from architecture spec"
    )
    latency_margin_pct: float | None = Field(
        None, description="Latency margin: positive=headroom, negative=exceeded"
    )
    power_budget_w: float | None = Field(
        None, description="Power budget from architecture spec"
    )
    power_margin_pct: float | None = Field(
        None, description="Power margin: positive=headroom, negative=exceeded"
    )

    # === Per-Operator Analysis ===
    operator_timings: list[OperatorTiming] = Field(
        default_factory=list, description="Per-operator timing analysis"
    )

    # === Critical Path ===
    critical_path: list[str] = Field(
        default_factory=list,
        description="Operator instance IDs on the critical path",
    )
    critical_path_latency_ms: float = Field(
        ..., description="Sum of latencies on critical path"
    )

    # === Bottleneck Identification ===
    bottleneck_operator: str | None = Field(
        None, description="Operator instance ID that limits throughput"
    )
    bottleneck_latency_ms: float | None = Field(
        None, description="Latency of bottleneck operator"
    )
    bottleneck_type: str | None = Field(
        None, description="Type of bottleneck: compute, memory, transfer"
    )

    # === Data Movement ===
    data_movement: DataMovementAnalysis | None = Field(
        None, description="Data movement analysis"
    )

    # === Scheduling Analysis ===
    scheduling: SchedulingAnalysisResult | None = Field(
        None, description="Multi-rate scheduling analysis"
    )

    # === Recommendations ===
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Warnings about the analysis"
    )

    model_config = {"extra": "forbid"}


class OperatorBenchmarkResult(BaseModel):
    """Measured benchmark result for a single operator."""

    operator_instance_id: str = Field(..., description="Instance ID in architecture")
    operator_id: str = Field(..., description="Reference to OperatorEntry ID")
    execution_target: str = Field(..., description="Where this ran: cpu, gpu, npu, etc.")

    # Measured timing (actual measurements)
    iterations: int = Field(..., description="Number of benchmark iterations")
    mean_latency_ms: float = Field(..., description="Mean measured latency in ms")
    std_latency_ms: float = Field(..., description="Standard deviation in ms")
    min_latency_ms: float = Field(..., description="Minimum latency in ms")
    max_latency_ms: float = Field(..., description="Maximum latency in ms")
    p50_latency_ms: float | None = Field(None, description="50th percentile (median)")
    p95_latency_ms: float | None = Field(None, description="95th percentile")
    p99_latency_ms: float | None = Field(None, description="99th percentile")

    # Memory
    peak_memory_mb: float | None = Field(None, description="Peak memory usage in MB")

    # Comparison with estimate
    estimated_latency_ms: float | None = Field(
        None, description="Previously estimated latency"
    )
    estimate_error_pct: float | None = Field(
        None, description="Error in estimate: (measured-estimated)/estimated * 100"
    )

    # Status
    success: bool = Field(True, description="Whether benchmark completed successfully")
    error_message: str | None = Field(None, description="Error if benchmark failed")


class ArchitectureBenchmarkResult(BaseModel):
    """Measured benchmark result for a complete architecture pipeline.

    Captures actual timing measurements from running the architecture,
    with per-operator breakdown and comparison to estimates.
    """

    # === Verdict ===
    verdict: Verdict = Field(..., description="Overall benchmark verdict")
    confidence: Confidence = Field(..., description="Confidence in measurements")
    summary: str = Field(..., description="One-sentence summary of benchmark results")

    # === Metadata ===
    architecture_id: str = Field(..., description="Architecture identifier")
    architecture_name: str = Field(..., description="Human-readable architecture name")
    hardware_id: str = Field(..., description="Hardware platform identifier")
    hardware_name: str | None = Field(None, description="Human-readable hardware name")
    variant_id: str | None = Field(None, description="Architecture variant if used")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Benchmark timestamp"
    )

    # === Benchmark Configuration ===
    iterations: int = Field(..., description="Number of pipeline iterations")
    warmup_iterations: int = Field(..., description="Warmup iterations (not counted)")
    input_source: str = Field(..., description="Input data source: synthetic, file, camera")

    # === Measured Metrics (End-to-End) ===
    measured_latency_ms: float = Field(..., description="Measured end-to-end latency")
    measured_latency_std_ms: float = Field(..., description="Standard deviation")
    measured_throughput_fps: float = Field(..., description="Measured throughput in FPS")
    measured_memory_mb: float | None = Field(None, description="Peak memory usage")
    measured_power_w: float | None = Field(None, description="Measured power consumption")

    # === Per-Operator Results ===
    operator_results: list[OperatorBenchmarkResult] = Field(
        default_factory=list, description="Per-operator benchmark results"
    )

    # === Comparison with Estimates ===
    estimated_latency_ms: float | None = Field(
        None, description="Previously estimated end-to-end latency"
    )
    latency_estimate_error_pct: float | None = Field(
        None, description="Error in latency estimate"
    )

    # === Constraint Checking ===
    latency_target_ms: float | None = Field(
        None, description="Target latency from architecture spec"
    )
    meets_latency_target: bool | None = Field(
        None, description="Whether measured latency meets target"
    )
    latency_margin_pct: float | None = Field(
        None, description="Margin from target: positive=headroom"
    )

    # === Bottleneck Analysis (from measurements) ===
    measured_bottleneck_operator: str | None = Field(
        None, description="Operator with highest measured latency"
    )
    measured_bottleneck_latency_ms: float | None = Field(
        None, description="Latency of bottleneck operator"
    )
    measured_bottleneck_contribution_pct: float | None = Field(
        None, description="Percentage of total latency from bottleneck"
    )

    # === Warnings and Notes ===
    warnings: list[str] = Field(
        default_factory=list, description="Warnings about the benchmark"
    )
    notes: str | None = Field(None, description="Additional notes")

    model_config = {"extra": "forbid"}
