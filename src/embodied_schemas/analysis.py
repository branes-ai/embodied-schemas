"""Graph analysis result schemas for DNN performance estimation.

Defines verdict-first Pydantic models for graph analysis outputs,
designed for consumption by LLM-based agentic workflows.

These schemas represent ESTIMATED performance (from roofline models,
energy models, etc.) as opposed to MEASURED performance (from benchmarks).
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from embodied_schemas.benchmarks import Confidence, Verdict


class Bottleneck(str, Enum):
    """Classification of performance bottleneck."""

    COMPUTE_BOUND = "compute-bound"
    MEMORY_BOUND = "memory-bound"
    BALANCED = "balanced"


class RooflineResult(BaseModel):
    """Roofline model analysis results.

    Provides latency estimation and bottleneck classification
    based on the roofline performance model.
    """

    latency_ms: float = Field(..., description="Estimated latency in milliseconds")
    bottleneck: Bottleneck = Field(..., description="Performance bottleneck type")
    utilization_pct: float = Field(
        ..., description="Hardware utilization percentage (0-100)"
    )
    arithmetic_intensity: float = Field(
        ..., description="Arithmetic intensity (FLOPs/byte)"
    )

    # Hardware limits
    peak_flops: float = Field(..., description="Peak hardware FLOPS capacity")
    peak_bandwidth_gbps: float = Field(
        ..., description="Peak memory bandwidth in GB/s"
    )

    # Achieved performance
    achieved_flops: float = Field(..., description="Achieved FLOPS for this workload")
    achieved_bandwidth_gbps: float = Field(
        ..., description="Achieved memory bandwidth in GB/s"
    )

    # Ridge point (where compute = memory bound)
    ridge_point: float | None = Field(
        None, description="Ridge point arithmetic intensity (FLOPs/byte)"
    )

    model_config = {"extra": "forbid"}


class EnergyResult(BaseModel):
    """Three-component energy model results.

    Breaks down energy consumption into compute, memory, and static components.
    """

    total_energy_mj: float = Field(
        ..., description="Total energy per inference in millijoules"
    )
    compute_energy_mj: float = Field(
        ..., description="Energy from compute operations in mJ"
    )
    memory_energy_mj: float = Field(
        ..., description="Energy from memory transfers in mJ"
    )
    static_energy_mj: float = Field(
        ..., description="Static/leakage energy during execution in mJ"
    )

    # Power metrics
    average_power_w: float = Field(
        ..., description="Average power consumption in watts"
    )
    peak_power_w: float | None = Field(
        None, description="Peak power consumption in watts"
    )
    tdp_w: float | None = Field(None, description="Thermal design power in watts")

    # Efficiency
    energy_efficiency_gflops_per_w: float | None = Field(
        None, description="Energy efficiency in GFLOPS/watt"
    )

    # Power gating (optional)
    power_gating_enabled: bool = Field(
        False, description="Whether power gating analysis was performed"
    )
    power_gating_savings_mj: float | None = Field(
        None, description="Energy saved by power gating in mJ"
    )

    model_config = {"extra": "forbid"}


class MemoryResult(BaseModel):
    """Memory footprint analysis results.

    Analyzes memory requirements and hardware fit.
    """

    peak_memory_mb: float = Field(
        ..., description="Peak memory usage in MB (weights + activations + workspace)"
    )
    weights_mb: float = Field(..., description="Model weights memory in MB")
    activations_mb: float = Field(..., description="Peak activation memory in MB")
    workspace_mb: float | None = Field(
        None, description="Additional workspace memory in MB"
    )

    # Hardware fit analysis
    fits_in_l2: bool = Field(
        ..., description="Whether working set fits in L2 cache"
    )
    fits_in_device_memory: bool = Field(
        ..., description="Whether model fits in device memory"
    )

    # Hardware capacities
    l2_cache_mb: float | None = Field(None, description="L2 cache size in MB")
    device_memory_mb: float | None = Field(
        None, description="Total device memory in MB"
    )

    # Memory pressure
    memory_utilization_pct: float | None = Field(
        None, description="Percentage of device memory used (0-100)"
    )

    model_config = {"extra": "forbid"}


class ConcurrencyResult(BaseModel):
    """Multi-level parallelism analysis results."""

    # Parallelism levels
    data_parallelism: int | None = Field(
        None, description="Available data parallelism (batch dimension)"
    )
    tensor_parallelism: int | None = Field(
        None, description="Available tensor parallelism (spatial dimensions)"
    )
    pipeline_parallelism: int | None = Field(
        None, description="Available pipeline parallelism (layer-wise)"
    )

    # Hardware mapping
    sm_utilization_pct: float | None = Field(
        None, description="GPU SM utilization percentage"
    )
    core_utilization_pct: float | None = Field(
        None, description="CPU core utilization percentage"
    )
    vector_utilization_pct: float | None = Field(
        None, description="Vector/SIMD unit utilization percentage"
    )

    model_config = {"extra": "forbid"}


class SubgraphBreakdown(BaseModel):
    """Per-subgraph analysis breakdown."""

    subgraph_id: str = Field(..., description="Subgraph identifier")
    op_types: list[str] = Field(..., description="Operation types in this subgraph")
    flops: int = Field(..., description="FLOPs in this subgraph")
    bytes_transferred: int = Field(..., description="Bytes transferred")
    latency_ms: float = Field(..., description="Subgraph latency in ms")
    energy_mj: float = Field(..., description="Subgraph energy in mJ")
    bottleneck: Bottleneck = Field(..., description="Subgraph bottleneck type")

    model_config = {"extra": "forbid"}


class GraphAnalysisResult(BaseModel):
    """Verdict-first graph analysis output for agentic workflows.

    This is the top-level result schema that combines all analysis components
    with a clear verdict for LLM consumption.
    """

    # === Verdict (required - enables LLM trust) ===
    verdict: Verdict = Field(..., description="Overall analysis verdict")
    confidence: Confidence = Field(..., description="Confidence in the analysis")
    summary: str = Field(
        ..., description="One-sentence summary of what was checked and found"
    )

    # === Metadata ===
    model_id: str = Field(..., description="Model identifier")
    hardware_id: str = Field(..., description="Hardware identifier")
    batch_size: int = Field(1, description="Batch size used for analysis")
    precision: str = Field("fp32", description="Numeric precision: fp32, fp16, int8")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Analysis timestamp"
    )
    analyzer_version: str = Field("1.0.0", description="Analyzer version")

    # === Key Metrics (derived, for quick access) ===
    latency_ms: float = Field(..., description="Estimated latency in milliseconds")
    throughput_fps: float = Field(..., description="Estimated throughput in FPS")
    energy_per_inference_mj: float = Field(
        ..., description="Energy per inference in millijoules"
    )
    peak_memory_mb: float = Field(..., description="Peak memory usage in MB")

    # === Detailed Breakdowns ===
    roofline: RooflineResult = Field(..., description="Roofline model analysis")
    energy: EnergyResult = Field(..., description="Energy breakdown analysis")
    memory: MemoryResult = Field(..., description="Memory footprint analysis")
    concurrency: ConcurrencyResult | None = Field(
        None, description="Parallelism analysis (optional)"
    )

    # === Subgraph Breakdown (optional) ===
    subgraphs: list[SubgraphBreakdown] | None = Field(
        None, description="Per-subgraph breakdown"
    )
    total_subgraphs: int | None = Field(None, description="Number of subgraphs")
    fusion_ratio: float | None = Field(
        None, description="Fusion ratio (ops fused / total ops)"
    )

    # === Constraint Checking (optional) ===
    constraint_metric: str | None = Field(
        None, description="Metric being constrained: latency, power, memory"
    )
    constraint_threshold: float | None = Field(
        None, description="Required threshold value"
    )
    constraint_actual: float | None = Field(
        None, description="Actual value achieved"
    )
    constraint_margin_pct: float | None = Field(
        None, description="Margin from requirement: positive=headroom, negative=exceeded"
    )

    # === Recommendations ===
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable suggestions for improvement",
    )
    warnings: list[str] = Field(
        default_factory=list, description="Warnings about the analysis"
    )

    model_config = {"extra": "forbid"}


class ComparisonResult(BaseModel):
    """Result from comparing multiple hardware targets.

    Useful for hardware selection workflows.
    """

    # Verdict
    verdict: Verdict = Field(..., description="Comparison verdict")
    confidence: Confidence = Field(..., description="Confidence in comparison")
    summary: str = Field(..., description="One-sentence comparison summary")

    # What was compared
    model_id: str = Field(..., description="Model identifier")
    hardware_ids: list[str] = Field(..., description="Hardware platforms compared")
    batch_size: int = Field(1, description="Batch size used")
    precision: str = Field("fp32", description="Numeric precision")

    # Rankings
    fastest_hardware: str = Field(..., description="Hardware with lowest latency")
    most_efficient_hardware: str = Field(
        ..., description="Hardware with best energy efficiency"
    )
    best_value_hardware: str | None = Field(
        None, description="Hardware with best performance per dollar"
    )

    # Individual results
    results: list[GraphAnalysisResult] = Field(
        ..., description="Individual analysis results"
    )

    # Recommendation
    recommended_hardware: str = Field(
        ..., description="Recommended hardware for this workload"
    )
    recommendation_reason: str = Field(
        ..., description="Why this hardware is recommended"
    )

    model_config = {"extra": "forbid"}


class BatchSweepResult(BaseModel):
    """Result from analyzing multiple batch sizes.

    Useful for finding optimal batch size for throughput or latency targets.
    """

    # Verdict
    verdict: Verdict = Field(..., description="Sweep verdict")
    confidence: Confidence = Field(..., description="Confidence in sweep")
    summary: str = Field(..., description="One-sentence sweep summary")

    # What was swept
    model_id: str = Field(..., description="Model identifier")
    hardware_id: str = Field(..., description="Hardware identifier")
    batch_sizes: list[int] = Field(..., description="Batch sizes analyzed")
    precision: str = Field("fp32", description="Numeric precision")

    # Optimal points
    optimal_throughput_batch: int = Field(
        ..., description="Batch size for maximum throughput"
    )
    optimal_latency_batch: int = Field(
        ..., description="Batch size for minimum latency"
    )
    optimal_efficiency_batch: int = Field(
        ..., description="Batch size for best energy efficiency"
    )

    # Individual results
    results: list[GraphAnalysisResult] = Field(
        ..., description="Individual analysis results"
    )

    # Scaling analysis
    throughput_scales_linearly: bool = Field(
        ..., description="Whether throughput scales linearly with batch size"
    )
    memory_limited_batch: int | None = Field(
        None, description="Batch size where memory becomes limiting"
    )

    model_config = {"extra": "forbid"}
