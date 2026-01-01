"""Operator schemas for embodied AI software architectures.

Defines data structures for reusable operators in embodied AI systems,
including perception, state estimation, planning, control, and reasoning
components that compose into complete application architectures.
"""

from enum import Enum
from pydantic import BaseModel, Field


class OperatorCategory(str, Enum):
    """Categories of operators in embodied AI systems."""

    PERCEPTION = "perception"
    STATE_ESTIMATION = "state_estimation"
    PLANNING = "planning"
    CONTROL = "control"
    REASONING = "reasoning"
    INFRASTRUCTURE = "infrastructure"


class ImplType(str, Enum):
    """Implementation type for operators."""

    PYTORCH = "pytorch"
    NUMPY = "numpy"
    CPP = "cpp"
    CUDA = "cuda"
    PYTHON = "python"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class IOSpec(BaseModel):
    """Specification for an operator input or output."""

    dtype: str = Field(
        ..., description="Data type: uint8, float32, int64, object, etc."
    )
    shape: list[str | int] = Field(
        ...,
        description="Shape with named dims or concrete values, e.g., ['H', 'W', 3] or [1, 640, 640, 3]",
    )
    description: str | None = Field(None, description="Human-readable description")
    optional: bool = Field(False, description="Whether this input/output is optional")


class ConfigParam(BaseModel):
    """Configuration parameter specification."""

    type: str = Field(..., description="Parameter type: string, number, boolean, array")
    default: str | float | int | bool | list | None = Field(
        None, description="Default value"
    )
    enum: list[str | float | int] | None = Field(
        None, description="Allowed values for enum types"
    )
    description: str | None = Field(None, description="Parameter description")
    min: float | None = Field(None, description="Minimum value for numeric types")
    max: float | None = Field(None, description="Maximum value for numeric types")


class OperatorPerfProfile(BaseModel):
    """Performance characteristics on a specific hardware target."""

    hardware_id: str = Field(..., description="Reference to HardwareEntry ID")
    latency_ms: float | None = Field(None, description="Inference latency in milliseconds")
    memory_mb: float | None = Field(None, description="Memory footprint in MB")
    power_w: float | None = Field(None, description="Power consumption in watts")
    throughput_fps: float | None = Field(None, description="Throughput in frames per second")
    conditions: str | None = Field(
        None, description="Test conditions: batch_size, precision, resolution, etc."
    )


class OperatorEntry(BaseModel):
    """Complete operator catalog entry.

    Represents a reusable component in embodied AI software architectures,
    such as a detector, tracker, state estimator, or controller.
    """

    # Identity
    id: str = Field(..., description="Unique identifier, e.g., yolo_detector_n")
    name: str = Field(..., description="Human-readable name")
    category: OperatorCategory = Field(..., description="Operator category")
    subcategory: str | None = Field(
        None, description="Subcategory within category, e.g., 'detection', 'tracking'"
    )

    # Implementation
    impl_type: ImplType = Field(..., description="Primary implementation type")
    reference_impl: str | None = Field(
        None, description="Reference implementation URL (GitHub, paper, etc.)"
    )

    # Interface specification
    inputs: dict[str, IOSpec] = Field(
        ..., description="Input specifications keyed by port name"
    )
    outputs: dict[str, IOSpec] = Field(
        ..., description="Output specifications keyed by port name"
    )
    config_schema: dict[str, ConfigParam] | None = Field(
        None, description="Configuration parameters schema"
    )

    # Per-hardware performance profiles
    perf_profiles: list[OperatorPerfProfile] = Field(
        default_factory=list,
        description="Performance profiles on specific hardware targets",
    )

    # Reference characteristics (quick lookup)
    reference_hardware: str | None = Field(
        None, description="Hardware ID for reference performance values"
    )
    typical_latency_ms: float | None = Field(
        None, description="Typical latency on reference hardware"
    )
    typical_memory_mb: float | None = Field(
        None, description="Typical memory on reference hardware"
    )
    compute_flops: float | None = Field(
        None, description="Compute operations (FLOPs) per inference"
    )

    # Dependencies
    requires_gpu: bool = Field(False, description="Whether GPU is required")
    requires_model: str | None = Field(
        None, description="Required ML model ID (if wrapping a DNN)"
    )
    python_deps: list[str] = Field(
        default_factory=list, description="Python package dependencies with versions"
    )

    # Metadata
    tags: list[str] = Field(
        default_factory=list, description="Searchable tags: real-time, edge, etc."
    )
    papers: list[str] = Field(
        default_factory=list, description="Reference papers or citations"
    )
    last_updated: str | None = Field(None, description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}
