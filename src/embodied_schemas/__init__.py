"""Embodied Schemas - Shared schemas and data for embodied AI codesign.

This package provides:
- Pydantic models for hardware, models, sensors, use cases, and benchmarks
- Factual data catalog with validated YAML specifications
- Registry API for querying the data catalog
"""

from embodied_schemas.hardware import (
    HardwareEntry,
    HardwareCapability,
    HardwareType,
    FormFactor,
    PhysicalSpec,
    EnvironmentalSpec,
    PowerSpec,
    PowerMode,
    InterfaceSpec,
)
from embodied_schemas.models import (
    ModelEntry,
    ModelType,
    ModelVariant,
    AccuracyBenchmark,
    MemoryRequirements,
)
from embodied_schemas.sensors import (
    SensorEntry,
    SensorCategory,
    CameraSpec,
    DepthSpec,
    LidarSpec,
)
from embodied_schemas.usecases import (
    UseCaseEntry,
    Constraint,
    ConstraintCriticality,
    SuccessCriterion,
)
from embodied_schemas.benchmarks import (
    BenchmarkResult,
    LatencyMetrics,
    PowerMetrics,
    MemoryMetrics,
    ThermalMetrics,
    Verdict,
    Confidence,
)
from embodied_schemas.registry import Registry

__version__ = "0.1.0"

__all__ = [
    # Hardware
    "HardwareEntry",
    "HardwareCapability",
    "HardwareType",
    "FormFactor",
    "PhysicalSpec",
    "EnvironmentalSpec",
    "PowerSpec",
    "PowerMode",
    "InterfaceSpec",
    # Models
    "ModelEntry",
    "ModelType",
    "ModelVariant",
    "AccuracyBenchmark",
    "MemoryRequirements",
    # Sensors
    "SensorEntry",
    "SensorCategory",
    "CameraSpec",
    "DepthSpec",
    "LidarSpec",
    # Use Cases
    "UseCaseEntry",
    "Constraint",
    "ConstraintCriticality",
    "SuccessCriterion",
    # Benchmarks
    "BenchmarkResult",
    "LatencyMetrics",
    "PowerMetrics",
    "MemoryMetrics",
    "ThermalMetrics",
    "Verdict",
    "Confidence",
    # Registry
    "Registry",
    # Version
    "__version__",
]
