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
from embodied_schemas.analysis import (
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
from embodied_schemas.scheduling import (
    OperatorTiming,
    DataTransfer,
    DataMovementAnalysis,
    ExecutionTargetUtilization,
    RateFeasibility,
    SchedulingAnalysisResult,
    ArchitectureAnalysisResult,
)
from embodied_schemas.gpu import (
    GPUEntry,
    GPUVendor,
    DieSpec,
    ComputeResources,
    ClockSpeeds,
    MemorySpec,
    MemoryType,
    TheoreticalPerformance,
    PowerSpec as GPUPowerSpec,
    EfficiencyMetrics,
    BoardSpec,
    FeatureSupport,
    MarketInfo,
    TargetMarket,
    GPUArchitectureSummary,
    Foundry,
    PCIeGen,
    PowerConnector,
)
from embodied_schemas.cpu import (
    CPUEntry,
    CPUVendor,
    CPUArchitecture,
    SocketType,
    TargetMarket as CPUTargetMarket,
    ProcessNode,
    CoreConfig,
    ClockSpeeds as CPUClockSpeeds,
    CacheSpec,
    MemorySpec as CPUMemorySpec,
    InstructionExtensions,
    PowerSpec as CPUPowerSpec,
    PlatformSpec,
    IntegratedGraphics,
    MarketInfo as CPUMarketInfo,
)
from embodied_schemas.operators import (
    OperatorEntry,
    OperatorCategory,
    IOSpec,
    ConfigParam,
    OperatorPerfProfile,
    ImplType,
)
from embodied_schemas.architectures import (
    SoftwareArchitecture,
    OperatorInstance,
    DataflowEdge,
    ArchitectureVariant,
    architecture_to_mermaid,
)
from embodied_schemas.registry import Registry
from embodied_schemas.loaders import (
    load_gpus,
    load_gpu_architectures,
    load_cpus,
    load_hardware,
    load_chips,
    load_models,
    load_sensors,
    load_usecases,
    load_benchmarks,
    load_operators,
    load_architectures,
)

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
    # Analysis (Graph Analysis)
    "Bottleneck",
    "RooflineResult",
    "EnergyResult",
    "MemoryResult",
    "ConcurrencyResult",
    "SubgraphBreakdown",
    "GraphAnalysisResult",
    "ComparisonResult",
    "BatchSweepResult",
    # Scheduling (Architecture Analysis)
    "OperatorTiming",
    "DataTransfer",
    "DataMovementAnalysis",
    "ExecutionTargetUtilization",
    "RateFeasibility",
    "SchedulingAnalysisResult",
    "ArchitectureAnalysisResult",
    # GPU
    "GPUEntry",
    "GPUVendor",
    "DieSpec",
    "ComputeResources",
    "ClockSpeeds",
    "MemorySpec",
    "MemoryType",
    "TheoreticalPerformance",
    "GPUPowerSpec",
    "EfficiencyMetrics",
    "BoardSpec",
    "FeatureSupport",
    "MarketInfo",
    "TargetMarket",
    "GPUArchitectureSummary",
    "Foundry",
    "PCIeGen",
    "PowerConnector",
    # CPU
    "CPUEntry",
    "CPUVendor",
    "CPUArchitecture",
    "SocketType",
    "CPUTargetMarket",
    "ProcessNode",
    "CoreConfig",
    "CPUClockSpeeds",
    "CacheSpec",
    "CPUMemorySpec",
    "InstructionExtensions",
    "CPUPowerSpec",
    "PlatformSpec",
    "IntegratedGraphics",
    "CPUMarketInfo",
    # Registry
    "Registry",
    # Operators
    "OperatorEntry",
    "OperatorCategory",
    "IOSpec",
    "ConfigParam",
    "OperatorPerfProfile",
    "ImplType",
    # Architectures
    "SoftwareArchitecture",
    "OperatorInstance",
    "DataflowEdge",
    "ArchitectureVariant",
    "architecture_to_mermaid",
    # Loaders
    "load_gpus",
    "load_gpu_architectures",
    "load_cpus",
    "load_hardware",
    "load_chips",
    "load_models",
    "load_sensors",
    "load_usecases",
    "load_benchmarks",
    "load_operators",
    "load_architectures",
    # Version
    "__version__",
]
