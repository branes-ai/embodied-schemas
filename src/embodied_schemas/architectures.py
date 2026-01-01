"""Software architecture schemas for embodied AI systems.

Defines data structures for representing complete application architectures
composed of operators connected via dataflow edges. Architectures capture
the composition, timing requirements, and variant configurations for
specific platform deployments.
"""

from enum import Enum
from pydantic import BaseModel, Field


class ExecutionTarget(str, Enum):
    """Standard execution target types for heterogeneous systems.

    Covers the range of accelerators found in embodied AI systems,
    from general-purpose CPUs to specialized AI accelerators.

    Accelerator Taxonomy
    --------------------

    **General-Purpose Processors (CPU)**
        Standard processors that handle OS, control logic, and orchestration.
        Can run any workload but not optimized for parallel AI compute.

    **Graphics Processing Units (GPU)**
        Massively parallel processors originally for graphics, now widely used
        for DNN training and inference. High throughput, high power.

    **Neural Processing Units (NPU)**
        Generic term for dedicated AI inference accelerators integrated into
        mobile/edge SoCs. NPUs are characterized by:

        - **Fixed-function matrix engines**: Hardwired MAC arrays optimized for
          convolutions, matrix multiplies, and common DNN operations
        - **Quantization support**: Native INT8/INT4 compute for efficiency
        - **On-chip memory hierarchy**: Large SRAM buffers to minimize DRAM access
        - **Power efficiency focus**: Designed for always-on, battery-powered use
        - **Limited flexibility**: Cannot run arbitrary compute like GPUs

        Examples: Qualcomm Hexagon NPU, Apple Neural Engine, Samsung NPU,
        MediaTek APU, Google Edge TPU, Intel NPU (Meteor Lake)

        NPU vs other accelerators:
        - NPU vs GPU: NPUs are more power-efficient but less flexible
        - NPU vs TPU: TPUs are Google's specific NPU architecture
        - NPU vs CVU: CVUs are vision-specific; NPUs handle broader DNN workloads

    **Tensor Processing Units (TPU)**
        Google's custom ASIC for neural network acceleration. Systolic array
        architecture optimized for matrix operations. Available as cloud TPUs
        and Edge TPU for embedded deployment.

    **Knowledge Processing Units (KPU)**
        Stillwater Supercomputing's architecture for high-precision inference
        and scientific computing. Supports posit arithmetic and extended precision.

    **Vision Processing Units (VPU)**
        Specialized for computer vision pipelines including depth estimation,
        stereo matching, and optical flow. Examples: Intel Movidius Myriad.

    **Computer Vision Units (CVU)**
        Dedicated CNN accelerators optimized for object detection and image
        classification. Examples: Hailo-8, Mobileye EyeQ, Ambarella CV series.

    **Digital Signal Processors (DSP)**
        Optimized for signal processing, sensor fusion, and audio. Often used
        alongside NPUs for pre/post-processing. Examples: Qualcomm Hexagon DSP.

    **Field-Programmable Gate Arrays (FPGA)**
        Reconfigurable hardware for custom dataflow architectures. Offers
        flexibility between ASICs and GPUs. Low latency, moderate power.

    **Research Architectures**
        Academic and experimental accelerators like MIT Eyeriss (systolic array)
        and dataflow architectures (SambaNova, Cerebras, Graphcore).
    """

    # General-purpose processors
    CPU = "cpu"  # Host CPU (ARM, x86, RISC-V)

    # Graphics and general-purpose GPU compute
    GPU = "gpu"  # NVIDIA CUDA, AMD ROCm, Intel oneAPI

    # Neural/AI accelerators - vendor-specific architectures
    NPU = "npu"  # Generic neural processing unit (see docstring for details)
    TPU = "tpu"  # Google Tensor Processing Unit
    KPU = "kpu"  # Stillwater Knowledge Processing Unit

    # Vision-specific accelerators
    VPU = "vpu"  # Intel Movidius, OAK-D Myriad
    CVU = "cvu"  # Computer Vision Unit (Hailo, Mobileye EyeQ)

    # Signal processing
    DSP = "dsp"  # Digital Signal Processor (Qualcomm Hexagon, TI C66x)

    # Reconfigurable
    FPGA = "fpga"  # Field-programmable gate array

    # Research architectures
    SYSTOLIC = "systolic"  # Systolic arrays (Eyeriss, etc.)
    DATAFLOW = "dataflow"  # Dataflow architectures (Wave, SambaNova)

    # Catch-all for custom accelerators
    CUSTOM = "custom"


# Mapping of execution targets to their characteristics
EXECUTION_TARGET_INFO = {
    ExecutionTarget.CPU: {
        "name": "CPU",
        "description": "General-purpose host processor",
        "typical_workloads": ["control", "scheduling", "sequential algorithms"],
        "vendors": ["Intel", "AMD", "ARM", "RISC-V"],
    },
    ExecutionTarget.GPU: {
        "name": "GPU",
        "description": "Graphics processing unit for parallel compute",
        "typical_workloads": ["DNN inference", "parallel compute", "image processing"],
        "vendors": ["NVIDIA", "AMD", "Intel", "Qualcomm"],
    },
    ExecutionTarget.NPU: {
        "name": "NPU",
        "description": (
            "Neural Processing Unit - dedicated AI inference accelerator integrated into "
            "mobile/edge SoCs. Features fixed-function matrix engines (MAC arrays), native "
            "INT8/INT4 quantization support, large on-chip SRAM buffers, and power-efficient "
            "design for always-on battery-powered applications. Less flexible than GPUs but "
            "significantly more power-efficient for supported operations."
        ),
        "typical_workloads": [
            "DNN inference",
            "quantized models (INT8/INT4)",
            "CNN classification",
            "transformer inference",
            "on-device AI",
        ],
        "vendors": [
            "Qualcomm (Hexagon NPU)",
            "Apple (Neural Engine)",
            "Samsung (NPU)",
            "MediaTek (APU)",
            "Google (Edge TPU)",
            "Intel (NPU in Meteor Lake)",
            "AMD (XDNA/Ryzen AI)",
        ],
        "characteristics": {
            "power_efficiency": "high",
            "flexibility": "low",
            "precision": ["INT8", "INT4", "FP16"],
            "typical_tops": "10-40 TOPS",
        },
    },
    ExecutionTarget.TPU: {
        "name": "TPU",
        "description": "Google Tensor Processing Unit",
        "typical_workloads": ["DNN inference", "matrix operations"],
        "vendors": ["Google"],
    },
    ExecutionTarget.KPU: {
        "name": "KPU",
        "description": "Stillwater Knowledge Processing Unit",
        "typical_workloads": ["high-precision inference", "scientific computing"],
        "vendors": ["Stillwater Supercomputing"],
    },
    ExecutionTarget.VPU: {
        "name": "VPU",
        "description": "Vision processing unit",
        "typical_workloads": ["computer vision", "depth processing"],
        "vendors": ["Intel Movidius", "Luxonis"],
    },
    ExecutionTarget.CVU: {
        "name": "CVU",
        "description": "Computer vision accelerator",
        "typical_workloads": ["object detection", "CNN inference"],
        "vendors": ["Hailo", "Mobileye", "Ambarella"],
    },
    ExecutionTarget.DSP: {
        "name": "DSP",
        "description": "Digital signal processor",
        "typical_workloads": ["audio", "sensor fusion", "signal processing"],
        "vendors": ["Qualcomm Hexagon", "TI", "Cadence"],
    },
    ExecutionTarget.FPGA: {
        "name": "FPGA",
        "description": "Field-programmable gate array",
        "typical_workloads": ["custom dataflow", "low-latency inference"],
        "vendors": ["AMD/Xilinx", "Intel/Altera", "Lattice"],
    },
    ExecutionTarget.SYSTOLIC: {
        "name": "Systolic Array",
        "description": "Systolic array architecture",
        "typical_workloads": ["matrix operations", "CNN inference"],
        "vendors": ["MIT Eyeriss", "Google TPU"],
    },
    ExecutionTarget.DATAFLOW: {
        "name": "Dataflow",
        "description": "Dataflow architecture",
        "typical_workloads": ["large models", "reconfigurable compute"],
        "vendors": ["SambaNova", "Cerebras", "Graphcore"],
    },
    ExecutionTarget.CUSTOM: {
        "name": "Custom",
        "description": "Custom or unspecified accelerator",
        "typical_workloads": [],
        "vendors": [],
    },
}


class DataflowEdge(BaseModel):
    """Connection between operators in an architecture.

    Represents a data dependency from one operator's output port
    to another operator's input port.
    """

    source_op: str = Field(..., description="Source operator instance ID")
    source_port: str = Field(..., description="Output port name on source operator")
    target_op: str = Field(..., description="Target operator instance ID")
    target_port: str = Field(..., description="Input port name on target operator")
    queue_size: int = Field(
        1, description="Buffer depth for asynchronous edges (1 = synchronous)"
    )


class OperatorInstance(BaseModel):
    """Instantiation of an operator within an architecture.

    References an OperatorEntry by ID and provides instance-specific
    configuration and execution parameters.
    """

    id: str = Field(..., description="Instance ID unique within this architecture")
    operator_id: str = Field(..., description="Reference to OperatorEntry ID")
    config: dict = Field(
        default_factory=dict,
        description="Operator-specific configuration overriding defaults",
    )
    rate_hz: float | None = Field(
        None, description="Execution rate in Hz (None = event-driven/triggered)"
    )
    priority: int = Field(
        0, description="Scheduling priority (higher = more urgent)"
    )
    execution_target: str | None = Field(
        None, description="Execution target: cpu, gpu, npu, dsp, etc."
    )


class ArchitectureVariant(BaseModel):
    """Alternative configuration of a base architecture.

    Allows defining hardware-optimized or accuracy-optimized variants
    by overriding operator selections and configurations without
    duplicating the full architecture definition.
    """

    id: str = Field(..., description="Unique variant identifier")
    name: str = Field(..., description="Human-readable variant name")
    description: str = Field(..., description="Description of variant purpose")
    operator_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Map instance_id -> replacement operator_id",
    )
    config_overrides: dict[str, dict] = Field(
        default_factory=dict,
        description="Map instance_id -> config dict overrides",
    )
    target_hardware: list[str] = Field(
        default_factory=list, description="Hardware IDs this variant targets"
    )
    expected_latency_ms: float | None = Field(
        None, description="Expected end-to-end latency on target hardware"
    )
    expected_power_w: float | None = Field(
        None, description="Expected power consumption on target hardware"
    )


class SoftwareArchitecture(BaseModel):
    """Complete embodied AI application architecture.

    Represents a fully specified software system composed of operator
    instances connected via dataflow edges. Includes platform bindings,
    timing requirements, resource budgets, and variant configurations.
    """

    # Identity
    id: str = Field(..., description="Unique architecture identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Description of the architecture purpose")

    # Platform binding
    platform_type: str = Field(
        ..., description="Platform type: drone, quadruped, vehicle, manipulator, etc."
    )
    sensors: list[str] = Field(
        default_factory=list,
        description="Sensor IDs used by this architecture",
    )
    actuators: list[str] = Field(
        default_factory=list,
        description="Actuator IDs controlled by this architecture",
    )

    # Composition
    operators: list[OperatorInstance] = Field(
        ..., description="Operator instances in this architecture"
    )
    dataflow: list[DataflowEdge] = Field(
        ..., description="Dataflow connections between operators"
    )

    # Timing requirements
    end_to_end_latency_ms: float | None = Field(
        None, description="Maximum end-to-end latency requirement in milliseconds"
    )
    min_throughput_fps: float | None = Field(
        None, description="Minimum throughput requirement in frames per second"
    )

    # Resource envelope
    power_budget_w: float | None = Field(
        None, description="Total power budget in watts"
    )
    memory_budget_mb: float | None = Field(
        None, description="Total memory budget in megabytes"
    )

    # Variants
    variants: list[ArchitectureVariant] = Field(
        default_factory=list,
        description="Alternative configurations of this architecture",
    )

    # Metadata
    reference_impl: str | None = Field(
        None, description="Reference implementation path or URL"
    )
    tags: list[str] = Field(
        default_factory=list, description="Searchable tags: real-time, perception, etc."
    )
    last_updated: str | None = Field(None, description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}


def architecture_to_mermaid(arch: SoftwareArchitecture) -> str:
    """Generate a Mermaid flowchart diagram from an architecture.

    Produces a left-to-right flowchart showing operator instances
    as nodes and dataflow edges as connections.

    Args:
        arch: The software architecture to visualize.

    Returns:
        Mermaid diagram source as a string.
    """
    lines = ["flowchart LR"]

    # Add operator nodes with rate annotations
    for op in arch.operators:
        if op.rate_hz:
            label = f'{op.id}["{op.operator_id}<br/>{op.rate_hz}Hz"]'
        else:
            label = f'{op.id}["{op.operator_id}"]'
        lines.append(f"    {label}")

    # Add blank line for readability
    lines.append("")

    # Add dataflow edges
    for edge in arch.dataflow:
        lines.append(f"    {edge.source_op} --> {edge.target_op}")

    return "\n".join(lines)
