"""Software architecture schemas for embodied AI systems.

Defines data structures for representing complete application architectures
composed of operators connected via dataflow edges. Architectures capture
the composition, timing requirements, and variant configurations for
specific platform deployments.
"""

from pydantic import BaseModel, Field


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
