"""Hardware platform and capability schemas.

Defines data structures for edge compute platforms, accelerators, and SoCs
used in embodied AI systems.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class HardwareType(str, Enum):
    """Types of hardware accelerators."""

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"
    FPGA = "fpga"
    DSP = "dsp"
    DPU = "dpu"  # Dataflow Processing Unit
    KPU = "kpu"  # Knowledge Processing Unit / AI accelerator


class ComputeParadigm(str, Enum):
    """Computation paradigm of the hardware."""

    VON_NEUMANN = "von_neumann"
    SIMD = "simd"
    DATAFLOW = "dataflow"
    SYSTOLIC_ARRAY = "systolic_array"
    RECONFIGURABLE = "reconfigurable"


class OperationType(str, Enum):
    """Types of operations hardware is optimized for."""

    GENERAL_PURPOSE = "general_purpose"
    MATRIX_MULTIPLY = "matrix_multiply"
    CONVOLUTION = "convolution"
    ATTENTION = "attention"
    SPARSE_OPS = "sparse_ops"
    INT_QUANTIZED = "int_quantized"
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class FormFactor(str, Enum):
    """Physical form factor of the hardware."""

    SOM = "som"  # System on Module
    PCIE = "pcie"
    USB = "usb"
    STANDALONE = "standalone"
    RACK = "rack"
    M2 = "m2"
    MXMX = "mxm"  # MXM module


class LifecycleStatus(str, Enum):
    """Product lifecycle status."""

    ANNOUNCED = "announced"
    NEW = "new"
    MATURE = "mature"
    LEGACY = "legacy"
    EOL = "eol"  # End of life


class Availability(str, Enum):
    """Product availability."""

    WIDELY_AVAILABLE = "widely_available"
    LIMITED = "limited"
    PROPRIETARY = "proprietary"
    CLOUD_ONLY = "cloud_only"


class PowerMode(BaseModel):
    """A power mode configuration for the hardware."""

    name: str = Field(..., description="Power mode name, e.g., '10W', '15W', 'MAX-N'")
    power_watts: float = Field(..., description="Power consumption in this mode")
    gpu_freq_mhz: int | None = Field(None, description="GPU frequency in this mode")
    cpu_freq_mhz: int | None = Field(None, description="CPU frequency in this mode")
    description: str | None = Field(None, description="Description of use case for this mode")


class PhysicalSpec(BaseModel):
    """Physical specifications for embodied systems."""

    weight_grams: float | None = Field(None, description="Weight in grams (module only)")
    dimensions_mm: list[float] | None = Field(
        None, description="Dimensions [L, W, H] in millimeters"
    )
    form_factor: FormFactor = Field(..., description="Physical form factor")
    mounting: str | None = Field(
        None, description="Mounting type: carrier_board, direct, rack, etc."
    )


class EnvironmentalSpec(BaseModel):
    """Environmental operating specifications."""

    operating_temp_c: list[float] = Field(
        ..., description="Operating temperature range [min, max] in Celsius"
    )
    storage_temp_c: list[float] | None = Field(
        None, description="Storage temperature range [min, max] in Celsius"
    )
    humidity_percent: list[float] | None = Field(
        None, description="Humidity range [min, max] percent, non-condensing"
    )
    ip_rating: str | None = Field(None, description="IP rating, e.g., IP65, IP67")
    vibration_g: float | None = Field(None, description="Max sustained vibration in g")
    shock_g: float | None = Field(None, description="Max shock in g")
    altitude_m: float | None = Field(None, description="Max operating altitude in meters")


class PowerSpec(BaseModel):
    """Power delivery and consumption specifications."""

    input_voltage_v: list[float] = Field(
        ..., description="Input voltage range [min, max] in volts"
    )
    power_modes: list[PowerMode] = Field(
        default_factory=list, description="Available power modes"
    )
    tdp_watts: float | None = Field(None, description="Thermal Design Power in watts")
    typical_power_watts: float | None = Field(
        None, description="Typical power consumption in watts"
    )
    battery_compatible: bool = Field(
        False, description="Whether the device is suitable for battery operation"
    )
    poe_support: bool = Field(False, description="Power over Ethernet support")


class InterfaceSpec(BaseModel):
    """Interface specifications for sensor and peripheral integration."""

    camera_csi: int = Field(0, description="Number of CSI camera ports")
    camera_usb: int = Field(0, description="Number of USB camera-capable ports")
    usb3: int = Field(0, description="Number of USB 3.x ports")
    usb2: int = Field(0, description="Number of USB 2.0 ports")
    usb_c: int = Field(0, description="Number of USB-C ports")
    pcie_lanes: int = Field(0, description="Total PCIe lanes available")
    pcie_gen: int | None = Field(None, description="PCIe generation (3, 4, 5)")
    gpio: int = Field(0, description="Number of GPIO pins")
    ethernet: str | None = Field(None, description="Ethernet speed: 1gbps, 10gbps, etc.")
    wifi: str | None = Field(None, description="WiFi standard: wifi6, wifi5, etc.")
    bluetooth: str | None = Field(None, description="Bluetooth version")
    can_bus: int = Field(0, description="Number of CAN bus interfaces")
    i2c: int = Field(0, description="Number of I2C interfaces")
    spi: int = Field(0, description="Number of SPI interfaces")
    uart: int = Field(0, description="Number of UART interfaces")
    hdmi: int = Field(0, description="Number of HDMI outputs")
    displayport: int = Field(0, description="Number of DisplayPort outputs")


class HardwareCapability(BaseModel):
    """Compute and memory capabilities of the hardware."""

    # Compute specifications
    peak_tops_int8: float | None = Field(None, description="Peak INT8 TOPS")
    peak_tflops_fp16: float | None = Field(None, description="Peak FP16 TFLOPS")
    peak_tflops_fp32: float | None = Field(None, description="Peak FP32 TFLOPS")
    peak_tflops_bf16: float | None = Field(None, description="Peak BF16 TFLOPS")

    # Memory specifications
    memory_gb: float = Field(..., description="Total memory in GB")
    memory_type: str | None = Field(None, description="Memory type: LPDDR5, HBM2e, etc.")
    memory_bandwidth_gbps: float | None = Field(
        None, description="Memory bandwidth in GB/s"
    )
    cache_hierarchy: dict[str, float] = Field(
        default_factory=dict, description="Cache sizes in MB: L1, L2, L3"
    )

    # Parallelism
    compute_units: int | None = Field(
        None, description="Number of compute units (cores, SMs, etc.)"
    )
    tensor_cores: int | None = Field(None, description="Number of tensor cores")
    simd_width: int | None = Field(None, description="SIMD width in bits")

    # Special features
    sparse_acceleration: bool = Field(False, description="Hardware sparse acceleration")
    int4_support: bool = Field(False, description="INT4 quantization support")

    # Software support
    frameworks: list[str] = Field(
        default_factory=list,
        description="Supported frameworks: PyTorch, TensorFlow, ONNX, etc.",
    )
    quantization_support: list[str] = Field(
        default_factory=list, description="Supported quantization: int8, fp16, etc."
    )
    inference_runtimes: list[str] = Field(
        default_factory=list,
        description="Inference runtimes: TensorRT, OpenVINO, etc.",
    )


class SoftwareSpec(BaseModel):
    """Software ecosystem specifications."""

    os: list[str] = Field(
        default_factory=list, description="Supported operating systems"
    )
    sdk: str | None = Field(None, description="Primary SDK name")
    sdk_version: str | None = Field(None, description="SDK version")
    frameworks: list[str] = Field(
        default_factory=list, description="Supported ML frameworks"
    )
    inference_runtimes: list[str] = Field(
        default_factory=list, description="Supported inference runtimes"
    )
    container_support: bool = Field(False, description="Docker/container support")


class HardwareEntry(BaseModel):
    """Complete hardware catalog entry for embodied AI platforms."""

    # Identity
    id: str = Field(..., description="Unique identifier, e.g., nvidia_jetson_nano_4gb")
    name: str = Field(..., description="Human-readable name")
    vendor: str = Field(..., description="Manufacturer name")
    model: str = Field(..., description="Model name/number")
    hardware_type: HardwareType = Field(..., description="Primary hardware type")

    # Architecture
    compute_paradigm: ComputeParadigm = Field(..., description="Compute architecture type")
    optimized_for: list[OperationType] = Field(
        default_factory=list, description="Operations this hardware is optimized for"
    )

    # Capabilities
    capabilities: HardwareCapability = Field(..., description="Compute and memory specs")

    # Physical (for embodied systems)
    physical: PhysicalSpec | None = Field(None, description="Physical specifications")

    # Environmental
    environmental: EnvironmentalSpec | None = Field(
        None, description="Environmental operating specs"
    )

    # Power
    power: PowerSpec | None = Field(None, description="Power specifications")

    # Interfaces
    interfaces: InterfaceSpec | None = Field(None, description="Interface specifications")

    # Software
    software: SoftwareSpec | None = Field(None, description="Software ecosystem")

    # Deployment classification
    suitable_for: list[str] = Field(
        default_factory=list,
        description="Deployment contexts: edge, cloud, robotics, drone, etc.",
    )
    target_applications: list[str] = Field(
        default_factory=list,
        description="Target applications: vision, nlp, robotics, etc.",
    )

    # Cost and availability
    cost_usd: float | None = Field(None, description="Approximate cost in USD")
    availability: Availability = Field(
        Availability.WIDELY_AVAILABLE, description="Product availability"
    )
    lifecycle_status: LifecycleStatus = Field(
        LifecycleStatus.MATURE, description="Product lifecycle status"
    )

    # Metadata
    notes: str = Field("", description="Additional notes")
    datasheet_url: str | None = Field(None, description="Link to datasheet")
    product_url: str | None = Field(None, description="Link to product page")
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    # For relationships
    chip_id: str | None = Field(
        None, description="Reference to underlying chip/SoC if applicable"
    )

    model_config = {"extra": "forbid"}


class ChipEntry(BaseModel):
    """Raw chip/SoC specifications (shared across multiple platforms)."""

    id: str = Field(..., description="Unique identifier, e.g., nvidia_orin_soc")
    name: str = Field(..., description="Chip name")
    vendor: str = Field(..., description="Manufacturer")
    architecture: str | None = Field(None, description="Architecture name: Ampere, Ada, etc.")
    process_node_nm: int | None = Field(None, description="Process node in nanometers")

    # Compute blocks
    cpu_cores: int | None = Field(None, description="Number of CPU cores")
    cpu_architecture: str | None = Field(None, description="CPU architecture: Cortex-A78, etc.")
    gpu_cores: int | None = Field(None, description="Number of GPU cores/SMs")
    gpu_architecture: str | None = Field(None, description="GPU architecture")
    npu_tops: float | None = Field(None, description="NPU performance in TOPS")
    dla_tops: float | None = Field(None, description="DLA performance in TOPS")

    # Memory interface
    memory_interface: str | None = Field(None, description="Memory interface: LPDDR5x, etc.")
    memory_channels: int | None = Field(None, description="Number of memory channels")
    max_memory_gb: float | None = Field(None, description="Maximum supported memory")

    # Metadata
    announcement_date: str | None = Field(None, description="Announcement date")
    datasheet_url: str | None = Field(None, description="Datasheet link")
    last_updated: str = Field(..., description="Last update date")

    model_config = {"extra": "forbid"}
