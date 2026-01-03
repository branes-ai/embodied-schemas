"""NPU and AI Accelerator schemas for dedicated inference hardware.

Covers dedicated neural processing units and AI accelerators that are
distinct from GPUs. These include:
- Edge AI accelerators (Hailo-8, Coral TPU)
- Integrated NPUs (Intel NPU, Qualcomm Hexagon, Apple Neural Engine)
- Datacenter AI accelerators (Groq, Cerebras, Graphcore)
"""

from enum import Enum
from pydantic import BaseModel, Field, computed_field


class NPUVendor(str, Enum):
    """NPU/AI accelerator vendors."""

    HAILO = "hailo"
    GOOGLE = "google"  # Coral TPU
    INTEL = "intel"  # Intel NPU
    QUALCOMM = "qualcomm"  # Hexagon
    APPLE = "apple"  # Neural Engine
    AMD = "amd"  # XDNA
    GROQ = "groq"
    CEREBRAS = "cerebras"
    GRAPHCORE = "graphcore"
    TENSTORRENT = "tenstorrent"
    MYTHIC = "mythic"
    SYNTIANT = "syntiant"


class NPUType(str, Enum):
    """NPU form factor and integration type."""

    DISCRETE = "discrete"  # Standalone accelerator (Hailo-8, Coral)
    INTEGRATED = "integrated"  # Built into SoC (Intel NPU, Apple ANE)
    DATACENTER = "datacenter"  # Server-class accelerator (Groq, Cerebras)
    EMBEDDED = "embedded"  # Embedded in edge device


class NPUInterface(str, Enum):
    """NPU host interface type."""

    PCIE = "pcie"
    M2 = "m2"
    USB = "usb"
    INTERNAL = "internal"  # Integrated into SoC
    CUSTOM = "custom"


class DataType(str, Enum):
    """Supported data types for inference."""

    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    BINARY = "binary"  # 1-bit weights


class ComputeSpec(BaseModel):
    """NPU compute specifications."""

    # Peak performance
    peak_tops_int8: float = Field(..., description="Peak INT8 TOPS")
    peak_tops_int4: float | None = Field(None, description="Peak INT4 TOPS")
    peak_tflops_fp16: float | None = Field(None, description="Peak FP16 TFLOPS")
    peak_tflops_bf16: float | None = Field(None, description="Peak BF16 TFLOPS")

    # Architecture
    compute_units: int | None = Field(
        None, description="Number of compute units/cores/engines"
    )
    mac_units: int | None = Field(
        None, description="Total MAC (multiply-accumulate) units"
    )
    vector_units: int | None = Field(None, description="Vector processing units")

    # Supported precisions
    supported_dtypes: list[DataType] = Field(
        default_factory=list, description="Supported data types"
    )

    # Sparsity
    sparse_support: bool = Field(False, description="Hardware sparse acceleration")
    sparse_speedup: float | None = Field(
        None, description="Speedup factor with 50% sparsity"
    )

    model_config = {"extra": "forbid"}


class MemorySpec(BaseModel):
    """NPU memory specifications."""

    # On-chip memory
    sram_mb: float | None = Field(None, description="On-chip SRAM in MB")
    l2_cache_mb: float | None = Field(None, description="L2 cache in MB")

    # External memory (if applicable)
    external_memory_gb: float | None = Field(
        None, description="External memory (DRAM/HBM) in GB"
    )
    external_memory_type: str | None = Field(
        None, description="External memory type: LPDDR4, HBM2, etc."
    )
    memory_bandwidth_gbps: float | None = Field(
        None, description="Memory bandwidth in GB/s"
    )

    # Host memory access
    uses_host_memory: bool = Field(
        False, description="Uses host system memory for model weights"
    )

    model_config = {"extra": "forbid"}


class PowerSpec(BaseModel):
    """NPU power specifications."""

    tdp_watts: float = Field(..., description="Thermal design power in watts")
    typical_power_watts: float | None = Field(
        None, description="Typical power during inference"
    )
    idle_power_watts: float | None = Field(None, description="Idle power consumption")
    peak_power_watts: float | None = Field(None, description="Peak power consumption")

    # Efficiency metrics (computed if not provided)
    tops_per_watt: float | None = Field(
        None, description="INT8 TOPS per watt efficiency"
    )

    model_config = {"extra": "forbid"}


class SoftwareSpec(BaseModel):
    """NPU software ecosystem specifications."""

    # SDK and tools
    sdk_name: str | None = Field(None, description="Primary SDK name")
    compiler_name: str | None = Field(None, description="Model compiler name")

    # Framework support
    supported_frameworks: list[str] = Field(
        default_factory=list,
        description="Supported ML frameworks: TensorFlow, PyTorch, ONNX, etc.",
    )

    # Model formats
    native_format: str | None = Field(
        None, description="Native model format: .hef, .tflite, .onnx, etc."
    )
    supported_formats: list[str] = Field(
        default_factory=list, description="Supported import formats"
    )

    # Operator coverage
    supported_ops: list[str] = Field(
        default_factory=list, description="Key supported operators/layers"
    )

    model_config = {"extra": "forbid"}


class PhysicalSpec(BaseModel):
    """NPU physical specifications."""

    form_factor: str = Field(..., description="Form factor: M.2, PCIe, USB, chip")
    dimensions_mm: tuple[float, float, float] | None = Field(
        None, description="Dimensions (L x W x H) in mm"
    )
    weight_grams: float | None = Field(None, description="Weight in grams")

    # Interface details
    interface: NPUInterface = Field(..., description="Host interface type")
    pcie_lanes: int | None = Field(None, description="PCIe lanes (if PCIe/M.2)")
    pcie_gen: str | None = Field(None, description="PCIe generation")

    model_config = {"extra": "forbid"}


class NPUEntry(BaseModel):
    """Complete NPU/AI accelerator catalog entry.

    Comprehensive specification for neural processing units and
    dedicated AI inference accelerators.
    """

    # Identity
    id: str = Field(
        ...,
        description="Unique identifier: {vendor}_{product}_{variant}",
    )
    name: str = Field(..., description="Full product name")
    vendor: NPUVendor = Field(..., description="NPU vendor")

    # Classification
    npu_type: NPUType = Field(..., description="NPU type/form factor")
    generation: str | None = Field(None, description="Product generation")

    # Compute
    compute: ComputeSpec = Field(..., description="Compute specifications")

    # Memory
    memory: MemorySpec | None = Field(None, description="Memory specifications")

    # Power
    power: PowerSpec = Field(..., description="Power specifications")

    # Software
    software: SoftwareSpec | None = Field(None, description="Software ecosystem")

    # Physical
    physical: PhysicalSpec | None = Field(None, description="Physical specifications")

    # Use cases
    target_applications: list[str] = Field(
        default_factory=list,
        description="Target applications: vision, NLP, audio, etc.",
    )

    # Metadata
    launch_date: str | None = Field(None, description="Launch date (YYYY-MM-DD)")
    msrp_usd: float | None = Field(None, description="MSRP in USD")
    datasheet_url: str | None = Field(None, description="Link to datasheet")
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}

    @computed_field
    @property
    def efficiency_tops_per_watt(self) -> float | None:
        """Compute efficiency metric if not provided."""
        if self.power.tops_per_watt is not None:
            return self.power.tops_per_watt
        if self.power.tdp_watts > 0:
            return self.compute.peak_tops_int8 / self.power.tdp_watts
        return None
