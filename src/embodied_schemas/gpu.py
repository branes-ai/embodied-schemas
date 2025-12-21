"""GPU-specific schemas for discrete and integrated graphics processors.

Comprehensive schema capturing specifications needed for:
- Performance estimation (TFLOPS, fill rates, memory bandwidth)
- Energy estimation (TDP, power modes, efficiency)
- Concurrency estimation (cores, compute units, tensor/RT cores)
- Benchmark performance (theoretical peaks, measured results)
- Cost analysis (MSRP, market pricing, value metrics)

Schema design inspired by TechPowerUp GPU Database structure.
"""

from enum import Enum
from pydantic import BaseModel, Field, computed_field


class GPUVendor(str, Enum):
    """GPU silicon vendors."""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    QUALCOMM = "qualcomm"
    ARM = "arm"  # Mali GPUs
    IMAGINATION = "imagination"  # PowerVR
    APPLE = "apple"


class Foundry(str, Enum):
    """Semiconductor foundries."""

    TSMC = "tsmc"
    SAMSUNG = "samsung"
    INTEL = "intel"
    GLOBALFOUNDRIES = "globalfoundries"
    SMIC = "smic"
    UMC = "umc"


class MemoryType(str, Enum):
    """GPU memory technologies."""

    GDDR5 = "gddr5"
    GDDR5X = "gddr5x"
    GDDR6 = "gddr6"
    GDDR6X = "gddr6x"
    GDDR7 = "gddr7"
    HBM = "hbm"
    HBM2 = "hbm2"
    HBM2E = "hbm2e"
    HBM3 = "hbm3"
    HBM3E = "hbm3e"
    LPDDR4 = "lpddr4"
    LPDDR4X = "lpddr4x"
    LPDDR5 = "lpddr5"
    LPDDR5X = "lpddr5x"
    DDR4 = "ddr4"
    DDR5 = "ddr5"
    UNIFIED = "unified"  # Shared system memory


class TargetMarket(str, Enum):
    """GPU target market segment."""

    CONSUMER_DESKTOP = "consumer_desktop"
    CONSUMER_MOBILE = "consumer_mobile"
    WORKSTATION = "workstation"
    DATACENTER = "datacenter"
    EMBEDDED = "embedded"
    AUTOMOTIVE = "automotive"
    CLOUD_GAMING = "cloud_gaming"
    CRYPTO_MINING = "crypto_mining"  # Historical


class PCIeGen(str, Enum):
    """PCIe generation."""

    PCIE_3 = "pcie_3.0"
    PCIE_4 = "pcie_4.0"
    PCIE_5 = "pcie_5.0"
    PCIE_6 = "pcie_6.0"


class PowerConnector(str, Enum):
    """GPU power connector types."""

    NONE = "none"  # Slot-powered only
    PIN_6 = "6-pin"
    PIN_8 = "8-pin"
    PIN_6_PLUS_8 = "6+8-pin"
    PIN_8X2 = "2x8-pin"
    PIN_8X3 = "3x8-pin"
    PIN_12VHPWR = "12VHPWR"  # 16-pin, 600W capable
    PIN_12V2X6 = "12V-2x6"  # Updated 12VHPWR


class DirectXVersion(str, Enum):
    """DirectX feature level support."""

    DX11 = "11"
    DX11_1 = "11.1"
    DX12 = "12"
    DX12_1 = "12.1"
    DX12_2 = "12.2"  # Ultimate


class ShaderModel(str, Enum):
    """Shader model version."""

    SM_5_0 = "5.0"
    SM_5_1 = "5.1"
    SM_6_0 = "6.0"
    SM_6_1 = "6.1"
    SM_6_2 = "6.2"
    SM_6_3 = "6.3"
    SM_6_4 = "6.4"
    SM_6_5 = "6.5"
    SM_6_6 = "6.6"
    SM_6_7 = "6.7"
    SM_6_8 = "6.8"


# =============================================================================
# Fabrication / Die Specifications
# =============================================================================


class DieSpec(BaseModel):
    """GPU die/chip fabrication specifications.

    Essential for understanding manufacturing constraints, yields, and
    fundamental performance/power characteristics.
    """

    gpu_name: str = Field(..., description="GPU chip codename, e.g., 'AD102', 'Navi 31', 'GA102'")
    architecture: str = Field(
        ..., description="GPU architecture name, e.g., 'Ada Lovelace', 'RDNA 3', 'Ampere'"
    )
    architecture_codename: str | None = Field(
        None, description="Internal codename if different, e.g., 'Blackwell'"
    )
    foundry: Foundry = Field(..., description="Semiconductor foundry")
    process_nm: int = Field(..., description="Process node in nanometers")
    process_name: str | None = Field(
        None, description="Foundry process name, e.g., 'N4', '4nm', '7nm'"
    )
    transistors_billion: float = Field(..., description="Transistor count in billions")
    die_size_mm2: float = Field(..., description="Die size in square millimeters")

    # Multi-chip designs (AMD RDNA 3, etc.)
    is_chiplet: bool = Field(False, description="Whether this is a chiplet/MCM design")
    num_dies: int = Field(1, description="Number of dies in package")
    chiplet_breakdown: dict[str, int] | None = Field(
        None, description="Chiplet counts by type, e.g., {'gcd': 1, 'mcd': 6}"
    )

    @computed_field
    @property
    def transistor_density_mtx_mm2(self) -> float | None:
        """Transistor density in millions per mm²."""
        if self.die_size_mm2 > 0:
            return (self.transistors_billion * 1000) / self.die_size_mm2
        return None


# =============================================================================
# Compute Resources / Shader Units
# =============================================================================


class ComputeResources(BaseModel):
    """GPU compute unit specifications.

    Captures the parallel processing resources across different vendor
    architectures (NVIDIA, AMD, Intel).
    """

    # Unified shader count (vendor-agnostic)
    shaders: int = Field(..., description="Total unified shader processors")

    # Vendor-specific naming (same as shaders, but for clarity)
    cuda_cores: int | None = Field(None, description="NVIDIA: CUDA cores")
    stream_processors: int | None = Field(None, description="AMD: Stream processors")
    xe_cores: int | None = Field(None, description="Intel: Xe cores (each has 16 vector engines)")

    # Compute units / Streaming Multiprocessors
    compute_units: int | None = Field(
        None, description="AMD CUs / Intel Xe cores / logical groupings"
    )
    streaming_multiprocessors: int | None = Field(None, description="NVIDIA: SM count")

    # Fixed-function units
    tmus: int = Field(..., description="Texture Mapping Units")
    rops: int = Field(..., description="Render Output Units (Raster Operations Pipelines)")

    # AI/ML acceleration
    tensor_cores: int | None = Field(None, description="NVIDIA: Tensor core count")
    tensor_core_gen: int | None = Field(None, description="Tensor core generation (1-5)")
    matrix_cores: int | None = Field(None, description="AMD: AI accelerator / matrix cores")
    xmx_engines: int | None = Field(None, description="Intel: XMX (Xe Matrix eXtensions) engines")

    # Ray tracing
    rt_cores: int | None = Field(None, description="NVIDIA: Ray Tracing core count")
    rt_core_gen: int | None = Field(None, description="RT core generation (1-4)")
    ray_accelerators: int | None = Field(None, description="AMD: Ray Accelerator count")
    ray_tracing_units: int | None = Field(None, description="Intel: Ray Tracing Units")

    # Execution characteristics
    warp_size: int = Field(32, description="Threads per warp/wavefront (NVIDIA=32, AMD=32/64)")
    max_warps_per_sm: int | None = Field(None, description="Maximum warps per SM")
    registers_per_sm_kb: int | None = Field(None, description="Register file size per SM in KB")


# =============================================================================
# Clock Speeds
# =============================================================================


class ClockSpeeds(BaseModel):
    """GPU clock frequency specifications."""

    base_clock_mhz: int = Field(..., description="Base/reference GPU clock in MHz")
    boost_clock_mhz: int = Field(..., description="Maximum boost GPU clock in MHz")
    game_clock_mhz: int | None = Field(None, description="AMD: Typical gaming clock in MHz")

    # Memory clocks
    memory_clock_mhz: int = Field(..., description="Memory clock in MHz (actual, not effective)")
    memory_effective_gbps: float | None = Field(
        None, description="Effective memory speed in Gbps (with data rate multiplier)"
    )

    # Overclocking headroom (optional, for enthusiast data)
    typical_oc_boost_mhz: int | None = Field(
        None, description="Typical overclock boost frequency"
    )


# =============================================================================
# Memory Subsystem
# =============================================================================


class MemorySpec(BaseModel):
    """GPU memory specifications."""

    memory_size_gb: float = Field(..., description="Total VRAM in GB")
    memory_type: MemoryType = Field(..., description="Memory technology")
    memory_bus_bits: int = Field(..., description="Memory bus width in bits")
    memory_bandwidth_gbps: float = Field(..., description="Memory bandwidth in GB/s")

    # ECC support (datacenter/workstation)
    ecc_support: bool = Field(False, description="Error Correcting Code memory support")

    # Infinity Cache / L2 / Last-level cache
    l2_cache_mb: float | None = Field(None, description="L2 cache size in MB")
    infinity_cache_mb: float | None = Field(None, description="AMD: Infinity Cache size in MB")
    smart_access_memory: bool = Field(False, description="AMD: SAM / Resizable BAR support")

    @computed_field
    @property
    def memory_bits_per_gb(self) -> float:
        """Bus width per GB of memory (useful for config detection)."""
        if self.memory_size_gb > 0:
            return self.memory_bus_bits / self.memory_size_gb
        return 0


# =============================================================================
# Theoretical Performance
# =============================================================================


class TheoreticalPerformance(BaseModel):
    """Theoretical peak performance metrics.

    Calculated from shader count and clock speeds.
    These are vendor-published or calculated peaks, not measured performance.
    """

    # Floating-point throughput
    fp32_tflops: float = Field(..., description="Peak FP32 (single precision) TFLOPS")
    fp16_tflops: float | None = Field(None, description="Peak FP16 (half precision) TFLOPS")
    fp64_tflops: float | None = Field(None, description="Peak FP64 (double precision) TFLOPS")
    bf16_tflops: float | None = Field(None, description="Peak BF16 TFLOPS")

    # Integer throughput
    int8_tops: float | None = Field(None, description="Peak INT8 TOPS")
    int4_tops: float | None = Field(None, description="Peak INT4 TOPS")

    # Tensor/Matrix throughput (with sparsity if applicable)
    tensor_tflops_fp16: float | None = Field(
        None, description="Tensor core FP16 TFLOPS (dense)"
    )
    tensor_tflops_fp16_sparse: float | None = Field(
        None, description="Tensor core FP16 TFLOPS with 2:4 sparsity"
    )
    tensor_tflops_int8: float | None = Field(None, description="Tensor core INT8 TOPS")
    tensor_tflops_fp8: float | None = Field(None, description="Tensor core FP8 TFLOPS")

    # Graphics fill rates
    pixel_rate_gpixels: float = Field(..., description="Pixel fill rate in GPixels/s")
    texture_rate_gtexels: float = Field(..., description="Texture fill rate in GTexels/s")

    # Ray tracing (when applicable)
    rt_tflops: float | None = Field(None, description="Ray tracing TFLOPS estimate")


# =============================================================================
# Power Specifications
# =============================================================================


class PowerSpec(BaseModel):
    """GPU power consumption and delivery specifications."""

    # Board power
    tdp_watts: int = Field(..., description="Thermal Design Power / TBP in watts")
    max_power_watts: int | None = Field(
        None, description="Maximum power limit (power limit max) in watts"
    )
    min_power_watts: int | None = Field(
        None, description="Minimum power limit in watts"
    )

    # Operating power states
    idle_power_watts: float | None = Field(None, description="Idle power consumption in watts")
    gaming_power_watts: float | None = Field(
        None, description="Typical gaming power consumption in watts"
    )
    video_playback_watts: float | None = Field(
        None, description="Video playback power consumption"
    )

    # Power delivery
    power_connectors: list[PowerConnector] = Field(
        default_factory=list, description="Required power connectors"
    )
    slot_power_watts: int = Field(75, description="Power available from PCIe slot")
    recommended_psu_watts: int | None = Field(
        None, description="Manufacturer recommended PSU wattage"
    )

    @computed_field
    @property
    def total_power_capacity_watts(self) -> int:
        """Total theoretical power delivery capacity."""
        connector_power = {
            PowerConnector.NONE: 0,
            PowerConnector.PIN_6: 75,
            PowerConnector.PIN_8: 150,
            PowerConnector.PIN_6_PLUS_8: 225,
            PowerConnector.PIN_8X2: 300,
            PowerConnector.PIN_8X3: 450,
            PowerConnector.PIN_12VHPWR: 600,
            PowerConnector.PIN_12V2X6: 600,
        }
        total = self.slot_power_watts
        for conn in self.power_connectors:
            total += connector_power.get(conn, 0)
        return total


# =============================================================================
# Efficiency Metrics
# =============================================================================


class EfficiencyMetrics(BaseModel):
    """Computed efficiency metrics for performance/power analysis."""

    perf_per_watt_tflops: float | None = Field(
        None, description="FP32 TFLOPS per watt (efficiency)"
    )
    perf_per_mm2_tflops: float | None = Field(
        None, description="FP32 TFLOPS per mm² die area"
    )
    bandwidth_per_watt_gbps: float | None = Field(
        None, description="Memory bandwidth per watt"
    )
    transistors_per_tflop_million: float | None = Field(
        None, description="Transistor efficiency"
    )


# =============================================================================
# Physical / Board Specifications
# =============================================================================


class BoardSpec(BaseModel):
    """Physical graphics card specifications."""

    # Form factor
    slot_width: float = Field(..., description="Slot width (2.0, 2.5, 3.0, etc.)")
    length_mm: int = Field(..., description="Card length in millimeters")
    height_mm: int | None = Field(None, description="Card height in millimeters")
    width_mm: int | None = Field(None, description="Card width/thickness in millimeters")

    # PCIe interface
    pcie_interface: PCIeGen = Field(..., description="PCIe generation")
    pcie_lanes: int = Field(16, description="Number of PCIe lanes")

    # Display outputs
    hdmi_ports: int = Field(0, description="Number of HDMI outputs")
    hdmi_version: str | None = Field(None, description="HDMI version, e.g., '2.1a'")
    displayport_ports: int = Field(0, description="Number of DisplayPort outputs")
    displayport_version: str | None = Field(None, description="DisplayPort version, e.g., '2.1'")
    usb_c_ports: int = Field(0, description="USB-C/VirtualLink ports")

    # Cooling
    fans: int | None = Field(None, description="Number of fans (reference design)")
    cooler_type: str | None = Field(
        None, description="Cooling solution: blower, open-air, liquid, passive"
    )


# =============================================================================
# API / Feature Support
# =============================================================================


class FeatureSupport(BaseModel):
    """Graphics API and feature support."""

    # Graphics APIs
    directx: DirectXVersion = Field(..., description="DirectX feature level")
    opengl: str | None = Field(None, description="OpenGL version, e.g., '4.6'")
    vulkan: str | None = Field(None, description="Vulkan version, e.g., '1.3'")
    opencl: str | None = Field(None, description="OpenCL version, e.g., '3.0'")
    metal: str | None = Field(None, description="Metal version (macOS), e.g., '3'")

    # Shader model
    shader_model: ShaderModel | None = Field(None, description="Shader model version")

    # Vendor-specific features
    cuda_compute: str | None = Field(
        None, description="NVIDIA: CUDA compute capability, e.g., '8.9'"
    )
    nvenc: bool = Field(False, description="NVIDIA: Hardware video encoder")
    nvenc_gen: int | None = Field(None, description="NVENC generation")
    nvdec: bool = Field(False, description="NVIDIA: Hardware video decoder")

    # Upscaling / frame generation
    dlss: bool = Field(False, description="NVIDIA: DLSS support")
    dlss_version: str | None = Field(None, description="DLSS version, e.g., '3.5'")
    fsr: bool = Field(False, description="AMD: FidelityFX Super Resolution")
    fsr_version: str | None = Field(None, description="FSR version, e.g., '3.0'")
    xess: bool = Field(False, description="Intel: XeSS support")
    frame_generation: bool = Field(False, description="AI frame generation support")

    # Display features
    max_resolution: str | None = Field(None, description="Maximum resolution, e.g., '7680x4320'")
    max_displays: int | None = Field(None, description="Maximum simultaneous displays")
    hdr_support: bool = Field(False, description="HDR output support")
    variable_refresh: bool = Field(False, description="VRR / Adaptive Sync / G-Sync support")


# =============================================================================
# Market / Availability
# =============================================================================


class MarketInfo(BaseModel):
    """Market positioning and pricing information."""

    launch_date: str = Field(..., description="Launch date (YYYY-MM-DD or YYYY-MM)")
    launch_msrp_usd: float | None = Field(None, description="Launch MSRP in USD")
    current_msrp_usd: float | None = Field(None, description="Current MSRP in USD")

    target_market: TargetMarket = Field(..., description="Target market segment")

    # Product positioning
    product_family: str | None = Field(
        None, description="Product family: GeForce RTX, Radeon RX, Arc, etc."
    )
    model_tier: str | None = Field(
        None, description="Tier within family: flagship, high-end, mid-range, entry"
    )

    # Availability
    is_available: bool = Field(True, description="Currently available for purchase")
    is_discontinued: bool = Field(False, description="Product has been discontinued")
    oem_only: bool = Field(False, description="Only available in prebuilt systems")


# =============================================================================
# Complete GPU Entry
# =============================================================================


class GPUEntry(BaseModel):
    """Complete GPU catalog entry.

    Comprehensive specification for a GPU SKU (specific model variant).
    Designed for:
    - Performance prediction and roofline modeling
    - Power/thermal analysis for embodied systems
    - Hardware selection and comparison
    - Cost/value analysis
    """

    # Identity
    id: str = Field(
        ...,
        description="Unique identifier: {vendor}_{family}_{model}_{variant}",
    )
    name: str = Field(..., description="Full product name, e.g., 'NVIDIA GeForce RTX 4090'")
    vendor: GPUVendor = Field(..., description="GPU vendor")

    # Core specifications
    die: DieSpec = Field(..., description="Die/fabrication specifications")
    compute: ComputeResources = Field(..., description="Compute unit specifications")
    clocks: ClockSpeeds = Field(..., description="Clock frequencies")
    memory: MemorySpec = Field(..., description="Memory specifications")
    performance: TheoreticalPerformance = Field(..., description="Theoretical peak performance")
    power: PowerSpec = Field(..., description="Power specifications")

    # Physical (for discrete GPUs)
    board: BoardSpec | None = Field(None, description="Physical board specifications")

    # Features and API support
    features: FeatureSupport | None = Field(None, description="API and feature support")

    # Market information
    market: MarketInfo = Field(..., description="Market and pricing information")

    # Computed efficiency metrics (optional, can be calculated)
    efficiency: EfficiencyMetrics | None = Field(None, description="Efficiency metrics")

    # Relationships
    reference_design: bool = Field(True, description="Is this a reference/Founders Edition?")
    aib_partner: str | None = Field(
        None, description="AIB partner for custom cards: ASUS, MSI, etc."
    )
    parent_gpu_id: str | None = Field(
        None, description="Parent GPU if this is a variant (e.g., Super, Ti, XT)"
    )

    # Metadata
    notes: str = Field("", description="Additional notes")
    datasheet_url: str | None = Field(None, description="Link to datasheet/spec page")
    techpowerup_url: str | None = Field(
        None, description="Link to TechPowerUp GPU Database entry"
    )
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}

    def compute_efficiency_metrics(self) -> EfficiencyMetrics:
        """Compute efficiency metrics from specifications."""
        perf_per_watt = None
        perf_per_mm2 = None
        bw_per_watt = None
        transistors_per_tflop = None

        fp32 = self.performance.fp32_tflops
        tdp = self.power.tdp_watts
        die_size = self.die.die_size_mm2
        transistors = self.die.transistors_billion
        bandwidth = self.memory.memory_bandwidth_gbps

        if fp32 and tdp:
            perf_per_watt = fp32 / tdp

        if fp32 and die_size:
            perf_per_mm2 = fp32 / die_size

        if bandwidth and tdp:
            bw_per_watt = bandwidth / tdp

        if transistors and fp32:
            transistors_per_tflop = (transistors * 1000) / fp32

        return EfficiencyMetrics(
            perf_per_watt_tflops=perf_per_watt,
            perf_per_mm2_tflops=perf_per_mm2,
            bandwidth_per_watt_gbps=bw_per_watt,
            transistors_per_tflop_million=transistors_per_tflop,
        )


# =============================================================================
# GPU Family/Architecture Summary
# =============================================================================


class GPUArchitectureSummary(BaseModel):
    """Summary of a GPU architecture/family for quick reference."""

    id: str = Field(..., description="Architecture identifier, e.g., 'nvidia_ada_lovelace'")
    name: str = Field(..., description="Architecture name")
    vendor: GPUVendor = Field(..., description="Vendor")
    codename: str | None = Field(None, description="Internal codename")

    # Fabrication
    foundry: Foundry = Field(..., description="Primary foundry")
    process_nm: int = Field(..., description="Process node in nm")

    # Generation info
    launch_year: int = Field(..., description="Year of first product launch")
    predecessor: str | None = Field(None, description="Previous architecture ID")
    successor: str | None = Field(None, description="Next architecture ID")

    # Key improvements
    key_features: list[str] = Field(
        default_factory=list, description="Key architectural features"
    )

    # SM/CU architecture changes
    fp32_per_sm: int | None = Field(None, description="FP32 units per SM/CU")
    tensor_cores_per_sm: int | None = Field(None, description="Tensor cores per SM")
    rt_cores_per_sm: int | None = Field(None, description="RT cores per SM")

    model_config = {"extra": "forbid"}
