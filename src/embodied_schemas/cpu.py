"""CPU-specific schemas for server, desktop, and mobile processors.

Comprehensive schema capturing specifications needed for:
- Performance estimation (cores, clocks, cache, memory bandwidth)
- Energy estimation (TDP, power modes)
- Workload suitability (instruction sets, accelerators)
- Platform compatibility (socket, chipset, memory support)
"""

from enum import Enum
from pydantic import BaseModel, Field, computed_field


class CPUVendor(str, Enum):
    """CPU silicon vendors."""

    INTEL = "intel"
    AMD = "amd"
    ARM = "arm"
    QUALCOMM = "qualcomm"
    APPLE = "apple"
    AMPERE = "ampere"
    AWS = "aws"  # Graviton
    NVIDIA = "nvidia"  # Grace
    NXP = "nxp"  # i.MX embedded processors
    STMICRO = "stmicro"  # STM32 microcontrollers
    GOOGLE = "google"  # Tensor mobile SoCs
    SAMSUNG = "samsung"  # Exynos mobile SoCs


class CPUArchitecture(str, Enum):
    """CPU microarchitecture families."""

    # Intel
    SAPPHIRE_RAPIDS = "sapphire_rapids"
    EMERALD_RAPIDS = "emerald_rapids"
    GRANITE_RAPIDS = "granite_rapids"
    RAPTOR_LAKE = "raptor_lake"
    ALDER_LAKE = "alder_lake"
    METEOR_LAKE = "meteor_lake"
    LUNAR_LAKE = "lunar_lake"
    ARROW_LAKE = "arrow_lake"

    # AMD
    ZEN4 = "zen4"
    ZEN4C = "zen4c"
    ZEN5 = "zen5"
    ZEN3 = "zen3"
    ZEN2 = "zen2"

    # ARM Server
    NEOVERSE_V2 = "neoverse_v2"
    NEOVERSE_V1 = "neoverse_v1"
    NEOVERSE_N2 = "neoverse_n2"
    NEOVERSE_N1 = "neoverse_n1"

    # ARM Application
    CORTEX_A78 = "cortex_a78"
    CORTEX_X4 = "cortex_x4"
    CORTEX_A720 = "cortex_a720"
    CORTEX_A55 = "cortex_a55"
    CORTEX_A53 = "cortex_a53"

    # ARM Microcontroller
    CORTEX_M7 = "cortex_m7"
    CORTEX_M4 = "cortex_m4"
    CORTEX_M33 = "cortex_m33"

    # Apple
    M1 = "m1"
    M2 = "m2"
    M3 = "m3"
    M4 = "m4"

    # Qualcomm
    ORYON = "oryon"

    # Other
    CUSTOM = "custom"


class SocketType(str, Enum):
    """CPU socket types."""

    # Intel Server
    LGA4677 = "lga4677"  # Sapphire/Emerald Rapids
    LGA4710 = "lga4710"  # Granite Rapids

    # Intel Desktop
    LGA1700 = "lga1700"  # Alder/Raptor Lake
    LGA1851 = "lga1851"  # Arrow Lake

    # AMD Server
    SP5 = "sp5"  # EPYC Genoa/Turin
    SP3 = "sp3"  # EPYC Milan/Rome

    # AMD Desktop
    AM5 = "am5"  # Ryzen 7000+
    AM4 = "am4"  # Ryzen 5000 and earlier
    STRX4 = "strx4"  # Threadripper
    STR5 = "str5"  # Threadripper 7000

    # Mobile/Embedded
    BGA = "bga"  # Soldered
    FCBGA = "fcbga"  # Flip-chip BGA

    # ARM Server
    SXM = "sxm"  # NVIDIA Grace
    OAM = "oam"  # OCP Accelerator Module
    LGA4926 = "lga4926"  # Ampere Altra
    LGA5964 = "lga5964"  # AmpereOne


class TargetMarket(str, Enum):
    """CPU target market segment."""

    DATACENTER = "datacenter"
    WORKSTATION = "workstation"
    DESKTOP_ENTHUSIAST = "desktop_enthusiast"
    DESKTOP_MAINSTREAM = "desktop_mainstream"
    MOBILE_PERFORMANCE = "mobile_performance"
    MOBILE_ULTRABOOK = "mobile_ultrabook"
    MOBILE_GAMING = "mobile_gaming"
    EMBEDDED = "embedded"
    EDGE = "edge"


class ProcessNode(str, Enum):
    """Semiconductor process nodes."""

    # Intel
    INTEL_7 = "intel_7"  # 10nm Enhanced
    INTEL_4 = "intel_4"
    INTEL_3 = "intel_3"
    INTEL_20A = "intel_20a"
    INTEL_18A = "intel_18a"

    # TSMC
    TSMC_N7 = "tsmc_n7"
    TSMC_N5 = "tsmc_n5"
    TSMC_N4 = "tsmc_n4"
    TSMC_N3 = "tsmc_n3"
    TSMC_N3E = "tsmc_n3e"
    TSMC_N28 = "tsmc_n28"

    # Samsung
    SAMSUNG_4LPP = "samsung_4lpp"
    SAMSUNG_14LPC = "samsung_14lpc"
    SAMSUNG_3GAE = "samsung_3gae"

    # Legacy/Embedded
    CUSTOM = "custom"  # Non-standard or undisclosed process


# =============================================================================
# Core Configuration
# =============================================================================


class CoreConfig(BaseModel):
    """CPU core configuration, supporting hybrid architectures."""

    # Total cores
    total_cores: int = Field(..., description="Total physical cores")
    total_threads: int = Field(..., description="Total threads (with SMT/HT)")

    # Hybrid architecture (Intel P+E cores, ARM big.LITTLE)
    p_cores: int | None = Field(None, description="Performance cores (big cores)")
    e_cores: int | None = Field(None, description="Efficiency cores (little cores)")
    p_core_threads: int | None = Field(None, description="Threads per P-core")
    e_core_threads: int | None = Field(None, description="Threads per E-core")

    # Core architecture names (for hybrid)
    p_core_arch: str | None = Field(None, description="P-core microarchitecture name")
    e_core_arch: str | None = Field(None, description="E-core microarchitecture name")

    # SMT/HT
    smt_enabled: bool = Field(True, description="Simultaneous multithreading enabled")
    threads_per_core: int = Field(2, description="Threads per core (1 if no SMT)")


class ClockSpeeds(BaseModel):
    """CPU clock frequency specifications."""

    base_clock_mhz: int = Field(..., description="Base frequency in MHz")
    boost_clock_mhz: int = Field(..., description="Max single-core turbo in MHz")
    all_core_boost_mhz: int | None = Field(
        None, description="All-core turbo frequency in MHz"
    )

    # Hybrid clocks
    p_core_base_mhz: int | None = Field(None, description="P-core base frequency")
    p_core_boost_mhz: int | None = Field(None, description="P-core max turbo")
    e_core_base_mhz: int | None = Field(None, description="E-core base frequency")
    e_core_boost_mhz: int | None = Field(None, description="E-core max turbo")


# =============================================================================
# Cache Hierarchy
# =============================================================================


class CacheSpec(BaseModel):
    """CPU cache hierarchy specifications."""

    l1_data_kb: int | None = Field(None, description="L1 data cache per core in KB")
    l1_instruction_kb: int | None = Field(
        None, description="L1 instruction cache per core in KB"
    )
    l2_cache_mb: float = Field(..., description="L2 cache total in MB")
    l3_cache_mb: float | None = Field(None, description="L3 cache total in MB")

    # Cache details
    l2_per_core: bool = Field(True, description="L2 is per-core (vs shared)")
    l3_shared: bool = Field(True, description="L3 is shared across all cores")

    # AMD 3D V-Cache
    vcache_mb: float | None = Field(None, description="3D V-Cache in MB (AMD)")


# =============================================================================
# Memory Controller
# =============================================================================


class MemorySpec(BaseModel):
    """CPU memory controller specifications."""

    max_memory_gb: int = Field(..., description="Maximum supported memory in GB")
    memory_channels: int = Field(..., description="Number of memory channels")
    memory_type: str = Field(..., description="Memory type: DDR5, DDR4, LPDDR5X, etc.")
    max_memory_speed_mts: int = Field(
        ..., description="Max memory speed in MT/s (megatransfers)"
    )
    memory_bandwidth_gbps: float | None = Field(
        None, description="Peak memory bandwidth in GB/s"
    )

    # ECC support
    ecc_support: bool = Field(False, description="ECC memory support")

    @computed_field
    @property
    def calculated_bandwidth_gbps(self) -> float:
        """Calculate theoretical bandwidth from channels and speed."""
        # Each channel is typically 64-bit (8 bytes)
        return (self.memory_channels * 8 * self.max_memory_speed_mts) / 1000


# =============================================================================
# Instruction Set Extensions
# =============================================================================


class InstructionExtensions(BaseModel):
    """CPU instruction set extensions and accelerators."""

    # SIMD
    avx: bool = Field(False, description="AVX support")
    avx2: bool = Field(False, description="AVX2 support")
    avx512: bool = Field(False, description="AVX-512 support")
    avx512_bf16: bool = Field(False, description="AVX-512 BF16 support")
    avx512_vnni: bool = Field(False, description="AVX-512 VNNI (INT8) support")
    avx512_fp16: bool = Field(False, description="AVX-512 FP16 support")

    # Intel AMX
    amx: bool = Field(False, description="Intel AMX support")
    amx_bf16: bool = Field(False, description="AMX BF16 tiles")
    amx_int8: bool = Field(False, description="AMX INT8 tiles")
    amx_fp16: bool = Field(False, description="AMX FP16 tiles")

    # ARM
    sve: bool = Field(False, description="ARM SVE support")
    sve2: bool = Field(False, description="ARM SVE2 support")
    sme: bool = Field(False, description="ARM SME (Scalable Matrix Extension)")
    neon: bool = Field(False, description="ARM NEON support")

    # AMD
    avx512_supported: bool = Field(False, description="AMD AVX-512 (Zen 4+)")

    # Security
    sgx: bool = Field(False, description="Intel SGX support")
    sev: bool = Field(False, description="AMD SEV support")
    tme: bool = Field(False, description="Total Memory Encryption")


# =============================================================================
# Power Specifications
# =============================================================================


class PowerSpec(BaseModel):
    """CPU power consumption specifications."""

    tdp_watts: int = Field(..., description="Thermal Design Power in watts")
    base_power_watts: int | None = Field(
        None, description="Base power (PBP) in watts"
    )
    max_turbo_power_watts: int | None = Field(
        None, description="Max turbo power (MTP) in watts"
    )

    # Configurable TDP
    configurable_tdp_down_watts: int | None = Field(
        None, description="Minimum configurable TDP"
    )
    configurable_tdp_up_watts: int | None = Field(
        None, description="Maximum configurable TDP"
    )

    # Efficiency
    idle_power_watts: float | None = Field(None, description="Idle power consumption")


# =============================================================================
# Platform / IO
# =============================================================================


class PlatformSpec(BaseModel):
    """CPU platform and I/O specifications."""

    socket: SocketType = Field(..., description="CPU socket type")
    pcie_lanes: int = Field(..., description="Total PCIe lanes from CPU")
    pcie_version: str = Field(..., description="PCIe version: 4.0, 5.0, etc.")

    # Chipset lanes (additional from chipset)
    chipset_pcie_lanes: int | None = Field(
        None, description="Additional PCIe lanes from chipset"
    )

    # CXL support
    cxl_support: bool = Field(False, description="CXL support")
    cxl_version: str | None = Field(None, description="CXL version: 1.1, 2.0, etc.")

    # USB
    usb4_support: bool = Field(False, description="USB4/Thunderbolt support")


# =============================================================================
# Integrated Graphics
# =============================================================================


class IntegratedGraphics(BaseModel):
    """Integrated GPU specifications (for CPUs with iGPU)."""

    has_igpu: bool = Field(..., description="Has integrated graphics")
    igpu_name: str | None = Field(
        None, description="iGPU name: Intel UHD 770, Radeon 780M, etc."
    )
    igpu_cores: int | None = Field(None, description="GPU execution units/cores")
    igpu_base_mhz: int | None = Field(None, description="iGPU base frequency")
    igpu_boost_mhz: int | None = Field(None, description="iGPU boost frequency")
    igpu_tflops_fp32: float | None = Field(None, description="iGPU FP32 TFLOPS")


# =============================================================================
# Market Information
# =============================================================================


class MarketInfo(BaseModel):
    """Market positioning and pricing information."""

    launch_date: str = Field(..., description="Launch date (YYYY-MM-DD or YYYY-QN)")
    launch_msrp_usd: float | None = Field(None, description="Launch MSRP in USD")
    current_msrp_usd: float | None = Field(None, description="Current MSRP in USD")

    target_market: TargetMarket = Field(..., description="Target market segment")
    product_family: str = Field(
        ..., description="Product family: Xeon Platinum, EPYC, Core i9, Ryzen 9, etc."
    )
    product_line: str | None = Field(
        None, description="Product line: 8400 series, 9004 series, etc."
    )

    is_available: bool = Field(True, description="Currently available")
    is_discontinued: bool = Field(False, description="Product discontinued")


# =============================================================================
# Complete CPU Entry
# =============================================================================


class CPUEntry(BaseModel):
    """Complete CPU catalog entry.

    Comprehensive specification for a CPU SKU. Designed for:
    - Performance prediction and workload analysis
    - Power/thermal analysis for embodied systems
    - Hardware selection and comparison
    - Platform compatibility checking
    """

    # Identity
    id: str = Field(
        ...,
        description="Unique identifier: {vendor}_{family}_{model}_{variant}",
    )
    name: str = Field(..., description="Full product name")
    vendor: CPUVendor = Field(..., description="CPU vendor")

    # Architecture
    architecture: CPUArchitecture = Field(..., description="Microarchitecture")
    process_node: ProcessNode | None = Field(None, description="Manufacturing process")

    # Core specifications
    cores: CoreConfig = Field(..., description="Core configuration")
    clocks: ClockSpeeds = Field(..., description="Clock frequencies")
    cache: CacheSpec = Field(..., description="Cache hierarchy")

    # Memory
    memory: MemorySpec = Field(..., description="Memory controller specs")

    # Instructions/Accelerators
    instructions: InstructionExtensions | None = Field(
        None, description="Instruction set extensions"
    )

    # Power
    power: PowerSpec = Field(..., description="Power specifications")

    # Platform
    platform: PlatformSpec | None = Field(None, description="Platform/IO specs")

    # Integrated graphics
    graphics: IntegratedGraphics | None = Field(
        None, description="Integrated graphics (if present)"
    )

    # Market
    market: MarketInfo = Field(..., description="Market information")

    # Metadata
    notes: str = Field("", description="Additional notes")
    datasheet_url: str | None = Field(None, description="Link to datasheet")
    ark_url: str | None = Field(None, description="Intel ARK or AMD product page URL")
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}

    @computed_field
    @property
    def threads_per_watt(self) -> float | None:
        """Compute threads per watt efficiency metric."""
        if self.power.tdp_watts > 0:
            return self.cores.total_threads / self.power.tdp_watts
        return None


# =============================================================================
# CPU Architecture Summary
# =============================================================================


class CPUArchitectureSummary(BaseModel):
    """Summary of a CPU microarchitecture for quick reference.

    Similar to GPUArchitectureSummary, this provides high-level architectural
    info for performance modeling and comparison across CPU generations.
    """

    id: str = Field(
        ...,
        description="Architecture identifier, e.g., 'intel_raptor_lake', 'amd_zen4'",
    )
    name: str = Field(..., description="Architecture name")
    vendor: CPUVendor = Field(..., description="CPU vendor")
    codename: str | None = Field(None, description="Internal codename")

    # Fabrication
    process_node: ProcessNode = Field(..., description="Manufacturing process node")
    process_nm: int | None = Field(None, description="Process size in nm (approximate)")

    # Generation info
    launch_year: int = Field(..., description="Year of first product launch")
    predecessor: str | None = Field(None, description="Previous architecture ID")
    successor: str | None = Field(None, description="Next architecture ID")

    # Target markets
    target_markets: list[TargetMarket] = Field(
        default_factory=list, description="Target market segments"
    )

    # Key architectural features
    key_features: list[str] = Field(
        default_factory=list, description="Key architectural improvements"
    )

    # Core architecture
    is_hybrid: bool = Field(False, description="Uses hybrid P+E core design")
    max_cores: int | None = Field(None, description="Maximum cores in family")
    max_threads: int | None = Field(None, description="Maximum threads in family")

    # Cache architecture (typical per-core values)
    l1_data_kb: int | None = Field(None, description="L1 data cache per core in KB")
    l1_inst_kb: int | None = Field(None, description="L1 instruction cache per core in KB")
    l2_per_core_mb: float | None = Field(None, description="L2 cache per core in MB")
    l3_per_core_mb: float | None = Field(None, description="L3 cache per core in MB (typical)")

    # Instruction extensions introduced
    new_instructions: list[str] = Field(
        default_factory=list,
        description="New instruction extensions introduced in this architecture",
    )

    # Performance characteristics
    ipc_improvement_pct: float | None = Field(
        None, description="IPC improvement vs predecessor (percentage)"
    )
    single_thread_improvement_pct: float | None = Field(
        None, description="Single-thread performance improvement vs predecessor"
    )

    # Memory support
    memory_types: list[str] = Field(
        default_factory=list, description="Supported memory types: DDR5, DDR4, LPDDR5X"
    )
    max_memory_channels: int | None = Field(None, description="Maximum memory channels")

    # Platform
    socket_types: list[SocketType] = Field(
        default_factory=list, description="Compatible socket types"
    )
    pcie_version: str | None = Field(None, description="PCIe version support")

    model_config = {"extra": "forbid"}
