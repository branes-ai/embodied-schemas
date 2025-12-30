# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-30

### Added

#### CPU Schema (`cpu.py`)
- Complete CPU schema for server, desktop, and mobile processors
- `CPUEntry` - Full processor specification with cores, clocks, cache, memory, power, platform
- `CoreConfig` - Hybrid architecture support (Intel P+E cores, ARM big.LITTLE)
- `ClockSpeeds` - Base/boost frequencies with per-core-type clocks
- `CacheSpec` - L1/L2/L3 hierarchy with 3D V-Cache support
- `MemorySpec` - Memory controller specs (channels, speed, ECC, computed bandwidth)
- `InstructionExtensions` - AVX-512, AMX, SVE, NEON, security extensions (SGX, SEV)
- `PowerSpec` - TDP with configurable power modes
- `PlatformSpec` - Socket, PCIe lanes, CXL support
- `IntegratedGraphics` - iGPU specs for APUs
- `MarketInfo` - Launch date, MSRP, target market
- Enums: `CPUVendor`, `CPUArchitecture`, `SocketType`, `TargetMarket`, `ProcessNode`
- Computed field: `threads_per_watt` efficiency metric

#### CPU Data (25 processors)
**Datacenter (8)**:
- Intel: Xeon 6980P (Granite Rapids), Xeon 6766E (Sierra Forest), Xeon Platinum 8592+ (Emerald Rapids), Xeon W9-3595X (Sapphire Rapids)
- AMD: EPYC 9965 (Turin 192-core), EPYC 9654 (Genoa), EPYC 9754 (Bergamo 128-core), EPYC 9575F (Turin V-Cache)

**Desktop (10)**:
- Intel: Core Ultra 9 285K (Arrow Lake), Core i9-14900K, i7-14700K, i7-12700K, i5-14600K, i3-14100
- AMD: Ryzen 9 9950X, Ryzen 9 7950X3D, Ryzen 7 9700X, Ryzen 5 9600X, Threadripper 7980X

**Mobile (4)**:
- Intel: Core Ultra 7 268V (Lunar Lake)
- AMD: Ryzen 9 8945HS, Ryzen 7 8845HS (Hawk Point)
- Apple: M4 Pro, M4 Max
- Qualcomm: Snapdragon X Elite X1E-84-100

#### Hardware Platform Data
- `nvidia_jetson_orin_nano_8gb` - Complete dev kit specification
  - Power modes (7W/15W), interfaces (CSI, USB, PCIe, GPIO)
  - Software ecosystem (JetPack, TensorRT, DeepStream)
  - Physical and environmental specs

#### Chip/SoC Data
- `nvidia_orin_nano_soc` - Orin Nano SoC specification (6x A78AE, 1024 CUDA cores, Ampere)

#### GPU Data Expansion (15 GPUs total)
**Datacenter AI GPUs**:
- NVIDIA: H100 SXM5, H200 SXM5, A100 SXM4, V100 SXM2, B100 SXM5, B200 SXM5, L40
- AMD: MI300X, MI250X, MI250
- Qualcomm: Cloud AI 100

**Consumer GPUs**:
- NVIDIA: RTX 4090, RTX 4080
- AMD: RX 7900 XTX
- Intel: Arc A770

#### Architecture Documentation
- `docs/architecture.md` - Explains data ownership split between embodied-schemas (datasheet facts) and graphs/hardware_registry (analysis params)

#### Registry Enhancements
- `cpus` CatalogView in Registry
- CPU query methods: `get_cpus_by_vendor()`, `get_cpus_by_architecture()`, `get_cpus_by_market()`, `get_cpus_by_socket()`

#### Testing
- `test_cpu_integration.py` - 20 tests for CPU loader, registry, data integrity, efficiency
- Total: 81 tests passing

### Changed
- GPU file naming standardized to `{model}_{form_factor}_{memory}_{memtype}.yaml`
- Updated `loaders.py` with `load_cpus()` function
- Updated `registry.py` with CPU catalog and query methods
- Updated `__init__.py` with CPU exports

## [0.2.2] - 2025-12-21

### Added
- `gpu.py` - Comprehensive GPU-specific schemas inspired by TechPowerUp GPU Database
  - `GPUEntry` - Complete GPU specification for discrete graphics cards
  - `DieSpec` - Fabrication specs (foundry, process, transistors, die size, chiplet support)
  - `ComputeResources` - Parallel compute units (shaders, CUDA/Stream processors, TMUs, ROPs, Tensor cores, RT cores)
  - `ClockSpeeds` - Base/boost/memory frequencies
  - `MemorySpec` - VRAM configuration (size, type, bus width, bandwidth, cache)
  - `TheoreticalPerformance` - Peak throughput (FP32/FP16/FP64 TFLOPS, tensor ops, fill rates)
  - `PowerSpec` - Power delivery (TDP, idle/gaming power, connectors, PSU recommendation)
  - `BoardSpec` - Physical card specs (slot width, length, PCIe, display outputs)
  - `FeatureSupport` - API support (DirectX, Vulkan, CUDA, DLSS/FSR, ray tracing)
  - `MarketInfo` - Pricing and availability (MSRP, launch date, target market)
  - `EfficiencyMetrics` - Computed metrics (perf/watt, perf/mm², bandwidth/watt)
  - `GPUArchitectureSummary` - Architecture family reference
  - Enums: `GPUVendor`, `Foundry`, `MemoryType`, `TargetMarket`, `PCIeGen`, `PowerConnector`, `DirectXVersion`, `ShaderModel`
- `data/gpus/` - GPU data directory structure (nvidia/, amd/, intel/)
- Example GPU entry: `data/gpus/nvidia/rtx_4090.yaml` - Complete RTX 4090 specification
- GPU schema tests in `test_schemas.py` - 4 new tests (19 total)
  - Minimal entry validation
  - Efficiency metric computation
  - Transistor density calculation
  - Extra field rejection

### Changed
- Updated `__init__.py` exports to include GPU schemas

## [0.2.1] - 2025-12-21

### Added
- CLAUDE.md - Claude Code guidance file for repository onboarding
  - Documents repository role as shared dependency for graphs and Embodied-AI-Architect
  - Build/test/development commands
  - Schema design patterns (verdict-first outputs, ID conventions)
  - Data ownership boundaries (datasheet specs vs roofline/calibration)
  - Guidelines for adding new data and making schema changes
  - Versioning and downstream compatibility notes

### Fixed
- Fixed date in docs/sessions/2025-12-20-initial-setup.md (was incorrectly 2024)

## [0.2.0] - 2025-12-20

### Added

#### Schema Models
- `hardware.py` - Complete hardware platform schemas
  - `HardwareEntry` - Full platform specification with capabilities, physical, environmental, power, and interface specs
  - `HardwareCapability` - Compute and memory capabilities (TOPS, TFLOPS, memory, bandwidth)
  - `ChipEntry` - Raw SoC/chip specifications
  - `PhysicalSpec` - Weight, dimensions, form factor for embodied systems
  - `EnvironmentalSpec` - Operating temperature, IP rating, vibration/shock ratings
  - `PowerSpec` - Power modes, TDP, battery compatibility
  - `InterfaceSpec` - CSI, USB, PCIe, GPIO, CAN bus counts
  - Enums: `HardwareType`, `FormFactor`, `ComputeParadigm`, `OperationType`, `LifecycleStatus`, `Availability`

- `models.py` - ML model schemas for perception workloads
  - `ModelEntry` - Complete model specification with architecture, I/O, accuracy, variants
  - `ModelVariant` - Quantization variants (fp32, fp16, int8) with accuracy delta
  - `ArchitectureSpec` - Backbone, neck, head, params, FLOPs
  - `AccuracyBenchmark` - mAP, mIoU, accuracy metrics per dataset
  - `MemoryRequirements` - Weights, activations, workspace memory
  - `InputSpec`, `OutputSpec` - Model I/O specifications
  - Enums: `ModelType`, `ModelFormat`, `DataType`

- `sensors.py` - Sensor schemas for perception systems
  - `SensorEntry` - Complete sensor specification
  - `CameraSpec` - Resolution, FPS, dynamic range, shutter type
  - `DepthSpec` - Stereo/ToF/structured light specs, range, accuracy
  - `LidarSpec` - Channels, points/sec, FoV, range
  - `ImuSpec` - Gyro/accel range, noise, sample rate
  - `SensorInterface`, `SensorPower`, `SensorPhysical`, `SensorEnvironmental`
  - Enums: `SensorCategory`, `InterfaceType`, `DepthType`

- `usecases.py` - Use case template schemas
  - `UseCaseEntry` - Application constraint templates for drones, quadrupeds, AMRs, etc.
  - `Constraint` - Min/max values with criticality levels
  - `SuccessCriterion` - Measurable success criteria with operators
  - `PerceptionRequirement` - Required tasks, target classes, detection range
  - `PlatformSpec` - Platform type, size class, indoor/outdoor
  - `RecommendedConfig` - Recommended hardware/model/sensor configurations
  - Enums: `ConstraintCriticality`, `PlatformType`, `Operator`

- `benchmarks.py` - Benchmark result schemas
  - `BenchmarkResult` - Complete benchmark with conditions, metrics, verdict
  - `AnalysisResult` - Tool output format with verdict, confidence, suggestion
  - `LatencyMetrics` - Mean, std, percentiles (p50, p90, p95, p99)
  - `ThroughputMetrics` - FPS, samples/sec, tokens/sec
  - `PowerMetrics` - Mean/peak power, energy per inference
  - `MemoryMetrics` - Model size, peak usage, GPU utilization
  - `ThermalMetrics` - Temperature, throttling status
  - `AccuracyMetrics` - Verification against expected accuracy
  - `BenchmarkConditions` - Power mode, batch size, environment
  - Enums: `Verdict`, `Confidence`

- `constraints.py` - Constraint ontology and tier definitions
  - `LatencyTier` - Ultra real-time (<10ms) to batch (>1s)
  - `PowerClass` - Ultra low power (<2W) to datacenter (>100W)
  - `MemoryClass`, `AccuracyClass` - Additional classifications
  - Pre-defined tier specifications with thresholds and use cases
  - Platform implication rules (drone → battery_powered → low_power)
  - Utility functions: `get_latency_tier()`, `get_power_class()`, `get_platform_implications()`

#### Data Infrastructure
- `loaders.py` - YAML loading with Pydantic validation
  - `load_yaml()`, `load_and_validate()` - Single file loading
  - `load_all_from_directory()` - Batch loading with validation
  - Category-specific loaders: `load_hardware()`, `load_models()`, `load_sensors()`, etc.
  - `validate_data_integrity()` - Full catalog validation

- `registry.py` - Unified data access API
  - `Registry` - Central access point for all catalog data
  - `CatalogView` - Queryable view with filtering support
  - Query methods: `get()`, `find()`, `find_one()`
  - Filter support: exact match, list membership, `_min`/`_max` comparisons
  - Relationship queries: `get_compatible_hardware()`, `get_compatible_models()`
  - Benchmark lookup: `get_benchmark()`, `get_benchmarks_for_model()`

#### Data Directory Structure
- `data/hardware/` - Hardware platforms by vendor (nvidia, qualcomm, hailo, google, intel, amd, raspberry_pi)
- `data/chips/` - Raw SoC specifications by vendor
- `data/models/` - ML models by type (detection, segmentation, depth, pose)
- `data/sensors/` - Sensors by category (cameras, depth, lidar)
- `data/usecases/` - Use case templates by platform (drone, quadruped, biped, amr, edge)
- `data/constraints/` - Tier definitions

#### Testing
- `tests/test_schemas.py` - Comprehensive schema validation tests (15 tests)
  - Hardware schema tests (minimal, full, physical specs, extra field rejection)
  - Model schema tests (variant, entry)
  - Sensor schema tests (camera entry)
  - Use case schema tests (constraint, criterion, entry)
  - Benchmark schema tests (metrics, result)
  - Constraint utility tests (tier classification)

#### Project Configuration
- `pyproject.toml` - Package configuration with dev dependencies
- `README.md` - Usage documentation and examples
- `LICENSE` - MIT license
- `.gitignore` - Python/IDE ignores

## [0.1.0] - 2025-12-20

### Added
- Initial repository creation
- Basic project structure

---

## Architecture Decisions

### Data Split with graphs Repository

This package owns **datasheet specs** (vendor-published facts):
- Hardware capabilities (memory, bandwidth, compute units)
- Power specifications and modes
- Physical specs (weight, dimensions, form factor)
- Environmental specs (temp range, IP rating)
- Interface specs (CSI, USB, PCIe counts)

The `graphs` repository owns **analysis-specific data**:
- `ops_per_clock` - Roofline model parameters
- `theoretical_peaks` - Computed performance ceilings
- Calibration data - Measured performance, efficiency curves
- Operation profiles - GEMM, CONV, attention benchmarks

See `Embodied-AI-Architect/docs/plans/shared-schema-repo-architecture.md` for full details.
