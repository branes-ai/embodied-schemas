# Session Log: GPU Schema Design

**Date**: December 21, 2025
**Focus**: Creating comprehensive GPU-specific schemas inspired by TechPowerUp GPU Database

---

## Context

Building on the initial schema repository setup from December 20th, this session focused on creating a dedicated GPU schema that captures all the fields needed for:
- Performance estimation (TFLOPS, fill rates, memory bandwidth)
- Energy estimation (TDP, power modes, efficiency)
- Concurrency estimation (cores, compute units, tensor/RT cores)
- Benchmark performance (theoretical peaks, measured results)
- Cost analysis (MSRP, market pricing, value metrics)

The schema design was inspired by TechPowerUp's GPU Database structure, which maintains comprehensive specifications for thousands of GPUs.

---

## Research Phase

### TechPowerUp GPU Database Analysis

The TechPowerUp GPU Database (https://www.techpowerup.com/gpu-specs/) tracks:
- **Die/Fabrication**: GPU name, architecture, foundry, process node, transistor count, die size
- **Compute Units**: Shaders, CUDA cores (NVIDIA), Stream Processors (AMD), Xe cores (Intel), TMUs, ROPs
- **AI Acceleration**: Tensor cores (NVIDIA), Matrix cores (AMD), XMX engines (Intel)
- **Ray Tracing**: RT cores (NVIDIA), Ray Accelerators (AMD)
- **Clocks**: Base, boost, game clock, memory clock
- **Memory**: Size, type (GDDR6/6X/7, HBM), bus width, bandwidth
- **Power**: TDP/TBP, power connectors, recommended PSU
- **Physical**: Slot width, card length, display outputs, PCIe interface
- **Market**: Launch date, MSRP, target market

### Additional Research

- **Tensor Cores vs CUDA Cores**: Understood the hierarchy of NVIDIA compute resources
- **AMD Stream Processors**: Mapped to shader count for cross-vendor compatibility
- **Power Specifications**: TDP vs TBP vs TGP terminology across vendors
- **Efficiency Metrics**: Performance per watt as key embedded systems metric

---

## Implementation

### Schema Structure

Created `src/embodied_schemas/gpu.py` with the following models:

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `DieSpec` | Fabrication/silicon | gpu_name, architecture, foundry, process_nm, transistors_billion, die_size_mm2, chiplet support |
| `ComputeResources` | Parallel compute | shaders, cuda_cores/stream_processors, tmus, rops, tensor_cores, rt_cores, warp_size |
| `ClockSpeeds` | Frequencies | base_clock_mhz, boost_clock_mhz, memory_clock_mhz, memory_effective_gbps |
| `MemorySpec` | VRAM configuration | memory_size_gb, memory_type, memory_bus_bits, memory_bandwidth_gbps, cache sizes |
| `TheoreticalPerformance` | Peak throughput | fp32_tflops, fp16_tflops, tensor_tflops, int8_tops, pixel/texture fill rates |
| `PowerSpec` | Power delivery | tdp_watts, idle/gaming power, power_connectors, recommended_psu_watts |
| `BoardSpec` | Physical card | slot_width, length_mm, pcie_interface, display outputs, cooling |
| `FeatureSupport` | API/features | DirectX, Vulkan, CUDA compute, DLSS/FSR, ray tracing |
| `MarketInfo` | Pricing/availability | launch_date, msrp, target_market, product_family |
| `EfficiencyMetrics` | Derived metrics | perf_per_watt_tflops, perf_per_mm2_tflops, bandwidth_per_watt |
| `GPUEntry` | Complete entry | Combines all above with identity, relationships, metadata |
| `GPUArchitectureSummary` | Architecture reference | Architecture family summary for quick lookup |

### Key Design Decisions

1. **Vendor-Agnostic Design**
   - `shaders` as unified count across all vendors
   - Vendor-specific fields (`cuda_cores`, `stream_processors`, `xe_cores`) as aliases
   - Supports NVIDIA, AMD, Intel, Qualcomm, ARM Mali, Apple

2. **Chiplet Support**
   - `is_chiplet` flag for MCM designs (AMD RDNA 3, Intel Ponte Vecchio)
   - `chiplet_breakdown` dict for die counts by type

3. **Computed Properties**
   - `transistor_density_mtx_mm2` - Derived from transistors/die size
   - `total_power_capacity_watts` - Calculated from connectors
   - `memory_bits_per_gb` - Useful for detecting memory config

4. **Efficiency Methods**
   - `compute_efficiency_metrics()` method on GPUEntry
   - Returns perf/watt, perf/mm², bandwidth/watt, transistors/TFLOP

5. **Comprehensive Enums**
   - `Foundry` - TSMC, Samsung, Intel, GlobalFoundries, SMIC, UMC
   - `MemoryType` - GDDR5/5X/6/6X/7, HBM/2/2e/3/3e, LPDDR4/4X/5/5X
   - `TargetMarket` - Consumer, workstation, datacenter, embedded, automotive
   - `PowerConnector` - 6-pin, 8-pin, 12VHPWR, 12V-2x6

### Example Entry

Created `data/gpus/nvidia/rtx_4090.yaml` as reference implementation:
- AD102 die (76.3B transistors, 608.5mm², TSMC N4)
- 16,384 CUDA cores, 512 Tensor cores, 128 RT cores
- 24GB GDDR6X, 384-bit bus, 1008 GB/s bandwidth
- 82.58 FP32 TFLOPS, 330 Tensor TFLOPS
- 450W TDP, 12VHPWR connector

### Tests Added

4 new tests in `tests/test_schemas.py`:
1. `test_gpu_entry_minimal` - Validates GPUEntry with required fields
2. `test_gpu_compute_efficiency` - Tests efficiency metric computation
3. `test_die_spec_transistor_density` - Tests computed transistor density
4. `test_gpu_entry_rejects_extra_fields` - Ensures schema strictness

All 19 tests pass.

---

## Files Created/Modified

### New Files
- `src/embodied_schemas/gpu.py` - GPU schema module (~500 lines)
- `src/embodied_schemas/data/gpus/nvidia/rtx_4090.yaml` - Example entry
- `src/embodied_schemas/data/gpus/amd/` - Directory for AMD GPUs
- `src/embodied_schemas/data/gpus/intel/` - Directory for Intel GPUs
- `docs/sessions/2025-12-21-gpu-schema.md` - This session log

### Modified Files
- `src/embodied_schemas/__init__.py` - Added GPU exports
- `tests/test_schemas.py` - Added GPU test class
- `CHANGELOG.md` - Added v0.2.2 entry

---

## Statistics

- **Lines of Code Added**: ~600 (gpu.py + tests + example YAML)
- **New Models**: 12 Pydantic models
- **New Enums**: 10 enumerations
- **Test Count**: 19 (4 new GPU tests)
- **Session Duration**: ~45 minutes

---

## Next Steps

1. **Seed more GPU data** - Add entries for RTX 4080, 4070 Ti, AMD RX 7900 XTX, Intel Arc A770
2. **Add GPU loader** - Extend `loaders.py` with `load_gpus()` function
3. **Registry integration** - Add GPU catalog to Registry class
4. **Cross-reference with hardware.py** - Link GPUEntry to HardwareEntry for SoC-integrated GPUs

---

## References

- [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/)
- [Tom's Hardware GPU Comparison Table](https://forums.tomshardware.com/threads/graphics-card-comparison-table-a-sortable-database-of-desktop-gpus.3163659/)
- [NVIDIA H100 Datasheet](https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf)
- [Tensor Cores vs CUDA Cores](https://www.wevolver.com/article/tensor-cores-vs-cuda-cores)
- [Graphics Card Power Consumption Testing](https://www.tomshardware.com/features/graphics-card-power-consumption-tested)
