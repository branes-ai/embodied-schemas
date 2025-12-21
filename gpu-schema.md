# GPU Schema

Here's the comprehensive GPU schema

  GPU Schema Summary (src/embodied_schemas/gpu.py)

Schema Hierarchy

  | Model                  | Purpose                | Key Fields                                                                                          |
  |------------------------|------------------------|-----------------------------------------------------------------------------------------------------|
  | DieSpec                | Fabrication/silicon    | gpu_name, architecture, foundry, process_nm, transistors_billion, die_size_mm2, chiplet support     |
  | ComputeResources       | Parallel compute units | shaders, cuda_cores/stream_processors, tmus, rops, tensor_cores, rt_cores, warp_size                |
  | ClockSpeeds            | Frequencies            | base_clock_mhz, boost_clock_mhz, memory_clock_mhz, memory_effective_gbps                            |
  | MemorySpec             | VRAM configuration     | memory_size_gb, memory_type, memory_bus_bits, memory_bandwidth_gbps, l2_cache_mb, infinity_cache_mb |
  | TheoreticalPerformance | Peak throughput        | fp32_tflops, fp16_tflops, tensor_tflops_*, int8_tops, pixel_rate_gpixels, texture_rate_gtexels      |
  | PowerSpec              | Power delivery         | tdp_watts, idle_power_watts, gaming_power_watts, power_connectors, recommended_psu_watts            |
  | BoardSpec              | Physical card          | slot_width, length_mm, pcie_interface, display outputs, cooling                                     |
  | FeatureSupport         | API/features           | DirectX, Vulkan, CUDA compute, DLSS/FSR, NVENC, ray tracing                                         |
  | MarketInfo             | Pricing/availability   | launch_date, launch_msrp_usd, target_market, product_family                                         |
  | EfficiencyMetrics      | Derived metrics        | perf_per_watt_tflops, perf_per_mm2_tflops, bandwidth_per_watt_gbps                                  |

Key Design Decisions

  1. Vendor-agnostic - Supports NVIDIA (CUDA), AMD (Stream Processors), Intel (Xe cores)
  2. Computed fields - Transistor density, power capacity auto-calculated
  3. Chiplet support - Handles MCM designs like AMD RDNA 3
  4. Efficiency methods - compute_efficiency_metrics() derives perf/watt, perf/mm²

Example Entry Created

  data/gpus/nvidia/rtx_4090.yaml - Full RTX 4090 specification with all fields populated.

Sources

  - https://www.techpowerup.com/gpu-specs/
  - https://forums.tomshardware.com/threads/graphics-card-comparison-table-a-sortable-database-of-desktop-gpus.3163659/
  - https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf
  - https://www.wevolver.com/article/tensor-cores-vs-cuda-cores
  - https://www.tomshardware.com/features/graphics-card-power-consumption-tested

