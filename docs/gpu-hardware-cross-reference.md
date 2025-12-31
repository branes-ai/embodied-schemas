# GPU-Hardware Cross-Reference

This document explains how GPUs and hardware platforms are linked in embodied-schemas.

## The Core Insight

**All GPUs operate within a hardware platform context.** There is no such thing as a GPU running in isolation:

| GPU | Platform Reality |
|-----|------------------|
| H100 SXM | Physically requires DGX/HGX baseboard - cannot operate outside it |
| H100 PCIe | Needs server with PCIe 5.0 lanes, 700W cooling, host CPUs |
| RTX 4090 | Needs workstation with 850W PSU, adequate airflow, fast CPU |
| Orin Nano | Part of Jetson module requiring carrier board |

The host platform directly affects GPU performance:

- **Kernel launch latency** - Host CPU speed determines how fast kernels are dispatched
- **Data transfer throughput** - PCIe topology, NUMA architecture, CPU memory bandwidth
- **Multi-GPU scaling** - NVLink fabric vs PCIe switches, topology configuration
- **Sustainable performance** - System cooling capacity, power delivery limits

## The Design

We model the complete system using two entry types with cross-references:

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   HardwareEntry                      GPUEntry                  │
│   (Platform specs)                   (GPU specs)               │
│                                                                │
│   ┌───────────────────┐             ┌───────────────────┐      │
│   │ DGX H100          │             │ H100 SXM5         │      │
│   │                   │   gpu_id    │                   │      │
│   │ - 2x Intel Xeon   │ ─────────►  │ - 16896 CUDA      │      │
│   │ - 8x H100 GPUs    │             │ - 80GB HBM3       │      │
│   │ - NVLink fabric   │  ◄───────── │ - 3.35 TB/s BW    │      │
│   │ - 10.2 kW power   │  embedded_  │ - 700W TDP        │      │
│   └───────────────────┘  in_hardware└───────────────────┘      │
│                          _ids                                  │
│                                                                │
│   ┌───────────────────┐             ┌───────────────────┐      │
│   │ Jetson Orin Nano  │             │ Orin Nano GPU     │      │
│   │                   │   gpu_id    │                   │      │
│   │ - 6x Cortex-A78   │ ─────────►  │ - 1024 CUDA       │      │
│   │ - 8GB LPDDR5      │             │ - 32 Tensor       │      │
│   │ - CSI/USB/GPIO    │  ◄───────── │ - 40 TOPS INT8    │      │
│   │ - 7W/15W modes    │  embedded_  │ - 15W TDP         │      │
│   └───────────────────┘  in_hardware└───────────────────┘      │
│                          _ids                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Why Two Entry Types?

**GPUEntry** captures GPU-specific specifications that are constant regardless of deployment:
- Compute resources (CUDA cores, tensor cores, SMs)
- Clock speeds (base, boost, memory)
- Memory interface (size, type, bandwidth, cache)
- Theoretical performance (TFLOPS, TOPS, fill rates)
- Features (CUDA compute capability, API support)

**HardwareEntry** captures platform specifications that define the deployment context:
- Host compute (CPU cores, architecture, memory)
- Physical specs (form factor, weight, dimensions)
- Power delivery (input voltage, modes, cooling capacity)
- Interfaces (PCIe topology, NVLink, network, storage)
- Software ecosystem (OS, SDK, drivers)

### Cross-Reference Fields

**In HardwareEntry:**
```yaml
gpu_id: nvidia_h100_sxm5_80gb_hbm3  # Primary GPU in this platform
```

**In GPUEntry:**
```yaml
embedded_in_hardware_ids:
  - nvidia_dgx_h100        # DGX H100 system
  - nvidia_hgx_h100_8gpu   # HGX 8-GPU baseboard
```

## Platform Categories

### Tightly Coupled (SoC/Module)

GPU and platform are designed, sold, and operate as a unit:

| Platform | GPU | Relationship |
|----------|-----|--------------|
| Jetson Orin Nano | Orin Nano GPU | Same silicon die |
| Jetson AGX Orin | Orin GPU | Same silicon die |
| Apple Mac Studio | M4 Max GPU | Same package |
| DGX H100 | 8x H100 SXM | Fixed configuration |

### Configurable (Server/Workstation)

GPU installs into platform, but platform specs still matter:

| Platform | GPU Options | Platform Impact |
|----------|-------------|-----------------|
| Dell PowerEdge R750xa | 2-4x A100/H100 PCIe | CPU, PCIe lanes, cooling |
| Supermicro GPU Server | 8x L40S | Power delivery, airflow |
| Custom Workstation | RTX 4090 | PSU, case airflow, CPU |

## Code Examples

### Complete System Query

```python
from embodied_schemas import Registry

registry = Registry.load()

# Get a hardware platform
platform = registry.hardware.get("nvidia_jetson_orin_nano_8gb")

# Get platform specs
print(f"Platform: {platform.name}")
print(f"Form factor: {platform.physical.form_factor.value}")
print(f"Power modes: {[m.name for m in platform.power.power_modes]}")
print(f"Interfaces: CSI={platform.interfaces.camera_csi}, USB3={platform.interfaces.usb3}")

# Get GPU specs via cross-reference
gpu = registry.get_gpu_for_hardware(platform.id)
print(f"\nGPU: {gpu.name}")
print(f"Compute: {gpu.compute.cuda_cores} CUDA, {gpu.compute.tensor_cores} Tensor")
print(f"Performance: {gpu.performance.int8_tops} TOPS INT8")
print(f"Memory: {gpu.memory.memory_size_gb}GB {gpu.memory.memory_type.upper()}")
```

Output:
```
Platform: NVIDIA Jetson Orin Nano 8GB Developer Kit
Form factor: som
Power modes: ['7W', '15W']
Interfaces: CSI=2, USB3=4

GPU: NVIDIA Jetson Orin Nano GPU
Compute: 1024 CUDA, 32 Tensor
Performance: 40.0 TOPS INT8
Memory: 8.0GB LPDDR5
```

### Finding All Platforms for a GPU

```python
from embodied_schemas import Registry

registry = Registry.load()

# An H100 might be deployed in multiple platform types
gpu = registry.gpus.get("nvidia_h100_sxm5_80gb_hbm3")

platforms = registry.get_hardware_with_gpu(gpu.id)
for platform in platforms:
    print(f"{platform.name}")
    print(f"  Host CPU: {platform.capabilities.compute_units} cores")
    print(f"  System power: {platform.power.tdp_watts}W")
    print(f"  GPU count: {len([g for g in platform... if 'gpu' in g])}")
```

### System-Level Analysis

```python
from embodied_schemas import Registry

registry = Registry.load()

# For AI inference, we need both GPU and platform specs
platform = registry.hardware.get("nvidia_jetson_orin_nano_8gb")
gpu = registry.get_gpu_for_hardware(platform.id)

# Calculate system-level metrics
system_power = platform.power.tdp_watts
gpu_tops = gpu.performance.int8_tops
tops_per_watt = gpu_tops / system_power

print(f"System: {platform.name}")
print(f"GPU: {gpu.name}")
print(f"System power: {system_power}W")
print(f"INT8 performance: {gpu_tops} TOPS")
print(f"System efficiency: {tops_per_watt:.2f} TOPS/W")

# Compare power modes
for mode in platform.power.power_modes:
    # GPU performance scales with frequency
    freq_ratio = mode.gpu_freq_mhz / gpu.clocks.boost_clock_mhz
    scaled_tops = gpu_tops * freq_ratio
    efficiency = scaled_tops / mode.power_watts
    print(f"  {mode.name}: {scaled_tops:.1f} TOPS, {efficiency:.2f} TOPS/W")
```

Output:
```
System: NVIDIA Jetson Orin Nano 8GB Developer Kit
GPU: NVIDIA Jetson Orin Nano GPU
System power: 15W
INT8 performance: 40.0 TOPS
System efficiency: 2.67 TOPS/W
  7W: 19.6 TOPS, 2.80 TOPS/W
  15W: 40.0 TOPS, 2.67 TOPS/W
```

### Comparing Deployment Options

```python
from embodied_schemas import Registry

registry = Registry.load()

# Compare the same workload across different platforms
platforms_to_compare = [
    "nvidia_jetson_orin_nano_8gb",
    "nvidia_dgx_h100",  # Future entry
]

print(f"{'Platform':<30} {'GPU':<20} {'TOPS':<10} {'Power':<10} {'TOPS/W':<10}")
print("-" * 80)

for platform_id in platforms_to_compare:
    platform = registry.hardware.get(platform_id)
    if not platform:
        continue

    gpu = registry.get_gpu_for_hardware(platform_id)
    if not gpu:
        continue

    tops = gpu.performance.int8_tops or 0
    power = platform.power.tdp_watts or gpu.power.tdp_watts
    efficiency = tops / power if power else 0

    print(f"{platform.name:<30} {gpu.name:<20} {tops:<10.0f} {power:<10.0f} {efficiency:<10.2f}")
```

## Data Modeling Guidelines

### When Adding a New GPU

1. **Create GPUEntry** with GPU-specific specs (cores, memory, TFLOPS)
2. **Create HardwareEntry** for each platform configuration that uses it
3. **Link via cross-references** in both directions

### Example: Adding DGX H100

```yaml
# data/gpus/nvidia/h100_sxm5_80gb_hbm3.yaml (already exists)
id: nvidia_h100_sxm5_80gb_hbm3
embedded_in_hardware_ids:
  - nvidia_dgx_h100
  - nvidia_hgx_h100_8gpu
# ... GPU specs

# data/hardware/nvidia/dgx_h100.yaml (new)
id: nvidia_dgx_h100
name: NVIDIA DGX H100
gpu_id: nvidia_h100_sxm5_80gb_hbm3
capabilities:
  # System-level specs
  peak_tops_int8: 32000.0  # 8x H100
  memory_gb: 640.0  # 8x 80GB
hardware_type: gpu
# Host CPUs
# cpu_id: intel_xeon_platinum_8480_lga4677  # Future link
power:
  tdp_watts: 10200  # Full system
# NVLink topology, networking, storage...
```

## Future Extensions

This unified model enables:

1. **Full system simulation** - Model CPU-GPU interactions, data movement
2. **TCO analysis** - Power, cooling, rack space across platform options
3. **Deployment planning** - Match workloads to appropriate platforms
4. **Performance prediction** - Account for host bottlenecks

Planned platform additions:
- DGX H100 / H200 / B200
- HGX baseboards
- Cloud instances (AWS p5, Azure ND H100)
- Dell/HPE/Supermicro GPU servers
- Apple Mac Studio / Mac Pro
