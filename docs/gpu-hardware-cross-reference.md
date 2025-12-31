# GPU-Hardware Cross-Reference

This document explains how GPUs and hardware platforms are linked in embodied-schemas, and when to use each type of entry.

## The Problem

Embodied AI systems use two distinct types of GPU configurations:

1. **Discrete GPUs** - Standalone graphics cards (RTX 4090, MI300X) that plug into a system
2. **Integrated GPUs** - GPUs embedded in SoCs (Jetson Orin, Apple M4, Qualcomm Snapdragon)

These have different characteristics:

| Aspect | Discrete GPU | Integrated GPU |
|--------|--------------|----------------|
| Memory | Dedicated VRAM (GDDR6X, HBM3) | Shared system RAM (LPDDR5) |
| Power | 75-700W, external power | 5-45W, battery compatible |
| Form factor | PCIe card | SoM, BGA package |
| Deployment | Workstation, server | Edge, robotics, mobile |

**The problem**: How do we model integrated GPUs that are part of a larger hardware platform?

## The Solution: Bidirectional Cross-References

We use two entry types with cross-reference fields:

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   HardwareEntry                      GPUEntry                  │
│   (Platform specs)                   (GPU specs)               │
│                                                                │
│   ┌───────────────────┐             ┌───────────────────┐      │
│   │ Jetson Orin Nano  │             │ Orin Nano GPU     │      │
│   │                   │   gpu_id    │                   │      │
│   │ - Power modes     │ ─────────►  │ - CUDA cores      │      │
│   │ - Interfaces      │             │ - Tensor cores    │      │
│   │ - Form factor     │  ◄───────── │ - Memory spec     │      │
│   │ - Software SDK    │  embedded_  │ - Performance     │      │
│   └───────────────────┘  in_hardware└───────────────────┘      │
│                          _ids                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### HardwareEntry

Captures **platform-level** specifications:
- Physical specs (weight, dimensions, form factor)
- Power delivery (input voltage, power modes, battery compatibility)
- Interfaces (CSI cameras, USB, PCIe, GPIO, CAN bus)
- Environmental (operating temperature, IP rating)
- Software ecosystem (OS, SDK, frameworks)
- Deployment context (edge, robotics, drone)

### GPUEntry

Captures **GPU-specific** specifications:
- Compute resources (CUDA cores, tensor cores, SMs)
- Clock speeds (base, boost, memory)
- Memory (size, type, bandwidth, cache)
- Performance (TFLOPS, TOPS, fill rates)
- Features (CUDA compute capability, API support)

### Cross-Reference Fields

**In HardwareEntry:**
```yaml
gpu_id: nvidia_orin_nano_gpu_8gb_lpddr5  # Reference to GPUEntry
```

**In GPUEntry:**
```yaml
embedded_in_hardware_ids:
  - nvidia_jetson_orin_nano_8gb  # List of HardwareEntry IDs
```

## When to Use Each Entry Type

### Use GPUEntry alone for discrete GPUs

Discrete GPUs don't need a HardwareEntry because they:
- Plug into any compatible system
- Have dedicated memory
- Don't define platform interfaces

```yaml
# data/gpus/nvidia/rtx_4090_pcie_24gb_gddr6x.yaml
id: nvidia_rtx_4090_pcie_24gb_gddr6x
name: NVIDIA GeForce RTX 4090
# ... full GPU specs, no embedded_in_hardware_ids needed
```

### Use both for integrated GPUs

Integrated GPUs need both entries because:
- GPU specs (cores, TFLOPS) live in GPUEntry
- Platform specs (interfaces, power modes) live in HardwareEntry
- Users may query from either direction

```yaml
# data/hardware/nvidia/jetson_orin_nano_8gb.yaml
id: nvidia_jetson_orin_nano_8gb
gpu_id: nvidia_orin_nano_gpu_8gb_lpddr5  # Link to GPU
# ... platform specs (CSI, USB, power modes, SDK)

# data/gpus/nvidia/orin_nano_gpu_8gb_lpddr5.yaml
id: nvidia_orin_nano_gpu_8gb_lpddr5
embedded_in_hardware_ids:
  - nvidia_jetson_orin_nano_8gb  # Back-reference to platform
# ... GPU specs (CUDA cores, tensor cores, TFLOPS)
```

## Code Examples

### Loading and Querying

```python
from embodied_schemas import Registry

registry = Registry.load()

# Get all GPUs
print(f"Total GPUs: {len(registry.gpus)}")  # 16

# Get all hardware platforms
print(f"Total hardware: {len(registry.hardware)}")  # 1
```

### Finding GPU Specs for a Hardware Platform

When you have a Jetson and want to know its GPU capabilities:

```python
from embodied_schemas import Registry

registry = Registry.load()

# Start with hardware platform
hardware = registry.hardware.get("nvidia_jetson_orin_nano_8gb")
print(f"Platform: {hardware.name}")
print(f"Power modes: {[m.name for m in hardware.power.power_modes]}")
# Output: ['7W', '15W']

# Get linked GPU specs
gpu = registry.get_gpu_for_hardware("nvidia_jetson_orin_nano_8gb")
if gpu:
    print(f"GPU: {gpu.name}")
    print(f"CUDA cores: {gpu.compute.cuda_cores}")
    print(f"Tensor cores: {gpu.compute.tensor_cores}")
    print(f"INT8 TOPS: {gpu.performance.int8_tops}")
    print(f"Memory: {gpu.memory.memory_size_gb}GB {gpu.memory.memory_type.upper()}")
```

Output:
```
Platform: NVIDIA Jetson Orin Nano 8GB Developer Kit
Power modes: ['7W', '15W']
GPU: NVIDIA Jetson Orin Nano GPU
CUDA cores: 1024
Tensor cores: 32
INT8 TOPS: 40.0
Memory: 8.0GB LPDDR5
```

### Finding Hardware Platforms for a GPU

When you have a GPU and want to know what platforms use it:

```python
from embodied_schemas import Registry

registry = Registry.load()

# Start with GPU
gpu = registry.gpus.get("nvidia_orin_nano_gpu_8gb_lpddr5")
print(f"GPU: {gpu.name}")

# Get platforms using this GPU
platforms = registry.get_hardware_with_gpu("nvidia_orin_nano_gpu_8gb_lpddr5")
for hw in platforms:
    print(f"Platform: {hw.name}")
    print(f"  Form factor: {hw.physical.form_factor.value}")
    print(f"  TDP: {hw.power.tdp_watts}W")
    print(f"  CSI cameras: {hw.interfaces.camera_csi}")
```

Output:
```
GPU: NVIDIA Jetson Orin Nano GPU
Platform: NVIDIA Jetson Orin Nano 8GB Developer Kit
  Form factor: som
  TDP: 15W
  CSI cameras: 2
```

### Comparing Discrete vs Integrated GPUs

```python
from embodied_schemas import Registry

registry = Registry.load()

# Compare RTX 4090 (discrete) vs Orin Nano (integrated)
rtx4090 = registry.gpus.get("nvidia_rtx_4090_pcie_24gb_gddr6x")
orin = registry.gpus.get("nvidia_orin_nano_gpu_8gb_lpddr5")

print(f"{'Metric':<20} {'RTX 4090':>15} {'Orin Nano':>15}")
print("-" * 50)
print(f"{'CUDA cores':<20} {rtx4090.compute.cuda_cores:>15,} {orin.compute.cuda_cores:>15,}")
print(f"{'Tensor cores':<20} {rtx4090.compute.tensor_cores:>15} {orin.compute.tensor_cores:>15}")
print(f"{'FP32 TFLOPS':<20} {rtx4090.performance.fp32_tflops:>15.1f} {orin.performance.fp32_tflops:>15.2f}")
print(f"{'INT8 TOPS':<20} {rtx4090.performance.tensor_tflops_int8:>15.0f} {orin.performance.int8_tops:>15.0f}")
print(f"{'TDP (W)':<20} {rtx4090.power.tdp_watts:>15} {orin.power.tdp_watts:>15}")
print(f"{'Memory (GB)':<20} {rtx4090.memory.memory_size_gb:>15.0f} {orin.memory.memory_size_gb:>15.0f}")
print(f"{'Memory type':<20} {rtx4090.memory.memory_type.upper():>15} {orin.memory.memory_type.upper():>15}")

# Check if integrated
rtx_platforms = registry.get_hardware_with_gpu(rtx4090.id)
orin_platforms = registry.get_hardware_with_gpu(orin.id)
print(f"{'Integrated':<20} {'No' if not rtx_platforms else 'Yes':>15} {'Yes' if orin_platforms else 'No':>15}")
```

Output:
```
Metric                       RTX 4090       Orin Nano
--------------------------------------------------
CUDA cores                     16,384           1,024
Tensor cores                      512              32
FP32 TFLOPS                      82.6            1.28
INT8 TOPS                         661              40
TDP (W)                           450              15
Memory (GB)                        24               8
Memory type                    GDDR6X          LPDDR5
Integrated                         No             Yes
```

### Filtering Embedded GPUs

```python
from embodied_schemas import Registry

registry = Registry.load()

# Find all GPUs that are embedded in hardware platforms
embedded_gpus = [
    gpu for gpu in registry.gpus
    if gpu.embedded_in_hardware_ids
]

print(f"Embedded GPUs: {len(embedded_gpus)}")
for gpu in embedded_gpus:
    platforms = registry.get_hardware_with_gpu(gpu.id)
    print(f"  {gpu.name}")
    for hw in platforms:
        print(f"    -> {hw.name} ({hw.power.tdp_watts}W)")
```

## Future Extensions

This pattern will be used for other SoC-based platforms:

- **Jetson Orin NX / AGX Orin** - Higher-end Jetson variants
- **Apple M4 Pro/Max** - Mac systems with integrated GPU
- **Qualcomm Snapdragon X Elite** - Windows laptops with Adreno GPU
- **AMD Ryzen AI** - Laptops with RDNA 3 iGPU

Each will have:
- A `GPUEntry` with GPU-specific specs
- A `HardwareEntry` with platform specs
- Cross-references linking them together
