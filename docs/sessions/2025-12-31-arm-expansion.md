# Session Log: ARM CPU and Hardware Expansion

**Date**: December 31, 2025
**Focus**: Expanding ARM CPU coverage, hardware platforms, and cross-references

---

## Context

This session built on the December 30 CPU schema work, expanding coverage to ARM server, embedded, and mobile processors. Also added hardware platforms (Jetson variants, Raspberry Pi, Hailo) with complete GPU and chip cross-references.

---

## Work Completed

### 1. ARM CPU Data (11 processors)

| Category | Processor | Architecture | Cores | Notable |
|----------|-----------|--------------|-------|---------|
| **Server** | Ampere Altra Max M128-30 | Neoverse N1 | 128 | 3.0 GHz, 250W |
| **Server** | AmpereOne A192-32X | Custom ARM | 192 | 3.2 GHz, 350W |
| **Server** | AWS Graviton3 | Neoverse V1 | 64 | Cloud-optimized |
| **Server** | AWS Graviton4 | Neoverse V2 | 96 | 7-chiplet design |
| **Server** | NVIDIA Grace | Neoverse V2 | 72 | 117 MB L3, 546 GB/s |
| **Embedded** | NXP i.MX 8M Plus | Cortex-A53 | 4 | 2.3 TOPS NPU |
| **Embedded** | NXP i.MX 93 | Cortex-A55 | 2 | 0.5 TOPS NPU |
| **MCU** | STM32H7 | Cortex-M7+M4 | 2 | 480 MHz, sub-watt |
| **Mobile** | Snapdragon 8 Gen 3 | Cortex-X4 | 8 | 45 TOPS Hexagon |
| **Mobile** | Google Tensor G4 | Cortex-X4 | 8 | Gemini Nano |
| **Mobile** | Samsung Exynos 2400 | Cortex-X4 | 10 | Xclipse 940 GPU |

### 2. Hardware Platforms (11 devices)

| Vendor | Device | Compute | TOPS | Power |
|--------|--------|---------|------|-------|
| NVIDIA | Jetson Orin NX 8GB | 1024 CUDA | 70 | 10-25W |
| NVIDIA | Jetson Orin NX 16GB | 2048 CUDA | 100 | 10-25W |
| NVIDIA | Jetson AGX Orin 32GB | 2048 CUDA | 200 | 15-60W |
| NVIDIA | Jetson AGX Orin 64GB | 2048 CUDA | 275 | 15-60W |
| NVIDIA | Jetson AGX Thor 128GB | Blackwell | 1040 | 50-100W |
| NVIDIA | Jetson TX2 8GB | 256 Pascal | 1.33 | 7.5-15W |
| Raspberry Pi | Pi 5 8GB | BCM2712 | - | 5-25W |
| Raspberry Pi | Pi 4 8GB | BCM2711 | - | 3-15W |
| Hailo | Hailo-8 M.2 | NPU | 26 | 2.5W |
| Hailo | Hailo-8L M.2 | NPU | 13 | 1.5W |
| Hailo | Hailo-10H M.2 | NPU | 40 | 3W |

### 3. GPU Entries for Cross-References (6 GPUs)

Embedded GPUs linked to hardware platforms via `embedded_in_hardware_ids`:

| GPU ID | Form Factor | Compute | Memory |
|--------|-------------|---------|--------|
| `nvidia_orin_nx_gpu_8gb` | Embedded | 1024 CUDA | 8GB LPDDR5 |
| `nvidia_orin_nx_gpu_16gb` | Embedded | 2048 CUDA | 16GB LPDDR5 |
| `nvidia_orin_gpu_32gb` | Embedded | 2048 CUDA | 32GB LPDDR5 |
| `nvidia_orin_gpu_64gb` | Embedded | 2048 CUDA | 64GB LPDDR5 |
| `nvidia_thor_gpu_128gb` | Embedded | Blackwell | 128GB LPDDR5X |
| `nvidia_tx2_gpu_8gb` | Embedded | 256 Pascal | 8GB LPDDR4 |

### 4. Chip/SoC Entries (9 chips)

| Chip | Vendor | Architecture | Notes |
|------|--------|--------------|-------|
| Orin NX SoC | NVIDIA | Ampere | 6-8 A78AE cores |
| Orin SoC | NVIDIA | Ampere | 12 A78AE cores |
| Thor SoC | NVIDIA | Blackwell | 16 cores, automotive |
| Tegra X2 | NVIDIA | Pascal | Legacy Denver + A57 |
| BCM2712 | Broadcom | Cortex-A76 | Pi 5 SoC |
| BCM2711 | Broadcom | Cortex-A72 | Pi 4 SoC |
| Hailo-8 | Hailo | NPU | 26 TOPS INT8 |
| Hailo-8L | Hailo | NPU | 13 TOPS INT8 |
| Hailo-10H | Hailo | NPU | 40 TOPS INT4 |

### 5. CPU Schema Expansion

Extended enums in `cpu.py`:

```python
# New vendors
CPUVendor: nxp, stmicro, google, samsung

# New architectures
CPUArchitecture:
  - Server: neoverse_v1, neoverse_n1
  - Application: cortex_a55, cortex_a53
  - MCU: cortex_m7, cortex_m4, cortex_m33

# New process nodes
ProcessNode: tsmc_n7, tsmc_n28, samsung_14lpc, samsung_3gae, custom

# New sockets
SocketType: lga4926 (Ampere Altra), lga5964 (AmpereOne)
```

---

## Schema Fixes Applied

1. **STM32 power values**: Changed from floats (0.5W) to integers (1W) due to schema constraint
2. **Mobile SoC pcie_version**: Changed `null` to `"0"` (string required)
3. **Mobile SoC graphics**: Renamed `igpu:` to `graphics:` with `has_igpu: true`

---

## Files Created/Modified

### CPU Data (11 files)
```
data/cpus/
├── ampere/
│   ├── altra_max_m128_30.yaml
│   └── ampereone_a192_32x.yaml
├── aws/
│   ├── graviton3.yaml
│   └── graviton4.yaml
├── google/
│   └── tensor_g4.yaml
├── nvidia/
│   └── grace_cpu.yaml
├── nxp/
│   ├── imx8m_plus.yaml
│   └── imx93.yaml
├── qualcomm/
│   └── snapdragon_8_gen3.yaml
├── samsung/
│   └── exynos_2400.yaml
└── stmicro/
    └── stm32h7_480mhz.yaml
```

### Hardware Data (11 files)
```
data/hardware/
├── nvidia/
│   ├── jetson_orin_nx_8gb.yaml
│   ├── jetson_orin_nx_16gb.yaml
│   ├── jetson_agx_orin_32gb.yaml
│   ├── jetson_agx_orin_64gb.yaml
│   ├── jetson_agx_thor_128gb.yaml
│   └── jetson_tx2_8gb.yaml
├── raspberry_pi/
│   ├── raspberry_pi_5_8gb.yaml
│   └── raspberry_pi_4_8gb.yaml
└── hailo/
    ├── hailo_8_m2.yaml
    ├── hailo_8l_m2.yaml
    └── hailo_10h_m2.yaml
```

### GPU Data (6 files)
```
data/gpus/nvidia/
├── orin_nx_gpu_8gb_lpddr5.yaml
├── orin_nx_gpu_16gb_lpddr5.yaml
├── orin_gpu_32gb_lpddr5.yaml
├── orin_gpu_64gb_lpddr5.yaml
├── thor_gpu_128gb_lpddr5x.yaml
└── tx2_gpu_8gb_lpddr4.yaml
```

### Chip Data (9 files)
```
data/chips/
├── nvidia/
│   ├── orin_nx_soc.yaml
│   ├── orin_soc.yaml
│   ├── thor_soc.yaml
│   └── tegra_x2_soc.yaml
├── broadcom/
│   ├── bcm2712.yaml
│   └── bcm2711.yaml
└── hailo/
    ├── hailo_8_chip.yaml
    ├── hailo_8l_chip.yaml
    └── hailo_10h_chip.yaml
```

### Schema
- `src/embodied_schemas/cpu.py` (modified - new enums)

---

## Test Results

```
81 passed in 5.24s
```

All tests passing after schema fixes.

---

## Cross-Reference Validation

Hardware → GPU → Chip links verified:

| Hardware | GPU | Chip |
|----------|-----|------|
| Jetson Orin NX 8GB | `nvidia_orin_nx_gpu_8gb` | `nvidia_orin_nx_soc` |
| Jetson Orin NX 16GB | `nvidia_orin_nx_gpu_16gb` | `nvidia_orin_nx_soc` |
| Jetson AGX Orin 32GB | `nvidia_orin_gpu_32gb` | `nvidia_orin_soc` |
| Jetson AGX Orin 64GB | `nvidia_orin_gpu_64gb` | `nvidia_orin_soc` |
| Jetson AGX Thor 128GB | `nvidia_thor_gpu_128gb` | `nvidia_thor_soc` |
| Jetson TX2 8GB | `nvidia_tx2_gpu_8gb` | `nvidia_tegra_x2_soc` |
| Pi 5 8GB | - | `broadcom_bcm2712` |
| Pi 4 8GB | - | `broadcom_bcm2711` |
| Hailo-8 M.2 | - | `hailo_8_chip` |
| Hailo-8L M.2 | - | `hailo_8l_chip` |
| Hailo-10H M.2 | - | `hailo_10h_chip` |

---

## Coverage Summary

| Category | Before | After | Added |
|----------|--------|-------|-------|
| CPUs | 25 | 36 | +11 |
| Hardware | 1 | 12 | +11 |
| GPUs | 15 | 21 | +6 |
| Chips | 1 | 10 | +9 |

---

## References

- Previous session: `docs/sessions/2025-12-30-cpu-schema.md`
- NVIDIA Jetson: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/
- Ampere Computing: https://amperecomputing.com/
- AWS Graviton: https://aws.amazon.com/ec2/graviton/
- NXP i.MX: https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/i-mx-applications-processors
- ST STM32H7: https://www.st.com/en/microcontrollers-microprocessors/stm32h7-series.html
