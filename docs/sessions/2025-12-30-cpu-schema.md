# Session Log: CPU Schema and Hardware Expansion

**Date**: December 30, 2025
**Focus**: Adding CPU schemas, expanding GPU catalog, documenting architecture

---

## Context

This session built on the GPU schema work from December 29, adding comprehensive CPU support and significantly expanding the hardware catalog with datacenter, desktop, and mobile processors.

---

## Work Completed

### 1. CPU Schema (`cpu.py`)

Created a complete CPU schema supporting server, desktop, and mobile processors:

| Class | Purpose |
|-------|---------|
| `CPUEntry` | Complete processor specification |
| `CoreConfig` | Hybrid architecture (P+E cores, big.LITTLE) |
| `ClockSpeeds` | Base/boost frequencies per core type |
| `CacheSpec` | L1/L2/L3 hierarchy, 3D V-Cache |
| `MemorySpec` | Memory controller with computed bandwidth |
| `InstructionExtensions` | AVX-512, AMX, SVE, NEON, SGX, SEV |
| `PowerSpec` | TDP with configurable modes |
| `PlatformSpec` | Socket, PCIe, CXL support |
| `IntegratedGraphics` | iGPU specifications |
| `MarketInfo` | Launch date, MSRP, target market |

**Enums**: `CPUVendor`, `CPUArchitecture`, `SocketType`, `TargetMarket`, `ProcessNode`

**Computed field**: `threads_per_watt` for efficiency comparison

### 2. GPU Catalog Expansion

Standardized naming convention: `{model}_{form_factor}_{memory}_{memtype}.yaml`

Added 11 new GPUs (15 total):

| Category | GPUs Added |
|----------|------------|
| **NVIDIA Datacenter** | H100 SXM5, H200 SXM5, A100 SXM4, V100 SXM2, B100 SXM5, B200 SXM5, L40 |
| **AMD Datacenter** | MI300X, MI250X, MI250 |
| **Qualcomm AI** | Cloud AI 100 |

### 3. CPU Data (25 processors)

**Datacenter (8)**:
- Intel Xeon: 6980P (Granite Rapids 128-core), 6766E (Sierra Forest 144-core), Platinum 8592+ (Emerald Rapids), W9-3595X (Sapphire Rapids)
- AMD EPYC: 9965 (Turin 192-core), 9654 (Genoa 96-core), 9754 (Bergamo 128-core), 9575F (Turin V-Cache)

**Desktop (10)**:
- Intel: Core Ultra 9 285K, i9-14900K, i7-14700K, i7-12700K, i5-14600K, i3-14100
- AMD: Ryzen 9 9950X, Ryzen 9 7950X3D, Ryzen 7 9700X, Ryzen 5 9600X, Threadripper 7980X

**Mobile (7)**:
- Intel: Core Ultra 7 268V (Lunar Lake)
- AMD: Ryzen 9 8945HS, Ryzen 7 8845HS (Hawk Point)
- Apple: M4 Pro, M4 Max
- Qualcomm: Snapdragon X Elite X1E-84-100

### 4. Hardware Platform Data

Added first hardware entry:
- `nvidia_jetson_orin_nano_8gb` - Complete developer kit specification
  - Power modes: 7W and 15W
  - Interfaces: 2x CSI, 4x USB3, PCIe Gen3, 40 GPIO
  - Software: JetPack 6.1, TensorRT, DeepStream, Triton

### 5. Chip/SoC Data

- `nvidia_orin_nano_soc` - Orin Nano SoC (6x Cortex-A78AE, 1024 CUDA cores, Ampere)

### 6. Architecture Documentation

Created `docs/architecture.md` explaining the data ownership split:
- **embodied-schemas**: Datasheet facts (vendor specs, Pydantic validated)
- **graphs/hardware_registry**: Analysis params (ops_per_clock, calibrations)

### 7. Registry Integration

Added CPU support to the registry:
- `cpus` CatalogView
- Query methods: `get_cpus_by_vendor()`, `get_cpus_by_architecture()`, `get_cpus_by_market()`, `get_cpus_by_socket()`

---

## Commits

| Commit | Description |
|--------|-------------|
| `bc6d3bf` | Complete GPU schema integration with loaders and registry |
| `d21b05d` | Add datacenter AI GPUs: H100, H200, A100, L40, MI300X |
| `d4a1820` | Add remaining AI GPUs: V100, B100, B200, MI250/X, Cloud AI 100 |
| `30174b9` | Standardize GPU file naming convention |
| `79670ed` | Add architecture doc explaining hardware data ownership |
| `186bdaf` | Add CPU schema with 22 processor entries |
| `6986039` | Add user hardware SKUs: i7-12700K, Ryzen 8845HS/8945HS, Jetson Orin Nano |

---

## Files Created

### Schema
- `src/embodied_schemas/cpu.py` - 422 lines

### CPU Data (25 files)
```
data/cpus/
├── amd/
│   ├── epyc_9575f_sp5.yaml
│   ├── epyc_9654_sp5.yaml
│   ├── epyc_9754_sp5.yaml
│   ├── epyc_9965_sp5.yaml
│   ├── ryzen_5_9600x_am5.yaml
│   ├── ryzen_7_8845hs_bga.yaml
│   ├── ryzen_7_9700x_am5.yaml
│   ├── ryzen_9_7950x3d_am5.yaml
│   ├── ryzen_9_8945hs_bga.yaml
│   ├── ryzen_9_9950x_am5.yaml
│   └── threadripper_7980x_str5.yaml
├── apple/
│   ├── m4_max_bga.yaml
│   └── m4_pro_bga.yaml
├── intel/
│   ├── core_i3_14100_lga1700.yaml
│   ├── core_i5_14600k_lga1700.yaml
│   ├── core_i7_12700k_lga1700.yaml
│   ├── core_i7_14700k_lga1700.yaml
│   ├── core_i9_14900k_lga1700.yaml
│   ├── core_ultra_7_268v_bga.yaml
│   ├── core_ultra_9_285k_lga1851.yaml
│   ├── xeon_6766e_lga4710.yaml
│   ├── xeon_6980p_lga4710.yaml
│   ├── xeon_platinum_8592_lga4677.yaml
│   └── xeon_w9_3595x_lga4677.yaml
└── qualcomm/
    └── snapdragon_x_elite_x1e_84_100_bga.yaml
```

### GPU Data (15 files)
```
data/gpus/
├── amd/
│   ├── mi250_oam_128gb_hbm2e.yaml
│   ├── mi250x_oam_128gb_hbm2e.yaml
│   ├── mi300x_oam_192gb_hbm3.yaml
│   └── rx_7900_xtx_pcie_24gb_gddr6.yaml
├── intel/
│   └── arc_a770_pcie_16gb_gddr6.yaml
├── nvidia/
│   ├── a100_sxm4_80gb_hbm2e.yaml
│   ├── b100_sxm5_192gb_hbm3e.yaml
│   ├── b200_sxm5_192gb_hbm3e.yaml
│   ├── h100_sxm5_80gb_hbm3.yaml
│   ├── h200_sxm5_141gb_hbm3e.yaml
│   ├── l40_pcie_48gb_gddr6.yaml
│   ├── rtx_4080_pcie_16gb_gddr6x.yaml
│   ├── rtx_4090_pcie_24gb_gddr6x.yaml
│   └── v100_sxm2_32gb_hbm2.yaml
└── qualcomm/
    └── cloud_ai_100_pcie_32gb_lpddr4x.yaml
```

### Hardware/Chips
- `data/hardware/nvidia/jetson_orin_nano_8gb.yaml`
- `data/chips/nvidia/orin_nano_soc.yaml`

### Documentation
- `docs/architecture.md`

### Tests
- `tests/test_cpu_integration.py` - 20 tests

---

## Test Results

```
81 tests passed in 2.57s
```

Breakdown:
- `test_analysis_schemas.py`: 20 tests
- `test_cpu_integration.py`: 20 tests (new)
- `test_gpu_integration.py`: 22 tests
- `test_schemas.py`: 19 tests

---

## Usage Examples

```python
from embodied_schemas import Registry, load_cpus

# Load CPUs directly
cpus = load_cpus()
for cpu_id, cpu in cpus.items():
    print(f"{cpu.name}: {cpu.threads_per_watt:.3f} threads/W")

# Use Registry for queries
registry = Registry.load()

# Query by vendor
intel_cpus = registry.get_cpus_by_vendor("intel")
amd_cpus = registry.get_cpus_by_vendor("amd")

# Query by architecture
zen5_cpus = registry.get_cpus_by_architecture("zen5")

# Query by market
datacenter_cpus = registry.get_cpus_by_market("datacenter")

# Query by socket
am5_cpus = registry.get_cpus_by_socket("am5")

# Check registry summary
print(registry.summary())
# {'hardware': 1, 'chips': 1, 'models': 0, 'sensors': 0,
#  'usecases': 0, 'benchmarks': 0, 'gpus': 15, 'gpu_architectures': 0, 'cpus': 25}
```

---

## Architecture Highlights

### CPU Schema Design Decisions

1. **Hybrid Architecture Support**: `CoreConfig` with `p_cores`, `e_cores`, and per-type clocks for Intel Alder/Raptor/Arrow Lake and ARM big.LITTLE

2. **Instruction Set Flexibility**: Separate booleans for AVX-512 variants (BF16, VNNI, FP16), AMX tiles, ARM SVE/SME, security extensions

3. **Memory Controller Specs**: `calculated_bandwidth_gbps` computed field from channels and speed

4. **Process Node Enums**: Both Intel (Intel 7/4/3/20A/18A) and TSMC (N5/N4/N3) nodes supported

5. **Socket Types**: Server (LGA4677, SP5), desktop (LGA1700, AM5), mobile (BGA)

---

## Next Steps

1. Add more CPU architectures as enum values when needed
2. Expand hardware entries for other dev kits (Jetson Orin NX, AGX Orin)
3. Add ARM server CPUs (Ampere Altra, AWS Graviton, NVIDIA Grace)
4. Create CPU architecture summary entries (like GPUArchitectureSummary)
5. Link Jetson GPUs to hardware entries via `gpu_id`

---

## References

- Previous sessions:
  - `docs/sessions/2025-12-21-gpu-schema.md`
  - `docs/sessions/2025-12-29-gpu-schema-completion.md`
- Intel ARK: https://ark.intel.com/
- AMD Product Pages: https://www.amd.com/en/products/
- TechPowerUp GPU Database: https://www.techpowerup.com/gpu-specs/
