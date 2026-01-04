# Session Log: Schema Refinement (Phase 3)

**Date**: January 3, 2026
**Focus**: CPU Architecture Summaries and NPU/Accelerator Schema

---

## Context

This session completed Phase 3 (Schema Refinement) from the project roadmap:
- Option 7: CPU Architecture Summaries
- Option 8: NPU/Accelerator Schema

Also added Registry query methods for the new catalogs.

---

## Work Completed

### 1. CPU Architecture Summaries

Created `CPUArchitectureSummary` class in `cpu.py` following the pattern from `GPUArchitectureSummary`:

```python
class CPUArchitectureSummary(BaseModel):
    id: str                           # e.g., 'intel_raptor_lake'
    name: str                         # e.g., 'Raptor Lake'
    vendor: CPUVendor
    codename: str | None
    process_node: ProcessNode
    launch_year: int
    predecessor: str | None
    successor: str | None
    key_features: list[str]
    is_hybrid: bool
    max_cores: int | None
    ipc_improvement_pct: float | None
    # ... cache, memory, socket fields
```

**Data Created (6 architectures)**:

| Architecture | Vendor | Launch | Process | Key Features |
|-------------|--------|--------|---------|--------------|
| Raptor Lake | Intel | 2022 | Intel 7 | Hybrid P+E, DDR5, PCIe 5.0 |
| Arrow Lake | Intel | 2024 | Intel 20A | Tile arch, NPU, no HT |
| Granite Rapids | Intel | 2024 | Intel 3 | 128 cores, AMX FP16, CXL 2.0 |
| Zen 4 | AMD | 2022 | TSMC N5 | AVX-512, 96 cores (EPYC) |
| Zen 5 | AMD | 2024 | TSMC N4 | 192 cores, 16% IPC |
| Neoverse V2 | ARM | 2023 | TSMC N5 | SVE2, SME, 96 cores |

### 2. NPU/Accelerator Schema

Created new module `npu.py` with comprehensive schema for AI accelerators:

**Classes**:
- `NPUEntry` - Complete accelerator specification
- `ComputeSpec` - TOPS, MAC units, data types, sparsity
- `MemorySpec` - SRAM, external memory, host memory
- `PowerSpec` - TDP, efficiency metrics
- `SoftwareSpec` - SDK, frameworks, operators
- `PhysicalSpec` - Form factor, interface

**Enums**:
- `NPUVendor` - hailo, google, intel, qualcomm, apple, amd, groq, etc.
- `NPUType` - discrete, integrated, datacenter, embedded
- `NPUInterface` - pcie, m2, usb, internal
- `DataType` - int4, int8, fp16, bf16, fp32, binary

**Data Created (4 NPUs)**:

| NPU | Vendor | Type | TOPS | TDP | Efficiency |
|-----|--------|------|------|-----|------------|
| Hailo-8 | Hailo | Discrete | 26 | 2.5W | 10.4 TOPS/W |
| Hexagon Gen 3 | Qualcomm | Integrated | 45 | 8W | 5.6 TOPS/W |
| Coral Edge TPU | Google | Discrete | 4 | 2W | 2.0 TOPS/W |
| Intel NPU (Meteor Lake) | Intel | Integrated | 10 | 6W | 1.7 TOPS/W |

### 3. Registry Updates

Added new CatalogViews and query methods to `registry.py`:

**New Views**:
- `cpu_architectures` - CPUArchitectureSummary entries
- `npus` - NPUEntry entries

**NPU Query Methods**:
```python
reg.get_npus_by_vendor("hailo")           # Filter by vendor
reg.get_npus_by_type("discrete")          # Filter by type
reg.get_npus_by_tops_range(10, 50)        # Performance range
reg.get_npus_by_efficiency(5.0)           # Min TOPS/watt
```

**CPU Architecture Query Methods**:
```python
reg.get_cpu_architectures_by_vendor("intel")
reg.get_cpu_architecture_for_cpu("intel_core_i9_14900k")
reg.get_cpus_for_architecture("intel_raptor_lake")
```

---

## Files Created

### Schema
- `src/embodied_schemas/npu.py` - NPU/accelerator schema (new)

### CPU Architectures (6 files)
```
data/cpu_architectures/
├── intel/
│   ├── raptor_lake.yaml
│   ├── arrow_lake.yaml
│   └── granite_rapids.yaml
├── amd/
│   ├── zen4.yaml
│   └── zen5.yaml
└── arm/
    └── neoverse_v2.yaml
```

### NPU Data (4 files)
```
data/npus/
├── hailo/
│   └── hailo_8.yaml
├── google/
│   └── coral_tpu.yaml
├── intel/
│   └── npu_meteor_lake.yaml
└── qualcomm/
    └── hexagon_gen3.yaml
```

---

## Files Modified

- `src/embodied_schemas/cpu.py` - Added `CPUArchitectureSummary`
- `src/embodied_schemas/loaders.py` - Added `load_cpu_architectures()`, `load_npus()`
- `src/embodied_schemas/registry.py` - Added views and query methods
- `src/embodied_schemas/__init__.py` - Added exports

---

## Test Results

```
81 passed in 6.52s
```

All tests passing.

---

## Registry Summary

```python
>>> Registry.load().summary()
{
    'hardware': 14,
    'chips': 12,
    'models': 10,
    'sensors': 4,
    'usecases': 4,
    'benchmarks': 0,
    'gpus': 22,
    'gpu_architectures': 0,
    'cpus': 36,
    'cpu_architectures': 6,
    'npus': 4,
    'operators': 20,
    'architectures': 3
}
```

**Total catalog entries**: 131

---

## Phase 3 Status

| Option | Status | Summary |
|--------|--------|---------|
| 7. CPU Architecture Summaries | ✅ Complete | 6 architectures (Intel, AMD, ARM) |
| 8. NPU/Accelerator Schema | ✅ Complete | New module + 4 NPU entries |

---

## References

- Previous session: `docs/sessions/2026-01-02-integration-validation.md`
- Intel Architectures: https://www.intel.com/content/www/us/en/architecture-and-technology
- AMD Zen: https://www.amd.com/en/technologies/zen-core
- ARM Neoverse: https://www.arm.com/products/silicon-ip-cpu/neoverse
- Hailo: https://hailo.ai/products/
- Google Coral: https://coral.ai/products/
