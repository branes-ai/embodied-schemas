# Architecture: Hardware Data Ownership

This document clarifies the relationship between `embodied-schemas` and the `graphs/hardware_registry`, and explains why both exist.

## Overview

Hardware data is intentionally split across two repositories with complementary responsibilities:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Hardware Data Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   embodied-schemas                    graphs/hardware_registry          │
│   (Datasheet Facts)                   (Analysis Parameters)             │
│                                                                         │
│   ┌─────────────────────┐            ┌─────────────────────┐            │
│   │ • Vendor specs      │            │ • ops_per_clock     │            │
│   │ • Physical specs    │            │ • Calibration data  │            │
│   │ • Market info       │            │ • Measured perf     │            │
│   │ • Pydantic schemas  │            │ • Roofline params   │            │
│   └──────────┬──────────┘            └──────────┬──────────┘            │
│              │                                  │                       │
│              │    ┌─────────────────────────────┘                       │
│              │    │                                                     │
│              ▼    ▼                                                     │
│         ┌─────────────────────┐                                         │
│         │  graphs/adapters/   │                                         │
│         │ convert_to_pydantic │                                         │
│         └──────────┬──────────┘                                         │
│                    │                                                    │
│                    ▼                                                    │
│         ┌───────────────────────┐                                       │
│         │ embodied-ai-architect │                                       │
│         │ (LLM tools, verdicts) │                                       │
│         └───────────────────────┘                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Ownership

### embodied-schemas Owns: Datasheet Facts

Static, vendor-published specifications that don't change with measurement:

| Category | Examples |
|----------|----------|
| **Compute specs** | CUDA cores, tensor cores, SMs, compute units |
| **Memory specs** | Size, type (HBM3, GDDR6X), bandwidth, bus width |
| **Power specs** | TDP, power connectors, idle/gaming power |
| **Physical specs** | Die size, transistor count, process node |
| **Market info** | Launch date, MSRP, availability |
| **Features** | CUDA compute capability, DirectX version, DLSS support |

**Example** (`embodied-schemas/data/gpus/nvidia/h100_sxm5_80gb_hbm3.yaml`):
```yaml
id: nvidia_h100_sxm5_80gb_hbm3
memory:
  memory_size_gb: 80.0
  memory_type: hbm3
  memory_bandwidth_gbps: 3350.0
power:
  tdp_watts: 700
market:
  launch_date: "2022-09-20"
  launch_msrp_usd: 30000.0
```

### graphs/hardware_registry Owns: Analysis Parameters

Dynamic, computed, or measured data used for roofline analysis:

| Category | Examples |
|----------|----------|
| **ops_per_clock** | Operations per clock cycle by precision (FP32, FP16, INT8) |
| **theoretical_peaks** | Computed peak FLOPS/TOPS at boost clock |
| **Calibration data** | Measured real-world performance at specific power modes |
| **Efficiency curves** | Power vs performance tradeoffs |
| **Operation profiles** | GEMM, CONV, attention micro-benchmarks |

**Example** (`graphs/hardware_registry/gpu/h100_sxm5.json`):
```json
{
  "id": "h100_sxm5",
  "ops_per_clock": {
    "fp32": 33838,
    "fp16": 999495,
    "int8": 1998990
  },
  "ops_per_clock_notes": "132 SMs x 128 CUDA cores/SM x 2 (FMA) = 33,792 FP32 ops/clock",
  "theoretical_peaks": {
    "fp32": 67000.0,
    "fp16": 1979000.0
  }
}
```

**Calibration data** (`graphs/hardware_registry/gpu/calibrations/`):
```json
{
  "metadata": {
    "gpu_clock": {"sm_clock_mhz": 612, "power_mode_name": "15W"},
    "framework": "pytorch"
  },
  "measured_performance": { ... }
}
```

## Why Two Repositories?

### 1. Different Change Frequencies

- **Datasheet specs** change only when new hardware launches
- **Calibration data** changes with each measurement run
- **ops_per_clock** may be refined as understanding improves

### 2. Different Validation Requirements

- **embodied-schemas**: Strict Pydantic validation, schema versioning
- **graphs**: Flexible JSON, rapid iteration during analysis development

### 3. Different Consumers

- **embodied-schemas**: Used by LLM tools, knowledge base, multiple downstream repos
- **graphs**: Used internally by roofline analyzer, not exposed to LLMs directly

### 4. Separation of Concerns

- **Facts** (what the vendor says) vs **Analysis** (what we compute/measure)
- Keeps graphs focused on analysis logic, not data management
- Allows embodied-schemas to serve multiple analysis engines

## Integration: The Adapter Layer

The `graphs/adapters/pydantic_output.py` bridges both systems:

```python
from embodied_schemas import GraphAnalysisResult, Verdict, Confidence

def convert_to_pydantic(
    analysis_result,      # From graphs internal analysis
    constraint_metric,    # e.g., "latency_ms"
    constraint_threshold  # e.g., 10.0
) -> GraphAnalysisResult:
    """Convert graphs analysis to embodied-schemas verdict-first output."""

    # Uses graphs hardware_registry for roofline analysis
    # Returns embodied-schemas Pydantic model with verdict
    return GraphAnalysisResult(
        verdict=Verdict.PASS,
        confidence=Confidence.HIGH,
        summary="Meets 10ms latency target with 20% headroom",
        ...
    )
```

## File Naming Convention

GPU files in embodied-schemas follow this pattern:
```
{model}_{form_factor}_{memory_size}_{memory_type}.yaml
```

Examples:
- `h100_sxm5_80gb_hbm3.yaml`
- `a100_sxm4_80gb_hbm2e.yaml`
- `rtx_4090_pcie_24gb_gddr6x.yaml`
- `mi300x_oam_192gb_hbm3.yaml`

This enables managing different SKUs (e.g., A100 40GB vs 80GB).

## Cross-References

### From embodied-schemas to graphs

GPU IDs in embodied-schemas should match hardware IDs in graphs where applicable:
- `nvidia_h100_sxm5_80gb_hbm3` ↔ `h100_sxm5` (graphs uses shorter IDs)

### From HardwareEntry to GPUEntry

For SoCs with integrated GPUs (Jetson, Apple M-series):
```yaml
# In hardware entry
gpu_id: nvidia_jetson_orin_gpu  # Reference to GPUEntry
```

```yaml
# In GPU entry
embedded_in_hardware_ids:
  - nvidia_jetson_orin_nano_8gb
  - nvidia_jetson_orin_nx_16gb
```

## Adding New Hardware

### To embodied-schemas (datasheet specs):

1. Create `data/gpus/{vendor}/{model}_{form}_{mem}_{type}.yaml`
2. Follow `GPUEntry` schema from `gpu.py`
3. Include `datasheet_url` for traceability
4. Run `pytest tests/test_gpu_integration.py`

### To graphs/hardware_registry (analysis params):

1. Create `hardware_registry/gpu/{id}.json`
2. Calculate `ops_per_clock` from architecture docs
3. Add calibration data after running benchmarks
4. Document calculation methodology in `*_notes` fields

## Version Compatibility

Both downstream repos pin to compatible versions:

```toml
# In graphs/pyproject.toml
[project.optional-dependencies]
schemas = ["embodied-schemas>=0.2.0,<1.0.0"]

# In embodied-ai-architect/pyproject.toml
dependencies = ["embodied-schemas>=0.2.0,<1.0.0"]
```

Breaking changes to embodied-schemas require coordinated updates to both repos.

## Summary

| Question | Answer |
|----------|--------|
| Why two repos? | Different data types, change frequencies, and consumers |
| Is there duplication? | No - datasheet facts vs analysis parameters |
| How do they integrate? | Through `graphs/adapters/convert_to_pydantic()` |
| Which is authoritative for specs? | embodied-schemas |
| Which is authoritative for roofline? | graphs/hardware_registry |
