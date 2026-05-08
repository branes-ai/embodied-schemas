# RFC 0001: Unified ComputeProduct Schema

**Status:** Draft
**Author:** Theo Omtzigt
**Date:** 2026-05-08
**Target completion:** ~3-4 weeks after RFC sign-off

---

## Summary

Replace the four category-specific hardware schemas (`GPUEntry`, `CPUEntry`,
`NPUEntry`, `ChipEntry`) with a single `ComputeProduct` schema covering every
type of compute product: monolithic chips, MCMs, multi-die packages,
multi-chip boards, and rack-level systems. Category-specific detail lives in
a discriminated `blocks` list. Hierarchy (board contains chips contains dies)
is expressed via a recursive `contains` reference.

This unification:
- Eliminates the gap where `DieSpec` (transistors, die size, foundry,
  process node) is defined and populated only for GPUs.
- Makes downstream queries uniform: "give me FP16 throughput, TDP, die size"
  works the same regardless of category.
- Follows the industry trend: modern compute products (Apple M-series,
  AMD MI300A, Grace-Hopper, Jetson Orin, B100, DGX) cross category lines and
  are increasingly mislabelled by single-category schemas.
- Reduces maintenance: one schema, one folder, one place to add a field
  when vendors disclose new info.

---

## Motivation

### Current state

`embodied-schemas` defines four parallel hardware schemas:

| Schema | File | Files | Folder |
|--------|------|-------|--------|
| `GPUEntry` | `gpu.py` | 22 | `data/gpus/<vendor>/` |
| `CPUEntry` | `cpu.py` | 36 | `data/cpus/<vendor>/` |
| `NPUEntry` | `npu.py` | 4 | `data/npus/<vendor>/` |
| `ChipEntry` | `hardware.py` | 12 | `data/chips/<vendor>/` |

Field coverage is inconsistent across the four schemas:

| Field | GPU | CPU | NPU | Chip |
|-------|-----|-----|-----|------|
| `transistors_billion` | yes (22/22) | no | no | no |
| `die_size_mm2` | yes (22/22) | no | no | no |
| Process node | `process_nm: int` + `process_name: str` | `process_node` enum | none | `process_node_nm: int` |
| Foundry | yes | no | no | no |
| Architecture | yes | yes | no (`generation`) | yes |
| Launch date | yes (`market.launch_date`) | yes (`market.launch_date`) | yes (`launch_date`) | yes (`announcement_date`) |
| MSRP | yes | yes | yes (`msrp_usd`) | no |
| Peak throughput | yes (FP/INT table) | partial | yes (TOPS only) | no |
| Memory BW | yes | yes (DDR rate) | yes (host) | yes |
| TDP | yes | yes | yes | no |

The parallel schemas have already drifted on field naming
(`process_nm` vs `process_node` vs `process_node_nm`), data shape
(`PowerSpec` defined independently in `gpu.py` and `cpu.py`), and feature
coverage (DieSpec only on GPUs).

### The industry has moved past clean categories

Modern compute products defy single-category labels:

| Product | What is it? |
|---------|-------------|
| Apple M3 Max | CPU + GPU + NPU + media engines, one die |
| AMD MI300A | CPU chiplets + GPU chiplets, MCM |
| NVIDIA Grace-Hopper | Grace CPU + Hopper GPU, MCM via NVLink-C2C |
| NVIDIA B100/B200 | dual-die GPU bridged by NV-HBI |
| Jetson Orin | CPU + GPU + 2x DLA + PVA + NVENC, all on one SoC |
| DGX H100 | 8x H100 + dual Sapphire Rapids + NVSwitch + ConnectX, board-level |
| Tesla Dojo | training tile that's neither chip-shaped nor server-shaped |

Forcing these into "GPU" or "CPU" buckets creates two failure modes:
1. **False label** (Jetson Orin lives in `chips/` but consumers want GPU
   throughput numbers).
2. **Lost detail** (B100 is "one GPU" in the registry, but its dual-die nature
   matters for fabric modelling and die-area accounting).

### Downstream consumers want uniform queries

The graphs/ repo (mappers, estimators) and Embodied-AI-Architect
(LLM orchestrator) ultimately want:

> Given a hardware id, return: FP16 peak, TDP, die size, transistor budget,
> launch year, process node.

This query should work the same for an H100, an i7-12700K, a Hailo-8, and a
Jetson Orin. Today it requires four code paths and one of them (CPU/NPU/Chip
die specs) doesn't have answers at all.

---

## Proposed Design

### Three concepts

1. **`ComputeProduct`** -- the unified spine. Same shape for every product type.
2. **`blocks: list[Block]`** -- discriminated union (`kind: cpu|gpu|npu|...`)
   for category-specific detail. A pure GPU has one `gpu` block. Jetson Orin
   has `[cpu, gpu, npu, npu]`.
3. **`contains: list[ProductRef]`** -- hierarchy. Board-level and rack-level
   products reference child products by id. The schema is recursive.

### Schema sketch (illustrative, non-final)

```yaml
id: nvidia_h100_sxm5_80gb
name: NVIDIA H100 SXM5 80GB
vendor: nvidia
kind: chip                     # chip | mcm | board | system

packaging:
  form_factor: sxm5            # chip | sxm | pcie_card | m2 | board | rack
  num_dies: 1                  # 2 for B100, 8+CPU+switch for DGX
  is_chiplet: false
  package_type: monolithic     # monolithic | mcm | chiplet | board | system

physical:
  die_size_mm2: 814            # SUM across dies in package
  transistors_billion: 80      # SUM across dies
  process_node_nm: 4
  process_node_name: "TSMC N4"
  foundry: tsmc

power:
  tdp_watts: 700
  modes: []                    # MAXN/30W/15W for Jetson; PL1/PL2 for Intel; empty for GPUs with single TDP

memory:
  on_package_gb: 80
  type: hbm3
  bandwidth_gbps: 3350
  l2_cache_mb: 50

peak_throughput:               # uniform precision dict, applies to whole product
  fp64: 33.45
  fp32: 66.91
  fp16: 989.4                  # tensor-core peak
  fp8:  1978.9
  int8: 1978.9

market:
  launch_date: "2022-09-20"
  launch_msrp_usd: 30000
  target_market: datacenter
  product_family: NVIDIA H100

contains: []                   # for board/rack: [{id: nvidia_h100_sxm5_80gb, count: 8}, ...]

blocks:
  - kind: gpu
    streaming_multiprocessors: 132
    cuda_cores: 16896
    tensor_cores: 528
    tensor_core_gen: 4
    base_clock_mhz: 1620
    boost_clock_mhz: 1980

provenance:
  source: nvidia_datasheet_2023_04
  datasheet_url: https://resources.nvidia.com/...
  last_updated: "2026-05-08"

extras: {}                     # vendor-specific or category-specific orphans
```

### Worked examples by product type

**Pure GPU (H100 SXM5)**: `kind: chip`, one `gpu` block, no `contains`.

**Pure CPU (i7-12700K)**: `kind: chip`, one `cpu` block:
```yaml
blocks:
  - kind: cpu
    p_cores: 8
    e_cores: 4
    isa: x86_64
    vector_ext: [avx2]
```

**Heterogeneous SoC (Jetson Orin AGX)**: `kind: chip`, multiple blocks:
```yaml
blocks:
  - kind: cpu
    isa: aarch64
    p_cores: 12
    p_core_arch: Cortex-A78AE
  - kind: gpu
    streaming_multiprocessors: 16
    cuda_cores: 2048
  - kind: npu
    name: DLA0
    dataflow: fixed_function
    peak_tops_int8: 25
  - kind: npu
    name: DLA1
    dataflow: fixed_function
    peak_tops_int8: 25
```

**Multi-die package (B100 SXM6)**: `kind: chip`, `num_dies: 2`, single
logical `gpu` block (because B100 presents as one GPU to software; the dual
die is a fabric detail surfaced in `extras` or future fabric blocks):
```yaml
packaging:
  num_dies: 2
  is_chiplet: true
  package_type: chiplet
physical:
  die_size_mm2: 1670           # 2x ~835
  transistors_billion: 208     # SUM
extras:
  inter_die_fabric: NV-HBI
  inter_die_bandwidth_gbps: 10000
```

**Board-level product (DGX H100)**: `kind: board`, `contains` references
chip-level products:
```yaml
kind: board
contains:
  - {id: nvidia_h100_sxm5_80gb, count: 8}
  - {id: intel_xeon_platinum_8480c, count: 2}
  - {id: nvidia_nvswitch_v3, count: 4}
  - {id: nvidia_connectx_7, count: 8}
peak_throughput:               # explicitly stated, NOT auto-summed
  fp16: 7912                   # 8x H100 = 7912 TFLOPS
power:
  tdp_watts: 10200             # whole-system, including networking + cooling
```

Top-level `peak_throughput` and `power.tdp_watts` are stated explicitly (with
a `provenance.source` citation) rather than auto-summed from `contains`.
Auto-summing is fragile: it loses fabric overhead, host CPU power, cooling,
and PSU efficiency.

---

## Migration Plan

### Phase 1 -- Design & RFC review (3-5 days)

- Pydantic models for `ComputeProduct`, `Packaging`, `Physical`, `Power`,
  `Memory`, `PeakThroughput`, `Market`, `Provenance`, plus `Block` discriminated
  union (`CPUBlock`, `GPUBlock`, `NPUBlock`, `DSPBlock`, `CGRABlock`,
  `MediaBlock`, `FabricBlock`).
- JSON Schema export for editor autocomplete.
- Decide tricky cases up-front and document each in this RFC's Decisions section:
  - **D1**: How does a CPU-with-iGPU split? (Two blocks: one CPUBlock + one
    GPUBlock under one product.)
  - **D2**: How does B100 dual-die look? (One product, `num_dies: 2`,
    `is_chiplet: true`, single logical GPUBlock; dual-die fabric in `extras`
    until we have a fabric block schema.)
  - **D3**: How are thermal modes represented uniformly? (`power.modes: []`
    list of typed `PowerMode` records; empty for GPUs with single TDP.)
  - **D4**: What lives in `peak_throughput` for a multi-block product?
    (Maximum across blocks for shared workloads; sum for parallel workloads.
    Authors state explicitly with a citation.)
  - **D5**: How are board/rack products' top-level peaks computed? (Stated
    explicitly with `provenance.source`, not auto-summed.)
- Reference fixtures: ~5 worked examples spanning categories
  (H100, i7-12700K, Jetson Orin, B100 dual-die, DGX H100 board).

**Exit criteria:** RFC merged, Pydantic models in `embodied_schemas/compute_product.py`
loading the 5 fixtures via tests.

### Phase 2 -- Migration tooling (2-3 days)

- `scripts/migrate_to_compute_product.py`: reads old YAMLs, emits new
  `data/products/<vendor>/<id>.yaml`. Mostly mechanical; flags
  low-confidence conversions for human review.
- Round-trip validator: load old, convert, load new, assert structural
  equivalence on the fields that survive migration.
- Field-mapping matrix in `docs/migration-map.md`: every old field → new home,
  with explicit "drop" / "extras" decisions for orphans.

**Exit criteria:** Converter runs cleanly on the 5 fixtures from Phase 1;
diff against hand-written fixtures is empty modulo the documented orphan list.

### Phase 3 -- Bulk migration (~1 week)

- Run converter on all 74 existing files.
- Hand-edit converter-flagged cases (estimated ~10-15 files needing review).
- **Backfill the gap data while we're in the file anyway:** die sizes and
  transistor counts for CPUs/NPUs/SoCs from datasheets / Wikichip /
  Anandtech. This is the only structured reason we'll have to touch all
  those files at once -- don't waste it.
- New folder layout: `data/products/<vendor>/<id>.yaml` (flat, no category
  folder).
- Deprecation notices in `data/{gpus,cpus,npus,chips}/README.md` pointing to
  the new location.

**Exit criteria:** All 74 files migrated; new test suite covers loading every
product file; backfill coverage report shows >80% of files have die size and
transistors populated (research/proprietary chips remain `None`).

### Phase 4 -- Update consumers (1-2 weeks, parallel-able)

- **embodied-schemas**: deprecate `GPUEntry`/`CPUEntry`/`NPUEntry` exports
  with shim adapters that read from `ComputeProduct` and translate to the
  legacy shape. Add `ComputeProduct` exports.
- **graphs/**: implement a `ComputeProduct` loader that populates
  `mapper.physical_spec` from the unified YAML via `base_id` lookup. Build
  `cli/list_hardware_resources.py` against the unified schema.
- **Embodied-AI-Architect**: update prompts or hardware-selection logic that
  uses category-keyed fields.

**Exit criteria:** Both consumer repos pass their test suites against the new
schema. Old shim adapters covered by deprecation tests with a target removal
date.

### Phase 5 -- Sunset legacy schemas (1-2 weeks lag, parallel with normal work)

- Remove deprecated Pydantic models (`GPUEntry`, `CPUEntry`, `NPUEntry`,
  `ChipEntry`).
- Remove `data/{gpus,cpus,npus,chips}/` folders (data already migrated in
  Phase 3).
- Update `CHANGELOG.md`, `architecture.md`, `README.md`.
- Final cross-repo verification: graphs/ and Embodied-AI-Architect on green.

**Exit criteria:** No imports of legacy schemas anywhere in the org.
embodied-schemas package version bumped to a major version (breaking change).

---

## Decisions

This section will accumulate concrete decisions as Phase 1 progresses.
Currently captured:

- **D1 (CPU-with-iGPU split):** Pending Phase 1 design.
- **D2 (B100 dual-die representation):** Pending Phase 1 design.
- **D3 (Thermal mode shape):** Pending Phase 1 design.
- **D4 (peak_throughput aggregation across blocks):** Pending Phase 1 design.
- **D5 (Board/rack peak/power computation):** Pending Phase 1 design.

---

## Risks

1. **Discriminated unions in YAML are not free.** Pydantic v2 handles them
   cleanly with a `kind:` discriminator; YAML editors and humans won't
   validate the union as easily as a flat shape. Plan for editor tooling
   (JSON Schema export) so authors get autocomplete.
2. **The 80/20 trap.** Around 20% of fields are genuinely block-specific.
   Resist the temptation to flatten them onto the spine -- that re-creates
   the original bag-of-optional-fields problem. Block discrimination is what
   makes this schema honest.
3. **Information loss during migration.** Some old fields don't have an
   obvious home (e.g., GPU `nvenc: false`, CPU `socket: lga1700`). Decide
   policy in Phase 1 and document in `docs/migration-map.md`: drop, move to a
   typed `block.platform_metadata`, or keep as `extras: dict[str, Any]`.
4. **External consumers.** If anything outside graphs/ and
   Embodied-AI-Architect consumes embodied-schemas, the deprecation window
   in Phase 4-5 needs to cover them too. Audit before Phase 5.
5. **Backfill quality.** Phase 3 backfill of CPU/NPU/SoC die data depends on
   public datasheets, which vary wildly in quality. Some chips will have
   unknown die size (e.g., proprietary KPUs); leave `None` and document why.

---

## Alternatives considered

### Alternative 1: Add `DieSpec` to existing per-category schemas

Lowest-effort fix for the immediate gap (die size missing on CPU/NPU/Chip).
Three edits: add `DieSpec` to `cpu.py`, `npu.py`, `hardware.py` (ChipEntry).

**Rejected** because it doesn't address the deeper problem -- four parallel
schemas continuing to drift -- and modern multi-category products
(Apple M-series, MI300A, Jetson Orin) still don't have a clean home.

### Alternative 2: Flat ComputeProduct (no discriminated blocks)

Single schema with all possible fields optional. Authors fill in what's
relevant.

**Rejected** because it produces records where 80% of fields are None for
most products, defeats validation, and creates ambiguity (does
`tensor_cores: 0` mean "not a GPU" or "GPU without tensor cores"?).

### Alternative 3: Stay with per-category, add cross-cutting `DieSpec` mixin

Keep four schemas but extract `DieSpec`, `PowerSpec`, `MarketInfo` into
shared mixin classes that all four import.

**Rejected** because it solves the duplication problem but not the modeling
problem -- multi-category products still don't have a clean home, and the
folder structure (`data/gpus/`, `data/cpus/`) still forces classification.

---

## Out of scope

- Workload / benchmark data. Schemas like `BenchmarkResult` and
  `WorkloadProfile` are independent of `ComputeProduct` and will not be
  changed by this RFC.
- Vendor sensors (cameras, IMUs, lidar). `SensorEntry` is in
  `embodied_schemas/sensors.py` and remains unchanged.
- Mission profiles, capability tiers, constraints. Independent.
- Detailed fabric/interconnect schema (NVLink topology, PCIe lane counts,
  CXL coherence). Captured under `extras` in Phase 1; promoted to a typed
  `FabricBlock` in a later RFC if cross-product fabric modeling matures.
