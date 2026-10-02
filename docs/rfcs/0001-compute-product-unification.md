# RFC 0001: Unified ComputeProduct Schema

**Status:** Accepted -- in progress (Phases 1-2 closed; Phase 3 in progress; S1 done; S2-S4, 4, 5 not started)
**Author:** Theo Omtzigt
**Date:** 2026-05-08
**Revised:** 2026-10-02 (rev 4)
**Target completion:** Phases 3-5 by end of Q4 2026; SWaP-C² phases S1-S3 before the 1.0 release

## Revision history

| Rev | Date | Change |
|-----|------|--------|
| 1 | 2026-05-08 | Initial draft (#7) |
| 2 | 2026-10-01 | Status brought in line with the implementation (schema v1-v13, 45 catalog products, package 0.15.0). Original schema sketch replaced by the as-built design. D1-D5 recorded as built (D4 partially; completed in rev 3). Migration plan re-baselined. **New requirement R1: track and estimate SWaP-C² per compute product.** |
| 3 | 2026-10-02 | Sign-off decisions: second C of SWaP-C² is Cooling; `HardwareEntry` (and `SystemConfiguration`) absorbed into `ComputeProduct` (D8); estimators live in `scripts/`; D4 resolved -- multi-block peak reported as sum, min and max over blocks. |
| 4 | 2026-10-02 | Cost basis decided (D9): SWaP-C² cost is variable unit cost only, at quantity 1 and 1K; NRE never enters. All sign-off questions closed. |

---

## Summary

Replace the four category-specific hardware schemas (`GPUEntry`, `CPUEntry`,
`NPUEntry`, `ChipEntry`) with a single `ComputeProduct` schema covering every
type of compute product: monolithic chips, MCMs, multi-die packages,
modules, multi-chip boards, and rack-level systems. Category-specific detail
lives in a discriminated `blocks` list attached to each die. Hierarchy (board
contains modules contains chips) is expressed via a recursive `contains`
reference.

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

**Rev 2 adds a requirement (R1):** every compute product must carry a
tracked-or-estimated **SWaP-C²** profile -- Size, Weight, Power, Cost, and
Cooling -- resolvable at each of its operating points. SWaP-C² is the primary
optimization metric for UAVs and other weight- and endurance-bound embodied
platforms, and the current schema cannot answer it.

---

## Status at a glance (2026-10-01)

| Phase | Plan (rev 1) | Actual |
|-------|--------------|--------|
| 1. Design | Models in `compute_product.py`; 5 reference examples; D1-D5 documented | **Mostly done.** Spine plus 9 block kinds shipped additively as schema v1-v13 (KPU #15, GPU #19, CPU #22, NPU #25, CGRA #34, DPU #36, TPU #38, DSP, IO). Decisions were made in code; this revision records them. Of the 5 planned examples, only i7-12700K and Jetson AGX Orin exist; H100, B100 and DGX H100 do not. |
| 2. Migration tooling | `scripts/migrate_to_compute_product.py`, round-trip validator, `docs/migration-map.md` | **Superseded.** Products were hand-authored, with a per-product test, rather than converted. Only `scripts/generate_compute_product_yamls.py` (KPU SKU generator) exists. No migration map. |
| 3. Bulk migration | All legacy files into `data/products/` | **Partial.** 45 products in `data/compute_products/` (folder name differs from plan) across 14 vendors. About a dozen of the 76 legacy files have a counterpart. All other products are new (KPU SKUs, TPUs, DSP IP, Xeon/AmpereOne). No datacenter GPU is migrated. |
| 4. Update consumers | Compatibility shims, then graphs and Embodied-AI-Architect switch | **KPU only.** `data/kpus/` retired; `load_kpus()` is a shim over `load_compute_products()` (#18). No GPU/CPU/NPU/chip shims. |
| 5. Sunset | Remove legacy models, major bump | **Not started.** `data/{gpus,cpus,npus,chips}/` hold 24/36/4/12 files, with no deprecation notices or warnings. |
| S. SWaP-C² (new) | -- | **S1 done** (0.16.0, 2026-10-02): schema, `resolve_swapc2`, D4 aggregation, first module (SECO SOM-SMARC-QCS6490). S2-S4 not started. See [R1](#r1). |

---

## Motivation

### Current state (as of rev 1)

`embodied-schemas` defined four parallel hardware schemas:

| Schema | File | Files | Folder |
|--------|------|-------|--------|
| `GPUEntry` | `gpu.py` | 22 | `data/gpus/<vendor>/` |
| `CPUEntry` | `cpu.py` | 36 | `data/cpus/<vendor>/` |
| `NPUEntry` | `npu.py` | 4 | `data/npus/<vendor>/` |
| `ChipEntry` | `hardware.py` | 12 | `data/chips/<vendor>/` |

Field coverage was inconsistent across the four schemas:

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

The parallel schemas had already drifted on field naming
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
Jetson Orin. Under rev 1 it required four code paths, and one of them
(CPU/NPU/Chip die specs) had no answers at all.

### UAV design is decided on SWaP-C², not on peak throughput (rev 2)

For a UAV, the compute payload is chosen by what it costs the airframe,
not by its TOPS. The relevant query is:

> At the operating point that meets my latency target, how big, how heavy,
> how power-hungry, how expensive, and how hard to cool is this compute
> product -- *including* the heatsink or fan it needs at that point?

Today the answer is scattered and incomplete:

| SWaP-C² axis | Where it lives today | Gap |
|--------------|---------------------|-----|
| Size | `HardwareEntry.physical.dimensions_mm` (dev kits only) | Not on `ComputeProduct` at all |
| Weight | `HardwareEntry.physical.weight_grams`, `CoolingSolutionEntry.weight_g` | Not on `ComputeProduct`; cooling weight does not scale with the power it removes |
| Power | `ComputeProduct.power` with per-profile TDP | Good. Missing input-rail / conversion loss and the cooling's own power draw (fans) |
| Cost | `Market.launch_msrp_usd` (17 of 45 products), `CoolingSolutionEntry.cost_usd` | No volume pricing, no estimate for pre-silicon parts (most KPU SKUs) |
| Cooling | `thermal_profiles[].cooling_solution_id` -> generic catalog entry | Catalog has one entry per mechanism (e.g. a single `active_fan` at 400 g), not sized to the profile |

The constraint side already exists: `CapabilityTierEntry.form_factor`
carries `max_compute_weight_kg` and `max_compute_volume_cm3` (e.g.
`micro_autonomy`: 100 g, 50 cm³). The product side has nothing to check
against it.

There is also a level-of-integration problem. `jetson_agx_orin_64gb` is
`kind: chip` and describes the GA10B die, but its `launch_msrp_usd: 1999`
is the price of the module. A UAV integrator buys the module (plus a
carrier and a heatsink), not the die. SWaP-C² is only meaningful when you
know which level of integration it describes.

---

## Design as built (schema v1-v13)

The implemented design differs from the rev 1 sketch in three structural
ways. Blocks hang off dies instead of the product. Physical facts live on
each `Die` instead of a product-level `physical` section. Power is organized
around named thermal profiles, each bound to a cooling solution.

### Shape

```yaml
id: nvidia_jetson_agx_orin_64gb
name: NVIDIA Jetson AGX Orin 64GB
vendor: nvidia
kind: chip                       # ProductKind: chip | mcm | chiplet | board | system
packaging:
  kind: monolithic               # PackagingKind
  num_dies: 1                    # physical die count
  package_type: flip_chip_bga
lifecycle: production            # LifecycleStatus: engineering_sample ... eol

dies:                            # >= 1; modeling units, see D2
- die_id: ga10b_gpu
  die_role: compute              # compute | memory | io | bridge | mixed
  process_node_id: samsung_8lpp  # -> data/process-nodes/
  die_size_mm2: 455.0
  transistors_billion: 17.0
  silicon_bin: {...}             # per-block transistor decomposition
  clocks: {...}
  blocks:                        # discriminated union on `kind`
  - kind: gpu                    # kpu | gpu | cpu | npu | cgra | dpu | tpu | dsp | io
    ...
  interconnects: []              # die_to_die, package_to_package, ...

performance:                     # product-level peak roll-up
  int8_tops: ...
  bf16_tflops: ...
  fp32_tflops: ...

power:
  tdp_watts: 30.0
  max_power_watts: 60.0
  min_power_watts: 15.0
  idle_power_watts: 5.0
  default_thermal_profile: 30W
  thermal_profiles:              # each names its cooling_solution_id
  - {name: 15W, tdp_watts: 15, clock_mhz: ..., cooling_solution_id: passive_heatsink_large}
  - {name: 30W, tdp_watts: 30, clock_mhz: ..., cooling_solution_id: active_fan}

market: {launch_date, launch_msrp_usd, target_market, product_family, model_tier, is_available}
confidence: theoretical          # DataConfidence: calibrated | interpolated | theoretical | unknown
datasheet_url: ...
last_updated: '2026-05-15'
```

### Divergence from the rev 1 sketch

| Rev 1 sketch | As built | Note |
|--------------|----------|------|
| `blocks` on the product | `dies[].blocks` | Lets chiplet products put different blocks on dies with different process nodes |
| `physical.{die_size_mm2, transistors_billion, process_node_*, foundry}` summed across dies | Per-`Die` fields, `process_node_id` reference into the process-node catalog | Foundry and node name come from the process-node entry |
| `power.modes: []` | `power.thermal_profiles` + `default_thermal_profile`, per-domain operating points | Richer than planned; carries DVFS and cooling binding |
| `peak_throughput: {fp64, fp32, fp16, fp8, int8}` | `performance` typed as `KPUTheoreticalPerformance` | See rename debt below |
| `memory` section | Inside block definitions (e.g. `KPUBlock.memory`) | No product-level memory summary yet |
| `contains: list[ProductRef]` | Not implemented | Deferred; becomes a prerequisite for R1 |
| `provenance` section | Flat `confidence`, `datasheet_url`, `notes` | |
| `extras: dict` | Not present (`extra: forbid` everywhere) | Orphans go in `notes` or a typed field |
| `data/products/` | `data/compute_products/` | |

### Rename debt

Several spine types still carry KPU names because v1 reused them verbatim:
`Power.thermal_profiles: list[KPUThermalProfile]`,
`ComputeProduct.performance: KPUTheoreticalPerformance`,
`Die.silicon_bin: KPUSiliconBin`, `Die.clocks: KPUClocks`. They are already
used for GPU, CPU, TPU and DSP products. They should be renamed to
vendor-neutral names (`ThermalProfile`, `TheoreticalPerformance`,
`SiliconBin`, `Clocks`), with the KPU names kept as aliases, before Phase 5.

---

## Decisions

- **D1 (CPU-with-iGPU / heterogeneous SoC): decided, not yet exercised.**
  A heterogeneous SoC is one `Die` with several entries in `Die.blocks`, one
  per compute fabric. The schema supports this today. However, no catalog
  product uses it yet. `jetson_agx_orin_64gb` carries only its GPU block,
  and the TI TDA4 and Qualcomm SoCs carry only their DSP block. Backfilling
  the remaining blocks (Orin CPU + 2x DLA + PVA, TDA4 A72 + C7x + MMA) is
  part of Phase 3.

- **D2 (multi-die packages): decided -- differs from rev 1.** Each
  distinct die type is a `Die` entry with its own `process_node_id`,
  `die_size_mm2` and blocks. `packaging.num_dies` records the *physical*
  die count. `dies[]` may aggregate identical chiplets into one modeling
  entry. Example: EPYC 9654 has `num_dies: 13` (12 CCDs + IOD) and two
  `dies[]` entries (`epyc_9654_compute_aggregate`, `genoa_iod`). Die-to-die
  links go in `Die.interconnects` with `level: die_to_die`, replacing the
  rev 1 plan to put them in `extras`. B100 is not yet in the catalog. When
  added, it is one product with `num_dies: 2` and one logical GPU block.

- **D3 (thermal modes): decided.** `power.thermal_profiles` is a non-empty
  list of named profiles. Each profile has a TDP, a clock, a
  `cooling_solution_id`, and optional per-power-domain operating points.
  `default_thermal_profile` names one of them and must resolve, which a
  validator checks. Single-TDP products have one profile.

- **D4 (peak throughput across blocks): decided (rev 3).** A product with
  several compute blocks reports its peak in **three forms: sum, min and
  max over its blocks**. Neither form alone answers the questions consumers
  ask:
  - **sum**: every block busy at once. Upper bound for a workload that
    partitions across all engines (e.g. Orin GPU + 2x DLA).
  - **max**: the best single block. What a workload that maps to one
    engine can reach.
  - **min**: the weakest block. Worst-case placement floor.

  Mechanics:
  - Each compute block exposes its own peak (`TheoreticalPerformance`:
    `int8_tops`, `bf16_tflops`, `fp32_tflops`, ...) at the default thermal
    profile. Authored from the datasheet, or derived from block parameters
    where a roll-up exists (KPU tiles, B5).
  - Each entry in `Die.blocks` is one block instance. Two DLAs are two
    entries and count twice in the sum.
  - Non-compute blocks (`kind: io`) are excluded.
  - Aggregation is **per precision, over the blocks that support that
    precision**. A block with no FP32 path is left out of the FP32 sum, min
    and max; it is not counted as zero, which would make every min zero. The
    result records how many blocks contributed to each precision.
  - The three forms are a **computed property** of `ComputeProduct`
    (`performance_by_aggregation: {sum, min, max}`), not authored YAML, so
    they cannot drift from the blocks.
  - `performance` stays as the vendor-stated headline, unchanged, with
    the existing B5 check for KPU products. For a single-block product,
    sum = min = max = the block's peak.
  - For `module`/`board`/`system` products (D6, D8), aggregation runs over
    the union of the product's own `Die.blocks` (if it has dies) and the
    compute blocks of all contained products, expanded by `count`. The
    D5 rule still applies to the stated headline.

- **D5 (board/rack top-level figures): decided as policy, refined by R1.**
  Throughput and TDP of board- and system-level products are *stated*, with
  a citation, not auto-summed from `contains`. Auto-summing loses fabric
  overhead, host power, cooling and PSU losses. **Refinement:** mass, volume
  and BOM cost *are* additive, so for those three axes a `contains`
  roll-up is computed as the default estimate. A stated value overrides it.
  See R1.

- **D6 (new): level of integration.** A `ComputeProduct` describes exactly
  one level of integration. Add `ProductKind.MODULE` (SoM, M.2 card, OAM,
  SXM board) between `chip` and `board`. The Jetson AGX Orin module becomes
  a `module` that `contains` the Orin SoC `chip`. Market and SWaP-C² data
  attach at the level being sold.

- **D7 (new): SWaP-C² is resolved per thermal profile.** Power and cooling
  vary with the operating point, so SWaP-C² is a function of
  `(product, thermal_profile)`, not a scalar per product. See R1.

- **D8 (new, rev 3): `HardwareEntry` is absorbed into `ComputeProduct`.**
  Dev kits, SoMs, M.2 cards, SBCs, mini-PCs, VPX/VNX cards and chassis
  become `ComputeProduct`s at `kind: module`, `board` or `system`.
  `SystemConfiguration` (no catalog data today) becomes a `kind: system`
  product whose `contains` entries carry slot assignments. Its `total_*`
  fields become the D5 mass/volume/cost roll-up. Spine changes this
  requires, all optional, so additive:

  | `HardwareEntry` field | New home |
  |-----------------------|----------|
  | `physical.weight_grams`, `dimensions_mm` | `swapc2.weight`, `swapc2.size` |
  | `physical.form_factor`, `mounting`, `vita_standard`, `sosa_profile`, `conduction_cooled`, `conformal_coated` | `packaging` (new optional fields) |
  | `environmental` | new top-level `environmental` (reuses `EnvironmentalSpec`) |
  | `power.power_modes` | `power.thermal_profiles` |
  | `power.input_voltage_v`, `battery_compatible`, `poe_support` | `swapc2.power` |
  | `interfaces` | new top-level `interfaces` (reuses `InterfaceSpec`) |
  | `software`, `capabilities.frameworks` / `quantization_support` / `inference_runtimes` | new top-level `software` (reuses `SoftwareSpec`, merged) |
  | `capabilities.memory_gb` / `memory_type` / `memory_bandwidth_gbps` | new top-level `memory` summary (the rev 1 sketch's `memory`) |
  | `capabilities.compute_units`, `tensor_cores`, `simd_width`, `sparse_acceleration` | the contained products' blocks |
  | `hardware_type`, `compute_paradigm`, `optimized_for` | dropped; derived from block kinds |
  | `cost_usd` | `swapc2.cost.unit_price_1_usd` |
  | `availability`, `lifecycle_status` | `lifecycle`, `market.is_available` |
  | `suitable_for`, `target_applications` | `market` (new optional lists) |
  | `chip_id`, `gpu_id` | `contains` |
  | `product_url` | top-level `product_url` |

  Validation changes: `chip`/`mcm`/`chiplet` products still require at
  least one `Die`. `module`/`board`/`system` products require
  `dies` or `contains` to be non-empty; a carrier board with no silicon of
  its own has empty `dies`. `contains` references must resolve to
  `ComputeProduct` ids, so the SoCs a module contains (Orin NX/Nano, TX2,
  BCM2711/2712, Hawk Point, Hailo-8L) must be migrated first. The
  unrelated `HardwareEntry`-based `LifecycleStatus` in `hardware.py` is
  removed with it, leaving one `LifecycleStatus`.

- **D9 (new, rev 4): SWaP-C² cost is variable unit cost, never NRE.**
  The Cost axis answers "what does one more unit add to the vehicle's
  bill of materials". It is recorded at exactly two quantities:
  `unit_price_1_usd` and `unit_price_1k_usd`. No other volume tiers are
  recorded. Non-recurring engineering never enters any SWaP-C² value:
  mask sets, design, IP licensing up-front fees, qualification, and tooling
  are neither stored nor amortized into a unit figure. Consequences:
  - For purchased products (modules, chips, cooling), the value is the
    integrator's unit price at that quantity.
  - For in-house or pre-silicon parts (KPU SKUs), the `scripts/` silicon
    estimator produces variable manufacturing cost: wafer cost / good dies
    per wafer + package + test. It must not add an NRE-amortization term,
    and `source` names the estimator so the figure isn't read as a quote.
  - Per-unit IP royalties are variable and may be included. Up-front
    license fees are NRE and may not.
  - `contains` roll-ups (D5) sum unit costs at the same quantity tier.

---

<a id="r1"></a>

## Requirement R1: SWaP-C² tracking and estimation

### Definition

SWaP-C² = **S**ize, **W**eight, **a**nd **P**ower, **C**ost, **C**ooling
(second C confirmed as Cooling, 2026-10-01).
For a compute product at operating point *p*:

| Axis | Quantity | Unit |
|------|----------|------|
| Size | Envelope L x W x H, volume, board footprint | mm, cm³, mm² |
| Weight | Mass | g |
| Power | Sustained electrical draw at *p*, at the input rail | W |
| Cost | Variable unit cost at quantity 1 and 1K; never NRE (D9) | USD |
| Cooling | Cooling solution required at *p*, and *its* size, weight, power and cost | -- |

The cooling axis is what makes a product's own SWaP-C numbers incomplete.
A 15 W module can need a 30 g fin stack passively or a 250 g heatsink at a
50 °C ambient. A 50 W profile can need a fan that adds weight, cost and
its own electrical load. The cooling contribution is therefore folded into
the other four axes when a profile is resolved.

### Requirements

- **R1.1 Track.** Vendor-published values (datasheet dimensions, mass,
  input power, list price) are recorded with a citation.
- **R1.2 Estimate.** When no published value exists, which is the normal
  case for pre-silicon KPU SKUs, IP blocks and bare SoCs, a value is
  estimated from a documented model. The model name is recorded with the
  value.
- **R1.3 Provenance on every value.** Each SWaP-C² number carries its
  basis (`datasheet | measured | derived | estimated`), a `DataConfidence`,
  and a source. An estimate must never read as a datasheet fact.
- **R1.4 Per operating point.** SWaP-C² resolves for each entry in
  `power.thermal_profiles`. Profile power plus the bound cooling solution
  gives power, weight, size and cost for that profile.
- **R1.5 Level-aware and rollable.** SWaP-C² attaches at the product's
  level of integration (D6). Modules, boards and systems roll up mass,
  volume and cost from `contains` (D5 refinement).
- **R1.6 Checkable.** A resolved SWaP-C² can be compared field by field
  against `CapabilityTierEntry.form_factor` and mission-profile power
  budgets.
- **R1.7 Non-breaking.** All new fields are optional. Adding them is a
  minor version bump.

### Proposed schema (illustrative, non-final)

A shared value wrapper carries provenance:

```python
class SourcedValue(BaseModel):
    """A SWaP-C² quantity with its provenance."""
    value: float
    basis: Literal["datasheet", "measured", "derived", "estimated"]
    confidence: DataConfidence
    source: str                      # citation, or estimator id + version for 'estimated'
    notes: str = ""


class SourcedDimensions(BaseModel):
    """An L x W x H envelope with its provenance (one source for all three)."""
    length_mm: float
    width_mm: float
    height_mm: float
    basis: Literal["datasheet", "measured", "derived", "estimated"]
    confidence: DataConfidence
    source: str
    notes: str = ""
```

A new optional `swapc2` section on `ComputeProduct` holds the
product's own physical facts, independent of operating point:

Shape only; the values below are placeholders, not catalog data:

```yaml
swapc2:
  size:
    dimensions_mm:   {length_mm: <L>, width_mm: <W>, height_mm: <H>,     # SourcedDimensions
                      basis: datasheet, confidence: calibrated,
                      source: "<module datasheet, section>"}
    volume_cm3:      {value: <L*W*H/1000>, basis: derived, confidence: calibrated,
                      source: "dimensions_mm"}
  weight:
    mass_g:          {value: <g>, basis: estimated, confidence: theoretical,
                      source: "module_mass_v1(area, layers, shield)"}
  power:
    input_voltage_v: [<min>, <max>]
    conversion_efficiency: {value: <0..1>, basis: estimated, confidence: theoretical,
                            source: "<converter model>"}
    battery_compatible: <bool>       # from HardwareEntry.power (D8)
    poe_support: <bool>              # from HardwareEntry.power (D8)
  cost:                              # variable unit cost only -- no NRE (D9)
    unit_price_1_usd:  {value: <usd>, basis: datasheet, confidence: calibrated,
                        source: "<vendor list price, date>"}
    unit_price_1k_usd: {value: <usd>, basis: estimated, confidence: theoretical,
                        source: "<volume-discount model>"}
```

Operating temperature, IP rating, vibration and shock do not go in
`swapc2`. They go in the top-level `environmental` section (D8).
The SWaP-C² fit check reads them, for example the ambient temperature
that sizes the cooling solution.

`Market.launch_msrp_usd` stays for compatibility. The default for
`swapc2.cost.unit_price_1_usd` is derived from it, with
`basis: datasheet`.

The **cooling** axis stays where D3 already put it, on
`thermal_profiles[].cooling_solution_id`. `CoolingSolutionEntry` is
extended so a profile resolves to real numbers:

| New `CoolingSolutionEntry` field | Purpose |
|----------------------------------|---------|
| `volume_cm3` / `dimensions_mm` | Size contribution |
| `parasitic_power_w` | Fan / pump electrical load, **measured at the input rail** (not behind the product's converter) |
| `mass_g_per_w`, `volume_cm3_per_w`, `cost_usd_per_w` (optional) | Sizing model, so one mechanism scales with dissipated power instead of using one fixed weight |
| `max_total_w` (exists) | Validates that the bound solution can remove the profile's TDP |

### Resolution

A pure, deterministic helper in this package resolves a profile, in the same
spirit as `check_performance_rollup`:

```python
def resolve_swapc2(product: ComputeProduct, profile: str | None = None,
                   cooling: dict[str, CoolingSolutionEntry] | None = None) -> ResolvedSWaPC2:
    """Product SWaP-C plus the bound cooling solution at one thermal profile."""
```

with:

- `power_w = profile.tdp_watts / conversion_efficiency + cooling.parasitic_power_w`.
  Power boundary: the vehicle's input rail. Product TDP is behind the
  product's own converter. Cooling parasitic power is already input-rail
  power, so conversion loss is not applied to it twice.
- `mass_g = product.mass_g + cooling.mass_g(profile.tdp_watts)`
- Size is an **envelope**, not a sum of part volumes. The cooling solution
  is assumed to mount on the product's top face. It occupies the product
  footprint, and its height is `cooling.volume_cm3(W) / footprint`. A
  cooling solution with a stated `dimensions_mm` uses that instead, with
  the footprint taken as the larger of the two. The result:
  `envelope_mm = (max L, max W, product H + cooling H)`, and
  `envelope_cm3` = its product. The fit check against
  `max_compute_volume_cm3` uses `envelope_cm3`.
- `cost_usd = product.unit_price(qty) + cooling.cost_usd(profile.tdp_watts)`, `qty` in {1, 1000}
- each result keeps the *weakest* confidence of its inputs, and reports
  `basis: estimated` if any input was estimated.

The resolved record is what graphs and Embodied-AI-Architect consume.

### Estimation models (filling gaps per R1.2)

| Gap | Estimator | New inputs needed here |
|-----|-----------|------------------------|
| Silicon cost (bare die, pre-silicon SKUs) | dies per wafer from `die_size_mm2`; yield from a defect-density model; plus per-unit package and test adders. Variable cost only: no mask set, design or tooling amortization (D9) | `ProcessNodeEntry.wafer_cost_usd`, `defect_density_per_cm2` (new, optional) |
| Package size | from die area and package type | package-type size table |
| Module mass / volume | from module dimensions, PCB layer count, shield/heat-spreader presence | none beyond `swapc2.size` |
| Cooling mass / volume / cost | `CoolingSolutionEntry` per-W sizing model at the profile's TDP and the tier's ambient temperature | fields listed above |
| Board/system mass, volume, cost | sum over `contains` (D5 refinement) | `contains`, `ProductKind.MODULE` |

**Ownership.** Per this repo's charter, datasheet facts and the
*inputs* to these models (wafer cost, cooling sizing coefficients) live
here. **Decided (rev 3):** the estimator code that writes
`basis: estimated` values into catalog YAMLs lives in this repo's
`scripts/`, next to `generate_compute_product_yamls.py`, so estimates are
reproducible and versioned with the data they produce. Each estimator has
an id and version, which goes into `SourcedValue.source`. The deterministic
`resolve_swapc2` roll-up lives here. **Airframe-coupled metrics live in graphs or
Embodied-AI-Architect.** Examples: the extra hover power a payload costs
(for a multirotor, P_hover ∝ m^(3/2), so dP/dm ≈ 1.5·P_hover/m_total),
endurance impact against a `BatteryEntry`, and SWaP-C²-weighted ranking.
These depend on the vehicle, not the compute product. The verdict-first SWaP-C² fit
check against a capability tier (`PASS` / `FAIL` / `PARTIAL` with the
limiting axis as `suggestion`) also lives downstream.

### Coverage target

- Every product with `target_market` in {edge, embodied, automotive} that
  is sold as a chip or module carries `swapc2` with at least estimated
  values on all five axes.
- Datacenter products: optional.
- IP blocks (Cadence, CEVA, Synopsys, Vitis DPU) have no physical form.
  They carry silicon-cost and power only, and are flagged as not resolvable
  for size and weight.

---

## Migration plan (re-baselined 2026-10-01)

Phases 1-2 are closed: their remaining items moved into the phases below.
The incremental "new block kind + hand-authored YAML + per-product test"
pattern has worked and replaces the converter approach.

### Phase 3 -- Finish catalog migration (remaining)

- Migrate the ~64 legacy files without a counterpart, prioritizing the
  RFC's reference products (H100, B100, DGX H100) and edge parts (Orin
  NX/Nano, i.MX 8M Plus, Hailo-8L, Raspberry Pi BCM2711/2712).
- D1 backfill: add the missing blocks to heterogeneous SoCs (Orin, Thor,
  TDA4, Qualcomm), each with its own peak so the D4 sum/min/max is meaningful.
- Migrate the SoCs that `data/hardware/` modules reference (prerequisite
  for S3).
- Deprecation notices in `data/{gpus,cpus,npus,chips}/README.md`.
- Write `docs/migration-map.md` after the fact: legacy field to new home.

**Exit:** every legacy id has a `ComputeProduct` counterpart, or a recorded reason it was dropped.

### Phase S -- SWaP-C² (new, R1)

- **S1 Schema (minor bump):** `SourcedValue`, `ComputeProduct.swapc2`,
  `ProductKind.MODULE`, `contains: list[ProductRef]` (with optional slot
  assignment), the D8 spine additions (`environmental`, `interfaces`,
  `software`, `memory`, packaging form-factor fields), the
  `performance_by_aggregation` property (D4), `CoolingSolutionEntry`
  extensions, optional `ProcessNodeEntry` cost fields, `resolve_swapc2()`.
  Schema tests plus one worked example per level (chip, module, board).
- **S2 Cooling and cost inputs:** re-author the cooling catalog with per-W
  sizing coefficients. Add wafer cost and defect density to the process
  nodes the catalog uses (TSMC N4/N7/16FFP, Samsung 8LPP, GF 12FDX, ...).
- **S3 Absorb `data/hardware/` (D8) and backfill:** convert all 20
  `HardwareEntry` files: 7 Jetson SoMs and 3 Hailo M.2 cards become
  `module`s; 2 Raspberry Pi, 2 AMD NUCs and the Elma JetSys 5330 become
  `board`/`system`; the 5 Elma VPX/VNX cards and chassis become
  `board`/`system`. Do the UAV-relevant modules first. Depends on the Phase 3
  migration of the SoCs they contain. `load_hardware()` becomes a shim
  over `load_compute_products()`, as `load_kpus()` already is. Populate
  `swapc2` from datasheets; run the `scripts/` estimators for KPU SKUs.
- **S4 Downstream:** graphs consumes `resolve_swapc2` for SWaP-C²-aware
  ranking. Embodied-AI-Architect adds a verdict-first SWaP-C² fit tool
  against capability tiers and mission profiles.

**Exit:** coverage target above is met; `resolve_swapc2` returns a full
record for every edge product at every thermal profile; downstream fit
check exists.

### Phase 4 -- Update consumers

- Compatibility shims (`load_gpus`, `load_cpus`, `load_npus`, `load_chips`,
  `load_hardware`)
  over `load_compute_products()`, following the `load_kpus()` pattern, with
  `DeprecationWarning`.
- Rename debt resolved (vendor-neutral names; KPU names kept as aliases).
- graphs and Embodied-AI-Architect read `ComputeProduct` directly. This
  includes `registry.py` hardware queries, which today filter
  `HardwareEntry` on weight, power and capabilities.

**Exit:** both consumer repos pass against the new schema with no legacy imports.

### Phase 5 -- Sunset legacy schemas

- Remove `GPUEntry`, `CPUEntry`, `NPUEntry`, `ChipEntry`, `HardwareEntry`,
  `SystemConfiguration`, and `hardware.LifecycleStatus`.
- Remove `data/{gpus,cpus,npus,chips,hardware}/`.
- Update `CHANGELOG.md`, `architecture.md`, `README.md`.
- Release as **1.0.0** (breaking). Downstream pins (`<1.0.0`) need an update.

---

## Risks

1. **Discriminated unions in YAML are not free.** Pydantic v2 handles them
   cleanly with a `kind:` discriminator; YAML editors and humans won't
   validate the union as easily as a flat shape. Plan for editor tooling
   (JSON Schema export) so authors get autocomplete.
2. **The 80/20 trap.** Around 20% of fields are genuinely block-specific.
   Resist flattening them onto the spine; that re-creates the original
   bag-of-optional-fields problem.
3. **Information loss during migration.** Some old fields have no obvious
   home (GPU `nvenc`, CPU `socket`). With `extra: forbid` there is no
   `extras` escape hatch, so each orphan needs a recorded drop or a typed
   field.
4. **External consumers.** Audit for consumers outside graphs and
   Embodied-AI-Architect before Phase 5.
5. **Backfill quality.** Public die data varies wildly. Leave `None` and
   document why, rather than guess.
6. **Estimates read as facts (R1).** A plausible estimated heatsink mass
   that propagates into a vehicle sizing decision is worse than a missing
   value. Mitigation: `basis` and `confidence` are mandatory on every
   `SourcedValue`, and resolution propagates the weakest confidence.
7. **Double counting across levels (D6).** If a module's mass already
   includes its heat spreader and the profile also binds a cooling
   solution, the spreader is counted twice. Mitigation: each
   `CoolingSolutionEntry` states what it covers ("added on top of the
   module as sold"), and level-specific examples go into tests.
8. **Generic cooling catalog.** Until S2 lands, the existing fixed-weight
   cooling entries (e.g. one 400 g `active_fan` for 15-60 W) produce
   misleading SWaP-C² for small UAVs. Treat cooling-derived numbers as
   `confidence: unknown` until then.
9. **Scope growth.** R1 and D8 pull `contains`, `MODULE`, and all of
   `HardwareEntry` into this RFC. Keep S1-S3 limited to the 20 existing
   hardware entries and what SWaP-C² needs. Full datacenter board/rack
   composition (DGX) is not a prerequisite.
10. **Spine creep from D8.** `interfaces`, `software` and `environmental`
    are not SWaP-C² data. They are added only because `HardwareEntry`
    already carries them. Reuse the existing sub-models verbatim rather
    than redesigning them in this RFC.
11. **Misreading the D4 forms.** `sum` is not achievable throughput. It
    assumes perfect partitioning and ignores shared memory bandwidth and
    power limits. Docstrings and downstream tools must label it as an upper
    bound and not substitute it for `performance`.

---

## Open questions for sign-off

None. All resolved 2026-10-02:

- SWaP-C² second C = Cooling.
- `HardwareEntry` absorbed into `ComputeProduct` (D8).
- Estimators live in `scripts/`.
- Multi-block peak = sum, min and max over blocks (D4).
- Cost = variable unit cost at quantity 1 and 1K; NRE never enters (D9).

---

## Alternatives considered

### Alternative 1: Add `DieSpec` to existing per-category schemas

Lowest-effort fix for the immediate gap (die size missing on CPU/NPU/Chip).

**Rejected** because it doesn't address the deeper problem -- four parallel
schemas continuing to drift -- and multi-category products still have no
clean home.

### Alternative 2: Flat ComputeProduct (no discriminated blocks)

Single schema with all possible fields optional.

**Rejected** because most fields would be None for most products, which
defeats validation and creates ambiguity (does `tensor_cores: 0` mean "not a
GPU" or "GPU without tensor cores"?).

### Alternative 3: Stay with per-category, add cross-cutting `DieSpec` mixin

**Rejected** because it solves duplication but not modeling: multi-category
products still have no clean home, and the folder structure still forces
classification.

### Alternative 4 (rev 2): SWaP-C² as a downstream-only computation

Keep the schema as is and let graphs compute SWaP-C² from whatever fields
exist.

**Rejected** because size, mass and price are datasheet facts that belong
in this repo. Leaving them out means every consumer re-scrapes them, without
provenance. The airframe-dependent *interpretation* of SWaP-C² does stay
downstream.

---

## Out of scope

- Workload / benchmark data (`BenchmarkResult`, `WorkloadProfile`).
- Sensors (`SensorEntry`). Sensor SWaP is a separate, later concern, but
  should reuse `SourcedValue`.
- Mission profiles, capability tiers, batteries: consumed read-only by
  the SWaP-C² fit check; their schemas are unchanged.
- Vehicle / airframe models (propulsion, hover power, endurance). These
  belong to graphs / Embodied-AI-Architect.
- Detailed fabric topology beyond `Interconnect` (NVLink topology, CXL
  coherence domains, `Switch` as a first-class entity).
