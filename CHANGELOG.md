# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.0] - 2026-10-03

NVIDIA Jetson entries classified as SKUs of their product families. RFC
0001 D6, phase S3, slice d.

- `market.product_family` is now **NVIDIA Jetson Orin** and **NVIDIA Jetson
  Thor** (was "Jetson Orin" / "Jetson Thor").
- Within a family, SKUs share the silicon and are differentiated by
  floorsweeping (units enabled) and memory configuration. Each entry is a
  SKU and keeps NVIDIA's SKU name as its id (`nvidia_jetson_agx_orin_64gb`,
  `nvidia_jetson_agx_thor_128gb`). A header in each YAML names the fields
  that define the SKU. For the AGX Orin 64GB that is 16 SMs × 128 = 2048
  CUDA cores, 64 Tensor cores, and 64 GB LPDDR5 at 204.8 GB/s.
- **Ids are unchanged.** The `nvidia_orin_soc_*` / `nvidia_thor_soc_*`
  rename proposed during review was withdrawn before release, so consumers
  need no change.
- `tests/test_compute_product_v14_jetson_family.py` pins the family, the
  SKU ids, and the AGX Orin 64GB floorsweep and memory.

## [0.20.0] - 2026-10-03

Cooling solutions take their own footprint, and the legacy Hailo hardware
figures are corrected. RFC 0001 phase S3, slice c.

**Cooling footprint.** `CoolingSolutionEntry.max_height_mm` (new,
optional).

- A cooling solution sized by volume fills the product's footprint up to
  this height. Beyond it, it takes its own, larger footprint at the limit:
  same aspect ratio, volume conserved. A real heatsink overhangs a small
  module the same way.
- The envelope's `notes` say when this happens.
- Set from the entries' existing `height<=Xmm` constraints: small 15, large
  40, fan 60 and vapor chamber 80 mm. A test keeps the two in step.
- Hailo-8 M.2 at 8.65 W: 95.8 × 50.2 × 17.6 mm, instead of a 78 mm stack on
  a 42 × 22 mm card.

**Legacy Hailo hardware corrected** (`data/hardware/hailo/`). Each file is
now anchored to one SKU and holds the source-DB figures:

| Entry | Was | Now |
|---|---|---|
| `hailo_8_m2` (2242 Key M) | 22×30×3.5 mm, 8 g, 5 W TDP, $99 | 22×42×2.626 mm, 6 g, 8.65 W TDP (3.3 W typical, 8.25 W peak), $179.99 |
| `hailo_8l_m2` (2280 B+M) | 22×30×3.5 mm, 6 g, 3 W, PCIe x1, $49 | 22×80×2.626 mm, mass unset, 6.6 W TDP (1.9 W typical), PCIe x2, $89 |
| `hailo_10h_m2` (2280 8 GB) | 22×42×4 mm, 12 g, 5 W, LPDDR4X, $199 | 22×80×2.8 mm, mass unset, 8.25 W max (2.5 W typical), LPDDR4, $229 |

- Source DB: Hailo-8L datasheet and UP Shop listing, plus 8 observations.
- `tests/test_legacy_hardware_hailo.py` holds the legacy files to the
  source DB and to the unified module products.

## [0.19.0] - 2026-10-03

Hailo M.2 modules, verified against Hailo's datasheets. RFC 0001 phase S3,
slice b. Two new `module` products. Source observations now carry a
required `category`.

**Catalog.** Every SWaP-C² value is a source-DB figure
(`observations/m2_modules.yaml`), and a test holds each YAML to it.

- `hailo_8_m2_2242_m` (HM218B1C2HAE) contains `hailo_hailo_8`:
  - 42 × 22 × 2.626 mm;
  - 6 g (reseller listing; Hailo publishes no mass);
  - profiles 2.4 / 3.3 W (MobileNet-SSD / ResNet-50 typical) and 8.65 W
    (datasheet TDP);
  - $179.99 at quantity 1.
- `hailo_10h_m2_2280_8gb` (HM22HB1C2FAE) contains `hailo_hailo_10h`:
  - 80 × 22 × 2.8 mm (1.5 + 0.5 mm components on the 0.8 mm M.2 board);
  - 8 GB LPDDR4;
  - profiles 2.5 W (typical) and 8.25 W (maximum);
  - $229 at quantity 1.
  - Mass is unresolved: no source gives one.
- **The legacy `HardwareEntry` Hailo data is wrong.** The datasheets
  contradict `hailo_8_m2`, `hailo_8l_m2` and `hailo_10h_m2` on:
  - thickness (2.626 mm, not 3.5 mm);
  - TDP (8.65 / 6.6 W, not 5 / 3 W);
  - the Hailo-10H size (2280, not 2242).

  Their masses and prices have no source. None of it is used.
- `hailo_8l_m2` is not converted: there is no Hailo-8L chip product yet.

**Source DB.**

- Six documents: the Hailo-8 and 10H datasheets, the Waveshare and UP Shop
  listings, and Hackaday's M.2 article for the 0.8 mm board.
- 17 observations.
- `Observation.category` (new, required): heatsink, fan, m2_module,
  process_node, method, material or standard. One subject has one category.
- `find(..., category=)` and `subjects(..., category=)` filter on it. The
  estimator and the validation tests use it instead of inferring a
  subject's kind from which quantities it has.

**Tests.** Per-vendor product counts now count chip-level SKUs
(`cp.dies`). Modules contain chips rather than add to them.

**Known limit.** The S1 envelope rule puts cooling on the product's
footprint. A sized passive sink for the Hailo-8 at 8.65 W (68 g, about
69 cm³) becomes a 78 mm stack on a 42 × 22 mm card. The mass is plausible;
the shape is not.

## [0.18.0] - 2026-10-03

Die cost for the products whose silicon can be costed. RFC 0001 phase S3,
slice a. Additive schema; 12 products gain a `swapc2.cost.die_cost_usd`.

- **`CostSpec.die_cost_usd`** (new, optional): the variable cost of the
  product's good die(s) from `silicon_cost_v1`. It is not a unit price,
  because it excludes package, test and margin. `resolve_swapc2` passes it
  through as `ResolvedSWaPC2.die_cost_usd` and never uses it as unit cost.
- **`scripts/swapc2_estimators.py products --write | --check`** (new): writes
  the die cost on every eligible product, and `--check` (also a test) fails
  on drift. A product is eligible when each die is modeled one-to-one
  (`num_dies == len(dies)`, so aggregated chiplets such as EPYC are out) on
  a node with both a sourced wafer price and a sourced D0. Today that means
  TSMC N5 and N7.
- **Catalog: 12 products.** Die-only, against list price where one exists:

  | Product | Die cost |
  |---|---|
  | 7 nm KPU SKUs | $1.83 (H64) to $70.54 (T768, 75% yield) |
  | Google TPU Edge Pro | $25.67 (list $450) |
  | Google TPU v4 | $173.93 |
  | Qualcomm QRB5165 | $12.94 |
  | Qualcomm SA8775P | $51.14 |
  | AmpereOne A128 / A192 | $202.78 (list $3,888 / $5,555) |

- The die-cost writer edits only `swapc2.cost.die_cost_usd`
  (`set_nested`). Sibling fields, comments and blank lines in an existing
  `swapc2` block are kept, and a missing `cost:` or `swapc2:` is created.
  When a file has no anchor key, both writers append the field.
- **`load_process_nodes(include_overlay=True)`** (new keyword). The
  die-cost writer passes `False`, so a confidential PDK overlay
  (`PROCESS_NODE_DATA_DIR`) never feeds figures written into public catalog
  data.
- **Downstream:** graphs' KPU golden snapshots for the six 7 nm SKUs now see
  `.input.swapc2`. That is a declared catalog-data change, and graphs
  regenerates them (`cli/kpu_golden_snapshot.py --update`) when it adopts
  0.18.0.

## [0.17.0] - 2026-10-02

SWaP-C² estimators and their sourced inputs. RFC 0001 phase S2. The cooling
catalog now scales with the heat it removes, and the process nodes carry
wafer-cost inputs. Additive schema; three cooling entries change values.

**Source database** (`embodied_schemas.sources`, `data/sources/`, new). It
holds the cited figures behind every estimated value.

- `documents.yaml` holds 22 source documents.
- `observations/*.yaml` holds 128 quoted figures for wafer price, defect
  density, heatsinks and fans. Each records its exact quote, the date of the
  figure, its basis (datasheet, list price, reported, model estimate,
  disclosed) and conditions such as airflow, quantity break or maturity.
- Query with `load_source_db()`, using `find`, `get` / `value`, `series`
  (trends) and `to_sqlite()` for ad hoc SQL.
- `tests/test_sources.py` validates the figures locally:
  - references, units per quantity and ranges;
  - physical bounds (heatsink density below solid aluminum; catalog R x V
    within Lee's bound; resistance falling with airflow);
  - orderings (fan typ <= max; volume discounts; CSET price rising with
    node);
  - cross-source agreement (wafer prices within 20%);
  - trend ordering.
- `validate_data_integrity()` now loads the source DB.

**Estimators** (`scripts/swapc2_estimators.py`, new). Every parameter is
derived from the source DB, not hard-coded. `nodes --check` holds the
process-node catalog to the DB the way `cooling --check` holds the cooling
catalog, and both run as tests. Per RFC R1.2 they live
in `scripts/`. Each writes `basis: estimated` values from a named, versioned
model whose parameters are sourced.

- `cooling_sizing_v1`: per-W heatsink volume, mass and cost from the
  volumetric-thermal-resistance method, `V / P = R_vol / dT`, with
  mass = V × effective density. A fan adds its own mass, volume, cost and
  electrical load.
  - `cooling --write` rewrites the owned fields in place and keeps comments.
  - `cooling --check` fails if a YAML differs from the model; a test runs it.
- `silicon_cost_v1`: variable cost of one good die = wafer price / (gross
  dies per wafer × Murphy yield). It includes no NRE (D9). Applying it to
  products is S3.

**Cooling catalog.** `passive_heatsink_small`, `passive_heatsink_large` and
`active_fan` are sized by `cooling_sizing_v1`.

- R_vol comes from Lee, "How to Select a Heat Sink" (Electronics Cooling,
  1995).
- Effective density is 0.95 g/cm³, the median of 13 catalog aluminum sinks.
- Cost is a least-squares fit to Alpha Novatech qty-1 prices.
- The `active_fan` fan is the Delta AFB0612EH-A (80 g, 4.56 W).
- dT is each entry's `junction_c_max - ambient_c_max`. That ignores the
  junction-to-sink drop, so the sinks are a lower bound.
- The fixed figures they replace understated small-sink mass at UAV power
  levels:

  | Entry | 0.16.0 | 0.17.0 |
  |---|---|---|
  | `passive_heatsink_small` | 30 g at any power | 40 / 79 / 119 g at 5 / 10 / 15 W |
  | `active_fan` | 400 g | 107-262 g over 15-100 W, plus 4.56 W fan load |

- Not sized, because no source supports it: the vapor chamber (mass), liquid
  and datacenter cooling, the fanless entry (no sink), and the SMARC spreader
  (SGeT leaves the material to the vendor).
- `CoolingSolutionEntry.sizing_source` (new, optional): the estimator id
  and the source of each parameter.

**`CoolingSolutionEntry.ambient_c_max` is now optional** (consumer-visible).

- A surface-rated entry may have no air rating. `smarc_heat_spreader_82x50`
  now carries only `surface_c_max: 85`, the limit SECO rates at the spreader
  plate; its air limit depends on the enclosure.
- An entry needs `ambient_c_max`, `surface_c_max` or both.
- `cooling_sizing_v1` refuses an entry without an air rating.
- Downstream code that formats `ambient_c_max` must handle `None`. graphs'
  cooling CLIs are fixed in branes-ai/graphs#346.

**Process nodes.** These are sourced inputs to `silicon_cost_v1`.

- `wafer_cost_usd` for TSMC N5, N7, N12, N16, 28HPM, 40 and 65 nm. Source:
  CSET (Khan & Mann 2020, Table 9), a 2020 USD model estimate for 300 mm
  wafers that excludes masks.
- `defect_density_per_cm2` from TSMC symposium disclosures:
  - N7: 0.09 at HVM+3Q;
  - N5: 0.10-0.11 at about HVM+1Q (0.10 used);
  - N6: "same defect density as N7".
- Left unset, because no reliable source exists: N4P, Samsung 8LPP, every
  GlobalFoundries node, Intel 7 and Intel 3, and D0 for the mature TSMC
  nodes. Without D0, `silicon_cost_v1` refuses those nodes rather than
  assume a yield.
- The die cost is therefore computable on N5 and N7 only.

## [0.16.0] - 2026-10-02

SWaP-C² (Size, Weight, Power, Cost, Cooling) for compute products, and
modules that contain chips. RFC 0001 phase S1 (requirement R1, decisions
D4, D6-D9). **Additive**: every existing catalog entry validates and
serializes exactly as in 0.15.0.

**SWaP-C²** (`embodied_schemas.swapc2`, new).

- `ComputeProduct.swapc2` (optional) holds the product's own size, mass,
  input power and variable unit cost.
- Every value is a `SourcedValue` with a `basis` (`datasheet`, `measured`,
  `derived` or `estimated`), a confidence and a source, so an estimate never
  reads as a datasheet fact.
- Cost is unit price at quantity 1 and 1K only. NRE never enters (D9).
- `resolve_swapc2(product, profile)` adds the cooling solution that the
  thermal profile binds, and returns:
  - power at the vehicle input rail: TDP / converter efficiency + fan power;
  - mass;
  - the envelope, with cooling on the top face;
  - unit cost.

  Each result keeps the weakest confidence of its inputs. An axis it cannot
  resolve is None, with a reason in `unresolved`.
- A module or board with no mass or price of its own rolls them up from
  `contains`, recursively, as an estimate.

**Levels of integration (D6, D8).**

- `ProductKind.MODULE` (new).
- `ComputeProduct.contains: list[ProductRef]` (with optional slot and role).
- A chip-level product still needs at least one die; a module, board or
  system needs dies or `contains`.
- `check_contains_references` reports unknown ids and cycles, and
  `validate_data_integrity` runs it.
- New optional sections carried over from `HardwareEntry`: `memory`,
  `environmental`, `interfaces`, `software` and `product_url`; packaging
  form-factor fields (`form_factor`, `mounting`, VITA / SOSA, conduction
  cooling, conformal coat); and `market.suitable_for` /
  `target_applications`.
- `FormFactor.SMARC` (new).

**Peak aggregation (D4).** `aggregate_peak(product, products)` and the
`performance_by_aggregation` property report peak ops/s as **sum, min and
max** over compute blocks.

- Aggregation is per precision, over the blocks that support it; a zero peak
  means unsupported.
- Contained products are expanded by count.
- IO blocks are skipped.
- A single-block product without a block peak uses its `performance`
  headline.

GPU, CPU, NPU, CGRA, DPU and TPU blocks gain an optional
`theoretical_performance` for this (DSP already had one; KPU is derived from
tiles).

**Cooling and process nodes.**

- `CoolingSolutionEntry` gains optional SWaP-C² fields:
  - `dimensions_mm` and `volume_cm3`;
  - `parasitic_power_w`, the fan / pump load at the input rail;
  - per-W sizing: `mass_g_per_w`, `volume_cm3_per_w`, `cost_usd_per_w`;
  - `basis`;
  - `surface_c_max`, for solutions rated at a surface (COM heat spreaders).
    `ambient_c_max` may not exceed it.
- `ProcessNodeEntry` gains optional silicon-cost estimator inputs:
  `wafer_cost_usd`, `wafer_diameter_mm`, `defect_density_per_cm2` and
  `wafer_cost_source`.

**Serialization.** New optional fields are left out of `model_dump()` while
unset (`embodied_schemas.serialization.omit_if_default`). Downstream golden
snapshots (graphs' KPU goldens) therefore see no change.

**Catalog.**

- `seco_som_smarc_qcs6490`: the first `module`. It is the SECO SMARC 2.1.1
  module around the Qualcomm QCS6490, and it contains `qualcomm_qcs6490`.
- `smarc_heat_spreader_82x50`: the SMARC standard heat spreader, with
  geometry from SGeT SMARC 2.1.1.
- SWaP-C² for the SECO module resolves power and an 82 x 50 x 8.5 mm
  envelope (34.9 cm³) at each profile.
- Mass and unit cost stay unresolved: SECO publishes no weight or 1K price
  for any SMARC module, and the QCS6490 module has no public price.

**Tests.** Catalog-wide tests that assumed every product has a die now
skip products without dies.

## [0.15.0] - 2026-09-19

FP16 energy per op for every process node, and BF16 re-derived from the same
source. **BF16 figures fall by about 54%**, and the KPU SKUs' declared TDPs
fall by 9-32% to match.

**Derivation** (`embodied_schemas.fp_energy`, new). Every node authors one
floating-point figure per logic library, `<class>:fp32`. The `fp16` and
`bf16` entries are now DERIVED from it, from Horowitz, "Computing's Energy
Problem", ISSCC 2014, Fig. 1.1.9 (45 nm):

- **FP16 = 0.326 x FP32**: an FMA is a multiply plus an add, FP16
  1.1 + 0.4 = 1.5 pJ against FP32 3.7 + 0.9 = 4.6 pJ.
- **BF16 = 0.230 x FP32**. BF16 is not in the table. The multiply is fitted
  as a + b*m^2 and the add as c + d*m in the significand width m, through
  the FP16 (m = 11) and FP32 (m = 24) points, then evaluated at BF16's m = 8.
  That gives 0.774 + 0.285 = 1.059 pJ. It is an extrapolation below the
  fitted range, and the exponent width (5 bits in FP16, 8 in BF16) is not
  modeled.

The previous BF16 figures were FP32 x ~0.5, with no stated source. The
derived figures are rounded to three significant figures and are
THEORETICAL. `tests/test_fp_energy.py` holds every node to the derivation.

**Schema.** `ProcessNodeEntry.energy_per_op_sources` (new, optional) gives a
per-entry source keyed like `energy_per_op_pj`. Each derived entry cites
its ratio and the paper. A key that names no energy entry is rejected.

**KPU tile classes.** Every FP16 mode that charged the BF16 anchor ("no fp16
anchor in the catalog; same width as bf16") now charges `balanced_logic:fp16`:

- `pe_int8_mac_i32`, `pe_bf16_fma` and `pe_fp16_lerp` (the lerp stays at
  1.3x, now of its own format);
- the FP16 mode of both `kpu_h64_auto1` SKUs;
- `pe_lns16_mac`, whose LNS-Madam ratios are relative to an FP16 PE.

**KPU SKU envelopes.** BF16 was every KPU's maximum-power precision. With it
cheaper, FP16 sets the TDP (INT8 on the heterogeneous H64). The model
computes 9-32% less than each profile declared. Holding the envelopes by
re-tuning Vdd, as 0.12.0 did, would put most mid and top profiles above
nominal Vdd, up to 1.05 V on a 0.8 V node. So the envelopes are corrected
instead: every profile's `tdp_watts` and the `tdp_watts`, `max_power_watts`
and `min_power_watts` roll-ups are set to what the power model computes, and
Vdd and clocks are unchanged. Profile names such as `10W` are now labels.
For example, T64 N16 is 2.2 / 4.3 / 7.1 W, and T768 is 21.1 / 41.5 / 68.3 W.

## [0.14.0] - 2026-09-17

The T768 `Matrix` tile class migrated to the `systolic` tile kind
(branes-ai/graphs#268 D8), the first catalog SKU to move onto a new kind
after shipping.

**Data.** `kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc`'s Matrix class was a PE-fabric
tile whose throughput (8192 INT8 ops per clock on an 8x8 array) carried a
systolic mechanism in a note. It is now a `SystolicTile` that states it:

- 8x8 weight-stationary cells in `hp_logic`, each a MAC unit with **64 lanes**
  in INT8 (128 ops per cell per clock) and 32 in BF16 / FP16. That matches
  the 0.3 Mtx per cell the `pe_matrix` silicon block already budgets, about
  4.7K transistors per lane.
- **Throughput unchanged:** 1353.8 INT8 TOPS, 676.9 BF16 TFLOPS, 27.1 FP32,
  756.1 INT4. `performance` now also declares the derived roll-up
  (`peak_ops_per_sec_by_precision`, `by_tile_kind`).
- **Energy basis unchanged:** no mode declares energy, so every op is still
  charged the node's `hp_logic` anchor. A systolic discount (the library's
  `systolic_int8_ws` derives 0.65x) would cut the computed TDP by about 22%
  and need a Vdd re-tune, so it is left to a separate model change.
- **Memory:** a systolic tile does not inherit the chip's L1 / L2, so the
  Matrix class declares its own `local_memory`, 4 KiB L1 and 32 KiB L2, the
  figures it used to inherit. As tile-carried silicon these are priced at
  0.052 Mtx/KiB, where this SKU's `l2_sram` block uses 0.022 (every other
  catalog SKU uses 0.052), so the chip gains about 74 Mtx (13.302 -> 13.376 B
  modeled) and under 3 mW of leakage. No profile's TDP moves by 0.05 W.
- The `pe_matrix` block stays the single count of the cells' logic
  (`mac.mtx` is unset, so nothing is counted twice). Its note no longer
  claims a per-PE L1 (graphs#268 F1).

**Tests.** `tests/test_kpu_catalog.py` adds `MIGRATED_KPU_SKU_IDS` (the T768)
beside the legacy and heterogeneous lists, and `SHIPPED_KPU_SKU_IDS` (legacy
plus migrated) for the contracts that survive a tile-kind change: block
round trip and key order, no overlays, the implicit mesh and its cluster
partition. The contracts that are about every tile being a `KPUTileSpec`
stay on the legacy list. A guard checks the three lists partition the
catalog and say what they claim. `tests/test_kpu_t768_systolic_d8.py` pins
the throughput, lanes, energy basis, single silicon count and memory.
`test_products_without_a_kpu_block_are_not_checked_against_tiles` now picks
a product without a KPU block explicitly; "not a legacy SKU" could have
picked a KPU.

## [0.13.0] - 2026-09-17

The default per-cluster DVFS partition on the twelve uniform KPU SKUs
(branes-ai/graphs#268 F2), from
`graphs/docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md`.

**Data.** Each uniform SKU gains `power_domains`: one `cluster` domain per
k x k block of compute sites, each naming its own `rail_id` and
`clock_domain_id` (one regulator and one PLL per cluster, as the design
specifies), plus one `uncore` domain for the memory PHYs, IO and control
logic. Following the design's table and its "recommended k=4":

| SKU family | mesh | cluster | clusters |
|---|---|---|---:|
| T64 | 8x8 | 2x2 | 16 |
| T128 | 16x8 | 4x4 | 8 |
| T256 | 16x16 | 4x4 | 16 |
| T512 | 32x16 | 4x4 | 32 |
| T768 | 32x24 | 4x4 | 48 |

No thermal profile declares `domain_operating_points`, so every cluster
runs at its profile's own Vdd and clock: **TDP, performance and area are
unchanged** (unrounded TDP delta 0.0 W on every profile). What a DVFS
policy then does with the clusters is a per-profile choice on top.

**Schema.** A `cluster` domain's `site_ranges` now resolve against the
implicit one-tile-per-site mesh (`noc.mesh_rows x noc.mesh_cols`) when there
is no checkerboard, which is what the `checkerboard` field already
documents `None` to mean. Previously a cluster domain required an explicit
checkerboard, which would have pushed a uniform SKU onto the heterogeneous
floorplan path just to name its DVFS clusters.

**Not modeled** (each needs a schema field or data the catalog does not
carry): the quadrant level, per-cluster regulator and PLL silicon and power,
per-cluster harvest, and process-variation bins. Clusters are not
`gateable`: in the design the tile is the gating unit and the quadrant the
region-gating unit; the cluster is the DVFS and floorsweeping unit.

## [0.12.0] - 2026-09-16

Vdd re-tune for the four `7nm_tsmc_hpc` KPU SKUs (branes-ai/graphs#268 F4).

Their `lp` and `default` profiles declared more TDP than the power model
computes -- the T512's `lp` claimed 10.3 W against a computed 8.9 W -- and
had done since leakage gained its Vdd scaling. `tsmc_n7` carries
`leakage_vdd_exponent: 4.5`, so below nominal Vdd the leakage term falls
steeply; these four SKUs kept the round 0.500 / 0.650 / 0.750 V they were
authored with, while the 16 nm family and the T768 were re-tuned at the
time.

The tell was which profile looked healthy. `boost` sits exactly at the
node's nominal 0.75 V, where the scaling is a no-op, so the one profile
that could not reveal the problem was the one that passed.

The declared envelope is the target and Vdd is the knob, so the Vdds move,
not the envelopes:

| SKU | `lp` | `default` |
|---|---|---|
| kpu_t64_32x32_lp5x4_7nm_tsmc_hpc | 0.500 -> 0.538 | 0.650 -> 0.668 |
| kpu_t128_32x32_lp5x8_7nm_tsmc_hpc | 0.500 -> 0.539 | 0.650 -> 0.663 |
| kpu_t256_32x32_lp5x16_7nm_tsmc_hpc | 0.500 -> 0.534 | 0.650 -> 0.660 |
| kpu_t512_32x32_lp5x32_7nm_tsmc_hpc | 0.500 -> 0.537 | 0.650 -> 0.662 |

All four converge on the same band and stay below the node's nominal
0.75 V. Every KPU profile in the catalog now computes the TDP it declares,
and graphs gains a `declared_tdp_matches_model` validator so this cannot
drift unnoticed again.

## [0.11.0] - 2026-09-16

**Breaking (data model):** `KPUMemorySubsystem.l1_kib_per_pe` is renamed
`l1_kib_per_tile`, and every KPU SKU gains an `l1_sram` silicon_bin block
(branes-ai/graphs#268 F1).

L1 was declared per PE, which is the wrong denominator. The KPU hierarchy
is a streaming pipeline, not a cache hierarchy:

- **L3** is the block linear-algebra reuse scratchpad, allocated to its
  own memory tile;
- **L2** reformats L3 blocks so streaming preparation is easy at fabric
  clock rates, and may sit in the L3 memory tile or the compute tile
  depending on the bank layout needed to feed L1;
- **L1** is tightly coupled to the fabric edges: it turns row and column
  fetches out of L2 into streams of operands pushed into the edges of the
  fabric. It is per compute **tile**.

A PE is an ALU with datapath registers and a small token CAM; it holds no
SRAM. At 0.052 Mtx/KiB the declared 4 KiB/PE is ~208,000 transistors per
PE, against the 6,000 the catalog's own `per_pe` block budgets for "MAC +
reg + sequencing" -- and binning it would have added 835-851% of the die.
The figure was right; the denominator was wrong.

The `l1_sram` block (`count_ref: l1_total_kib`) adds 0.17-0.45% of die
area per SKU. Leakage rises by under 1 mW, two orders of magnitude below
the 0.1 W the thermal profiles declare their TDP at, so **no Vdd re-tune
was needed** -- measured rather than assumed.

The field descriptions now carry the role of each level, so the next
reader does not have to infer the denominator.

## [0.10.0] - 2026-09-16

The first heterogeneous KPU SKUs in the catalog: `kpu_h64_auto1` at
`tsmc_n16` and `tsmc_n7` (branes-ai/graphs#268 E1).

**Data, not schema.** No schema change: these are the first products to
use the capability phases B1-B6 added. 64 compute sites on an 8x8
checkerboard carrying three PE-fabric datapath classes (INT8 MAC, LNS16,
min-plus), a weight-stationary systolic array at a 1x2 footprint, and an
ISP -> SGM -> VIO stream-linked chain of fixed-function cores, the VIO at
2x2 absorbing the memory cells it covers. Every tile class comes from the
tile-class library; the two variants differ only in process node, clock
and thermal envelope, and the systolic accumulator's SRAM library
(`tsmc_n7` offers `sram_hp`, `tsmc_n16` does not).

**What this means for consumers.**
- `load_compute_products()` returns 14 KPU products, not 12.
- `load_kpus()` likewise returns 14, and the legacy `KPUEntry` view is not
  lossy for them: `KPUEntry.kpu_architecture` is the same
  `KPUArchitectureBase`, so tile kinds, the checkerboard and the NoC
  overlays all survive and round-trip.
- Code that assumed every catalog KPU tile is a `KPUTileSpec`, or that no
  catalog SKU declares a checkerboard, overlays or a non-`pe_fabric` tile
  kind, needs to say which SKUs it means. The B1-B6 backward-compatibility
  tests now scope themselves to an explicit list of the twelve SKUs that
  predate this work (`tests/kpu_catalog.py`), rather than to "whatever is
  in the catalog" -- deriving that set would have made the contract
  vacuous the moment a SKU changed shape.

## [0.9.0] - 2026-09-15

Phases B2-B6 of the KPU heterogeneous-tile sprint (branes-ai/graphs#268;
#90-#94):
- interconnect overlays;
- systolic and fixed-function tile kinds, with function cores;
- the checkerboard, power domains and per-domain operating points;
- the performance roll-up by tile kind;
- the tile-class library, with the `tsmc_n40` / `tsmc_n65` anchor nodes.

Additive: every catalog YAML loads unchanged. Three checks are stricter, and
every catalog entry already satisfies them:
- `total_tiles` must equal `sum(num_tiles)`;
- `compute_product.Power.default_thermal_profile` must name a profile;
- `load_kpus()` warns about KPU products it cannot express as a `KPUEntry`.

### Added

- **KPU tile-class library** (branes-ai/graphs#268 Phase B6).
  - New module `kpu_tile_class.py`. `KPUTileClassEntry` is a one-tile
    template (its `tile_class_id` is the entry id, `num_tiles` 1) plus
    provenance (`sources`, `confidence`, `ref_node_id`).
    `instantiate(num_tiles, **overrides)` returns a self-contained SKU tile.
  - `KPUTileBase` gains the optional `tile_class_ref`, which names the
    library entry a tile came from. It is informational only: SKUs stay
    self-contained, so validators never need the library. It is serialized
    last, so existing key positions are unchanged.
  - `load_kpu_tile_classes()` reads `data/kpu-tile-classes/<id>.yaml`. A
    private overlay directory can be named by `KPU_TILE_DATA_DIR`; when an
    id is in both, the higher-confidence entry wins, as for
    `PROCESS_NODE_DATA_DIR`. The two overlays now share one merge helper,
    with behavior unchanged.
  - **Initial entries**, all THEORETICAL and cited:
    - `pe_int8_mac_i32`: the legacy-equivalent INT8-primary tile plus its
      datapath.
    - `pe_bf16_fma`.
    - `pe_lns16_mac`: LNS8 energy from LNS-Madam; the LNS16 energy is an
      assumption, flagged in the entry.
    - `pe_fp16_lerp` and `pe_minplus_i16`: energies derived from the
      Horowitz ISSCC 2014 45 nm table.
    - `systolic_int8_ws`: TPU v1 organization.
    - `ff_isp_raw2yuv`: Darkroom, SIGGRAPH 2014.
    - `ff_vio_stereo_inertial`: Navion, JSSC 2019.
    - `ff_stereo_sgm`: Li et al., ISSCC 2017.
- **Scaling-anchor process nodes** `tsmc_n65` and `tsmc_n40`, both
  THEORETICAL. They are the reference nodes for the 65 nm (Navion) and
  40/45 nm (SGM, Darkroom, Horowitz) figures. Both are derived from
  `tsmc_n28hpm` / `tsmc_n16` by C x V^2 and area scaling, and
  cross-checked against the cited silicon. No KPU SKU targets them.
- `load_kpus()` now **warns** about a product that has a KPU block but
  cannot be expressed as a legacy `KPUEntry` (several dies, or a KPU block
  beside other blocks), and about a KPU product whose process node is
  missing. It no longer drops these silently. Products without a KPU block
  are still skipped silently.

- **Performance roll-up by tile kind** (branes-ai/graphs#268 Phase B5).
  `KPUTheoreticalPerformance` gains three optional fields:
  - `peak_ops_per_sec_by_precision`: ops/s by precision, with the same name
    and meaning as `TheoreticalPerformance`;
  - `by_tile_kind`: the same map per programmable tile kind (`pe_fabric`,
    `systolic`);
  - `fixed_function_throughput`: work units/s per `function_id`.

  Fixed-function tiles never enter the ops/s fields or the legacy TOPS.
  - **Consistency:** when the peak map is set, the legacy
    `int8_tops` / `bf16_tflops` / `fp32_tflops` / `int4_tops` must equal it
    to their stated precision, and `by_tile_kind` must sum to it.
  - **`derive_kpu_performance(tiles, clock_mhz)`:** computes every field
    from the tiles. It reproduces the legacy numbers of all 12 catalog SKUs
    exactly, at the default thermal profile's clock with the graphs
    generator's rounding.
  - **Architecture check:** `KPUEntry` and `ComputeProduct` check a
    declared roll-up against their tiles at the default profile's clock. A
    legacy-only performance block is not checked, as before.
  - New: the `PROGRAMMABLE_TILE_KINDS` export, and a `default_profile`
    property on `KPUPowerSpec` and `compute_product.Power`.
  - `compute_product.Power` now rejects a `default_thermal_profile` that
    names no profile, as `KPUPowerSpec` already did. All 43 catalog
    products pass.
- **Checkerboard** (branes-ai/graphs#268 Phase B4): the new optional
  `KPUArchitectureBase.checkerboard` (`CheckerboardSpec`) makes the
  compute-site grid explicit.
  - Fields: `compute_sites` (rows x cols), an optional `memory_cell`
    (restates `memory.l3_kib_per_tile`), `placement` `auto` / `explicit`,
    `placement_map` (a tile_class_id or `.` per site) and `spare_sites`.
  - Checks: site accounting (`sum(num_tiles * footprint sites) +
    spare_sites == rows * cols`), a NoC mesh equal to the grid, footprints
    that fit the grid, and, for an explicit map, that every class tiles into
    exactly `num_tiles` footprint rectangles.
- **Power domains** (new module `power_domain.py`, re-exported from
  `compute_block_common`; block-kind neutral):
  - `PowerDomain`: a `cluster` (grid `site_ranges`), `tile_class`
    (`members`) or `uncore` domain, with `rail_id`, `clock_domain_id` and
    `gateable`.
  - `SiteRange` and `DomainOperatingPoint` (clock, Vdd, `gated`,
    activity).
  - Exposed as the new optional `KPUArchitectureBase.power_domains`. It
    checks unique ids, known members, one tile_class domain per class,
    cluster ranges inside the grid and not overlapping, and that every tile
    `power_domain_id` names a defined, consistent domain.
- **`KPUThermalProfile`** gains the optional `domain_operating_points`
  (keyed by domain_id) and `tdp_scenario` (the activity of every tile
  class, which the TDP is derived from). `KPUEntry` and `ComputeProduct`
  check both against the architecture: known domains, gating only on
  gateable domains, and every tile class listed exactly once.

- **KPU tile kinds** (branes-ai/graphs#268 Phase B3).
  - `KPUArchitectureBase.tiles` is now `list[AnyKPUTile]`, discriminated on
    `tile_kind`; a missing `tile_kind` means `pe_fabric`, so every catalog
    YAML loads unchanged. `scalar` / `io_bridge` stay reserved and are
    rejected.
  - `KPUTileBase` holds the shared fields: `tile_kind`, `tile_type`,
    `tile_class_id`, `num_tiles`, `notes`, `footprint`, `local_memory`,
    `power_domain_id`, `placement`.
  - `KPUTileSpec` (alias `PEFabricTile`) is the programmable domain-flow
    fabric, unchanged. Its serialized key order is preserved exactly.
  - `SystolicTile`: a fixed-schedule GEMM / conv array, with `array_rows` x
    `array_cols` cells, one `mac` unit (MAC / FMA, one mode per format),
    `dataflow` and `supported_kernels`. `ops_per_tile_per_clock` and
    fill / drain cycles are derived.
  - `FixedFunctionTile`: wraps a `FunctionCore`. Its
    `ops_per_tile_per_clock` is always empty, so it never adds to
    programmable TOPS, and its memory is declared in the core.
  - `KPUArchitectureBase` now rejects a `total_tiles` that differs from
    `sum(num_tiles)` over the tile classes. Every catalog SKU already
    satisfies this.
- **Function cores** (new module `function_core.py`, architecture-neutral):
  `FunctionCore` is one encapsulated compute segment (ISP, SGM, VIO,
  radar, ...). It carries:
  - a dotted `function_id`, an input / output contract with config limits,
    and its numeric formats;
  - throughput in `WorkUnit`s per clock;
  - energy per unit at a reference node, split into logic / SRAM fractions;
  - `ops_equivalent_per_unit` (reporting only), IO bytes per unit, local
    memory, and silicon given as transistors or as an area at a reference
    node.

  The same definition can be placed in a KPU checkerboard or, later, as a
  standalone SoC block.
- **`local_memory.py`**: `LocalMemory` / `LocalMemoryLevel` /
  `LocalMemoryScope` moved out of `kpu.py`, so function cores can use them
  (still importable from `embodied_schemas.kpu`). Adds the `state` level.

- **Interconnect overlays** (new module `overlay.py`; branes-ai/graphs#268
  Phase B2). All are statically configured links: no routed networks and no
  routing tables.
  - **PE level:** `FabricInterconnect`, a nearest-neighbor base plus
    `FabricOverlay`s. The overlay kinds are row/col broadcast, express
    (`span`), reduction tree, transpose, butterfly and segmented bus, each
    with a per-row / col / tile scope. Exposed as the new optional
    `KPUTileSpec.interconnect`.
  - **Tile level:** `NoCOverlay`: express channels (`span` in mesh hops),
    stream links (an ordered producer -> consumer chain of `tile_class_id`s)
    and multicast trees. Exposed as the new optional `KPUNoCSpec.overlays`.
- **Validation:**
  - span limits per overlay kind;
  - overlays must fit the PE array (spans, square arrays for transpose,
    power-of-two axes for butterfly);
  - express channels must fit the mesh;
  - NoC overlay endpoints must reference existing tile classes;
  - overlay ids must be unique.

Backward compatible: every catalog YAML loads unchanged. Downstream,
`model_dump()` gains `noc.overlays` and `tiles[].interconnect` (B2), plus
`checkerboard`, `power_domains` and the thermal profiles'
`domain_operating_points` / `tdp_scenario` (B4), plus `performance`'s
`peak_ops_per_sec_by_precision` / `by_tile_kind` / `fixed_function_throughput`
(B5), plus `tiles[].tile_class_ref` (B6). All are `null` for catalog SKUs.

## [0.8.0] - 2026-09-14

Phase B1 of the KPU heterogeneous-tile sprint (branes-ai/graphs#268, #88).
Additive and backward compatible: every catalog YAML loads unchanged.

### Added

- **Datapath schemas** (new module `datapath.py`; branes-ai/graphs#268 Phase
  B1).
  - Number formats (`parse_number_format`, `NumberFormatName`):
    `int<n>` / `uint<n>`, named floats, `lns<n>`, `posit<n>[_<es>]` and
    `fixed<i>.<f>`.
  - `OpKind`, plus the software-equivalent ops-counting convention
    `DEFAULT_OPS_PER_INVOCATION` (MAC / FMA = 2, lerp = 3, min-plus = 2, ...).
  - `FunctionalUnit` with alternative `UnitMode`s, so a multi-precision unit
    counts its area once.
  - `PEDatapath`, with `ops_per_pe_per_clock()` and a projection onto legacy
    precision keys.
  - `RelativeEnergy` / `AbsoluteEnergy` (`EnergyRef`), energy referenced to a
    ProcessNode anchor op or given at a reference node.
- **New optional `KPUTileSpec` fields**, all backward compatible (every
  catalog YAML loads unchanged):
  - `tile_kind` (`KPUTileKind`; `pe_fabric` for all tiles today).
  - `tile_class_id` (defaults to a slug of `tile_type`).
  - `datapath`: when declared, it must reproduce `ops_per_tile_per_clock`.
  - `footprint`, `local_memory`, `power_domain_id` and `placement`.
- **`KPUArchitectureBase` validation:** `tile_class_id` must be unique, and
  `placement.adjacent_to` must reference existing tile classes.

Downstream note: `model_dump()` of a KPU tile now includes the new keys, so
catalog YAML emitted by the graphs generator and the graphs KPU golden
snapshots' `input` section gain them.

## [0.7.0] - 2026-09-14

First release with the unified `ComputeProduct` schema and the silicon
catalogs (process nodes, cooling, KPU SKUs). The `graphs` estimators need
this release: 0.6.0 predates `compute_product`, `process_node`, `kpu` and
`cooling_solution` entirely.

### Added

- **Unified `ComputeProduct` schema.** RFC 0001
  (`docs/rfcs/0001-compute-product-unification.md`, #7) defines it:
  products, dies, and a discriminated `blocks` union. Loaded with
  `load_compute_products()` (#15, #16). There are nine block kinds:
  - KPU (v1, #15)
  - GPU (v2, #19)
  - CPU (v3, #22)
  - NPU (v4, #25), including `KVCacheSpec` for transformer NPUs (#30)
  - CGRA (v5, #34)
  - DPU (v6, #36)
  - TPU (v7, #38)
  - DSP (v9, #43)
  - IO (v13, #79), the first non-compute block, used for multi-die chiplet
    IODs
- **`compute_block_common`**: vendor-neutral types shared across block kinds
  (#40):
  - a unified `TheoreticalPerformance` (#41, #42);
  - a unified `ThermalProfile` (#45, #46);
  - the `OnDieFabric` inheritance base (#47, #48);
  - the `DramAttachment` discriminator and the `dram_attachment` field
    (#49, #51).
- **Silicon catalogs: `ProcessNodeEntry`, `CoolingSolutionEntry` and KPU
  SKUs** (#9).
  - Per-circuit-class density, leakage and energy per op.
  - Per-profile Vdd, plus SRAM / NoC / DRAM energies.
  - Optional `leakage_vdd_exponent` for Vdd-scaled leakage (#84).
  - The `PROCESS_NODE_DATA_DIR` private overlay (#12, #13).
- **KPU thermal profiles and SKUs:**
  - per-(profile, precision) efficiency on `KPUThermalProfile` (#10);
  - FP16 throughput on KPU tile classes (#11);
  - the KPU SKU naming convention, plus the 12FDX / N7 sweeps (#14).
- **`KPUArchitectureBase`** (#86): the KPU architectural field set, defined
  once and shared by `KPUArchitecture` and `KPUBlock`. Convert between the
  two with `KPUBlock.from_architecture(arch)` and `block.to_architecture()`.
  Serialized output is byte-identical to before.
- **Catalog data** (ComputeProduct YAMLs unless noted):
  - **KPUs:** 12 Stillwater KPU SKUs (#16, #17).
  - **GPUs:** Jetson AGX Orin 64GB (#20), Jetson AGX Thor 128GB (#21),
    H100 PCIe (#73), T4 PCIe (#74).
  - **CPUs:** Core i7-12700K (#24); EPYC 9654 / 9754 / 9965 (#63-#65),
    re-authored as multi-die IOBlock layouts (#80-#82); AmpereOne A192 /
    A128 (#66, #67); Xeon 8490H / 8592+ / 6980P (#69-#71).
  - **NPUs:** Hailo-8 (#26), Hailo-10H (#31), Coral Edge TPU (#33).
  - **CGRA:** Plasticine v2 (#35).
  - **DPU:** Vitis AI B4096 (#37).
  - **TPUs:** v4 (#39), v1 (#75), v3 (#76), v5p (#77), Edge Pro (#78).
  - **DSPs:** Cadence Vision Q8 (#44), Synopsys ARC EV7x (#53), CEVA
    NeuPro-M NPM11 (#54), TI TDA4VM / TDA4AL / TDA4VH / TDA4VL (#55-#58),
    Qualcomm SA8775P / QRB5165 / QCS6490 (#59-#61).
  - **Process nodes:** 14 in total, including GF 28nm (#32), TSMC N4P (#21),
    Intel 7 (#24), TSMC N6 (#61), Intel 3 (#71) and TSMC 28HPM (#75).
  - **Board-level hardware** (`data/hardware/elma`, not ComputeProducts):
    6 Elma Electronic SKUs.

### Changed

- **Breaking:** the legacy `data/kpus/` catalog was retired.
  `load_kpus()` is now a compatibility shim that projects KPU
  ComputeProducts onto `KPUEntry` (#18).
- **Breaking:** the CGRA `host_dram_*` fields were renamed to
  `external_dram_*` (#50).
- **Breaking:** `dram_attachment` is now required when
  `has_external_dram=True` (#52).
- KPU profile Vdd values were re-tuned so the round catalog TDPs still hold
  under Vdd-scaled leakage (#85).

### Fixed

- `memory_bus_width_bits` on the Jetson AGX Thor 128GB (512 -> 256) and the
  Jetson Orin Nano 8GB (64 -> 128) YAMLs (#8, #83).

## [0.6.0] - 2026-04-13

### Added

- Mission-profile, battery and capability-tier schemas, plus the PyPI
  publish workflow. (This entry was reconstructed when 0.7.0 was released;
  0.6.0 shipped without a changelog entry.)

## [0.5.0] - 2026-01-03

### Added

#### CPU Architecture Summaries (`CPUArchitectureSummary`)

New schema for modeling CPU microarchitecture families, similar to `GPUArchitectureSummary`:
- Fields: process node, launch year, predecessor/successor, key features, IPC improvements
- Cache architecture, instruction extensions, memory support, socket compatibility

**Data (6 architectures)**:
- Intel: Raptor Lake (2022), Arrow Lake (2024), Granite Rapids (2024)
- AMD: Zen 4 (2022), Zen 5 (2024)
- ARM: Neoverse V2 (2023)

#### NPU/AI Accelerator Schema (`npu.py`)

New module for dedicated neural processing units and AI accelerators:
- `NPUEntry` - Complete accelerator specification
- `ComputeSpec` - TOPS, MAC units, supported data types, sparsity
- `MemorySpec` - SRAM, external memory, host memory usage
- `PowerSpec` - TDP, efficiency (TOPS/watt)
- `SoftwareSpec` - SDK, frameworks, supported operators
- `PhysicalSpec` - Form factor, interface (M.2, PCIe, USB, integrated)
- Enums: `NPUVendor`, `NPUType`, `NPUInterface`, `DataType`

**Data (4 NPUs)**:
| NPU | Vendor | TOPS | Efficiency |
|-----|--------|------|------------|
| Hailo-8 | Hailo | 26 | 10.4 TOPS/W |
| Hexagon Gen 3 | Qualcomm | 45 | 5.6 TOPS/W |
| Coral Edge TPU | Google | 4 | 2.0 TOPS/W |
| Intel NPU (Meteor Lake) | Intel | 10 | 1.7 TOPS/W |

#### Registry Query Methods

**NPU queries**:
- `get_npus_by_vendor()` - Filter by vendor
- `get_npus_by_type()` - Filter by type (discrete, integrated, datacenter)
- `get_npus_by_tops_range()` - Filter by performance range
- `get_npus_by_efficiency()` - Filter by minimum TOPS/watt

**CPU architecture queries**:
- `get_cpu_architectures_by_vendor()` - Filter by vendor
- `get_cpu_architecture_for_cpu()` - Get architecture for a CPU
- `get_cpus_for_architecture()` - Get CPUs using an architecture

### Changed

- Registry now includes `cpu_architectures` and `npus` CatalogViews
- `summary()` includes counts for new catalogs
- Updated loaders with `load_cpu_architectures()` and `load_npus()`
- Updated `__init__.py` exports with all new types

## [0.4.1] - 2026-01-02

### Added

#### AMD Hawk Point Chip Entries (2 chips)
- `amd_hawk_point_8845hs` - Ryzen 7 8845HS (Zen 4 + RDNA 3, 8 cores, 12 GPU CUs, 16 TOPS NPU)
- `amd_hawk_point_8945hs` - Ryzen 9 8945HS (Zen 4 + RDNA 3, 8 cores, 12 GPU CUs, 16 TOPS NPU)

These chip entries complete the cross-references from CPU entries `amd_hawk_point_8845hs` and `amd_hawk_point_8945hs`.

### Fixed

- Fixed cross-reference validation by adding missing AMD Hawk Point chip entries
- Verified graphs integration: all analysis schemas, adapters, and loaders compatible

### Validated

- **Graphs Integration**: All tests pass
  - Analysis schemas: `RooflineResult`, `EnergyResult`, `MemoryResult`, `GraphAnalysisResult`
  - Adapters: `convert_to_pydantic`, `convert_roofline_to_pydantic`, etc.
  - Data loaders: 14 hardware, 22 GPUs, 36 CPUs, 10 models, 4 sensors, 4 use cases
  - Cross-references: Hardware → GPU links verified

## [0.4.0] - 2025-12-31

### Added

#### ARM CPU Expansion (11 processors)

**Server CPUs**:
- Ampere: Altra Max M128-30 (128 cores, Neoverse N1), AmpereOne A192-32X (192 cores)
- AWS: Graviton3 (64 cores, Neoverse V1), Graviton4 (96 cores, Neoverse V2)
- NVIDIA: Grace CPU (72 cores, Neoverse V2)

**Embedded Processors**:
- NXP: i.MX 8M Plus (Quad A53 + 2.3 TOPS NPU), i.MX 93 (Dual A55 + 0.5 TOPS NPU)
- ST Micro: STM32H7 (Dual-core Cortex-M7 + M4, 480 MHz)

**Mobile SoCs**:
- Qualcomm: Snapdragon 8 Gen 3 (Cortex-X4 + A720 + A520)
- Google: Tensor G4 (Cortex-X4 + A720 + A520, Mali-G715)
- Samsung: Exynos 2400 (10-core, Xclipse 940 GPU)

#### Hardware Platforms (11 devices)

**NVIDIA Jetson**:
- Jetson Orin NX 8GB/16GB, AGX Orin 32GB/64GB, AGX Thor 128GB, TX2 8GB

**Raspberry Pi**:
- Raspberry Pi 5 8GB, Raspberry Pi 4 8GB

**Hailo Accelerators**:
- Hailo-8 M.2 (26 TOPS), Hailo-8L M.2 (13 TOPS), Hailo-10H M.2 (40 TOPS)

#### Embedded GPU Entries (6 GPUs)

Cross-referenced GPUs for hardware platforms:
- Orin NX GPU 8GB/16GB, Orin GPU 32GB/64GB, Thor GPU 128GB, TX2 GPU 8GB

#### Chip/SoC Entries (9 chips)

- NVIDIA: Orin NX SoC, Orin SoC, Thor SoC, Tegra X2 SoC
- Broadcom: BCM2712 (Pi 5), BCM2711 (Pi 4)
- Hailo: Hailo-8, Hailo-8L, Hailo-10H chips

#### CPU Schema Expansion

New vendors: `nxp`, `stmicro`, `google`, `samsung`

New architectures:
- Server: `neoverse_v1`, `neoverse_n1`
- Application: `cortex_a55`, `cortex_a53`
- Microcontroller: `cortex_m7`, `cortex_m4`, `cortex_m33`

New process nodes: `tsmc_n7`, `tsmc_n28`, `samsung_14lpc`, `samsung_3gae`, `custom`

New sockets: `lga4926` (Ampere Altra), `lga5964` (AmpereOne)

#### Software Architecture Schema (`architectures.py`)

New schema for modeling software architectures as operator graphs:
- `SoftwareArchitecture` - Complete architecture with stages and flows
- `ArchitectureStage` - Processing stage with operators and QoS
- `ArchitectureOperator` - Operator instance in stage
- `DataFlow` - Data flow between stages
- Enums: `StageRole`, `DataFlowType`, `StageExecutionMode`

Reference architectures:
- `drone_perception_v1` - Drone perception pipeline
- `simple_adas_v1` - ADAS perception pipeline
- `pick_and_place_v1` - Manipulation pipeline

#### Operator Schema (`operators.py`)

New schema for modeling computational operators:
- `Operator` - Complete operator with compute, I/O, QoS
- `ComputeProfile` - FLOPs, memory, parallelization
- `OperatorIO` - Input/output specifications
- `QoSRequirements` - Latency, throughput, accuracy
- Enums: `OperatorCategory`, `OperatorType`, `ComputeParadigm`

20 reference operators across categories:
- Perception: YOLO detectors (N/S/M/L/X), depth estimation, tracking
- Reasoning: behavior classifier, collision detector, trajectory predictor
- State Estimation: Kalman filters, scene graph manager
- Control: PID controller, A* path planner
- Infrastructure: image preprocessor, NMS postprocessor

### Changed

- GPU-Hardware cross-reference: All embedded GPUs now link to their hardware platforms
- Standardized hardware entry structure with complete cross-references

#### Seed Data: Models, Sensors, Use Cases

**Models (10)**:
- Detection: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, RT-DETR-L
- Depth: MiDaS Small v2.1, Depth Anything V2 Small
- Segmentation: YOLOv8n-seg, SAM ViT-B
- Pose: RTMPose-M

**Sensors (4)**:
- Cameras: Arducam IMX477 (12MP)
- Depth: Intel RealSense D435i, Luxonis OAK-D Pro
- LiDAR: Livox Mid-360

**Use Cases (4)**:
- Drone: obstacle avoidance, visual inspection
- Quadruped: security patrol
- AMR: warehouse navigation

### Fixed

- STM32 power values rounded to integers (schema constraint)
- Mobile SoC `pcie_version: null` changed to `"0"` (string required)
- Mobile SoC `igpu:` renamed to `graphics:` (schema field name)

## [0.3.0] - 2025-12-30

### Added

#### CPU Schema (`cpu.py`)
- Complete CPU schema for server, desktop, and mobile processors
- `CPUEntry` - Full processor specification with cores, clocks, cache, memory, power, platform
- `CoreConfig` - Hybrid architecture support (Intel P+E cores, ARM big.LITTLE)
- `ClockSpeeds` - Base/boost frequencies with per-core-type clocks
- `CacheSpec` - L1/L2/L3 hierarchy with 3D V-Cache support
- `MemorySpec` - Memory controller specs (channels, speed, ECC, computed bandwidth)
- `InstructionExtensions` - AVX-512, AMX, SVE, NEON, security extensions (SGX, SEV)
- `PowerSpec` - TDP with configurable power modes
- `PlatformSpec` - Socket, PCIe lanes, CXL support
- `IntegratedGraphics` - iGPU specs for APUs
- `MarketInfo` - Launch date, MSRP, target market
- Enums: `CPUVendor`, `CPUArchitecture`, `SocketType`, `TargetMarket`, `ProcessNode`
- Computed field: `threads_per_watt` efficiency metric

#### CPU Data (25 processors)
**Datacenter (8)**:
- Intel: Xeon 6980P (Granite Rapids), Xeon 6766E (Sierra Forest), Xeon Platinum 8592+ (Emerald Rapids), Xeon W9-3595X (Sapphire Rapids)
- AMD: EPYC 9965 (Turin 192-core), EPYC 9654 (Genoa), EPYC 9754 (Bergamo 128-core), EPYC 9575F (Turin V-Cache)

**Desktop (10)**:
- Intel: Core Ultra 9 285K (Arrow Lake), Core i9-14900K, i7-14700K, i7-12700K, i5-14600K, i3-14100
- AMD: Ryzen 9 9950X, Ryzen 9 7950X3D, Ryzen 7 9700X, Ryzen 5 9600X, Threadripper 7980X

**Mobile (4)**:
- Intel: Core Ultra 7 268V (Lunar Lake)
- AMD: Ryzen 9 8945HS, Ryzen 7 8845HS (Hawk Point)
- Apple: M4 Pro, M4 Max
- Qualcomm: Snapdragon X Elite X1E-84-100

#### Hardware Platform Data
- `nvidia_jetson_orin_nano_8gb` - Complete dev kit specification
  - Power modes (7W/15W), interfaces (CSI, USB, PCIe, GPIO)
  - Software ecosystem (JetPack, TensorRT, DeepStream)
  - Physical and environmental specs

#### Chip/SoC Data
- `nvidia_orin_nano_soc` - Orin Nano SoC specification (6x A78AE, 1024 CUDA cores, Ampere)

#### GPU Data Expansion (15 GPUs total)
**Datacenter AI GPUs**:
- NVIDIA: H100 SXM5, H200 SXM5, A100 SXM4, V100 SXM2, B100 SXM5, B200 SXM5, L40
- AMD: MI300X, MI250X, MI250
- Qualcomm: Cloud AI 100

**Consumer GPUs**:
- NVIDIA: RTX 4090, RTX 4080
- AMD: RX 7900 XTX
- Intel: Arc A770

#### Architecture Documentation
- `docs/architecture.md` - Explains data ownership split between embodied-schemas (datasheet facts) and graphs/hardware_registry (analysis params)

#### Registry Enhancements
- `cpus` CatalogView in Registry
- CPU query methods: `get_cpus_by_vendor()`, `get_cpus_by_architecture()`, `get_cpus_by_market()`, `get_cpus_by_socket()`

#### Testing
- `test_cpu_integration.py` - 20 tests for CPU loader, registry, data integrity, efficiency
- Total: 81 tests passing

### Changed
- GPU file naming standardized to `{model}_{form_factor}_{memory}_{memtype}.yaml`
- Updated `loaders.py` with `load_cpus()` function
- Updated `registry.py` with CPU catalog and query methods
- Updated `__init__.py` with CPU exports

## [0.2.2] - 2025-12-21

### Added
- `gpu.py` - Comprehensive GPU-specific schemas inspired by TechPowerUp GPU Database
  - `GPUEntry` - Complete GPU specification for discrete graphics cards
  - `DieSpec` - Fabrication specs (foundry, process, transistors, die size, chiplet support)
  - `ComputeResources` - Parallel compute units (shaders, CUDA/Stream processors, TMUs, ROPs, Tensor cores, RT cores)
  - `ClockSpeeds` - Base/boost/memory frequencies
  - `MemorySpec` - VRAM configuration (size, type, bus width, bandwidth, cache)
  - `TheoreticalPerformance` - Peak throughput (FP32/FP16/FP64 TFLOPS, tensor ops, fill rates)
  - `PowerSpec` - Power delivery (TDP, idle/gaming power, connectors, PSU recommendation)
  - `BoardSpec` - Physical card specs (slot width, length, PCIe, display outputs)
  - `FeatureSupport` - API support (DirectX, Vulkan, CUDA, DLSS/FSR, ray tracing)
  - `MarketInfo` - Pricing and availability (MSRP, launch date, target market)
  - `EfficiencyMetrics` - Computed metrics (perf/watt, perf/mm², bandwidth/watt)
  - `GPUArchitectureSummary` - Architecture family reference
  - Enums: `GPUVendor`, `Foundry`, `MemoryType`, `TargetMarket`, `PCIeGen`, `PowerConnector`, `DirectXVersion`, `ShaderModel`
- `data/gpus/` - GPU data directory structure (nvidia/, amd/, intel/)
- Example GPU entry: `data/gpus/nvidia/rtx_4090.yaml` - Complete RTX 4090 specification
- GPU schema tests in `test_schemas.py` - 4 new tests (19 total)
  - Minimal entry validation
  - Efficiency metric computation
  - Transistor density calculation
  - Extra field rejection

### Changed
- Updated `__init__.py` exports to include GPU schemas

## [0.2.1] - 2025-12-21

### Added
- CLAUDE.md - Claude Code guidance file for repository onboarding
  - Documents repository role as shared dependency for graphs and Embodied-AI-Architect
  - Build/test/development commands
  - Schema design patterns (verdict-first outputs, ID conventions)
  - Data ownership boundaries (datasheet specs vs roofline/calibration)
  - Guidelines for adding new data and making schema changes
  - Versioning and downstream compatibility notes

### Fixed
- Fixed date in docs/sessions/2025-12-20-initial-setup.md (was incorrectly 2024)

## [0.2.0] - 2025-12-20

### Added

#### Schema Models
- `hardware.py` - Complete hardware platform schemas
  - `HardwareEntry` - Full platform specification with capabilities, physical, environmental, power, and interface specs
  - `HardwareCapability` - Compute and memory capabilities (TOPS, TFLOPS, memory, bandwidth)
  - `ChipEntry` - Raw SoC/chip specifications
  - `PhysicalSpec` - Weight, dimensions, form factor for embodied systems
  - `EnvironmentalSpec` - Operating temperature, IP rating, vibration/shock ratings
  - `PowerSpec` - Power modes, TDP, battery compatibility
  - `InterfaceSpec` - CSI, USB, PCIe, GPIO, CAN bus counts
  - Enums: `HardwareType`, `FormFactor`, `ComputeParadigm`, `OperationType`, `LifecycleStatus`, `Availability`

- `models.py` - ML model schemas for perception workloads
  - `ModelEntry` - Complete model specification with architecture, I/O, accuracy, variants
  - `ModelVariant` - Quantization variants (fp32, fp16, int8) with accuracy delta
  - `ArchitectureSpec` - Backbone, neck, head, params, FLOPs
  - `AccuracyBenchmark` - mAP, mIoU, accuracy metrics per dataset
  - `MemoryRequirements` - Weights, activations, workspace memory
  - `InputSpec`, `OutputSpec` - Model I/O specifications
  - Enums: `ModelType`, `ModelFormat`, `DataType`

- `sensors.py` - Sensor schemas for perception systems
  - `SensorEntry` - Complete sensor specification
  - `CameraSpec` - Resolution, FPS, dynamic range, shutter type
  - `DepthSpec` - Stereo/ToF/structured light specs, range, accuracy
  - `LidarSpec` - Channels, points/sec, FoV, range
  - `ImuSpec` - Gyro/accel range, noise, sample rate
  - `SensorInterface`, `SensorPower`, `SensorPhysical`, `SensorEnvironmental`
  - Enums: `SensorCategory`, `InterfaceType`, `DepthType`

- `usecases.py` - Use case template schemas
  - `UseCaseEntry` - Application constraint templates for drones, quadrupeds, AMRs, etc.
  - `Constraint` - Min/max values with criticality levels
  - `SuccessCriterion` - Measurable success criteria with operators
  - `PerceptionRequirement` - Required tasks, target classes, detection range
  - `PlatformSpec` - Platform type, size class, indoor/outdoor
  - `RecommendedConfig` - Recommended hardware/model/sensor configurations
  - Enums: `ConstraintCriticality`, `PlatformType`, `Operator`

- `benchmarks.py` - Benchmark result schemas
  - `BenchmarkResult` - Complete benchmark with conditions, metrics, verdict
  - `AnalysisResult` - Tool output format with verdict, confidence, suggestion
  - `LatencyMetrics` - Mean, std, percentiles (p50, p90, p95, p99)
  - `ThroughputMetrics` - FPS, samples/sec, tokens/sec
  - `PowerMetrics` - Mean/peak power, energy per inference
  - `MemoryMetrics` - Model size, peak usage, GPU utilization
  - `ThermalMetrics` - Temperature, throttling status
  - `AccuracyMetrics` - Verification against expected accuracy
  - `BenchmarkConditions` - Power mode, batch size, environment
  - Enums: `Verdict`, `Confidence`

- `constraints.py` - Constraint ontology and tier definitions
  - `LatencyTier` - Ultra real-time (<10ms) to batch (>1s)
  - `PowerClass` - Ultra low power (<2W) to datacenter (>100W)
  - `MemoryClass`, `AccuracyClass` - Additional classifications
  - Pre-defined tier specifications with thresholds and use cases
  - Platform implication rules (drone → battery_powered → low_power)
  - Utility functions: `get_latency_tier()`, `get_power_class()`, `get_platform_implications()`

#### Data Infrastructure
- `loaders.py` - YAML loading with Pydantic validation
  - `load_yaml()`, `load_and_validate()` - Single file loading
  - `load_all_from_directory()` - Batch loading with validation
  - Category-specific loaders: `load_hardware()`, `load_models()`, `load_sensors()`, etc.
  - `validate_data_integrity()` - Full catalog validation

- `registry.py` - Unified data access API
  - `Registry` - Central access point for all catalog data
  - `CatalogView` - Queryable view with filtering support
  - Query methods: `get()`, `find()`, `find_one()`
  - Filter support: exact match, list membership, `_min`/`_max` comparisons
  - Relationship queries: `get_compatible_hardware()`, `get_compatible_models()`
  - Benchmark lookup: `get_benchmark()`, `get_benchmarks_for_model()`

#### Data Directory Structure
- `data/hardware/` - Hardware platforms by vendor (nvidia, qualcomm, hailo, google, intel, amd, raspberry_pi)
- `data/chips/` - Raw SoC specifications by vendor
- `data/models/` - ML models by type (detection, segmentation, depth, pose)
- `data/sensors/` - Sensors by category (cameras, depth, lidar)
- `data/usecases/` - Use case templates by platform (drone, quadruped, biped, amr, edge)
- `data/constraints/` - Tier definitions

#### Testing
- `tests/test_schemas.py` - Comprehensive schema validation tests (15 tests)
  - Hardware schema tests (minimal, full, physical specs, extra field rejection)
  - Model schema tests (variant, entry)
  - Sensor schema tests (camera entry)
  - Use case schema tests (constraint, criterion, entry)
  - Benchmark schema tests (metrics, result)
  - Constraint utility tests (tier classification)

#### Project Configuration
- `pyproject.toml` - Package configuration with dev dependencies
- `README.md` - Usage documentation and examples
- `LICENSE` - MIT license
- `.gitignore` - Python/IDE ignores

## [0.1.0] - 2025-12-20

### Added
- Initial repository creation
- Basic project structure

---

## Architecture Decisions

### Data Split with graphs Repository

This package owns **datasheet specs** (vendor-published facts):
- Hardware capabilities (memory, bandwidth, compute units)
- Power specifications and modes
- Physical specs (weight, dimensions, form factor)
- Environmental specs (temp range, IP rating)
- Interface specs (CSI, USB, PCIe counts)

The `graphs` repository owns **analysis-specific data**:
- `ops_per_clock` - Roofline model parameters
- `theoretical_peaks` - Computed performance ceilings
- Calibration data - Measured performance, efficiency curves
- Operation profiles - GEMM, CONV, attention benchmarks

See `Embodied-AI-Architect/docs/plans/shared-schema-repo-architecture.md` for full details.
