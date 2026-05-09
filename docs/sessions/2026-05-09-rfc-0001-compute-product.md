# Session 2026-05-09 -- RFC 0001 ComputeProduct + YAML bug report

**Repo:** `branes-ai/embodied-schemas`
**Landed on `main`:** PR #7 (RFC 0001), commit `5a6e0f4`
**Filed:** issue #8

This session's work in this repo was driven by the cross-repo arc happening in
`branes-ai/graphs` (PRs #133-#143), where a chip-level `PhysicalSpec` data
class was added as a sibling to `HardwareResourceModel` and a YAML loader was
written to source `PhysicalSpec` data live from this repo's catalogs. Two
artifacts came back upstream:

1. An architectural assessment proposing a unified `ComputeProduct` schema to
   replace the four parallel category schemas (GPU / CPU / NPU / Chip) -- now
   living in this repo as **RFC 0001**.
2. A bug report documenting two YAMLs that disagree with their own announced
   bandwidth math -- filed as **issue #8**.

## RFC 0001: Unified ComputeProduct Schema

The original assessment lives in `branes-ai/graphs:docs/assessments/
compute-product-unification.md`; the version merged here (`docs/rfcs/
0001-compute-product-unification.md`) is the consumer-side critique
turned into a concrete migration proposal.

### What's wrong with the four-schema status quo

Counting `die_spec` coverage across the four catalogs revealed the gap:

| Catalog | Entries | `die_spec` populated |
|---------|---------|----------------------|
| GPU     | 22      | 22 (100%)            |
| CPU     | varies  | 0                    |
| NPU     | 4       | 0                    |
| Chip    | varies  | 0                    |

The four schemas don't share a die-level metadata model, so adding
fabrication data (die size, transistors, foundry, process node) to one
schema doesn't generalize to the others. Worse, modern automotive /
embodied SoCs are heterogeneous (CPU + GPU + NPU + DSP on one die), and
the current scheme forces them into a primary-category bin with the rest
of the silicon's compute blocks treated as second-class fields.

### The proposed shape

`ComputeProduct` is a single top-level schema with:

- A `blocks` field: discriminated union over compute-block kinds (CPU /
  GPU / NPU / DSP / FPGA). One physical product can have many blocks.
- A `contains` hierarchy: products can contain other products (a board
  contains a module, a module contains an SoC, an SoC contains dies).
  This lets `die_spec` live at the level it physically belongs to (the
  die) without being duplicated up the hierarchy.
- A unified fab/spec metadata bundle that's reused across all block
  kinds.

The RFC keeps the existing per-category loaders working through the
transition, and outlines a 4-phase migration: (1) introduce the schema
alongside existing ones, (2) populate from existing entries via a
generated converter, (3) move loaders to `ComputeProduct` while keeping
shims, (4) remove deprecated paths.

### Status

**Phase 1 done** -- the RFC document is in the repo. Phases 2-4 are
outstanding work and intentionally not yet scheduled, since the consumer
(graphs) currently has a working YAML loader that reads the per-category
schemas via `KNOWN_OVERRIDES` for documented bugs. That works as a bridge
until phase 4 of this RFC lands.

## Issue #8: YAML memory-bus-width bugs

The graphs-side `PhysicalSpec` loader was given a bandwidth-math
invariant (`BW = bus_width / 8 * DRAM_rate`, ±5%) as a sanity check on
loaded values. It immediately flagged two existing YAMLs in this repo:

| SKU | YAML says | Math says | Vendor source |
|-----|-----------|-----------|---------------|
| Jetson Thor 128GB | 512-bit | 256-bit (273 GB/s ÷ 8.533 GT/s LPDDR5X) | NVIDIA Jetson Thor announcement blog: "256-bit LPDDR5X" |
| Jetson Orin Nano 8GB | 64-bit | 128-bit (68 GB/s ÷ 4.267 GT/s LPDDR5-4267) | NVIDIA Orin Nano datasheet: 128-bit |

Both look like transcription errors (likely doubling/halving from a
different unit interpretation). The interim workaround on the consumer
side is a `KNOWN_OVERRIDES` table in `physical_spec_loader.py` that
applies field-level corrections at load time and cites issue #8.

### Recommendation for this repo

Adopt the bandwidth-math check as a schema-level validator (e.g., in the
`MemorySpec` Pydantic model). It's a one-liner that would have caught
both these bugs at YAML import time:

```python
@model_validator(mode='after')
def check_bandwidth_consistency(self):
    if self.peak_bandwidth_gb_s and self.bus_width_bits and self.dram_rate_gtps:
        expected = (self.bus_width_bits / 8) * self.dram_rate_gtps
        if abs(self.peak_bandwidth_gb_s - expected) / expected > 0.05:
            raise ValueError(
                f"Bandwidth math inconsistent: stated {self.peak_bandwidth_gb_s} GB/s, "
                f"derived {expected:.1f} GB/s from {self.bus_width_bits}b * {self.dram_rate_gtps} GT/s"
            )
    return self
```

Filing this as a follow-up in this session would make sense; for now the
consumer-side override is the bridge.

## Cross-repo deployment context

What this repo's RFC and YAML data unblocked downstream this session:

- **graphs** got a single-source-of-truth YAML loader for chip-level fab
  metadata. Five hardware factories (H100 SXM5, Orin AGX 64GB, Orin NX
  16GB, Orin Nano 8GB, Thor 128GB) now read live from these YAMLs
  instead of carrying inline `PhysicalSpec(...)` literals.
- **embodied-ai-architect** got a dynamic hardware catalog that lives off
  the graphs registry, which itself is keyed by `base_id` references
  back to this repo. The chain is now: orchestrator → graphs registry
  → embodied-schemas YAML.

## Next steps for this repo

- **Phase 2 of RFC 0001**: write the converter from existing per-category
  YAMLs to `ComputeProduct` form, in this repo. Targeted but not
  scheduled.
- **Resolve issue #8**: correct the two Jetson YAMLs and (ideally) add
  the bandwidth-math validator so this class of bug is caught at the
  source. The graphs-side `KNOWN_OVERRIDES` table will retire
  automatically once these are fixed.
- **Coverage pass for `die_spec` on non-GPU catalogs**: 0% on CPUs / NPUs
  / SoCs / Chips today. Filling these in (or migrating to
  `ComputeProduct`) is what unblocks the remaining 41 mappers in graphs
  from getting populated `PhysicalSpec`.
