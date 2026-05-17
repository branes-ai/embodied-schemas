"""NPU compute block for ``ComputeProduct`` (v4 schema, additive).

PR 2 of the NPU sprint scoped at ``graphs#187``. Adds the fourth
member of the ``Block`` discriminated union after KPU (v1), GPU (v2),
and CPU (v3). Modeled directly off the field set audited in
``graphs/docs/designs/npu-compute-product-schema-extension.md``.

Design choice: same per-architecture-types rule the prior sprints
established -- ship NPU-specific sub-types (``NPUComputeFabric``,
``NPUMemorySubsystem``, ``NPUOnDieFabric``, ``NPUThermalProfile``,
``NPUTheoreticalPerformance``) rather than generalize. With only 4
architectures the right unification shape isn't obvious yet; defer
rename + unify to v5 (the design doc's "5th sprint" milestone).

**Second cross-block-kind type reuse**: ``NPUOnDieFabric.confidence``
reuses ``DataConfidence`` from ``process_node``. (The first cross-
block-kind reuse was CPU's ``ClockDomain`` from ``gpu_block``.)
These two data points justify carving out a vendor-neutral
``compute_block_common`` module when the v5 unification sprint lands.

Hailo-8 reference SKU specifics that shaped the design:

  - **All on-chip memory** (no external DRAM). The optional DRAM
    fields are gated by ``has_external_dram: bool``; a
    ``model_validator`` enforces consistency. Hailo-10H (LPDDR4X) and
    Coral Edge TPU (also SRAM-only) populate the same shape.
  - **Single thermal profile, no DVFS**. ``NPUThermalProfile`` uses
    a scalar ``clock_mhz`` plus ``dvfs_enabled`` flag (defaulting
    False); ``ClockDomain`` is NOT required because most NPUs ship
    one clock. (When an NPU does have DVFS, the schema can grow to
    accept ``ClockDomain`` per the GPU/CPU pattern.)
  - **Quantization-first precision support**. NPUs typically ship
    INT4 / INT8 only; ``NPUComputeFabric`` validator enforces that
    at least one of those is present (no FP-only fabrics).
  - **No SIMD efficiency**. NPUs are pure dataflow; the
    ``simd_efficiency_by_op_kind`` concept (CPU-only) doesn't apply
    and isn't present on ``NPUBlock``.

``KVCacheSpec`` (issue #27) adds the transformer-specific KV cache
description as an optional ``NPUBlock.kv_cache`` field. Hailo-8 leaves
it None (CNN-class NPU, no KV cache); Hailo-10H populates it for
generative inference (LPDDR4X-backed cache with ring-buffer streaming).

Future-deferred (v5+):

  - Integrated NPUs (Intel NPU, Qualcomm Hexagon, Apple ANE) with
    shared LPDDR with the host CPU complex. Needs cross-block
    memory link concepts -- v5 chiplet-style scope.
  - Datacenter AI accelerators (Groq, Cerebras, Graphcore,
    Tenstorrent) -- much larger surface area than edge NPUs.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.gpu import MemoryType
from embodied_schemas.process_node import CircuitClass, DataConfidence


# ---------------------------------------------------------------------------
# Helper enums
# ---------------------------------------------------------------------------

class NPUDataflowKind(str, Enum):
    """The dataflow scheduling discipline a fabric implements.

    Hailo's "structure-driven graph mapping" is STRUCTURE_DRIVEN.
    Google Edge TPU's systolic array is SYSTOLIC. Spatial dataflow
    machines (Cerebras, Wave Computing) are SPATIAL. KPUs (Stillwater)
    use OUTPUT_STATIONARY by convention; some NPUs (Mythic, certain
    embedded NPUs) use WEIGHT_STATIONARY or INPUT_STATIONARY for
    matvec-heavy workloads.
    """

    STRUCTURE_DRIVEN = "structure_driven"
    SYSTOLIC = "systolic"
    SPATIAL = "spatial"
    WEIGHT_STATIONARY = "weight_stationary"
    OUTPUT_STATIONARY = "output_stationary"
    INPUT_STATIONARY = "input_stationary"


class NPUNoCTopology(str, Enum):
    """On-die fabric topology for NPUs. Edge NPUs (Hailo, Coral) use
    2D meshes of dataflow units. Systolic arrays have an implicit
    DATAFLOW_RING. Crossbar is used for small accelerators with
    few units."""

    MESH_2D = "mesh_2d"
    DATAFLOW_RING = "dataflow_ring"
    SYSTOLIC = "systolic"
    CROSSBAR = "crossbar"


class NPUSramLayout(str, Enum):
    """How shared on-chip SRAM is organized across dataflow units."""

    SHARED = "shared"          # single shared SRAM bank visible to all units
    PARTITIONED = "partitioned"  # banked, one slice per cluster of units


class KVCacheStreamingKind(str, Enum):
    """How a transformer NPU streams its KV cache through the dataflow.

    Hailo-10H uses RING_BUFFER (rolling overwrite as context advances).
    Sliding-window attention models (Mistral-style) use SLIDING_WINDOW.
    PagedAttention-style runtimes (vLLM influence on hardware) use
    PAGE_BASED. PRECOMPUTED is the rare static-context case (prompt-
    pinned NPUs that never roll the cache).
    """

    RING_BUFFER = "ring_buffer"
    SLIDING_WINDOW = "sliding_window"
    PAGE_BASED = "page_based"
    PRECOMPUTED = "precomputed"


# ---------------------------------------------------------------------------
# Compute fabric (single dataflow fabric on most NPUs)
# ---------------------------------------------------------------------------

class NPUComputeFabric(BaseModel):
    """One compute fabric on an NPU. Most NPUs ship a single fabric
    (Hailo-8: 32 dataflow units, 500 INT8 ops/unit/clock). Multi-
    fabric NPUs would carry multiple entries; not common today.

    Mirrors ``GPUComputeFabric`` / ``CPUComputeFabric`` field-by-field
    where possible. The shape difference is in the discriminator
    (``dataflow_kind`` instead of ``fabric_kind`` / ``isa_extension``)
    and the energy baseline (``energy_per_op_int8_pj`` instead of
    ``energy_per_flop_fp32_pj`` -- NPUs don't ship FP32 so the FP32
    baseline is meaningless).
    """

    dataflow_kind: NPUDataflowKind = Field(...)
    circuit_class: CircuitClass = Field(
        ..., description="Standard-cell library used for this fabric"
    )
    ops_per_unit_per_clock: dict[str, int] = Field(
        ...,
        description=(
            "Ops per dataflow unit per clock keyed on precision name. "
            "Hailo-8: {'int8': 500, 'int4': 1000}. Coral: {'int8': 64}. "
            "Hailo-10H: {'int8': 500, 'int4': 1000} (same dataflow as "
            "Hailo-8 plus KV cache extensions)."
        ),
    )
    energy_per_op_int8_pj: float = Field(
        ..., gt=0,
        description=(
            "Energy per INT8 op in picojoules at the fabric's nominal "
            "operating point. NPUs are INT8-dominant; FP32 baseline "
            "doesn't apply. Hailo-8 16nm dataflow: ~0.34 pJ per INT8 op."
        ),
    )
    energy_scaling: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Multiplier on ``energy_per_op_int8_pj`` for each precision. "
            "Typical: {'int4': 0.5} since INT4 packs 2x into the same "
            "datapath. INT8 baseline is 1.0 implicitly (omit from this "
            "dict)."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_int_precision_required(self) -> "NPUComputeFabric":
        """NPUs must ship at least one of INT4 / INT8 -- the dominant
        inference precisions. Catches typo'd YAMLs that only declare
        FP precisions (which would be wrong for a real NPU)."""
        precisions = {k.lower() for k in self.ops_per_unit_per_clock}
        if not ({"int4", "int8"} & precisions):
            raise ValueError(
                "NPUComputeFabric.ops_per_unit_per_clock must include at "
                "least one of {'int4', 'int8'} (NPUs are inference "
                f"accelerators dominated by integer quantization); got: "
                f"{sorted(precisions)}"
            )
        return self


# ---------------------------------------------------------------------------
# Memory subsystem (SRAM-dominant or SRAM-only)
# ---------------------------------------------------------------------------

class NPUMemorySubsystem(BaseModel):
    """NPU memory hierarchy: SRAM-dominant. Hailo-8 and Coral Edge
    TPU have no external DRAM at all; Hailo-10H has LPDDR4X for
    transformer model weights + KV cache.

    The chip-shared L3 / coherence concepts from GPU / CPU /
    integrated CPUs don't apply -- NPUs are inference-only, run
    compiler-routed dataflow, and have no cache coherence (compiler
    handles all data movement explicitly)."""

    # On-chip SRAM bandwidth -- replaces the GPU/CPU "DRAM bandwidth"
    # field because most NPUs are SRAM-resident
    on_chip_bandwidth_gbps: float = Field(..., gt=0)

    # Per-dataflow-unit SRAM partition (software-managed scratchpad).
    # Always private per-unit; the compiler statically allocates it.
    sram_kib_per_unit: int = Field(..., gt=0)

    # Inter-unit shared SRAM (the "LLC" of NPU-land). Acts above the
    # per-unit SRAM partitions; smaller than CPU L3 in absolute terms
    # but performs the same role.
    shared_sram_kib: int = Field(..., ge=0)
    shared_sram_layout: NPUSramLayout = Field(NPUSramLayout.SHARED)

    # External DRAM. Gated by has_external_dram bool; when False all
    # the dram_* fields must be None / 0 (validator enforces). Hailo-8:
    # False. Coral: False. Hailo-10H: True with LPDDR4X / 4-8GB.
    has_external_dram: bool = Field(False)
    external_dram_type: MemoryType | None = Field(default=None)
    external_dram_size_gb: float | None = Field(default=None, ge=0)
    external_dram_bandwidth_gbps: float | None = Field(default=None, ge=0)

    # Energy per byte for the dominant memory tier (on-chip SRAM).
    # ~2 pJ/B for SRAM on 16nm; higher (~20 pJ/B) when DRAM is
    # involved -- carried on external_dram_* fields when populated.
    sram_access_energy_pj_per_byte: float = Field(..., gt=0)

    # Cache coherence. NPU default is "none" since compiler-routed
    # dataflow has no host-coherent cache. Free-form string in case
    # future host-coherent NPUs (NVIDIA NVLink-C2C-style) show up.
    coherence_protocol: str = Field(
        "none",
        description="Common: 'none' (NPU default), 'pcie' (host DMA), 'nvlink-c2c'",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_external_dram_consistency(self) -> "NPUMemorySubsystem":
        """When ``has_external_dram=True`` the external_dram_* fields
        must all be populated; when False they must all be None / 0.
        Catches typo'd YAMLs that toggled one field without the other.
        Mirrors the L3/L4 consistency pattern from CPU sprint."""
        if self.has_external_dram:
            missing = []
            if self.external_dram_type is None:
                missing.append("external_dram_type")
            if self.external_dram_size_gb is None or self.external_dram_size_gb <= 0:
                missing.append("external_dram_size_gb")
            if (self.external_dram_bandwidth_gbps is None
                    or self.external_dram_bandwidth_gbps <= 0):
                missing.append("external_dram_bandwidth_gbps")
            if missing:
                raise ValueError(
                    f"has_external_dram=True requires all of "
                    f"external_dram_type, external_dram_size_gb, "
                    f"external_dram_bandwidth_gbps to be populated; "
                    f"missing/zero: {missing}"
                )
        else:
            extras = []
            if self.external_dram_type is not None:
                extras.append("external_dram_type")
            if self.external_dram_size_gb is not None and self.external_dram_size_gb > 0:
                extras.append("external_dram_size_gb")
            if (self.external_dram_bandwidth_gbps is not None
                    and self.external_dram_bandwidth_gbps > 0):
                extras.append("external_dram_bandwidth_gbps")
            if extras:
                raise ValueError(
                    f"has_external_dram=False requires external_dram_* "
                    f"fields to be None / 0; got populated: {extras}"
                )
        return self


# ---------------------------------------------------------------------------
# On-die fabric (dataflow mesh, often low-confidence)
# ---------------------------------------------------------------------------

class NPUOnDieFabric(BaseModel):
    """NPU on-die interconnect between dataflow units. Most edge NPUs
    use 2D meshes of dataflow units (Hailo: 8x4, estimated; Coral:
    unknown). NPU vendors typically don't publish NoC details, so
    the ``confidence`` field defaults to THEORETICAL.

    SECOND cross-block-kind type reuse: ``confidence`` field uses
    ``DataConfidence`` from ``process_node`` rather than an NPU-
    specific enum. (The first was CPU's ``ClockDomain`` reuse from
    ``gpu_block``.)
    """

    topology: NPUNoCTopology = Field(...)
    bisection_bandwidth_gbps: float = Field(..., gt=0)
    unit_count: int = Field(
        ..., gt=0,
        description="Number of fabric endpoints (= num_dataflow_units typically)",
    )
    flit_size_bytes: int = Field(..., gt=0)

    # Mesh-specific (optional; only populated when topology=MESH_2D)
    mesh_rows: int | None = Field(default=None, gt=0)
    mesh_cols: int | None = Field(default=None, gt=0)

    hop_latency_ns: float = Field(..., ge=0)
    pj_per_flit_per_hop: float = Field(..., ge=0)
    routing_distance_factor: float = Field(1.0, gt=0)

    confidence: DataConfidence = Field(
        DataConfidence.THEORETICAL,
        description=(
            "Provenance of NoC numbers. NPU vendors rarely publish "
            "fabric details so THEORETICAL is the dominant case."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_mesh_dims(self) -> "NPUOnDieFabric":
        """When topology=MESH_2D both mesh_rows and mesh_cols should
        be set (and their product should match unit_count); when
        topology is anything else both should be None."""
        is_mesh = self.topology == NPUNoCTopology.MESH_2D
        if is_mesh:
            if self.mesh_rows is None or self.mesh_cols is None:
                raise ValueError(
                    f"topology=MESH_2D requires both mesh_rows and "
                    f"mesh_cols to be set; got rows={self.mesh_rows}, "
                    f"cols={self.mesh_cols}"
                )
            if self.mesh_rows * self.mesh_cols != self.unit_count:
                raise ValueError(
                    f"topology=MESH_2D: mesh_rows * mesh_cols "
                    f"({self.mesh_rows} * {self.mesh_cols} = "
                    f"{self.mesh_rows * self.mesh_cols}) must equal "
                    f"unit_count ({self.unit_count})"
                )
        else:
            if self.mesh_rows is not None or self.mesh_cols is not None:
                raise ValueError(
                    f"topology={self.topology.value} requires mesh_rows "
                    f"and mesh_cols to be None; got rows={self.mesh_rows}, "
                    f"cols={self.mesh_cols}"
                )
        return self


# ---------------------------------------------------------------------------
# Thermal profile (single operating point on most NPUs)
# ---------------------------------------------------------------------------

# NPUThermalProfile is now an alias of the unified ``ThermalProfile``
# from ``compute_block_common`` (v9 sprint PR 3 -- branes-ai/graphs#215).
# The class body was byte-identical to 4 other per-block-kind classes
# (CGRA/DPU/TPU/DSP); the unified type accepts the same data shape.
#
# This alias preserves backward compat for callers that import
# ``NPUThermalProfile`` -- ``isinstance(x, NPUThermalProfile)``
# AND ``isinstance(x, ThermalProfile)`` are both True because
# they refer to the same class object.
from embodied_schemas.compute_block_common import ThermalProfile
NPUThermalProfile = ThermalProfile


# ---------------------------------------------------------------------------
# Theoretical performance roll-up
# ---------------------------------------------------------------------------

# NPUTheoreticalPerformance is now an alias of the unified
# ``TheoreticalPerformance`` from ``compute_block_common`` (v8 sprint
# PR 3 -- branes-ai/graphs#208). The class body was byte-identical to
# 5 other per-block-kind classes (CPU/GPU/CGRA/DPU/TPU); the unified
# type accepts the same data shape (NPU's INT4/INT8-only data lives
# in ``peak_ops_per_sec_by_precision`` exactly as before).
#
# This alias preserves backward compat for callers that import
# ``NPUTheoreticalPerformance`` -- ``isinstance(x, NPUTheoreticalPerformance)``
# AND ``isinstance(x, TheoreticalPerformance)`` are both True because
# they refer to the same class.
#
# The empty-FP set is still the NPU norm (most ship INT4/INT8 only);
# the unified type's optional ``sparse_peak_ops_per_sec_by_precision``
# field (GPU-only today) defaults to None for NPU SKUs.
from embodied_schemas.compute_block_common import TheoreticalPerformance
NPUTheoreticalPerformance = TheoreticalPerformance


# ---------------------------------------------------------------------------
# KV cache (transformer-capable NPUs only; issue #27)
# ---------------------------------------------------------------------------

class KVCacheSpec(BaseModel):
    """Describes the KV cache management surface of a transformer-
    capable NPU.

    Carried as an optional ``NPUBlock.kv_cache`` field. CNN-class NPUs
    (Hailo-8, Coral Edge TPU) leave it None. Transformer NPUs (Hailo-
    10H, future Tenstorrent / Groq inference SKUs) populate it.

    The fields describe the architectural KV cache capability, not the
    per-deployment configuration: ``max_context_length`` is the chip's
    upper bound, not what a particular model will use.
    """

    max_context_length: int = Field(
        ..., gt=0,
        description=(
            "Maximum sequence length (in tokens) the KV cache can hold. "
            "Architectural upper bound, not a deployment knob. "
            "Hailo-10H targets ~8192 tokens; future LLM NPUs reach 32K+."
        ),
    )
    kv_cache_kib_per_layer: int = Field(
        ..., gt=0,
        description=(
            "Per-transformer-layer KV cache footprint in KiB at the "
            "chip's native quantization. Derived from model class * "
            "max_context_length * head_dim, but published as a chip-"
            "level attribute so the SKU YAML can be authored without "
            "pinning to one model."
        ),
    )
    num_layers_supported: int = Field(
        ..., gt=0,
        description=(
            "Number of transformer layers whose KV cache the NPU can "
            "hold simultaneously (SRAM-resident + DRAM-offloaded "
            "combined). Hailo-10H: ~32 layers for the targeted 7B-class "
            "models."
        ),
    )
    quantization: dict[str, str] = Field(
        ...,
        description=(
            "Per-tier quantization for the K and V projections. Common "
            "values: {'k': 'int8', 'v': 'int8'} for symmetric, "
            "{'k': 'int8', 'v': 'int4'} for V-asymmetric (Hailo-10H "
            "style). Keys must be 'k' and 'v'."
        ),
    )
    streaming_strategy: KVCacheStreamingKind = Field(
        ...,
        description=(
            "How the cache rolls as new tokens arrive. RING_BUFFER is "
            "the common case for autoregressive decode; SLIDING_WINDOW "
            "for windowed attention; PAGE_BASED for runtimes that map "
            "KV blocks to pages."
        ),
    )
    has_offload_to_dram: bool = Field(
        ...,
        description=(
            "True when KV entries that don't fit in on-chip SRAM spill "
            "to external DRAM. Hailo-10H: True (LPDDR4X holds the "
            "overflow). Groq LPU: False (huge on-chip SRAM, no DRAM). "
            "When True, ``NPUMemorySubsystem.has_external_dram`` must "
            "also be True -- enforced by NPUBlock validator."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_quantization_keys(self) -> "KVCacheSpec":
        """``quantization`` must contain both 'k' and 'v' entries and
        no others. Catches typo'd YAMLs that named the tiers 'key' /
        'value' or omitted one tier."""
        keys = set(self.quantization)
        if keys != {"k", "v"}:
            raise ValueError(
                f"KVCacheSpec.quantization must have exactly keys "
                f"{{'k', 'v'}}; got {sorted(keys)}"
            )
        return self


# ---------------------------------------------------------------------------
# NPUBlock (the discriminated union member)
# ---------------------------------------------------------------------------

class NPUBlock(BaseModel):
    """NPU compute block. Carries the NPU-specific architectural
    description: dataflow unit hierarchy, on-chip-SRAM-dominant
    memory subsystem, dataflow on-die fabric, and NPU-only scheduler
    attributes (high default occupancy, single concurrent model).

    The discriminator value ``BlockKind.NPU`` is wired in
    ``compute_product.py``. Imports are arranged so this module does
    not import from ``compute_product`` (compute_product imports from
    here). Same pattern as GPUBlock and CPUBlock.
    """

    kind: Literal["npu"] = Field(
        "npu",
        description="Discriminator -- always 'npu' for NPUBlock",
    )

    # Dataflow unit hierarchy
    num_dataflow_units: int = Field(
        ..., gt=0,
        description=(
            "Number of dataflow processing elements. Hailo-8: 32. "
            "Coral Edge TPU: 1 (the systolic array is treated as a "
            "single 'unit' even though it has 4096 multipliers internally). "
            "Hailo-10H: 40."
        ),
    )
    lanes_per_unit: int = Field(
        1, gt=0,
        description=(
            "SIMD lane count per dataflow unit, or 1 for scalar dataflow. "
            "Most NPUs are single-lane (the parallelism comes from the "
            "unit count); some wider NPUs (Coral) have multi-lane units."
        ),
    )

    # Single compute fabric in the common case (Hailo, Coral); could
    # grow to multiple fabrics for an NPU with heterogeneous units.
    compute_fabrics: list[NPUComputeFabric] = Field(..., min_length=1)

    # Precisions supported chip-wide -- union of compute_fabrics[*].ops_per_unit_per_clock
    multi_precision_alu: list[str] = Field(default_factory=list)

    memory: NPUMemorySubsystem = Field(...)
    noc: NPUOnDieFabric = Field(...)

    # NPU-only scheduler / mapper attributes
    min_occupancy: float = Field(
        0.8, ge=0.0, le=1.0,
        description=(
            "Higher default (0.8) than GPU (0.3) or CPU (0.4) because "
            "the dataflow compiler statically allocates resources. "
            "Real NPU deployments routinely hit 0.85-0.95 occupancy "
            "because the compiler pre-maps the entire model graph."
        ),
    )
    max_concurrent_models: int = Field(
        1, gt=0,
        description=(
            "Maximum number of distinct compiled models the NPU can "
            "switch between without recompilation. Most edge NPUs run "
            "a single compiled model at a time (max_concurrent_models=1); "
            "datacenter NPUs may support more."
        ),
    )
    wave_quantization: int = Field(
        1, gt=0,
        description="NPUs don't wave-quantize; default 1 keeps shape consistent",
    )

    # Transformer-specific KV cache surface. Optional: CNN-class NPUs
    # (Hailo-8, Coral) leave it None; transformer NPUs (Hailo-10H,
    # future Tenstorrent / Groq inference SKUs) populate it.
    kv_cache: KVCacheSpec | None = Field(
        default=None,
        description=(
            "Optional KV cache description for transformer-capable NPUs. "
            "None for CNN-class NPUs."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_noc_unit_count_matches(self) -> "NPUBlock":
        """noc.unit_count should equal num_dataflow_units. Catches
        YAMLs where the NoC was authored against a different unit
        count than the block declared."""
        if self.noc.unit_count != self.num_dataflow_units:
            raise ValueError(
                f"noc.unit_count ({self.noc.unit_count}) must equal "
                f"num_dataflow_units ({self.num_dataflow_units})"
            )
        return self

    @model_validator(mode="after")
    def _validate_kv_cache_dram_consistency(self) -> "NPUBlock":
        """When ``kv_cache.has_offload_to_dram=True`` the memory
        subsystem must have ``has_external_dram=True`` -- a KV cache
        that overflows to nowhere doesn't make sense. Catches mismatched
        YAMLs that declared DRAM-offloaded KV cache on an SRAM-only NPU."""
        if (self.kv_cache is not None
                and self.kv_cache.has_offload_to_dram
                and not self.memory.has_external_dram):
            raise ValueError(
                "kv_cache.has_offload_to_dram=True requires "
                "memory.has_external_dram=True (KV cache cannot offload "
                "to nonexistent external DRAM)"
            )
        return self
