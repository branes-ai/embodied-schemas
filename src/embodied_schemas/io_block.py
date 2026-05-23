"""IO compute block for ``ComputeProduct`` (v13 schema, additive).

PR 2 of the IOBlock mini-sprint scoped at ``graphs#245``. Adds the
ninth member of the ``Block`` discriminated union after the 8
category block kinds (KPU/GPU/CPU/NPU/CGRA/DPU/TPU/DSP). Modeled
directly off the field set audited in the paper exercise at
``graphs/docs/designs/ioblock-compute-product-schema-extension.md``.

Design choice: **first non-compute block kind in the catalog**. The
8 prior block kinds all represent compute fabrics (with peak ops/sec).
IOBlock represents the non-compute silicon on an I/O die: memory
controllers, PCIe controllers, inter-socket coherence links,
intra-package coherence fabric, security processor, power-management
controller. There is **no ``performance: TheoreticalPerformance``
field** on IOBlock -- an IOD has no peak ops/sec; its "performance"
is captured by discrete typed bandwidth fields (memory_bandwidth_gbps,
pcie_lanes * generation, inter_socket aggregate).

IOBlock-specific architectural patterns the schema covers:

  - **Memory controller ownership** -- the IOD physically owns the
    DRAM IMC + PHY. ``IOMemorySubsystem`` mirrors the shape of
    ``CPUMemorySubsystem`` for the DRAM-facing fields but omits the
    cache hierarchy (L3 lives on CCDs in AMD chiplet products).
  - **PCIe surface** -- ``pcie_lanes`` + ``pcie_generation``. First
    block kind to model PCIe directly; prior block kinds assumed
    the host provided the PCIe surface.
  - **CXL surface** -- piggybacks on PCIe physical layer; ``cxl_supported``
    + ``cxl_version``. Free-form version string because CXL spec
    versioning is volatile (1.1 / 2.0 / 3.0 / 3.1 ...).
  - **Inter-socket coherence links** -- ``inter_socket_links:
    list[InterSocketLink]``. One entry per link kind (AMD G-link,
    Intel UPI, NVIDIA NVLink-C2C, etc.). Per-link bandwidth + count.
  - **Intra-package coherence fabric** -- ``coherence_fabric:
    IOOnDieFabric``. Inherits from ``OnDieFabric`` (v10) and contributes
    a typed ``IOFabricTopology`` enum
    (INFINITY_FABRIC / MESH_EXTENSION / EMIB_HUB / CXL_FABRIC).
  - **Security + management** -- ``security_processor_kind`` +
    ``power_management_controller`` + ``boot_rom_present``. Discrete
    boolean / free-form fields for v13; consolidate into a sub-object
    in v14+ if multi-tenant attestation / secure-boot-chain
    semantics need richer modeling.

AMD Genoa IOD (reference SKU for this sprint) specifics:

  - 12-channel DDR5-4800 (memory_controllers=12, memory_bandwidth_gbps=460.8)
  - 128 PCIe Gen5 lanes (pcie_lanes=128, pcie_generation=PCIE_5)
  - CXL 1.1 over PCIe physical (cxl_supported=True, cxl_version="1.1")
  - 4x G-link inter-socket links @ ~250 GB/s each (for 2P configurations)
  - INFINITY_FABRIC coherence_fabric topology
  - AMD PSP security processor
  - PMC + boot ROM present

Multi-SKU schema coverage (3 chiplet AMD SKUs in the catalog):

  - **EPYC 9654 (Genoa, this sprint reference)**: TSMC N6 IOD, 12-ch
    DDR5-4800, 128 PCIe Gen5, CXL 1.1, Infinity Fabric.
  - **EPYC 9754 (Bergamo)**: reuses the SAME Genoa IOD die unchanged.
    Schema-wise identical to 9654's IOBlock.
  - **EPYC 9965 (Turin Dense)**: TSMC N6 Turin IOD, 12-ch DDR5-6000,
    128 PCIe Gen5, CXL 2.0, Infinity Fabric. Same IOBlock shape, new
    field values.

Intel Xeon SPR/EMR/GNR are explicitly **out of scope** -- their tile
architectures integrate memory + IO into each compute tile (no
separate IOD). They stay as single-die products with CPUBlock; the
paper exercise documents this design call.

Future-deferred (v14+):

  - ``MEMORY`` block kind for HBM stacks as separate dies (next future
    block kind per the v9 BlockKind docstring).
  - ``BRIDGE`` block kind for EMIB / silicon-interposer modeling.
  - iGPU-on-IOD multi-block representation (AMD APUs); the existing
    multi-block-per-die infrastructure supports this already; v13 of
    IOBlock doesn't require schema changes for it.
  - Refined IOMemorySubsystem with per-channel bandwidth (vs aggregate).
  - ``ManagementSilicon`` sub-object collapsing the security + PMC + boot
    ROM fields once multi-tenant attestation / secure-boot-chain
    semantics warrant richer modeling.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from embodied_schemas.compute_block_common import (
    MemoryType,
    OnDieFabric,
    PCIeGen,
)


# ---------------------------------------------------------------------------
# IO fabric topology enum
# ---------------------------------------------------------------------------

class IOFabricTopology(str, Enum):
    """The intra-package coherence fabric topology on an I/O die.

    INFINITY_FABRIC is AMD's chiplet fabric (used by Genoa/Bergamo/
    Turin IODs). MESH_EXTENSION is a placeholder for Intel SPR-style
    tile-internal fabric (not used by current chiplet SKUs in the
    catalog -- Intel tiles don't ship a separate IOD). EMIB_HUB and
    CXL_FABRIC are placeholders for future products."""

    INFINITY_FABRIC = "infinity_fabric"
    MESH_EXTENSION = "mesh_extension"
    EMIB_HUB = "emib_hub"
    CXL_FABRIC = "cxl_fabric"


class IOOnDieFabric(OnDieFabric):
    """IO-block-specific on-die fabric. Inherits the 7 shared NoC fields
    + mesh dims + confidence from ``OnDieFabric`` (v10 base) and
    contributes the typed ``IOFabricTopology`` enum.

    Pattern matches the 8 prior per-kind OnDieFabric subclasses
    (CPUOnDieFabric, NPUOnDieFabric, etc.)."""

    topology: IOFabricTopology = Field(...)


# ---------------------------------------------------------------------------
# IO memory subsystem
# ---------------------------------------------------------------------------

class IOMemorySubsystem(BaseModel):
    """DRAM-facing memory subsystem owned by the I/O die. Mirrors the
    shape of ``CPUMemorySubsystem`` for the DRAM-facing fields but
    omits the cache hierarchy (in AMD chiplet products, L3 lives on
    the CCDs, not on the IOD).

    ``memory_controllers`` is the channel count (12 for AMD Genoa /
    Turin IODs; varies for other products). ``memory_type`` uses the
    shared ``MemoryType`` enum from ``compute_block_common``.
    """

    memory_type: MemoryType = Field(..., description="Main-memory technology (DDR5, HBM3, LPDDR5X, ...)")
    memory_size_gb: float = Field(..., gt=0, description="Maximum addressable DRAM per socket")
    memory_bus_bits: int = Field(..., gt=0, description="Total bus width across all channels (e.g., 768 = 12 channels x 64 bits)")
    memory_bandwidth_gbps: float = Field(..., gt=0, description="Aggregate peak bandwidth across all channels")
    memory_controllers: int = Field(..., gt=0, description="Channel count (e.g., 12 for AMD Genoa IOD)")
    ecc_supported: bool = Field(True, description="Datacenter IODs all support ECC; client IODs often don't")
    memory_controller_energy_pj_per_byte: float | None = Field(
        None, ge=0,
        description="Per-byte access energy at the IMC level (distinct from PHY-only); useful for IO-side roofline modeling",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Inter-socket coherence link sub-type
# ---------------------------------------------------------------------------

class InterSocketLink(BaseModel):
    """One kind of inter-socket coherence link. AMD G-link (used by
    Genoa/Bergamo/Turin) and Intel UPI (Sapphire Rapids+) are the
    primary instances. NVIDIA NVLink-C2C (Grace Hopper) is a future
    instance. Per-link bandwidth (one direction) + the number of links
    of this kind."""

    name: str = Field(..., description="Link kind, e.g., 'AMD G-link', 'Intel UPI 2.0'")
    bandwidth_per_link_gbps: float = Field(..., gt=0, description="One-direction bandwidth per link")
    link_count: int = Field(..., gt=0, description="Number of links of this kind on this IOD")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# IOBlock (discriminated union member)
# ---------------------------------------------------------------------------

class IOBlock(BaseModel):
    """I/O die block. Represents the non-compute silicon on an IOD:
    memory controllers, PCIe controllers, CXL interfaces, inter-socket
    coherence links, intra-package coherence fabric, security
    processor, power-management controller.

    The discriminator value ``BlockKind.IO`` is wired in
    ``compute_product.py``. Imports are arranged so this module does
    not import from ``compute_product`` (compute_product imports from
    here). Same pattern as the 8 prior block kinds.

    **No ``performance`` field** -- IODs have no peak ops/sec. Their
    "performance" is captured by ``memory.memory_bandwidth_gbps`` +
    ``pcie_lanes * pcie_generation`` aggregate + ``inter_socket_links``
    aggregate + ``coherence_fabric.bisection_bandwidth_gbps`` -- all
    as discrete typed fields. Forcing them into the generic
    ``TheoreticalPerformance.peak_ops_per_sec_by_precision`` dict
    loses type safety; see the paper exercise at
    ``graphs/docs/designs/ioblock-compute-product-schema-extension.md``
    section 6 for the design discussion.
    """

    kind: Literal["io"] = Field(
        "io",
        description="Discriminator -- always 'io' for IOBlock",
    )

    # Memory subsystem (DRAM controllers + PHY)
    memory: IOMemorySubsystem = Field(...)

    # Intra-package coherence fabric (Infinity Fabric routers etc.)
    coherence_fabric: IOOnDieFabric = Field(...)

    # PCIe surface
    pcie_lanes: int = Field(..., gt=0, description="Total PCIe lanes exposed by this IOD")
    pcie_generation: PCIeGen = Field(..., description="PCIe generation supported")

    # CXL surface (piggybacks on PCIe physical layer)
    cxl_supported: bool = Field(False)
    cxl_version: str | None = Field(
        None,
        description="CXL spec version, e.g., '1.1', '2.0', '3.0'. Free-form because CXL version space is volatile.",
    )

    # Inter-socket coherence links
    inter_socket_links: list[InterSocketLink] = Field(
        default_factory=list,
        description="One entry per link kind (e.g., one for AMD G-link or Intel UPI). Empty for single-socket-only IODs (e.g., client APUs).",
    )

    # Security + management silicon
    security_processor_kind: str | None = Field(
        None,
        description="Free-form kind label, e.g., 'AMD PSP', 'Intel TXT/SGX', 'ARM TrustZone-CSE'",
    )
    security_processor_die_area_mm2: float | None = Field(None, ge=0)
    power_management_controller: bool = Field(
        False,
        description="Does this IOD own the PMC / SMU? Most datacenter IODs do; client SoCs sometimes put it on the CPU die.",
    )
    boot_rom_present: bool = Field(False)

    # Energy + power roll-ups (optional in v13; most datasheets don't publish)
    idle_power_watts: float | None = Field(
        None, ge=0,
        description="IODs have non-trivial idle power (memory PHY refresh, PSP heartbeat). ~30-50W for datacenter IODs.",
    )
    pcie_aggregate_energy_pj_per_byte: float | None = Field(None, ge=0)
    inter_socket_aggregate_energy_pj_per_byte: float | None = Field(None, ge=0)

    model_config = {"extra": "forbid"}
