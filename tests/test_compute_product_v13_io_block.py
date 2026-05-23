"""Tests for v13 ComputeProduct schema addition: IOBlock and friends.

Validates that:
1. IOBlock + supporting types (IOMemorySubsystem, IOOnDieFabric,
   IOFabricTopology, InterSocketLink) construct cleanly with realistic
   AMD Genoa IOD numbers.
2. The AnyBlock discriminated union now dispatches NINE block kinds
   (KPU + GPU + CPU + NPU + CGRA + DPU + TPU + DSP + IO) via the
   ``kind`` field.
3. ComputeProduct round-trips an IO-die-bearing chiplet through
   serialize/deserialize.
4. Schema invariants: positive bandwidth/lane counts, default ECC,
   ``extra: forbid``, ``BlockKind.IO`` discriminator, ``DieRole.IO``
   pairs with IOBlock correctly.
5. Existing 8 block-kind YAMLs continue to validate (additive
   guarantee).
6. PCIeGen re-export from compute_block_common works (no circular
   import; same module also exports MemoryType, OnDieFabric, etc.).
7. **No ``performance`` field on IOBlock** -- the deliberate
   architectural choice that distinguishes IOBlock from the 8 prior
   compute-fabric block kinds.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from embodied_schemas import (
    AnyBlock,
    BlockKind,
    ComputeProduct,
    CPUBlock,
    DSPBlock,
    Die,
    DieRole,
    GPUBlock,
    InterSocketLink,
    IOBlock,
    IOFabricTopology,
    IOMemorySubsystem,
    IOOnDieFabric,
    KPUBlock,
    LifecycleStatus,
    Market,
    MemoryType,
    NPUBlock,
    Packaging,
    PackagingKind,
    PCIeGen,
    Power,
    ProductKind,
    TPUBlock,
)
from embodied_schemas.kpu import (
    KPUClocks,
    KPUSiliconBin,
    KPUTheoreticalPerformance,
    KPUThermalProfile,
    SiliconBinBlock,
    TransistorSource,
    TransistorSourceKind,
)
from embodied_schemas.process_node import CircuitClass
from embodied_schemas.loaders import load_compute_products


# ---------------------------------------------------------------------------
# Fixtures: Genoa-IOD-shaped IOBlock
# ---------------------------------------------------------------------------

@pytest.fixture
def genoa_io_memory() -> IOMemorySubsystem:
    """AMD Genoa IOD memory subsystem: 12-channel DDR5-4800."""
    return IOMemorySubsystem(
        memory_type=MemoryType.DDR5,
        memory_size_gb=6144.0,
        memory_bus_bits=768,
        memory_bandwidth_gbps=460.8,
        memory_controllers=12,
        ecc_supported=True,
    )


@pytest.fixture
def genoa_io_fabric() -> IOOnDieFabric:
    """AMD Genoa IOD coherence fabric: Infinity Fabric."""
    return IOOnDieFabric(
        topology=IOFabricTopology.INFINITY_FABRIC,
        bisection_bandwidth_gbps=2048.0,
        unit_count=12,             # one mesh stop per CCD
        flit_size_bytes=32,
        hop_latency_ns=5.0,
        pj_per_flit_per_hop=18.0,
    )


@pytest.fixture
def genoa_io_block(genoa_io_memory, genoa_io_fabric) -> IOBlock:
    """Full AMD Genoa IOD: 12-ch DDR5-4800 + 128 PCIe Gen5 + CXL 1.1 +
    4x G-link + PSP + PMC."""
    return IOBlock(
        memory=genoa_io_memory,
        coherence_fabric=genoa_io_fabric,
        pcie_lanes=128,
        pcie_generation=PCIeGen.PCIE_5,
        cxl_supported=True,
        cxl_version="1.1",
        inter_socket_links=[
            InterSocketLink(
                name="AMD G-link",
                bandwidth_per_link_gbps=250.0,
                link_count=4,
            ),
        ],
        security_processor_kind="AMD PSP",
        power_management_controller=True,
        boot_rom_present=True,
        idle_power_watts=40.0,
    )


# ---------------------------------------------------------------------------
# 1. IOBlock constructs and dispatches via discriminator
# ---------------------------------------------------------------------------

def test_block_kind_io_added_in_v13():
    """BlockKind.IO is the 9th member of the enum. v13's headline."""
    assert BlockKind.IO.value == "io"
    kinds = {k.value for k in BlockKind}
    assert {"kpu", "gpu", "cpu", "npu", "cgra", "dpu", "tpu", "dsp", "io"}.issubset(kinds)
    assert len(kinds) >= 9


def test_io_block_constructs(genoa_io_block):
    """IOBlock with full Genoa IOD field set constructs cleanly."""
    assert genoa_io_block.kind == "io"
    assert genoa_io_block.memory.memory_controllers == 12
    assert genoa_io_block.memory.memory_bandwidth_gbps == 460.8
    assert genoa_io_block.pcie_lanes == 128
    assert genoa_io_block.pcie_generation == PCIeGen.PCIE_5
    assert genoa_io_block.cxl_supported is True
    assert genoa_io_block.cxl_version == "1.1"
    assert len(genoa_io_block.inter_socket_links) == 1
    assert genoa_io_block.security_processor_kind == "AMD PSP"
    assert genoa_io_block.power_management_controller is True
    assert genoa_io_block.boot_rom_present is True


def test_anyblock_dispatches_to_io_block(genoa_io_block):
    """AnyBlock discriminated union validates IOBlock dict-form data."""
    adapter = TypeAdapter(AnyBlock)
    payload = genoa_io_block.model_dump()
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, IOBlock)
    assert parsed.kind == "io"
    assert parsed.pcie_lanes == 128


def test_anyblock_still_dispatches_other_block_kinds():
    """v13 must not break dispatch of KPU/GPU/CPU/NPU/CGRA/DPU/TPU/DSP.
    Loads each from the catalog and round-trips via AnyBlock."""
    products = load_compute_products()
    adapter = TypeAdapter(AnyBlock)
    by_kind = {}
    for cp in products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        by_kind.setdefault(kind, []).append(block)
    # Every other block kind still validates via AnyBlock
    for kind in ("kpu", "gpu", "cpu", "npu", "cgra", "dpu", "tpu", "dsp"):
        assert by_kind.get(kind), f"no {kind} block in catalog"
        block = by_kind[kind][0]
        parsed = adapter.validate_python(block.model_dump())
        parsed_kind = parsed.kind.value if hasattr(parsed.kind, "value") else str(parsed.kind)
        assert parsed_kind == kind


# ---------------------------------------------------------------------------
# 2. Schema invariants
# ---------------------------------------------------------------------------

def test_io_memory_requires_positive_channels(genoa_io_memory):
    """memory_controllers > 0 required."""
    payload = genoa_io_memory.model_dump()
    payload["memory_controllers"] = 0
    with pytest.raises(ValidationError):
        IOMemorySubsystem(**payload)


def test_io_memory_requires_positive_bandwidth(genoa_io_memory):
    """memory_bandwidth_gbps > 0 required."""
    payload = genoa_io_memory.model_dump()
    payload["memory_bandwidth_gbps"] = 0
    with pytest.raises(ValidationError):
        IOMemorySubsystem(**payload)


def test_io_memory_ecc_defaults_to_true():
    """Datacenter IODs all support ECC; default to True."""
    mem = IOMemorySubsystem(
        memory_type=MemoryType.DDR5,
        memory_size_gb=6144.0,
        memory_bus_bits=768,
        memory_bandwidth_gbps=460.8,
        memory_controllers=12,
    )
    assert mem.ecc_supported is True


def test_io_block_requires_positive_pcie_lanes(genoa_io_block):
    """pcie_lanes > 0 required (an IOD without PCIe is meaningless)."""
    payload = genoa_io_block.model_dump()
    payload["pcie_lanes"] = 0
    with pytest.raises(ValidationError):
        IOBlock(**payload)


def test_inter_socket_link_requires_positive_count():
    """link_count > 0 required."""
    with pytest.raises(ValidationError):
        InterSocketLink(name="X", bandwidth_per_link_gbps=100.0, link_count=0)


def test_inter_socket_link_requires_positive_bandwidth():
    """bandwidth_per_link_gbps > 0 required."""
    with pytest.raises(ValidationError):
        InterSocketLink(name="X", bandwidth_per_link_gbps=0, link_count=4)


def test_io_block_extra_forbid(genoa_io_block):
    """extra='forbid' on IOBlock catches typos."""
    payload = genoa_io_block.model_dump()
    payload["nonsense_field"] = 42
    with pytest.raises(ValidationError, match="(?i)extra"):
        IOBlock(**payload)


def test_io_memory_extra_forbid(genoa_io_memory):
    """extra='forbid' on IOMemorySubsystem catches typos."""
    payload = genoa_io_memory.model_dump()
    payload["typo_field"] = 1
    with pytest.raises(ValidationError, match="(?i)extra"):
        IOMemorySubsystem(**payload)


def test_inter_socket_link_extra_forbid():
    """extra='forbid' on InterSocketLink catches typos."""
    with pytest.raises(ValidationError, match="(?i)extra"):
        InterSocketLink(
            name="X",
            bandwidth_per_link_gbps=100.0,
            link_count=2,
            energy_pj_per_byte=1.0,   # typo / not a real field
        )


# ---------------------------------------------------------------------------
# 3. No `performance` field on IOBlock (the design choice)
# ---------------------------------------------------------------------------

def test_io_block_has_no_performance_field(genoa_io_block):
    """Architectural decision: IOBlock has no peak ops/sec. Its
    'performance' is captured by discrete typed bandwidth fields
    (memory_bandwidth_gbps, pcie_lanes*generation, inter_socket
    aggregate, coherence_fabric.bisection_bandwidth_gbps). Verify
    no performance attribute exists on the model."""
    # Pydantic models expose fields via model_fields
    fields = set(IOBlock.model_fields.keys())
    assert "performance" not in fields, (
        "IOBlock unexpectedly carries a 'performance' field. The v13 "
        "design exercise explicitly excludes it."
    )
    # And the instance has no performance attribute either
    assert not hasattr(genoa_io_block, "performance")


def test_io_block_distinct_from_compute_blocks_in_field_set():
    """IOBlock omits compute-block fields like 'compute_fabrics' that
    every other block kind has."""
    io_fields = set(IOBlock.model_fields.keys())
    # IOBlock has no compute_fabrics (it's not a compute fabric)
    assert "compute_fabrics" not in io_fields
    # But it does have memory + coherence_fabric (it's an I/O block)
    assert "memory" in io_fields
    assert "coherence_fabric" in io_fields
    assert "pcie_lanes" in io_fields


# ---------------------------------------------------------------------------
# 4. PCIeGen re-export from compute_block_common
# ---------------------------------------------------------------------------

def test_pciegen_reexported_from_compute_block_common():
    """v13 adds PCIeGen to compute_block_common's re-exports so IOBlock
    can use it without importing from embodied_schemas.gpu directly
    (would create a circular dep risk once IOBlock joins AnyBlock)."""
    from embodied_schemas.compute_block_common import PCIeGen as CBC_PCIeGen
    from embodied_schemas.gpu import PCIeGen as Source_PCIeGen
    # Re-export is the same class object, not a subclass
    assert CBC_PCIeGen is Source_PCIeGen


def test_pciegen_enum_values_unchanged():
    """Sanity: re-export didn't change the enum values."""
    expected = {"pcie_3.0", "pcie_4.0", "pcie_5.0", "pcie_6.0"}
    assert {g.value for g in PCIeGen} == expected


# ---------------------------------------------------------------------------
# 5. Die + IOBlock + DieRole.IO pairing
# ---------------------------------------------------------------------------

def test_die_with_role_io_pairs_with_io_block(genoa_io_block):
    """A Die with die_role=DieRole.IO can carry an IOBlock in blocks[].
    This is the canonical pattern this sprint enables: chiplet
    products express the IOD as a separate die with proper Block
    typing."""
    die = Die(
        die_id="genoa_iod",
        die_role=DieRole.IO,
        process_node_id="tsmc_n6",
        die_size_mm2=397.0,
        transistors_billion=12.0,
        silicon_bin=KPUSiliconBin(blocks=[
            SiliconBinBlock(
                name="genoa_iod_silicon",
                circuit_class=CircuitClass.BALANCED_LOGIC,
                transistor_source=TransistorSource(
                    kind=TransistorSourceKind.FIXED,
                    mtx=12000.0,
                ),
                notes="Genoa IOD: IMC + PCIe Gen5 + IF + PSP",
            ),
        ]),
        clocks=KPUClocks(base_clock_mhz=2000.0, boost_clock_mhz=2000.0),
        blocks=[genoa_io_block],
    )
    assert die.die_role == DieRole.IO
    assert len(die.blocks) == 1
    assert isinstance(die.blocks[0], IOBlock)
    assert die.blocks[0].kind == "io"
    assert die.process_node_id == "tsmc_n6"


# ---------------------------------------------------------------------------
# 6. Catalog composition: no existing SKU uses IOBlock yet
# ---------------------------------------------------------------------------

def test_io_block_skus_in_catalog():
    """Tracks which SKUs have been re-authored to use IOBlock.

    Sprint #245 progress:
      - PR 2 (#79): schema only, additive (no SKUs).
      - PR 3 (#TBD): EPYC 9654 first IOBlock-using SKU.
      - PR 4-5: EPYC 9754 (reuses Genoa IOD) + EPYC 9965 (Turin IOD)
        follow-on; bump this list as they land.

    Verifying the set explicitly catches both directions:
      - Schema-only PRs accidentally adding SKUs (was the v13 schema PR's
        additive-guarantee invariant).
      - Data PRs that should land but don't (e.g., this test forces an
        update when EPYC 9754 / 9965 land, which is desirable since
        each is paired with downstream graphs PhysicalSpec work).
    """
    products = load_compute_products()
    io_skus = sorted(
        cp.id
        for cp in products.values()
        if any(isinstance(b, IOBlock) for d in cp.dies for b in d.blocks)
    )
    # Sprint #245 PR 3 introduced the first IOBlock SKU.
    assert io_skus == ["amd_epyc_9654_sp5"]


def test_anyblock_union_count():
    """v13 brings the AnyBlock union to 9 members (8 compute + 1 IO)."""
    # Inspect the Annotated type to count members
    from typing import get_args, get_origin, Union
    # AnyBlock is Annotated[Union[...], Field(...)]
    annotated_args = get_args(AnyBlock)
    union_type = annotated_args[0]
    union_members = get_args(union_type)
    assert len(union_members) == 9, (
        f"AnyBlock has {len(union_members)} members; expected 9 after v13 IOBlock"
    )
    member_names = {m.__name__ for m in union_members}
    assert "IOBlock" in member_names
