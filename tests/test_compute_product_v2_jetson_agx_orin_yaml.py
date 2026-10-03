"""Tests for the first GPU SKU YAML: nvidia_jetson_agx_orin_64gb.

Validates that the catalog YAML at
``data/compute_products/nvidia/jetson_agx_orin_64gb.yaml`` loads as a
fully-formed ComputeProduct + GPUBlock with the field values reported
in the NVIDIA Jetson AGX Orin Series Technical Brief and the
graphs-side mapper that this SKU was migrated from.

Pinning specific field values here doubles as a contract test: any
future YAML edit that drifts off the public spec sheet will break this
test and force a deliberate update.
"""

import pytest

from embodied_schemas import (
    BlockKind,
    ComputeProduct,
    GPUBlock,
    GPUFabricKind,
    GPUL1Kind,
    GPUL2Topology,
    GPUNoCTopology,
    LifecycleStatus,
    PackagingKind,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def agx_orin(all_products) -> ComputeProduct:
    cp = all_products.get("nvidia_jetson_agx_orin_64gb")
    if cp is None:
        pytest.fail("nvidia_jetson_agx_orin_64gb missing from catalog")
    return cp


# ---------------------------------------------------------------------------
# Catalog-level: GPU SKU sits alongside the 12 v1 KPU SKUs
# ---------------------------------------------------------------------------

def test_catalog_now_includes_gpu_sku(all_products):
    """v1 had 12 KPU products; v2 first data PR adds 1 GPU. Total >= 13."""
    assert len(all_products) >= 13
    nvidia_skus = [
        sku for sku, cp in all_products.items() if cp.vendor == "nvidia"
    ]
    assert "nvidia_jetson_agx_orin_64gb" in nvidia_skus


def test_kpu_skus_still_load_after_gpu_addition(all_products):
    """Additive guarantee: adding nvidia/ vendor directory must not
    perturb stillwater/ KPU loading."""
    stillwater_skus = [
        sku for sku, cp in all_products.items() if cp.vendor == "stillwater"
    ]
    assert len(stillwater_skus) >= 12
    for sku, cp in all_products.items():
        if cp.vendor == "stillwater":
            assert cp.dies[0].blocks[0].kind == BlockKind.KPU


# ---------------------------------------------------------------------------
# AGX Orin identity / packaging / lifecycle
# ---------------------------------------------------------------------------

def test_agx_orin_identity(agx_orin):
    assert agx_orin.id == "nvidia_jetson_agx_orin_64gb"
    assert agx_orin.vendor == "nvidia"
    assert agx_orin.packaging.kind == PackagingKind.MONOLITHIC
    assert agx_orin.packaging.num_dies == 1
    assert agx_orin.lifecycle == LifecycleStatus.PRODUCTION


def test_agx_orin_die_references_samsung_8lpp(agx_orin):
    """AGX Orin's GA10B die ships on Samsung 8LPP. The catalog must
    reference the existing process node entry by id; loader will fail
    if the id doesn't resolve."""
    die = agx_orin.dies[0]
    assert die.process_node_id == "samsung_8lpp"
    assert die.die_role.value == "compute"
    assert die.die_size_mm2 == pytest.approx(455.0, rel=0.05)
    assert die.transistors_billion == pytest.approx(17.0, rel=0.1)


def test_agx_orin_silicon_bin_reconciles_with_die_total(agx_orin):
    """Silicon-bin block transistor sum should be within ~10% of the
    die-level transistor count. Catches drift between per-block
    estimates and the chip total."""
    die = agx_orin.dies[0]
    sb_total_mtx = sum(
        b.transistor_source.mtx
        for b in die.silicon_bin.blocks
        if b.transistor_source.mtx is not None
    )
    declared_mtx = die.transistors_billion * 1000.0
    rel_err = abs(sb_total_mtx - declared_mtx) / declared_mtx
    assert rel_err < 0.10, (
        f"silicon_bin sum {sb_total_mtx:.0f} Mtx differs from declared "
        f"{declared_mtx:.0f} Mtx by {rel_err*100:.1f}% (>10% threshold)"
    )


# ---------------------------------------------------------------------------
# GPUBlock structural fields (the migration source of truth)
# ---------------------------------------------------------------------------

def test_agx_orin_block_is_gpu(agx_orin):
    block = agx_orin.dies[0].blocks[0]
    assert isinstance(block, GPUBlock)
    assert block.kind == "gpu"


def test_agx_orin_sm_hierarchy(agx_orin):
    """16 Ampere SMs * 128 CUDA cores/SM = 2048 CUDA cores;
    4 Tensor cores/SM * 16 = 64 Tensor cores."""
    block = agx_orin.dies[0].blocks[0]
    assert block.num_sms == 16
    assert block.cuda_cores_per_sm == 128
    assert block.tensor_cores_per_sm == 4
    assert block.warp_size == 32
    assert block.num_sms * block.cuda_cores_per_sm == 2048
    assert block.num_sms * block.tensor_cores_per_sm == 64


def test_agx_orin_compute_fabrics(agx_orin):
    """Two compute fabrics: CUDA cores + Tensor cores."""
    block = agx_orin.dies[0].blocks[0]
    assert len(block.compute_fabrics) == 2
    kinds = {f.fabric_kind for f in block.compute_fabrics}
    assert {GPUFabricKind.CUDA_CORE, GPUFabricKind.TENSOR_CORE} == kinds

    cuda = next(f for f in block.compute_fabrics
                if f.fabric_kind == GPUFabricKind.CUDA_CORE)
    assert cuda.units_per_sm == 128
    assert cuda.ops_per_unit_per_clock["fp32"] == 2  # FMA

    tensor = next(f for f in block.compute_fabrics
                  if f.fabric_kind == GPUFabricKind.TENSOR_CORE)
    assert tensor.units_per_sm == 4
    assert tensor.ops_per_unit_per_clock["fp16"] == 256
    assert tensor.ops_per_unit_per_clock["int8"] == 512


def test_agx_orin_memory_subsystem(agx_orin):
    """LPDDR5 / 64 GB / 256-bit / 204.8 GB/s; per-SM L1 128 KiB unified;
    4 MiB shared L2 (LLC); no L3."""
    mem = agx_orin.dies[0].blocks[0].memory
    assert mem.memory_type == MemoryType.LPDDR5
    assert mem.memory_size_gb == 64.0
    assert mem.memory_bus_bits == 256
    assert mem.memory_bandwidth_gbps == pytest.approx(204.8)
    assert mem.memory_controllers == 8
    assert mem.l1_kib_per_sm == 128
    assert mem.l1_kind == GPUL1Kind.UNIFIED
    assert mem.l2_total_kib == 4 * 1024
    assert mem.l2_topology == GPUL2Topology.SHARED_LLC
    assert mem.l3_present is False
    assert mem.l3_total_kib == 0
    assert mem.coherence_protocol == "none"


def test_agx_orin_noc_is_crossbar(agx_orin):
    """SM-to-L2 crossbar with 16 SMs as ports."""
    noc = agx_orin.dies[0].blocks[0].noc
    assert noc.topology == GPUNoCTopology.CROSSBAR
    assert noc.unit_count == 16   # v10 rename: was controller_count
    assert noc.flit_size_bytes == 32
    assert noc.bisection_bandwidth_gbps == pytest.approx(2048.0)


def test_agx_orin_scheduler_attrs(agx_orin):
    block = agx_orin.dies[0].blocks[0]
    assert block.min_occupancy == pytest.approx(0.3)
    assert block.max_concurrent_kernels == 8
    assert block.wave_quantization == 4


def test_agx_orin_thermal_profiles_cover_nvpmodel(agx_orin):
    """All four NVIDIA nvpmodel profiles (15W / 30W / 50W / MAXN)
    present and the chip default is 30W."""
    power = agx_orin.power
    assert power.tdp_watts == 30.0
    assert power.default_thermal_profile == "30W"
    profile_names = {p.name for p in power.thermal_profiles}
    assert profile_names == {"15W", "30W", "50W", "MAXN"}


def test_agx_orin_thermal_profile_efficiencies_in_unit_range(agx_orin):
    """Per-precision efficiency factors must lie in [0, 1] -- the
    Pydantic validator enforces this on construction; this test pins
    the YAML actually exercises the validator path."""
    for profile in agx_orin.power.thermal_profiles:
        if profile.efficiency_factor_by_precision is None:
            continue
        for precision, factor in profile.efficiency_factor_by_precision.items():
            assert 0.0 <= factor <= 1.0


def test_agx_orin_round_trips_through_serialize(agx_orin):
    """Serialize -> deserialize must round-trip a full ComputeProduct
    with a GPU die (catches Pydantic discriminator dispatch bugs)."""
    payload = agx_orin.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    assert isinstance(rebuilt.dies[0].blocks[0], GPUBlock)
    assert rebuilt.dies[0].blocks[0].num_sms == 16
    assert rebuilt.power.default_thermal_profile == "30W"
