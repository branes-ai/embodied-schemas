"""Tests for the second GPU SKU YAML: nvidia_jetson_agx_thor_128gb.

Catalog gain: 12 KPU + 2 GPU = 14 products. Validates the new YAML
loads as a fully-formed ``ComputeProduct`` + ``GPUBlock`` with values
matching the NVIDIA Jetson Thor product brief and the graphs-side
mapper. Also pins the new ``tsmc_n4p`` process node.

Same test shape as ``test_compute_product_v2_jetson_agx_orin_yaml.py``.
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
from embodied_schemas.loaders import load_compute_products, load_process_nodes


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def thor(all_products) -> ComputeProduct:
    cp = all_products.get("nvidia_jetson_agx_thor_128gb")
    if cp is None:
        pytest.fail("nvidia_jetson_agx_thor_128gb missing from catalog")
    return cp


@pytest.fixture(scope="module")
def thor_compute_die(thor):
    """Pick the compute die by role rather than positional index, so
    tests don't break if the YAML grows additional dies (chiplet
    products, IO die, HBM stacks)."""
    die = next((d for d in thor.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Thor has no compute die")
    return die


@pytest.fixture(scope="module")
def thor_gpu_block(thor_compute_die) -> GPUBlock:
    """Pick the GPUBlock by type rather than positional index, so tests
    don't break when CPU / DLA / PVA blocks are added to this die."""
    block = next(
        (b for b in thor_compute_die.blocks if isinstance(b, GPUBlock)),
        None,
    )
    if block is None:
        pytest.fail("Thor compute die has no GPUBlock")
    return block


# ---------------------------------------------------------------------------
# Process node: tsmc_n4p added by this PR
# ---------------------------------------------------------------------------

def test_tsmc_n4p_process_node_present():
    """N4P is the new process node introduced for Blackwell-Tegra."""
    nodes = load_process_nodes()
    assert "tsmc_n4p" in nodes
    n4p = nodes["tsmc_n4p"]
    assert n4p.foundry.value == "tsmc"
    assert n4p.node_nm == 4
    # SRAM density unchanged from N5 -- industry-wide scaling stall
    assert n4p.densities["sram_hd"].mtx_per_mm2 == 380.0


# ---------------------------------------------------------------------------
# Catalog: Thor sits alongside Orin AGX as the second GPU SKU
# ---------------------------------------------------------------------------

def test_catalog_now_includes_two_gpu_skus(all_products):
    """Tighten to ==2: a future GPU SKU addition is a deliberate
    catalog change and should fire this test as a reminder to update
    here (and any consumers that hardcoded the count)."""
    nvidia_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "nvidia")
    assert nvidia_skus == [
        "nvidia_jetson_agx_orin_64gb",
        "nvidia_jetson_agx_thor_128gb",
    ]


def test_kpu_skus_unaffected_by_thor_addition(all_products):
    """Tighten to ==12: protects against a KPU SKU silently disappearing
    from the catalog while we're touching the GPU vendor directory."""
    stillwater_skus = [s for s, cp in all_products.items() if cp.vendor == "stillwater"]
    assert len(stillwater_skus) == 12


# ---------------------------------------------------------------------------
# Thor identity / die / silicon_bin reconciliation
# ---------------------------------------------------------------------------

def test_thor_identity(thor):
    assert thor.id == "nvidia_jetson_agx_thor_128gb"
    assert thor.vendor == "nvidia"
    assert thor.packaging.kind == PackagingKind.MONOLITHIC
    assert thor.packaging.num_dies == 1
    assert thor.lifecycle == LifecycleStatus.PRODUCTION


def test_thor_die_references_tsmc_n4p(thor_compute_die):
    """Thor (T200 silicon) ships on TSMC N4P."""
    die = thor_compute_die
    assert die.process_node_id == "tsmc_n4p"
    assert die.die_role.value == "compute"
    assert die.die_size_mm2 == pytest.approx(600.0, rel=0.05)
    assert die.transistors_billion == pytest.approx(30.0, rel=0.1)


def test_thor_silicon_bin_reconciles_with_die_total(thor_compute_die):
    die = thor_compute_die
    sb_total_mtx = sum(
        b.transistor_source.mtx
        for b in die.silicon_bin.blocks
        if b.transistor_source.mtx is not None
    )
    declared_mtx = die.transistors_billion * 1000.0
    rel_err = abs(sb_total_mtx - declared_mtx) / declared_mtx
    assert rel_err < 0.10, (
        f"silicon_bin sum {sb_total_mtx:.0f} Mtx differs from declared "
        f"{declared_mtx:.0f} Mtx by {rel_err*100:.1f}% (>10%)"
    )


# ---------------------------------------------------------------------------
# GPUBlock structural fields
# ---------------------------------------------------------------------------

def test_thor_block_is_gpu(thor_gpu_block):
    assert isinstance(thor_gpu_block, GPUBlock)
    assert thor_gpu_block.kind == "gpu"


def test_thor_sm_hierarchy(thor_gpu_block):
    """64 Blackwell SMs * 128 CUDA cores = 8192 cores;
    4 Tensor cores/SM * 64 = 256 Tensor cores."""
    block = thor_gpu_block
    assert block.num_sms == 64
    assert block.cuda_cores_per_sm == 128
    assert block.tensor_cores_per_sm == 4
    assert block.num_sms * block.cuda_cores_per_sm == 8192
    assert block.num_sms * block.tensor_cores_per_sm == 256


def test_thor_compute_fabrics(thor_gpu_block):
    block = thor_gpu_block
    assert len(block.compute_fabrics) == 2
    cuda = next(f for f in block.compute_fabrics
                if f.fabric_kind == GPUFabricKind.CUDA_CORE)
    assert cuda.units_per_sm == 128
    assert cuda.energy_per_flop_fp32_pj == pytest.approx(1.3, rel=0.05)
    tensor = next(f for f in block.compute_fabrics
                  if f.fabric_kind == GPUFabricKind.TENSOR_CORE)
    assert tensor.units_per_sm == 4
    # Blackwell 5th-gen TC: 64 FP16 ops/clock per TC (different from
    # Ampere's 256 -- Blackwell rebalanced toward more SMs at lower
    # per-TC throughput).
    assert tensor.ops_per_unit_per_clock["fp16"] == 64
    assert tensor.ops_per_unit_per_clock["int8"] == 64


def test_thor_memory_subsystem(thor_gpu_block):
    """LPDDR5X / 128 GB / 256-bit / 273 GB/s; per-SM L1 256 KiB
    (doubled vs Orin's 128 KiB); 8 MiB shared L2 (LLC); no L3."""
    mem = thor_gpu_block.memory
    assert mem.memory_type == MemoryType.LPDDR5X
    assert mem.memory_size_gb == 128.0
    assert mem.memory_bandwidth_gbps == pytest.approx(273.0)
    assert mem.l1_kib_per_sm == 256
    assert mem.l1_kind == GPUL1Kind.UNIFIED
    assert mem.l2_total_kib == 8 * 1024
    assert mem.l2_topology == GPUL2Topology.SHARED_LLC
    assert mem.l3_present is False


def test_thor_noc_is_2d_mesh(thor_gpu_block):
    """Thor's 64 SMs need a mesh (crossbar quadratic at this width).
    AGX Orin (16 SMs) uses CROSSBAR; Thor switches to MESH_2D."""
    noc = thor_gpu_block.noc
    assert noc.topology == GPUNoCTopology.MESH_2D
    assert noc.unit_count == 64   # v10 rename: was controller_count
    assert noc.bisection_bandwidth_gbps == pytest.approx(4096.0)


def test_thor_thermal_profiles_cover_all_three_modes(thor):
    """Thor ships three nvpmodel profiles (no MAXN; the 100W mode is
    the unconstrained equivalent)."""
    power = thor.power
    assert power.tdp_watts == 60.0
    assert power.default_thermal_profile == "60W"
    profile_names = {p.name for p in power.thermal_profiles}
    assert profile_names == {"30W", "60W", "100W"}


def test_thor_round_trips_through_serialize(thor):
    payload = thor.model_dump(mode="json")
    rebuilt = ComputeProduct.model_validate(payload)
    rebuilt_block = next(
        b for d in rebuilt.dies
        if d.die_role.value == "compute"
        for b in d.blocks
        if isinstance(b, GPUBlock)
    )
    assert rebuilt_block.num_sms == 64
