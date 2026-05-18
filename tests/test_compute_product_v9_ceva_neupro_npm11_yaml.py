"""Tests for the third DSP SKU YAML: ceva_neupro_m_npm11.

Catalog gain: brings DSP count from 2 to 3. Second multi-fabric DSP
SKU (after Synopsys EV7x).

Notable difference from EV7x: this YAML places the TENSOR fabric
FIRST (vs EV7x where VPU vector is first by accident of mapper order).
The loader's pick-first convention then puts the canonical INT8
path (20 TOPS) on the chip-level compute_units / energy surfaces,
so the parity test for CEVA expects ZERO drifts on those fields
(unlike EV7x's 3 documented drifts).
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
    DramAttachment,
    DSPBlock,
    DSPDeploymentKind,
    DSPFabricKind,
    LifecycleStatus,
    PackagingKind,
)
from embodied_schemas.gpu import MemoryType
from embodied_schemas.loaders import load_compute_products


@pytest.fixture(scope="module")
def all_products() -> dict[str, ComputeProduct]:
    return load_compute_products()


@pytest.fixture(scope="module")
def npm11(all_products) -> ComputeProduct:
    cp = all_products.get("ceva_neupro_m_npm11")
    if cp is None:
        pytest.fail("ceva_neupro_m_npm11 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def npm11_block(npm11) -> DSPBlock:
    block = npm11.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: CEVA NPM11 lives in new ceva/ vendor directory
# ---------------------------------------------------------------------------

def test_catalog_includes_ceva_neupro_npm11(all_products):
    ceva_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "ceva")
    assert ceva_skus == ["ceva_neupro_m_npm11"]


def test_catalog_dsp_count_is_now_three(all_products):
    """Cadence + Synopsys + CEVA = 3 DSP SKUs."""
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count == 3


# ---------------------------------------------------------------------------
# Identity / packaging
# ---------------------------------------------------------------------------

def test_npm11_identity(npm11):
    assert npm11.id == "ceva_neupro_m_npm11"
    assert npm11.name == "CEVA NeuPro-M NPM11"
    assert npm11.vendor == "ceva"
    assert npm11.lifecycle == LifecycleStatus.PRODUCTION


def test_npm11_packaging(npm11):
    assert npm11.packaging.kind == PackagingKind.MONOLITHIC
    assert npm11.packaging.package_type == "ip_core"


def test_npm11_process_node(npm11):
    assert npm11.dies[0].process_node_id == "tsmc_n16"


# ---------------------------------------------------------------------------
# DSPBlock content: tensor + vector multi-fabric (tensor FIRST)
# ---------------------------------------------------------------------------

def test_npm11_deployment_kind_standalone_ip(npm11_block):
    assert npm11_block.deployment_kind == DSPDeploymentKind.STANDALONE_IP


def test_npm11_has_two_fabrics(npm11_block):
    """tensor + vector pairing canonical for NPM11."""
    assert len(npm11_block.compute_fabrics) == 2


def test_npm11_tensor_fabric_is_first(npm11_block):
    """**Fabric ordering matters**: tensor is fabric[0] so loader's
    pick-first convention picks tensor for chip-level surfaces.
    This is the canonical pattern; EV7x's VPU-first ordering causes
    documented drifts."""
    tensor = npm11_block.compute_fabrics[0]
    assert tensor.fabric_kind == DSPFabricKind.TENSOR_MATRIX
    assert tensor.num_units == 64
    assert "int8" in tensor.ops_per_unit_per_clock
    assert "int4" in tensor.ops_per_unit_per_clock
    assert "int16" in tensor.ops_per_unit_per_clock
    # Vector lives on fabric[1], not fabric[0]
    assert "fp16" not in tensor.ops_per_unit_per_clock


def test_npm11_vector_fabric_is_second(npm11_block):
    """Vector fabric handles FP16 only (NPM11 doesn't ship FP32)."""
    vector = npm11_block.compute_fabrics[1]
    assert vector.fabric_kind == DSPFabricKind.VECTOR_SIMD
    assert vector.num_units == 64
    assert "fp16" in vector.ops_per_unit_per_clock


def test_npm11_int8_peak_matches_marketed(npm11_block):
    """64 tensor units * 312 ops/cycle * 1.0 GHz = 19.97 TOPS (~20 marketed)."""
    int8_peak = npm11_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"]
    assert int8_peak == pytest.approx(20e12, rel=0.01)


def test_npm11_int4_peak_is_2x_int8(npm11_block):
    """INT4: 2x INT8 throughput (40 TOPS) -- NeuPro-M's INT4 capability."""
    int4_peak = npm11_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int4"]
    assert int4_peak == pytest.approx(40e12, rel=0.01)


def test_npm11_fp16_peak_matches_marketed(npm11_block):
    """64 vector units * 156 ops/cycle * 1.0 GHz = 9.98 TFLOPS (~10 marketed)."""
    fp16_peak = npm11_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["fp16"]
    assert fp16_peak == pytest.approx(10e12, rel=0.01)


# ---------------------------------------------------------------------------
# Memory subsystem
# ---------------------------------------------------------------------------

def test_npm11_memory_typical_chip_attached(npm11_block):
    mem = npm11_block.memory
    assert mem.has_external_dram is True
    assert mem.external_dram_bandwidth_kind == "typical"
    assert mem.external_dram_type == MemoryType.LPDDR5
    assert mem.external_dram_bandwidth_gbps == 50.0
    assert mem.dram_attachment == DramAttachment.CHIP_ATTACHED


def test_npm11_l1_l2_consistent(npm11_block):
    mem = npm11_block.memory
    assert mem.l1_size_bytes_per_unit == 64 * 1024
    assert mem.l2_size_bytes_total == 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Single thermal profile
# ---------------------------------------------------------------------------

def test_npm11_single_thermal_profile(npm11_block):
    assert len(npm11_block.thermal_profiles) == 1
    assert npm11_block.default_thermal_profile_name == "2W"
    profile = npm11_block.thermal_profiles[0]
    assert profile.name == "2W"
    assert profile.tdp_watts == 2.0
    assert profile.dvfs_enabled is False


def test_npm11_default_precision(npm11_block):
    assert npm11_block.default_precision == "int8"


# ---------------------------------------------------------------------------
# Chip-level power
# ---------------------------------------------------------------------------

def test_npm11_chip_power_envelope(npm11):
    assert npm11.power.tdp_watts == 2.0
    assert npm11.power.default_thermal_profile == "2W"
