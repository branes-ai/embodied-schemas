"""Tests for the second DSP SKU YAML: synopsys_arc_ev7x.

Catalog gain: brings DSP count from 1 to 2. First SKU to exercise:
  - **Multi-fabric DSPBlock**: 2 ``DSPComputeFabric`` entries
    (VPU vector + DNN tensor accelerator) -- the canonical multi-
    fabric shape (Cadence Vision Q8 was atypical single-fabric)
  - ``DSPFabricKind.TENSOR_MATRIX`` (DNN accelerator path)
  - Brand new ``synopsys/`` vendor directory

Same shape as ``test_compute_product_v9_cadence_vision_q8_yaml.py``.
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
def ev7x(all_products) -> ComputeProduct:
    cp = all_products.get("synopsys_arc_ev7x")
    if cp is None:
        pytest.fail("synopsys_arc_ev7x missing from catalog")
    return cp


@pytest.fixture(scope="module")
def ev7x_block(ev7x) -> DSPBlock:
    block = ev7x.dies[0].blocks[0]
    assert isinstance(block, DSPBlock)
    return block


# ---------------------------------------------------------------------------
# Catalog: Synopsys EV7x lives in new synopsys/ vendor directory
# ---------------------------------------------------------------------------

def test_catalog_includes_synopsys_arc_ev7x(all_products):
    synopsys_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "synopsys")
    assert synopsys_skus == ["synopsys_arc_ev7x"]


def test_catalog_dsp_count_is_now_two(all_products):
    """Cadence Vision Q8 + Synopsys ARC EV7x = 2 DSP SKUs."""
    dsp_count = sum(
        1 for cp in all_products.values()
        if cp.dies[0].blocks[0].kind == "dsp"
    )
    assert dsp_count == 2


# ---------------------------------------------------------------------------
# Identity / packaging
# ---------------------------------------------------------------------------

def test_ev7x_identity(ev7x):
    assert ev7x.id == "synopsys_arc_ev7x"
    assert ev7x.name == "Synopsys ARC EV7x (4-core)"
    assert ev7x.vendor == "synopsys"
    assert ev7x.lifecycle == LifecycleStatus.PRODUCTION


def test_ev7x_packaging(ev7x):
    assert ev7x.packaging.kind == PackagingKind.MONOLITHIC
    assert ev7x.packaging.package_type == "ip_core"


def test_ev7x_process_node(ev7x):
    assert ev7x.dies[0].process_node_id == "tsmc_n16"


# ---------------------------------------------------------------------------
# DSPBlock content: multi-fabric (the first one in the catalog)
# ---------------------------------------------------------------------------

def test_ev7x_deployment_kind_standalone_ip(ev7x_block):
    """IP cores must use STANDALONE_IP (verified by v9 validator)."""
    assert ev7x_block.deployment_kind == DSPDeploymentKind.STANDALONE_IP


def test_ev7x_has_two_fabrics(ev7x_block):
    """**First multi-fabric DSP SKU**. The DSPBlock.compute_fabrics
    list with min_length=1 now exercises length 2 end-to-end."""
    assert len(ev7x_block.compute_fabrics) == 2


def test_ev7x_first_fabric_is_vpu_vector(ev7x_block):
    """First fabric: 4 VPU cores, 512-bit SIMD vector. Carries
    FP32 / INT16 / INT32."""
    vpu = ev7x_block.compute_fabrics[0]
    assert vpu.fabric_kind == DSPFabricKind.VECTOR_SIMD
    assert vpu.num_units == 4
    assert "fp32" in vpu.ops_per_unit_per_clock
    assert "int16" in vpu.ops_per_unit_per_clock
    assert "int32" in vpu.ops_per_unit_per_clock
    # No INT8 on the VPU path (DNN accelerator handles INT8)
    assert "int8" not in vpu.ops_per_unit_per_clock


def test_ev7x_second_fabric_is_dnn_tensor(ev7x_block):
    """Second fabric: 128-unit DNN accelerator. INT8 only."""
    dnn = ev7x_block.compute_fabrics[1]
    assert dnn.fabric_kind == DSPFabricKind.TENSOR_MATRIX
    assert dnn.num_units == 128
    assert "int8" in dnn.ops_per_unit_per_clock
    # FP32 lives on the VPU path; not on the DNN
    assert "fp32" not in dnn.ops_per_unit_per_clock


def test_ev7x_int8_peak_matches_marketed(ev7x_block):
    """128 units * 273 ops/cycle * 1.0 GHz = 34.94 TOPS (~35 marketed)."""
    int8_peak = ev7x_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"]
    assert int8_peak == pytest.approx(35e12, rel=0.01)


def test_ev7x_fp32_peak_matches_marketed(ev7x_block):
    """4 VPUs * 2.2 GFLOPS = 8.8 GFLOPS marketed."""
    fp32_peak = ev7x_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["fp32"]
    assert fp32_peak == pytest.approx(8.8e9, rel=0.01)


# ---------------------------------------------------------------------------
# Memory subsystem: typical-integration + chip_attached (v11+v12)
# ---------------------------------------------------------------------------

def test_ev7x_memory_typical_chip_attached(ev7x_block):
    """IP cores must use external_dram_bandwidth_kind: typical AND
    dram_attachment: chip_attached (IP cores assume the SoC integrator
    puts DDR on the host SoC die)."""
    mem = ev7x_block.memory
    assert mem.has_external_dram is True
    assert mem.external_dram_bandwidth_kind == "typical"
    assert mem.external_dram_type == MemoryType.LPDDR4
    assert mem.external_dram_bandwidth_gbps == 60.0
    assert mem.dram_attachment == DramAttachment.CHIP_ATTACHED


def test_ev7x_l1_l2_consistent(ev7x_block):
    mem = ev7x_block.memory
    assert mem.l1_size_bytes_per_unit == 32 * 1024
    assert mem.l2_size_bytes_total == 4 * 1024 * 1024
    assert mem.l2_bandwidth_gbps is not None and mem.l2_bandwidth_gbps > 0


# ---------------------------------------------------------------------------
# Single thermal profile (multi-profile exercised by automotive SoCs later)
# ---------------------------------------------------------------------------

def test_ev7x_single_thermal_profile(ev7x_block):
    """IP core has 1 thermal profile."""
    assert len(ev7x_block.thermal_profiles) == 1
    assert ev7x_block.default_thermal_profile_name == "5W"
    profile = ev7x_block.thermal_profiles[0]
    assert profile.name == "5W"
    assert profile.tdp_watts == 5.0
    assert profile.dvfs_enabled is False


def test_ev7x_default_precision(ev7x_block):
    assert ev7x_block.default_precision == "int8"


def test_ev7x_not_vliw(ev7x_block):
    """ARCv2 is RISC; vliw_issue_width=None."""
    assert ev7x_block.vliw_issue_width is None


# ---------------------------------------------------------------------------
# Chip-level power
# ---------------------------------------------------------------------------

def test_ev7x_chip_power_envelope(ev7x):
    assert ev7x.power.tdp_watts == 5.0
    assert ev7x.power.default_thermal_profile == "5W"
