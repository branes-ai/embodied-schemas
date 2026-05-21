"""Tests for the first DSP SKU YAML: cadence_tensilica_vision_q8.

Catalog gain: 12 KPU + 2 GPU + 1 CPU + 3 NPU + 1 CGRA + 1 DPU +
1 TPU + 1 DSP = 22 ComputeProducts. First SKU to exercise:
  - ``BlockKind.DSP`` and the full ``DSPBlock`` schema (landed in #43)
  - ``DSPFabricKind.VECTOR_SIMD``
  - ``DSPDeploymentKind.STANDALONE_IP`` (first IP-core SKU)
  - ``external_dram_bandwidth_kind: 'typical'`` discriminator
  - Brand new ``cadence/`` vendor directory

Same shape as ``test_compute_product_v7_tpu_v4_yaml.py``.
"""

import pytest

from embodied_schemas import (
    ComputeProduct,
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
def cadence_q8(all_products) -> ComputeProduct:
    cp = all_products.get("cadence_tensilica_vision_q8")
    if cp is None:
        pytest.fail("cadence_tensilica_vision_q8 missing from catalog")
    return cp


@pytest.fixture(scope="module")
def cadence_q8_compute_die(cadence_q8):
    die = next((d for d in cadence_q8.dies if d.die_role.value == "compute"), None)
    if die is None:
        pytest.fail("Cadence Vision Q8 has no compute die")
    return die


@pytest.fixture(scope="module")
def cadence_q8_block(cadence_q8_compute_die) -> DSPBlock:
    block = next(
        (b for b in cadence_q8_compute_die.blocks if isinstance(b, DSPBlock)),
        None,
    )
    if block is None:
        pytest.fail("Cadence Vision Q8 compute die has no DSPBlock")
    return block


# ---------------------------------------------------------------------------
# Catalog: Cadence Vision Q8 lives in new cadence/ vendor directory
# ---------------------------------------------------------------------------

def test_catalog_includes_cadence_vision_q8(all_products):
    cadence_skus = sorted(s for s, cp in all_products.items() if cp.vendor == "cadence")
    assert cadence_skus == ["cadence_tensilica_vision_q8"]


def test_catalog_has_22_total_products(all_products):
    """At least 12 KPU + 2 GPU + 1 CPU + 3 NPU + 1 CGRA + 1 DPU + 1 TPU
    + 1 DSP = 22 at this SKU's PR baseline. Subset semantics so further
    DSP follow-ups (graphs#223) don't regress this test."""
    counts_by_kind: dict[str, int] = {}
    for cp in all_products.values():
        block = cp.dies[0].blocks[0]
        kind = block.kind.value if hasattr(block.kind, "value") else str(block.kind)
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
    assert counts_by_kind.get("kpu", 0) >= 12
    assert counts_by_kind.get("gpu", 0) >= 2
    assert counts_by_kind.get("cpu", 0) >= 1
    assert counts_by_kind.get("npu", 0) >= 3
    assert counts_by_kind.get("cgra", 0) >= 1
    assert counts_by_kind.get("dpu", 0) >= 1
    assert counts_by_kind.get("tpu", 0) >= 1
    assert counts_by_kind.get("dsp", 0) >= 1


def test_other_vendors_unaffected_by_cadence_addition(all_products):
    """Additive guarantee: adding Cadence Vision Q8 must not perturb
    other vendor directories."""
    counts_by_vendor: dict[str, int] = {}
    for cp in all_products.values():
        counts_by_vendor[cp.vendor] = counts_by_vendor.get(cp.vendor, 0) + 1
    assert counts_by_vendor.get("stillwater") == 12
    assert counts_by_vendor.get("nvidia") == 2
    assert counts_by_vendor.get("intel") == 4  # i7-12700k + 3 Xeons: 8490H + 8592+ + 6980P (sprint #68 PR 1-3)
    assert counts_by_vendor.get("hailo") == 2
    assert counts_by_vendor.get("stanford") == 1
    assert counts_by_vendor.get("xilinx") == 1
    assert counts_by_vendor.get("google") == 2
    # New cadence/ vendor directory
    assert counts_by_vendor.get("cadence") == 1


# ---------------------------------------------------------------------------
# Identity / packaging
# ---------------------------------------------------------------------------

def test_cadence_q8_identity(cadence_q8):
    assert cadence_q8.id == "cadence_tensilica_vision_q8"
    assert cadence_q8.name == "Cadence Tensilica Vision Q8"
    assert cadence_q8.vendor == "cadence"
    assert cadence_q8.kind.value == "chip"
    assert cadence_q8.lifecycle == LifecycleStatus.PRODUCTION


def test_cadence_q8_packaging(cadence_q8):
    """IP core uses MONOLITHIC packaging with ``package_type: ip_core``
    -- distinguishing it from shipping silicon products."""
    assert cadence_q8.packaging.kind == PackagingKind.MONOLITHIC
    assert cadence_q8.packaging.num_dies == 1
    assert cadence_q8.packaging.package_type == "ip_core"


def test_cadence_q8_process_node(cadence_q8_compute_die):
    """16nm TSMC is the typical integration node for this generation."""
    assert cadence_q8_compute_die.process_node_id == "tsmc_n16"


# ---------------------------------------------------------------------------
# DSPBlock content
# ---------------------------------------------------------------------------

def test_cadence_q8_block_is_dsp(cadence_q8_block):
    assert isinstance(cadence_q8_block, DSPBlock)
    assert cadence_q8_block.kind == "dsp"


def test_cadence_q8_deployment_kind(cadence_q8_block):
    """Cadence Vision Q8 is a pure IP core; deployment_kind must be
    STANDALONE_IP. This is the first IP-core SKU in the catalog."""
    assert cadence_q8_block.deployment_kind == DSPDeploymentKind.STANDALONE_IP


def test_cadence_q8_single_simd_fabric(cadence_q8_block):
    """Vision Q8 is single-fabric (atypical for DSPs; most SoC DSPs
    have 2 fabrics). The schema accommodates 1+ via min_length=1."""
    assert len(cadence_q8_block.compute_fabrics) == 1
    fabric = cadence_q8_block.compute_fabrics[0]
    assert fabric.fabric_kind == DSPFabricKind.VECTOR_SIMD
    assert fabric.num_units == 32


def test_cadence_q8_native_precisions(cadence_q8_block):
    """Vision-optimized: INT8 / INT16 native; FP32 / FP16 also."""
    fabric = cadence_q8_block.compute_fabrics[0]
    assert "int8" in fabric.ops_per_unit_per_clock
    assert "int16" in fabric.ops_per_unit_per_clock
    assert "fp32" in fabric.ops_per_unit_per_clock
    assert "fp16" in fabric.ops_per_unit_per_clock
    # INT8 and INT16 both at 119 ops/cycle/unit (1024-bit SIMD)
    assert fabric.ops_per_unit_per_clock["int8"] == 119
    assert fabric.ops_per_unit_per_clock["int16"] == 119


def test_cadence_q8_int8_peak_matches_marketed(cadence_q8_block):
    """32 units * 119 ops/cycle * 1.0 GHz = 3.81 TOPS (Cadence claims 3.8)."""
    int8_peak = cadence_q8_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["int8"]
    assert int8_peak == pytest.approx(3.8e12, rel=0.01)


def test_cadence_q8_fp32_peak_matches_marketed(cadence_q8_block):
    """32 units * 4 ops/cycle * 1.0 GHz = 128 GFLOPS (Cadence claims 129)."""
    fp32_peak = cadence_q8_block.theoretical_performance \
        .peak_ops_per_sec_by_precision["fp32"]
    assert fp32_peak == pytest.approx(129e9, rel=0.01)


# ---------------------------------------------------------------------------
# Memory subsystem: typical-integration discriminator
# ---------------------------------------------------------------------------

def test_cadence_q8_memory_typical_integration(cadence_q8_block):
    """IP cores must use ``external_dram_bandwidth_kind: 'typical'``.
    Cadence Vision Q8 is the **first SKU** to exercise this load-bearing
    discriminator that distinguishes IP-core estimates from SoC measured
    bandwidth."""
    mem = cadence_q8_block.memory
    assert mem.has_external_dram is True
    assert mem.external_dram_bandwidth_kind == "typical"
    assert mem.external_dram_type == MemoryType.LPDDR4
    assert mem.external_dram_bandwidth_gbps == 40.0


def test_cadence_q8_l1_l2_consistent(cadence_q8_block):
    """L1: 32 KiB per unit; L2: 1 MiB shared with bandwidth set."""
    mem = cadence_q8_block.memory
    assert mem.l1_size_bytes_per_unit == 32 * 1024
    assert mem.l2_size_bytes_total == 1 * 1024 * 1024
    assert mem.l2_bandwidth_gbps is not None
    assert mem.l2_bandwidth_gbps > 0


# ---------------------------------------------------------------------------
# Single thermal profile (multi-profile validated in follow-up SKUs)
# ---------------------------------------------------------------------------

def test_cadence_q8_single_thermal_profile(cadence_q8_block):
    """IP core has 1 thermal profile. Multi-profile DVFS exercised
    by automotive follow-ups (SA8775P 3 profiles, TDA4VM 2 profiles)."""
    assert len(cadence_q8_block.thermal_profiles) == 1
    assert cadence_q8_block.default_thermal_profile_name == "1W"
    profile = cadence_q8_block.thermal_profiles[0]
    assert profile.name == "1W"
    assert profile.tdp_watts == 1.0
    assert profile.cooling_solution_id == "passive_fanless"
    assert profile.dvfs_enabled is False


def test_cadence_q8_default_precision(cadence_q8_block):
    """Vision Q8 is vision-optimized; INT8 is the canonical workload."""
    assert cadence_q8_block.default_precision == "int8"


def test_cadence_q8_not_vliw(cadence_q8_block):
    """Vision Q8 is primarily SIMD-only (not VLIW); vliw_issue_width=None."""
    assert cadence_q8_block.vliw_issue_width is None


# ---------------------------------------------------------------------------
# Chip-level power
# ---------------------------------------------------------------------------

def test_cadence_q8_chip_power_envelope(cadence_q8):
    """1W TDP @ 1.0 GHz sustained."""
    assert cadence_q8.power.tdp_watts == 1.0
    assert cadence_q8.power.default_thermal_profile == "1W"
    assert cadence_q8.power.max_power_watts >= 1.0
