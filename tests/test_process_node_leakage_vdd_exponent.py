"""Contract tests for the optional leakage_vdd_exponent on ProcessNodeEntry.

Added for branes-ai/graphs#154: the KPU power model derived leakage from
leakage_w_per_mm2 at nominal Vdd only, overstating leakage by 2-3x for
low-power DVFS profiles that drop Vdd well below nominal. The node now carries
an optional exponent so consumers can scale leakage as (Vdd/Vnom)^exp.

The field is backward-compatible (default None = flat leakage). These tests pin
the schema default and the two FinFET nodes the Stillwater KPU SKUs target.
"""

import pytest

from embodied_schemas.loaders import load_process_nodes
from embodied_schemas.process_node import ProcessNodeEntry


@pytest.fixture(scope="module")
def nodes():
    return load_process_nodes()


def test_field_defaults_to_none(nodes):
    """Nodes that don't author the exponent keep legacy flat-leakage behavior.
    gf_28nm (bulk planar, pre-DVFS-heavy) does not set it."""
    node = nodes.get("gf_28nm")
    assert node is not None
    assert node.leakage_vdd_exponent is None


def test_field_rejects_negative(nodes):
    """Exponent is a magnitude; negative values are nonsensical (leakage must
    fall, not rise, with lower Vdd). Rebuild a real node with a bad exponent."""
    base = nodes["tsmc_n16"].model_dump()
    base["leakage_vdd_exponent"] = -1.0
    with pytest.raises(ValueError):
        ProcessNodeEntry.model_validate(base)


def test_tsmc_n16_carries_exponent(nodes):
    node = nodes.get("tsmc_n16")
    assert node is not None
    # Mixed-Vt FinFET node: blended leakage-power exponent ~4.
    assert node.leakage_vdd_exponent == pytest.approx(4.0)


def test_tsmc_n7_carries_exponent(nodes):
    node = nodes.get("tsmc_n7")
    assert node is not None
    # N7 is more Vdd-sensitive than N16 (stronger DIBL, ULVT corner).
    assert node.leakage_vdd_exponent == pytest.approx(4.5)
    assert node.leakage_vdd_exponent > nodes["tsmc_n16"].leakage_vdd_exponent
