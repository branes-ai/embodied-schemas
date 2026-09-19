"""The half-precision energy derivation and the catalog's use of it.

Every process node's ``<class>:fp16`` and ``<class>:bf16`` figures are
DERIVED from its ``<class>:fp32`` figure by ``embodied_schemas.fp_energy``
(Horowitz ISSCC 2014). These tests pin the derivation's arithmetic and hold
every catalog entry to it, so a hand-edited figure, or a new fp32 figure
without its derived companions, fails here.
"""

import pytest
from pydantic import ValidationError

from embodied_schemas.fp_energy import (
    DERIVED_FORMATS,
    derivation_source,
    derived_energy_pj,
    fma_pj_45nm,
    ratio_to_fp32,
)
from embodied_schemas.loaders import load_process_nodes
from embodied_schemas.process_node import ProcessNodeEntry


def test_fp16_is_horowitz_fma_ratio():
    # Fig. 1.1.9: FP16 mult 1.1 + add 0.4, FP32 mult 3.7 + add 0.9.
    assert fma_pj_45nm("fp16") == pytest.approx(1.5)
    assert fma_pj_45nm("fp32") == pytest.approx(4.6)
    assert ratio_to_fp32("fp16") == pytest.approx(1.5 / 4.6)


def test_bf16_width_fit_passes_through_both_measured_points():
    # The fits reproduce Horowitz's FP16 and FP32 rows exactly; BF16 is the
    # fit at m = 8: multiply 0.4086 + 0.005714 * 64, add -0.0231 + 0.03846 * 8.
    assert fma_pj_45nm("bf16") == pytest.approx(0.7743 + 0.2846, abs=1e-3)
    assert ratio_to_fp32("bf16") == pytest.approx(0.230, abs=5e-4)


def test_narrower_significand_costs_less():
    assert ratio_to_fp32("bf16") < ratio_to_fp32("fp16") < 1.0


def test_derived_figures_round_to_three_significant_figures():
    assert derived_energy_pj(2.0, "fp16") == 0.652
    assert derived_energy_pj(2.0, "bf16") == 0.46


def test_only_half_precision_is_derived():
    with pytest.raises(KeyError):
        derived_energy_pj(1.0, "int8")


NODES = load_process_nodes()


def _float_classes(node):
    return sorted(k.split(":")[0] for k in node.energy_per_op_pj if k.endswith(":fp32"))


@pytest.mark.parametrize("node_id", sorted(NODES))
def test_every_node_holds_to_the_derivation(node_id):
    node = NODES[node_id]
    for cc in _float_classes(node):
        fp32 = node.energy_per_op_pj[f"{cc}:fp32"]
        for fmt in DERIVED_FORMATS:
            key = f"{cc}:{fmt}"
            assert node.energy_per_op_pj.get(key) == derived_energy_pj(fp32, fmt), key
            assert node.energy_per_op_sources.get(key) == derivation_source(cc, fmt), key


@pytest.mark.parametrize("node_id", sorted(NODES))
def test_no_half_precision_figure_without_fp32(node_id):
    node = NODES[node_id]
    classes = set(_float_classes(node))
    for key in node.energy_per_op_pj:
        cc, _, fmt = key.partition(":")
        if fmt in DERIVED_FORMATS:
            assert cc in classes, f"{key} has no {cc}:fp32 to derive from"


def test_sources_must_name_an_energy_entry():
    node = NODES["samsung_8lpp"].model_dump()
    node["energy_per_op_sources"] = {"hp_logic:fp8": "made up"}
    with pytest.raises(ValidationError, match="hp_logic:fp8"):
        ProcessNodeEntry.model_validate(node)
