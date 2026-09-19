"""Half-precision FMA energy, derived from a node's FP32 figure.

The process-node catalog authors one floating-point figure per library,
``<class>:fp32``. The half-precision figures ``<class>:fp16`` and
``<class>:bf16`` are DERIVED from it with this module, so that every node
applies the same ratio and cites the same source.

**Source.** M. Horowitz, "Computing's Energy Problem (and what we can do
about it)", ISSCC 2014, Fig. 1.1.9 (45 nm, 0.9 V),
doi:10.1109/ISSCC.2014.6757323:

    ==========  =======  ============
    format      add, pJ  multiply, pJ
    ==========  =======  ============
    FP16        0.4      1.1
    FP32        0.9      3.7
    ==========  =======  ============

An FMA is taken as one multiply plus one add, so FP16 costs 1.5 pJ against
4.6 pJ for FP32: a ratio of 1.5 / 4.6 = 0.326.

**BF16** is not in the table. It is derived from the same two rows by the
significand width m (with the hidden bit: FP32 24, FP16 11, BF16 8):

- the multiplier's partial-product array grows as m^2, so the multiply is
  fitted as a + b * m^2 through the FP16 and FP32 points;
- alignment, addition and normalization grow as m, so the add is fitted as
  c + d * m through the same two points.

At m = 8 the fit gives a 0.774 pJ multiply and a 0.285 pJ add, 1.059 pJ in
all: a ratio of 0.230 to FP32. BF16 lies outside the fitted range (8 < 11),
so this is an extrapolation. The constant terms also absorb the exponent
logic, which is 5 bits wide in FP16 but 8 in BF16 and FP32; that difference
is not modeled.

**What the ratio is applied to.** The catalog's per-op figure includes the
operand registers and clocking, not only the arithmetic. Applying an
arithmetic ratio to it assumes that overhead scales the same way. The
derived figures are THEORETICAL.
"""

from __future__ import annotations

HOROWITZ_SOURCE = (
    "M. Horowitz, 'Computing's Energy Problem (and what we can do about it)', "
    "ISSCC 2014, Fig. 1.1.9 (45 nm, 0.9 V); doi:10.1109/ISSCC.2014.6757323"
)

# Fig. 1.1.9, pJ per operation at 45 nm.
HOROWITZ_45NM_PJ = {
    "fp16": {"add": 0.4, "mult": 1.1},
    "fp32": {"add": 0.9, "mult": 3.7},
}

# Significand bits, including the hidden bit.
SIGNIFICAND_BITS = {"fp32": 24, "fp16": 11, "bf16": 8}

# Formats derived from the node's FP32 figure.
DERIVED_FORMATS = ("fp16", "bf16")

SIGNIFICANT_FIGURES = 3


def _fit(width_power: int) -> tuple[float, float]:
    """Coefficients (constant, slope) of ``e = constant + slope * m**width_power``."""
    op = "mult" if width_power == 2 else "add"
    m_lo, m_hi = SIGNIFICAND_BITS["fp16"], SIGNIFICAND_BITS["fp32"]
    e_lo, e_hi = HOROWITZ_45NM_PJ["fp16"][op], HOROWITZ_45NM_PJ["fp32"][op]
    slope = (e_hi - e_lo) / (m_hi ** width_power - m_lo ** width_power)
    return e_lo - slope * m_lo ** width_power, slope


def fma_pj_45nm(fmt: str) -> float:
    """FMA energy in pJ at 45 nm: Horowitz's figures, or the width fit for BF16."""
    if fmt in HOROWITZ_45NM_PJ:
        row = HOROWITZ_45NM_PJ[fmt]
        return row["mult"] + row["add"]
    if fmt not in SIGNIFICAND_BITS:
        raise KeyError(f"no significand width for {fmt!r}")
    m = SIGNIFICAND_BITS[fmt]
    total = 0.0
    for power in (2, 1):
        constant, slope = _fit(power)
        total += constant + slope * m ** power
    return total


def ratio_to_fp32(fmt: str) -> float:
    """``fmt`` FMA energy over FP32 FMA energy."""
    return fma_pj_45nm(fmt) / fma_pj_45nm("fp32")


def _round_sig(value: float, figures: int = SIGNIFICANT_FIGURES) -> float:
    return float(f"{value:.{figures}g}")


def derived_energy_pj(fp32_pj: float, fmt: str) -> float:
    """The catalog figure for ``fmt``, given the node's FP32 figure."""
    if fmt not in DERIVED_FORMATS:
        raise KeyError(f"{fmt!r} is not derived from fp32; derived: {DERIVED_FORMATS}")
    return _round_sig(fp32_pj * ratio_to_fp32(fmt))


def derivation_source(circuit_class: str, fmt: str) -> str:
    """The ``energy_per_op_sources`` text for a derived entry."""
    how = ("FP16/FP32 FMA energy" if fmt == "fp16"
           else "BF16/FP32 FMA energy, BF16 fitted by significand width")
    return (f"DERIVED: {circuit_class}:fp32 x {ratio_to_fp32(fmt):.3f}, {how} from "
            "Horowitz ISSCC 2014 Fig. 1.1.9 (doi:10.1109/ISSCC.2014.6757323); "
            "see embodied_schemas.fp_energy")
