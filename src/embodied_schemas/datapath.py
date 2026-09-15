"""Datapath schemas: number formats, operators, functional units, PE datapaths.

A processing element's *datapath* is the set of functional units it carries
(an INT8xINT8->INT32 MAC, an LNS MAC, a BF16 FMA, a lerp or min-plus unit,
...). Ops per clock, area and energy of a PE follow from its datapath, so the
datapath is declared once and everything else is derived.

This module is architecture-neutral. It is first used by KPU ``pe_fabric``
tiles (``kpu.KPUTileSpec.datapath``, graphs#268 Phase B1), and is shaped so the
same types can describe custom datapaths in other compute blocks later.

Counting convention (graphs#268 D3): an *op* is one software-equivalent scalar
arithmetic operation, the same unit the workload op counts use. A MAC or FMA
is 2 ops, a lerp ``a + t*(b - a)`` is 3, a min-plus ``min(a, b + c)`` is 2,
and so on (``DEFAULT_OPS_PER_INVOCATION``). A unit can override the default
when its operator is counted differently.

Ops per clock, by ``"<op>:<format>"`` key:

- Units whose modes share a key run concurrently, so their ops add up.
- Different keys are alternative operating modes of the datapath (a
  multi-precision PE runs INT8 *or* INT4 in a given cycle, not both).

This matches the legacy ``ops_per_tile_per_clock`` semantics, where each
precision's value is the peak when running that precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import AfterValidator, BaseModel, Field, model_validator

from embodied_schemas.process_node import CircuitClass, DataConfidence

# ---------------------------------------------------------------------------
# Number formats
# ---------------------------------------------------------------------------


class NumberFormatFamily(str, Enum):
    INT = "int"  # two's-complement integer
    UINT = "uint"  # unsigned integer
    FLOAT = "float"  # IEEE-style / brain / tensor float
    LNS = "lns"  # logarithmic number system
    POSIT = "posit"  # posit (Posit Standard 2022)
    FIXED = "fixed"  # fixed point, <integer bits>.<fraction bits>


@dataclass(frozen=True)
class NumberFormatSpec:
    """Parsed metadata of a number-format name."""

    name: str
    family: NumberFormatFamily
    bits: int
    exponent_bits: int | None = None  # float: exponent field width
    es: int | None = None  # posit: exponent size
    integer_bits: int | None = None  # fixed: integer bits (incl. sign)
    fraction_bits: int | None = None  # fixed: fraction bits


# Named float formats: name -> (bits, exponent bits). These are the precision
# strings the catalog and the graphs ``Precision`` enum already use.
_NAMED_FLOATS: dict[str, tuple[int, int]] = {
    "fp64": (64, 11),
    "fp32": (32, 8),
    "tf32": (19, 8),
    "fp16": (16, 5),
    "bf16": (16, 8),
    "fp8": (8, 4),  # alias of fp8_e4m3 as used by the graphs Precision enum
    "fp8_e4m3": (8, 4),
    "fp8_e5m2": (8, 5),
    "fp4": (4, 2),
}
_INT_WIDTHS = {2, 4, 8, 16, 32, 64}
_LNS_WIDTHS = {8, 16, 32}
_POSIT_WIDTHS = {8, 16, 32, 64}

_INT_RE = re.compile(r"^(u?int)(\d+)$")
_LNS_RE = re.compile(r"^lns(\d+)$")
_POSIT_RE = re.compile(r"^posit(\d+)(?:_(\d))?$")
_FIXED_RE = re.compile(r"^fixed(\d+)\.(\d+)$")


def parse_number_format(name: str) -> NumberFormatSpec:
    """Parse a number-format name.

    Accepted names:

    - ``int<n>`` / ``uint<n>`` for n in {2, 4, 8, 16, 32, 64}
    - named floats: ``fp64 fp32 tf32 fp16 bf16 fp8 fp8_e4m3 fp8_e5m2 fp4``
    - ``lns<n>`` for n in {8, 16, 32}
    - ``posit<n>`` (es = 2, per the Posit Standard 2022) or
      ``posit<n>_<es>``, for n in {8, 16, 32, 64} and es in 0..4
    - ``fixed<i>.<f>``: i integer bits (incl. sign, >= 1), f fraction bits,
      i + f <= 64

    Raises ``ValueError`` for anything else.
    """
    if name in _NAMED_FLOATS:
        bits, exp = _NAMED_FLOATS[name]
        return NumberFormatSpec(name, NumberFormatFamily.FLOAT, bits, exponent_bits=exp)

    m = _INT_RE.match(name)
    if m:
        bits = int(m.group(2))
        if bits not in _INT_WIDTHS:
            raise ValueError(
                f"number format {name!r}: integer width must be one of {sorted(_INT_WIDTHS)}"
            )
        family = NumberFormatFamily.UINT if m.group(1) == "uint" else NumberFormatFamily.INT
        return NumberFormatSpec(name, family, bits)

    m = _LNS_RE.match(name)
    if m:
        bits = int(m.group(1))
        if bits not in _LNS_WIDTHS:
            raise ValueError(
                f"number format {name!r}: LNS width must be one of {sorted(_LNS_WIDTHS)}"
            )
        return NumberFormatSpec(name, NumberFormatFamily.LNS, bits)

    m = _POSIT_RE.match(name)
    if m:
        bits = int(m.group(1))
        es = int(m.group(2)) if m.group(2) is not None else 2
        if bits not in _POSIT_WIDTHS or not 0 <= es <= 4:
            raise ValueError(
                f"number format {name!r}: posit width must be one of "
                f"{sorted(_POSIT_WIDTHS)} and es in 0..4"
            )
        return NumberFormatSpec(name, NumberFormatFamily.POSIT, bits, es=es)

    m = _FIXED_RE.match(name)
    if m:
        ibits, fbits = int(m.group(1)), int(m.group(2))
        if ibits < 1 or ibits + fbits > 64:
            raise ValueError(
                f"number format {name!r}: need integer bits >= 1 and "
                f"integer + fraction bits <= 64"
            )
        return NumberFormatSpec(
            name,
            NumberFormatFamily.FIXED,
            ibits + fbits,
            integer_bits=ibits,
            fraction_bits=fbits,
        )

    raise ValueError(
        f"unknown number format {name!r}. Expected int<n>/uint<n>, "
        f"{'/'.join(_NAMED_FLOATS)}, lns<n>, posit<n>[_<es>] or fixed<i>.<f>"
    )


def _validate_number_format(name: str) -> str:
    parse_number_format(name)
    return name


NumberFormatName = Annotated[str, AfterValidator(_validate_number_format)]
"""A validated number-format name (see ``parse_number_format``)."""


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class OpKind(str, Enum):
    """Operator implemented by a functional unit."""

    MAC = "mac"  # a*b + acc
    FMA = "fma"  # fused a*b + c (floating point)
    ADD = "add"
    MUL = "mul"
    DIV = "div"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
    MIN_PLUS = "min_plus"  # min(a, b + c): SGM / wavefront / shortest path
    CMP_SELECT = "cmp_select"  # compare-and-select (max/min, argmax)
    ABS_DIFF = "abs_diff"  # |a - b| (SAD)
    LERP = "lerp"  # a + t*(b - a)
    POPCOUNT_XOR = "popcount_xor"  # Hamming distance on a word
    CORDIC = "cordic"  # CORDIC rotation (sin/cos/atan2/...)
    EXP_LUT = "exp_lut"  # table-based exp / activation


DEFAULT_OPS_PER_INVOCATION: dict[OpKind, float] = {
    OpKind.MAC: 2,
    OpKind.FMA: 2,
    OpKind.ADD: 1,
    OpKind.MUL: 1,
    OpKind.DIV: 1,
    OpKind.SQRT: 1,
    OpKind.RSQRT: 1,
    OpKind.MIN_PLUS: 2,
    OpKind.CMP_SELECT: 1,
    OpKind.ABS_DIFF: 2,
    OpKind.LERP: 3,
    OpKind.POPCOUNT_XOR: 2,
    OpKind.CORDIC: 1,
    OpKind.EXP_LUT: 1,
}
"""Software-equivalent scalar ops per invocation (graphs#268 D3)."""

# Ops whose ``<op>:<format>`` throughput maps onto a legacy precision key of
# ``ops_per_tile_per_clock`` (the catalog has always counted MAC / FMA).
LEGACY_PRECISION_OPS = frozenset({OpKind.MAC, OpKind.FMA})


def op_key(op: OpKind, operand_format: str) -> str:
    """The ``"<op>:<format>"`` key used for datapath throughput."""
    return f"{op.value}:{operand_format}"


# ---------------------------------------------------------------------------
# Energy references
# ---------------------------------------------------------------------------


def _validate_energy_anchor(anchor: str) -> str:
    try:
        cls, fmt = anchor.split(":", 1)
    except ValueError:
        raise ValueError(
            f"energy anchor {anchor!r} must be '<circuit_class>:<format>', "
            f"e.g. 'balanced_logic:int8'"
        ) from None
    try:
        CircuitClass(cls)
    except ValueError:
        raise ValueError(f"energy anchor {anchor!r}: unknown circuit class {cls!r}") from None
    parse_number_format(fmt)
    return anchor


EnergyAnchor = Annotated[str, AfterValidator(_validate_energy_anchor)]
"""``"<circuit_class>:<format>"``, a key into ``ProcessNodeEntry.energy_per_op_pj``."""


class RelativeEnergy(BaseModel):
    """Energy per invocation as a ratio to a ProcessNode anchor op.

    Scales with the process node automatically. ``anchor="balanced_logic:int8",
    ratio=0.9`` means 0.9x the node's balanced-logic INT8 energy per op.
    """

    kind: Literal["relative"] = "relative"
    anchor: EnergyAnchor
    ratio: float = Field(..., gt=0)
    confidence: DataConfidence = DataConfidence.THEORETICAL
    source: str = ""

    model_config = {"extra": "forbid"}


class AbsoluteEnergy(BaseModel):
    """Energy per invocation measured or published at a reference node.

    ``circuit_class`` names the library whose node-to-node energy ratio is
    used to retarget the figure to another process node.
    """

    kind: Literal["absolute"] = "absolute"
    pj: float = Field(..., gt=0, description="Energy per invocation at ref_node, pJ")
    ref_node_id: str = Field(
        ..., min_length=1, description="ProcessNodeEntry id the figure was taken at"
    )
    circuit_class: CircuitClass = Field(
        CircuitClass.BALANCED_LOGIC,
        description="Library used to scale the figure to another node",
    )
    confidence: DataConfidence = DataConfidence.THEORETICAL
    source: str = ""

    model_config = {"extra": "forbid"}


EnergyRef = Annotated[Union[RelativeEnergy, AbsoluteEnergy], Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Functional units and PE datapaths
# ---------------------------------------------------------------------------


class UnitMode(BaseModel):
    """One operating mode of a functional unit (a number format it runs)."""

    operand_format: NumberFormatName
    accumulate_format: NumberFormatName | None = None
    lanes: int = Field(1, ge=1, description="Parallel invocations per cycle in this mode")
    issue_interval_cycles: int = Field(
        1, ge=1, description="Cycles between issues (1 = fully pipelined)"
    )
    energy: EnergyRef | None = Field(None, description="Energy per invocation")

    model_config = {"extra": "forbid"}


class FunctionalUnit(BaseModel):
    """One physical functional unit in a PE.

    A multi-precision unit is one entry with several ``modes`` (INT8, INT4,
    ...). Its area (``mtx``) is counted once, and its modes are alternatives.
    """

    unit_id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    op: OpKind
    modes: list[UnitMode] = Field(..., min_length=1)
    ops_per_invocation: float | None = Field(
        None,
        gt=0,
        description="Override of DEFAULT_OPS_PER_INVOCATION[op] for this unit",
    )
    mtx: float | None = Field(None, ge=0, description="Transistors per unit instance (M)")
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _unique_mode_formats(self) -> "FunctionalUnit":
        fmts = [m.operand_format for m in self.modes]
        dup = sorted({f for f in fmts if fmts.count(f) > 1})
        if dup:
            raise ValueError(f"unit {self.unit_id!r}: duplicate mode formats {dup}")
        return self

    @property
    def resolved_ops_per_invocation(self) -> float:
        if self.ops_per_invocation is not None:
            return self.ops_per_invocation
        return DEFAULT_OPS_PER_INVOCATION[self.op]

    def ops_per_clock(self) -> dict[str, float]:
        """Ops per clock of this unit in each mode, by ``"<op>:<format>"``."""
        opi = self.resolved_ops_per_invocation
        return {
            op_key(self.op, m.operand_format): m.lanes * opi / m.issue_interval_cycles
            for m in self.modes
        }


class PEDatapath(BaseModel):
    """The datapath of one processing element.

    ``circuit_class`` is the standard-cell library the datapath is built in.
    Leave it unset in a tile to inherit the tile's ``pe_circuit_class``; when
    it is set, the tile validator requires the two to agree.
    """

    datapath_id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    circuit_class: CircuitClass | None = None
    functional_units: list[FunctionalUnit] = Field(..., min_length=1)
    operand_regs: int = Field(2, ge=0, description="PE-local operand registers")
    accumulator_bits: int | None = Field(None, gt=0)
    token_match: bool = Field(
        True,
        description="Domain-flow token / coordinate match per PE (a real structure; "
        "the schedule itself is encoded in the fabric topology)",
    )
    mtx_per_pe: float | None = Field(
        None,
        ge=0,
        description="Total transistors per PE (M), when known as one figure",
    )
    confidence: DataConfidence = DataConfidence.THEORETICAL
    source: str = ""
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _unique_unit_ids(self) -> "PEDatapath":
        ids = [u.unit_id for u in self.functional_units]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"datapath {self.datapath_id!r}: duplicate unit_id {dup}")
        return self

    def ops_per_pe_per_clock(self) -> dict[str, float]:
        """Peak ops per PE per clock, by ``"<op>:<format>"``.

        Units sharing a key run concurrently and add up; different keys are
        alternative modes (see the module docstring).
        """
        out: dict[str, float] = {}
        for unit in self.functional_units:
            for key, ops in unit.ops_per_clock().items():
                out[key] = out.get(key, 0.0) + ops
        return out

    def legacy_precision_ops_per_pe(self) -> dict[str, float]:
        """Projection onto the legacy precision keys of ``ops_per_tile_per_clock``.

        Only MAC / FMA throughput maps onto a precision (``mac:int8`` ->
        ``int8``). Custom operators (min_plus, lerp, ...) have no legacy
        precision key.

        ``mac:<p>`` and ``fma:<p>`` are different keys, i.e. alternative
        modes, so the precision's peak is the larger of the two, not their
        sum. Units sharing one key are concurrent and are already summed by
        ``ops_per_pe_per_clock``.
        """
        out: dict[str, float] = {}
        legacy_ops = {op.value for op in LEGACY_PRECISION_OPS}
        for key, ops in self.ops_per_pe_per_clock().items():
            op, fmt = key.split(":", 1)
            if op in legacy_ops:
                out[fmt] = max(out.get(fmt, 0.0), ops)
        return out
