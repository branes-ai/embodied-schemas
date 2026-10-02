"""SWaP-C² estimators (RFC 0001 R1.2, phase S2).

Estimates fill SWaP-C² values that no vendor publishes. Per the RFC, the
estimator code lives here and writes ``basis: estimated`` values into the
catalog YAMLs, so every estimate is reproducible from a named, versioned
model and its sourced parameters.

Estimators:

- ``cooling_sizing_v1``: per-W mass, volume and cost of a cooling solution.
  Heatsink volume follows the volumetric-thermal-resistance method:
  required sink-to-ambient resistance ``R = dT / P`` and volume
  ``V = R_vol / R``, so ``V / P = R_vol / dT``. Mass is volume times the
  heatsink's effective density (solid fraction x material density). A
  fan or pump adds a fixed mass, cost and electrical load.
- ``silicon_cost_v1``: variable cost of one good die. Gross dies per wafer,
  times Murphy's yield model, divided into the processed-wafer price.
  Variable cost only (RFC D9): no mask, design or other NRE term.

Run:
    python scripts/swapc2_estimators.py cooling --check   # fail if YAMLs differ
    python scripts/swapc2_estimators.py cooling --write   # rewrite the fields
    python scripts/swapc2_estimators.py silicon <die_mm2> <process_node_id>
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from embodied_schemas.loaders import load_cooling_solutions, load_process_nodes
from embodied_schemas.process_node import DataConfidence, ProcessNodeEntry
from embodied_schemas.swapc2 import SourcedValue, ValueBasis

REPO_ROOT = Path(__file__).resolve().parent.parent
COOLING_DIR = REPO_ROOT / "src" / "embodied_schemas" / "data" / "cooling-solutions"

COOLING_SIZING = "cooling_sizing_v1"
SILICON_COST = "silicon_cost_v1"


# ---------------------------------------------------------------------------
# silicon_cost_v1
# ---------------------------------------------------------------------------


def gross_dies_per_wafer(die_mm2: float, wafer_mm: float = 300.0) -> int:
    """Whole dies per wafer: wafer area over die area, less the partial dies
    lost at the edge (``pi * d / sqrt(2 * A)``). No edge exclusion or scribe."""
    if die_mm2 <= 0 or wafer_mm <= 0:
        raise ValueError("die area and wafer diameter must be positive")
    radius = wafer_mm / 2.0
    dpw = math.pi * radius**2 / die_mm2 - math.pi * wafer_mm / math.sqrt(2.0 * die_mm2)
    return max(int(dpw), 0)


def murphy_yield(die_mm2: float, d0_per_cm2: float) -> float:
    """Murphy's yield model, ``((1 - exp(-A * D0)) / (A * D0))^2`` with ``A``
    in cm^2 (B. T. Murphy, "Cost-size optima of monolithic integrated
    circuits", Proc. IEEE 52(12), 1964). 1.0 for a defect-free process."""
    if die_mm2 <= 0 or d0_per_cm2 < 0:
        raise ValueError("die area must be positive and D0 non-negative")
    ad = die_mm2 / 100.0 * d0_per_cm2
    if ad == 0:
        return 1.0
    return ((1.0 - math.exp(-ad)) / ad) ** 2


@dataclass(frozen=True)
class DieCost:
    """One good die's variable cost and the figures behind it."""

    gross_dies: int
    yield_fraction: float
    good_dies: float
    cost_usd: SourcedValue


def die_cost(die_mm2: float, node: ProcessNodeEntry) -> DieCost:
    """Variable cost of one good die of ``die_mm2`` on ``node`` (RFC D9).

    Raises:
        ValueError: the node states no ``wafer_cost_usd`` or
            ``defect_density_per_cm2``, or no good die fits on the wafer.
    """
    if node.wafer_cost_usd is None or node.defect_density_per_cm2 is None:
        raise ValueError(f"{node.id}: wafer_cost_usd and defect_density_per_cm2 are required")
    wafer_mm = node.wafer_diameter_mm or 300.0
    gross = gross_dies_per_wafer(die_mm2, wafer_mm)
    y = murphy_yield(die_mm2, node.defect_density_per_cm2)
    good = gross * y
    if good <= 0:
        raise ValueError(f"no good die of {die_mm2} mm^2 fits a {wafer_mm} mm wafer")
    cost = SourcedValue(
        value=node.wafer_cost_usd / good,
        basis=ValueBasis.ESTIMATED,
        confidence=_weaker(node.confidence, DataConfidence.THEORETICAL),
        source=(
            f"{SILICON_COST}: wafer_cost_usd / (gross dies x Murphy yield), "
            f"{node.id} ({node.wafer_cost_source or node.source})"
        ),
        notes=f"{gross} gross dies, yield {y:.3f}; die only, no package or test",
    )
    return DieCost(gross_dies=gross, yield_fraction=y, good_dies=good, cost_usd=cost)


_ORDER = [
    DataConfidence.CALIBRATED,
    DataConfidence.INTERPOLATED,
    DataConfidence.THEORETICAL,
    DataConfidence.UNKNOWN,
]


def _weaker(a: DataConfidence, b: DataConfidence) -> DataConfidence:
    return max(a, b, key=_ORDER.index)


# ---------------------------------------------------------------------------
# cooling_sizing_v1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """A model parameter and where it comes from."""

    value: float
    source: str


@dataclass(frozen=True)
class CoolingSizing:
    """Sizing inputs for one cooling-solution entry.

    ``r_vol``: volumetric thermal resistance of the heatsink for its airflow
    regime, cm^3 * C / W. ``delta_t``: design sink-to-ambient temperature
    rise, C. ``solid_fraction`` and ``density`` give the heatsink's effective
    density, g / cm^3. ``base_*``: fixed parts (fan, pump, mounting) that do
    not scale with W. ``cost_per_cm3``: heatsink cost per cm^3, if sourced.
    """

    r_vol: Param
    delta_t: Param
    solid_fraction: Param
    density: Param
    base_mass_g: Param | None = None
    base_volume_cm3: Param | None = None
    base_cost_usd: Param | None = None
    parasitic_power_w: Param | None = None
    cost_per_cm3: Param | None = None


def size_cooling(sizing: CoolingSizing) -> dict[str, float]:
    """The SWaP-C² fields ``cooling_sizing_v1`` writes for one entry."""
    volume_per_w = sizing.r_vol.value / sizing.delta_t.value
    fields = {
        "volume_cm3_per_w": volume_per_w,
        "mass_g_per_w": volume_per_w * sizing.solid_fraction.value * sizing.density.value,
    }
    if sizing.cost_per_cm3 is not None:
        fields["cost_usd_per_w"] = volume_per_w * sizing.cost_per_cm3.value
    for name in ("base_mass_g", "base_volume_cm3", "base_cost_usd", "parasitic_power_w"):
        param = getattr(sizing, name)
        if param is not None:
            fields[_BASE_FIELD[name]] = param.value
    return {k: round(v, 4) for k, v in fields.items()}


_BASE_FIELD = {
    "base_mass_g": "weight_g",
    "base_volume_cm3": "volume_cm3",
    "base_cost_usd": "cost_usd",
    "parasitic_power_w": "parasitic_power_w",
}


def sizing_source(sizing: CoolingSizing) -> str:
    """The ``sizing_source`` citation for an entry sized by ``sizing``."""
    parts = [f"{COOLING_SIZING} (scripts/swapc2_estimators.py)"]
    for name in CoolingSizing.__dataclass_fields__:
        param = getattr(sizing, name)
        if param is not None:
            parts.append(f"{name}={param.value:g} [{param.source}]")
    return "; ".join(parts)


# Shared, sourced parameters. Sources read 2026-10-02.
_LEE = 'S. Lee, "How to Select a Heat Sink", Electronics Cooling, June 1995, Table 2'
_R_VOL_NATURAL_SMALL = Param(
    500.0,
    f"{_LEE}: natural convection 500-800 cm3 C/W; the low end is for ~100-200 cm3 sinks. "
    "Small catalog sinks measure 214-474 (Alpha N30-25B / N50-15B / N80-40B, Toradex SMARC "
    "sink: R x envelope), so 500 is conservative (larger, heavier) for UAV-scale sinks",
)
_R_VOL_NATURAL_LARGE = Param(650.0, f"{_LEE}: natural convection 500-800 cm3 C/W, midpoint")
_R_VOL_FORCED_2_5 = Param(115.0, f"{_LEE}: 2.5 m/s (500 lfm) 80-150 cm3 C/W, midpoint")
_SOLID_FRACTION = Param(
    0.352,
    "0.95 g/cm3 effective density (median mass / envelope volume of 13 catalog aluminum "
    "heatsinks: Alpha Novatech N30-25B/N50-15B/N80-40B, ATS KRP/KRA/SF/MF, Wakefield "
    "655-53AB/698-100AB, Toradex SMARC sink; spread 0.57-1.90) / 2.70",
)
_AL_6063 = Param(2.70, "6063-T5 aluminum, 2.7 g/cc (QuickParts material datasheet)")
_SINK_COST_PER_CM3 = Param(
    0.0475,
    "linear fit cost = 5.23 + 0.0475 x V(cm3) through Alpha Novatech N30-25B ($6.30, 22.5 cm3) "
    "and N80-40B ($17.38, 256 cm3), Luxeon Star qty-1 prices 2026-10-02; predicts N50-15B "
    "($6.88) as $7.01",
)
_SINK_BASE_COST = 5.23
_SINK_BASE_COST_SRC = "fixed term of the Alpha Novatech qty-1 price fit (see cost_per_cm3)"


def _delta_t(junction_c: float, ambient_c: float) -> Param:
    return Param(
        junction_c - ambient_c,
        f"entry junction_c_max {junction_c:g} - ambient_c_max {ambient_c:g}; ignores the "
        "junction-to-sink drop, so the sink is a lower bound",
    )


# Sized cooling entries.
COOLING_PARAMS: dict[str, CoolingSizing] = {
    "passive_heatsink_small": CoolingSizing(
        r_vol=_R_VOL_NATURAL_SMALL,
        delta_t=_delta_t(100.0, 40.0),
        solid_fraction=_SOLID_FRACTION,
        density=_AL_6063,
        base_cost_usd=Param(_SINK_BASE_COST, _SINK_BASE_COST_SRC),
        cost_per_cm3=_SINK_COST_PER_CM3,
    ),
    "passive_heatsink_large": CoolingSizing(
        r_vol=_R_VOL_NATURAL_LARGE,
        delta_t=_delta_t(105.0, 50.0),
        solid_fraction=_SOLID_FRACTION,
        density=_AL_6063,
        base_cost_usd=Param(_SINK_BASE_COST, _SINK_BASE_COST_SRC),
        cost_per_cm3=_SINK_COST_PER_CM3,
    ),
    "active_fan": CoolingSizing(
        r_vol=_R_VOL_FORCED_2_5,
        delta_t=_delta_t(105.0, 45.0),
        solid_fraction=_SOLID_FRACTION,
        density=_AL_6063,
        base_mass_g=Param(80.0, 'Delta AFB0612EH-A 60 mm 12 V fan datasheet: "80 GRAMS"'),
        base_volume_cm3=Param(91.44, "Delta AFB0612EH-A envelope 60 x 60 x 25.4 mm"),
        base_cost_usd=Param(
            _SINK_BASE_COST + 10.08,
            "Delta AFB0612EH-A $10.08 (Avnet, qty 1620, via findchips 2026-10-02) + "
            + _SINK_BASE_COST_SRC,
        ),
        parasitic_power_w=Param(
            4.56, 'Delta AFB0612EH-A datasheet: "4.56 (MAX. 5.76) W" rated input'
        ),
        cost_per_cm3=_SINK_COST_PER_CM3,
    ),
}

# Fields cooling_sizing_v1 owns in a cooling YAML (written or removed).
_OWNED = (
    "weight_g",
    "volume_cm3",
    "cost_usd",
    "parasitic_power_w",
    "mass_g_per_w",
    "volume_cm3_per_w",
    "cost_usd_per_w",
    "basis",
    "sizing_source",
)


def expected_fields(entry_id: str) -> dict[str, object]:
    """The owned fields an entry's YAML should hold."""
    sizing = COOLING_PARAMS[entry_id]
    fields: dict[str, object] = dict(size_cooling(sizing))
    fields["basis"] = ValueBasis.ESTIMATED.value
    fields["sizing_source"] = sizing_source(sizing)
    return fields


def _yaml_scalar(value: object) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(float(value)) if isinstance(value, float) else str(value)


def render(text: str, fields: dict[str, object]) -> str:
    """``text`` with the owned top-level fields set to ``fields``: existing
    lines are replaced in place, missing ones inserted before ``source:``,
    and owned fields absent from ``fields`` removed. Comments are kept."""
    lines = text.splitlines(keepends=True)
    out, seen = [], set()
    for line in lines:
        m = re.match(r"^([a-z_0-9]+):", line)
        key = m.group(1) if m else None
        if key in _OWNED:
            if key in fields and key not in seen:
                out.append(f"{key}: {_yaml_scalar(fields[key])}\n")
                seen.add(key)
            continue
        if key == "source":
            out.extend(f"{k}: {_yaml_scalar(v)}\n" for k, v in fields.items() if k not in seen)
            seen.update(fields)
        out.append(line)
    return "".join(out)


def cooling_paths() -> dict[str, Path]:
    """Cooling YAML path by entry id."""
    paths = {}
    for path in sorted(COOLING_DIR.glob("*.yaml")):
        m = re.search(r"^id:\s*(\S+)", path.read_text(encoding="utf-8"), re.M)
        if m:
            paths[m.group(1)] = path
    return paths


def run_cooling(write: bool) -> int:
    """Check (or rewrite) every sized cooling YAML. Returns a process status."""
    paths = cooling_paths()
    status = 0
    for entry_id in COOLING_PARAMS:
        path = paths[entry_id]
        text = path.read_text(encoding="utf-8")
        new = render(text, expected_fields(entry_id))
        if new == text:
            continue
        if write:
            path.write_text(new, encoding="utf-8")
            print(f"wrote {path.relative_to(REPO_ROOT)}")
        else:
            print(f"FAIL: {path.relative_to(REPO_ROOT)} differs from {COOLING_SIZING}")
            status = 1
    if write:
        # The rewritten entries must still validate.
        load_cooling_solutions()
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cool = sub.add_parser("cooling", help="per-W sizing of the cooling catalog")
    mode = cool.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    sil = sub.add_parser("silicon", help="variable cost of one good die")
    sil.add_argument("die_mm2", type=float)
    sil.add_argument("process_node_id")
    args = parser.parse_args(argv)

    if args.cmd == "cooling":
        return run_cooling(write=args.write)
    node = load_process_nodes()[args.process_node_id]
    cost = die_cost(args.die_mm2, node)
    print(
        f"{args.die_mm2} mm^2 on {node.id}: {cost.gross_dies} gross dies, "
        f"yield {cost.yield_fraction:.3f}, ${cost.cost_usd.value:,.2f} per good die"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
