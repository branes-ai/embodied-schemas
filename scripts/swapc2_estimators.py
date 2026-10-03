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

Every parameter is derived from the source database (``data/sources/``,
``embodied_schemas.sources``): quoted figures with their documents. The
``cooling`` and ``nodes`` commands write the derived values into the
catalog, and ``--check`` fails if the catalog and the source DB disagree.

Run:
    python scripts/swapc2_estimators.py cooling --check   # fail if YAMLs differ
    python scripts/swapc2_estimators.py cooling --write   # rewrite the fields
    python scripts/swapc2_estimators.py nodes --check     # process-node inputs vs source DB
    python scripts/swapc2_estimators.py silicon <die_mm2> <process_node_id>
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from embodied_schemas.cooling_solution import CoolingSolutionEntry
from embodied_schemas.loaders import load_cooling_solutions, load_process_nodes
from embodied_schemas.process_node import DataConfidence, ProcessNodeEntry
from embodied_schemas.sources import SourceDB, load_source_db
from embodied_schemas.swapc2 import SourcedValue, ValueBasis

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "embodied_schemas" / "data"
COOLING_DIR = DATA_DIR / "cooling-solutions"
PROCESS_NODE_DIR = DATA_DIR / "process-nodes"

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


# ---------------------------------------------------------------------------
# Parameters derived from the source database (data/sources/)
# ---------------------------------------------------------------------------
#
# Every parameter is computed from quoted observations, and its source string
# names their keys, so a parameter changes only when the recorded figures do.

_DB: SourceDB | None = None


def source_db() -> SourceDB:
    """The source database (loaded once)."""
    global _DB
    if _DB is None:
        _DB = load_source_db()
    return _DB


def r_vol(db: SourceDB, regime: str, pick: str) -> Param:
    """Lee's volumetric thermal resistance for ``regime``: the range's
    ``min`` (small sinks) or its ``mid``point."""
    (obs,) = db.find("volumetric_thermal_resistance", regime)
    value = obs.value_min if pick == "min" else obs.value
    return Param(value, f"{obs.key} ({pick} of {obs.value_min:g}-{obs.value_max:g})")


def heatsink_subjects(db: SourceDB) -> list[str]:
    """Catalog heatsinks: subjects with mass and an envelope and no fan power."""
    fans = set(db.subjects("power"))
    return [s for s in db.subjects("mass") if s not in fans and db.find("length", s)]


def envelope_cm3(db: SourceDB, subject: str) -> float:
    """L x W x H of a subject, in cm^3."""
    dims = [db.find(q, subject)[0].value for q in ("length", "width", "height")]
    return dims[0] * dims[1] * dims[2] / 1000.0


def effective_density(db: SourceDB) -> tuple[float, list[str]]:
    """Median mass / envelope volume of the catalog heatsinks, g / cm^3."""
    subjects = heatsink_subjects(db)
    dens = [db.find("mass", s)[0].value / envelope_cm3(db, s) for s in subjects]
    return statistics.median(dens), subjects


def solid_fraction(db: SourceDB) -> Param:
    density, subjects = effective_density(db)
    al = db.get("aluminum_6063_t5.material_density@quickparts_al_6063_t5")
    return Param(
        round(density / al.value, 4),
        f"median mass/envelope {density:.3f} g/cm3 of {len(subjects)} heatsinks "
        f"({', '.join(subjects)}) / {al.key}",
    )


def sink_price_fit(db: SourceDB) -> tuple[Param, Param]:
    """Least-squares ``price = a + b * V`` over the heatsinks with a qty-1
    price: returns (a in USD, b in USD / cm^3)."""
    points = [
        (envelope_cm3(db, o.subject), o.value, o.key)
        for o in db.find("unit_price", variant="qty_1")
        if o.subject in heatsink_subjects(db)
    ]
    n = len(points)
    mx = sum(v for v, _, _ in points) / n
    my = sum(p for _, p, _ in points) / n
    b = sum((v - mx) * (p - my) for v, p, _ in points) / sum((v - mx) ** 2 for v, _, _ in points)
    a = my - b * mx
    keys = ", ".join(k for _, _, k in points)
    return (
        Param(round(a, 4), f"intercept of least-squares price = a + b x envelope over {keys}"),
        Param(round(b, 6), f"slope of least-squares price = a + b x envelope over {keys}"),
    )


def _fan(db: SourceDB, subject: str, quantity: str, variant: str | None = None) -> Param:
    (obs,) = db.find(quantity, subject, variant=variant)
    return Param(obs.value, obs.key)


def delta_t(entry: CoolingSolutionEntry) -> Param:
    return Param(
        entry.junction_c_max - entry.ambient_c_max,
        f"entry junction_c_max {entry.junction_c_max:g} - ambient_c_max "
        f"{entry.ambient_c_max:g}; ignores the junction-to-sink drop, so the sink is a lower bound",
    )


# What each sized entry is: (airflow regime, R_vol pick, fan subject or None).
# R_vol 'min' for the small sink: Lee's low end is for ~100-200 cm3 sinks, and
# small catalog sinks measure lower still (R x envelope), so it is conservative.
SIZED_ENTRIES: dict[str, tuple[str, str, str | None]] = {
    "passive_heatsink_small": ("natural_convection", "min", None),
    "passive_heatsink_large": ("natural_convection", "mid", None),
    "active_fan": ("forced_air_2_5_m_s", "mid", "delta_afb0612eh_a"),
}

# Fan price used: the lowest-quantity Avnet break on the access date.
FAN_PRICE_VARIANT = {"delta_afb0612eh_a": "qty_1620_avnet"}


def cooling_sizing(entry_id: str, db: SourceDB | None = None) -> CoolingSizing:
    """The ``cooling_sizing_v1`` inputs for a sized entry, from the source DB."""
    db = db or source_db()
    entry = load_cooling_solutions()[entry_id]
    regime, pick, fan = SIZED_ENTRIES[entry_id]
    sink_base, sink_slope = sink_price_fit(db)
    al = db.get("aluminum_6063_t5.material_density@quickparts_al_6063_t5")
    sizing = dict(
        r_vol=r_vol(db, regime, pick),
        delta_t=delta_t(entry),
        solid_fraction=solid_fraction(db),
        density=Param(al.value, al.key),
        base_cost_usd=sink_base,
        cost_per_cm3=sink_slope,
    )
    if fan is not None:
        price = _fan(db, fan, "unit_price", FAN_PRICE_VARIANT[fan])
        dims = [_fan(db, fan, q) for q in ("length", "width", "height")]
        sizing.update(
            base_mass_g=_fan(db, fan, "mass"),
            base_volume_cm3=Param(
                round(dims[0].value * dims[1].value * dims[2].value / 1000.0, 4),
                f"{fan} envelope ({', '.join(d.source for d in dims)})",
            ),
            base_cost_usd=Param(
                round(sink_base.value + price.value, 4),
                f"{price.source} + sink price-fit intercept",
            ),
            parasitic_power_w=_fan(db, fan, "power", "typ"),
        )
    return CoolingSizing(**sizing)


# ---------------------------------------------------------------------------
# YAML writer (comment-preserving, owned top-level fields only)
# ---------------------------------------------------------------------------


def _yaml_scalar(value: object) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(float(value)) if isinstance(value, float) else str(value)


def render(text: str, fields: dict[str, object], owned: tuple[str, ...]) -> str:
    """``text`` with the ``owned`` top-level fields set to ``fields``: lines
    replaced in place (block-scalar continuations included), missing fields
    inserted before ``source:``, and owned fields absent from ``fields``
    removed. Comments and every other field are kept."""
    out, seen, skipping = [], set(), False
    for line in text.splitlines(keepends=True):
        if skipping and line[:1] in (" ", "\t") and line.strip():
            continue
        skipping = False
        m = re.match(r"^([a-z_0-9]+):", line)
        key = m.group(1) if m else None
        if key in owned:
            skipping = True
            if key in fields and key not in seen:
                out.append(f"{key}: {_yaml_scalar(fields[key])}\n")
                seen.add(key)
            continue
        if key == "source":
            out.extend(f"{k}: {_yaml_scalar(v)}\n" for k, v in fields.items() if k not in seen)
            seen.update(fields)
        out.append(line)
    return "".join(out)


def _check_or_write(path: Path, new: str, label: str, write: bool) -> int:
    text = path.read_text(encoding="utf-8")
    if new == text:
        return 0
    rel = path.relative_to(REPO_ROOT)
    if write:
        path.write_text(new, encoding="utf-8")
        print(f"wrote {rel}")
        return 0
    print(f"FAIL: {rel} differs from {label}")
    return 1


# ---------------------------------------------------------------------------
# cooling: sized cooling-solution entries
# ---------------------------------------------------------------------------

# Fields cooling_sizing_v1 owns in a cooling YAML (written or removed).
COOLING_OWNED = (
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


def expected_cooling_fields(entry_id: str) -> dict[str, object]:
    """The owned fields a sized entry's YAML should hold."""
    sizing = cooling_sizing(entry_id)
    fields: dict[str, object] = dict(size_cooling(sizing))
    fields["basis"] = ValueBasis.ESTIMATED.value
    fields["sizing_source"] = sizing_source(sizing)
    return fields


def _paths_by_id(directory: Path) -> dict[str, Path]:
    paths = {}
    for path in sorted(directory.glob("**/*.yaml")):
        m = re.search(r"^id:\s*(\S+)", path.read_text(encoding="utf-8"), re.M)
        if m:
            paths[m.group(1)] = path
    return paths


def run_cooling(write: bool) -> int:
    """Check (or rewrite) every sized cooling YAML. Returns a process status."""
    paths = _paths_by_id(COOLING_DIR)
    status = 0
    for entry_id in SIZED_ENTRIES:
        path = paths[entry_id]
        new = render(
            path.read_text(encoding="utf-8"), expected_cooling_fields(entry_id), COOLING_OWNED
        )
        status |= _check_or_write(path, new, COOLING_SIZING, write)
    if write:
        load_cooling_solutions()  # the rewritten entries must still validate
    return status


# ---------------------------------------------------------------------------
# nodes: silicon_cost_v1 inputs on the process-node catalog
# ---------------------------------------------------------------------------

# Process node -> (wafer-price observation key, D0 observation key). A node
# with neither is left without silicon-cost inputs.
NODE_INPUTS: dict[str, tuple[str | None, str | None]] = {
    "tsmc_n5": (
        "tsmc_n5.wafer_price@cset_2020_ai_chips",
        "tsmc_n5.defect_density@anandtech_2020_08_25_tsmc_d0",
    ),
    "tsmc_n6": (None, "tsmc_n6.defect_density@tomshw_2020_08_24_tsmc_symposium"),
    "tsmc_n7": (
        "tsmc_n7.wafer_price@cset_2020_ai_chips",
        "tsmc_n7.defect_density@anandtech_2020_08_25_tsmc_d0",
    ),
    "tsmc_n12": ("tsmc_n12.wafer_price@cset_2020_ai_chips", None),
    "tsmc_n16": ("tsmc_n16.wafer_price@cset_2020_ai_chips", None),
    "tsmc_n28hpm": ("tsmc_n28hpm.wafer_price@cset_2020_ai_chips", None),
    "tsmc_n40": ("tsmc_n40.wafer_price@cset_2020_ai_chips", None),
    "tsmc_n65": ("tsmc_n65.wafer_price@cset_2020_ai_chips", None),
}

NODE_OWNED = (
    "wafer_cost_usd",
    "wafer_diameter_mm",
    "defect_density_per_cm2",
    "wafer_cost_source",
)


def expected_node_fields(node_id: str, db: SourceDB | None = None) -> dict[str, object]:
    """The silicon-cost input fields a node's YAML should hold."""
    db = db or source_db()
    price_key, d0_key = NODE_INPUTS[node_id]
    fields: dict[str, object] = {}
    cites = []
    if price_key:
        price = db.get(price_key)
        fields["wafer_cost_usd"] = price.value
        fields["wafer_diameter_mm"] = float(price.conditions.get("wafer_mm", 300.0))
        cites.append(f"wafer_cost_usd = {price_key} ({price.basis.value}, {price.as_of} USD)")
    else:
        cites.append("no sourced wafer price")
    if d0_key:
        d0 = db.get(d0_key)
        fields["defect_density_per_cm2"] = d0.value
        cites.append(f"defect_density_per_cm2 = {d0_key} ({d0.conditions.get('maturity')})")
    else:
        cites.append("no sourced D0")
    fields["wafer_cost_source"] = "source DB (data/sources/): " + "; ".join(cites)
    return fields


def run_nodes(write: bool) -> int:
    """Check (or rewrite) the silicon-cost inputs of the mapped nodes."""
    paths = _paths_by_id(PROCESS_NODE_DIR)
    status = 0
    for node_id in NODE_INPUTS:
        path = paths[node_id]
        new = render(path.read_text(encoding="utf-8"), expected_node_fields(node_id), NODE_OWNED)
        status |= _check_or_write(path, new, "the source DB", write)
    if write:
        load_process_nodes()
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
        ("cooling", "per-W sizing of the cooling catalog"),
        ("nodes", "silicon-cost inputs on the process-node catalog"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        mode = cmd.add_mutually_exclusive_group(required=True)
        mode.add_argument("--check", action="store_true")
        mode.add_argument("--write", action="store_true")
    sil = sub.add_parser("silicon", help="variable cost of one good die")
    sil.add_argument("die_mm2", type=float)
    sil.add_argument("process_node_id")
    args = parser.parse_args(argv)

    if args.cmd == "cooling":
        return run_cooling(write=args.write)
    if args.cmd == "nodes":
        return run_nodes(write=args.write)
    node = load_process_nodes()[args.process_node_id]
    cost = die_cost(args.die_mm2, node)
    print(
        f"{args.die_mm2} mm^2 on {node.id}: {cost.gross_dies} gross dies, "
        f"yield {cost.yield_fraction:.3f}, ${cost.cost_usd.value:,.2f} per good die"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
