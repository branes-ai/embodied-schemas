"""Power domains and per-domain operating points (branes-ai/graphs#268 Phase B4).

A power domain is a set of silicon that shares a supply rail, a clock and a
gating decision. The types are block-kind neutral: a domain describes a
region of a 2D grid of unit cells (a KPU compute-site checkerboard today; GPU
SM / GPC or TPU MXU clusters later), a named set of unit classes, or the
uncore. See ``graphs/docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md``
for the cluster rationale (tile -> cluster -> quadrant -> chip).

Types:

- ``SiteRange``: an inclusive rectangle of grid sites.
- ``PowerDomain``: one domain (``cluster``, ``tile_class`` or ``uncore``).
- ``DomainOperatingPoint``: the clock / Vdd / gating / activity of one
  domain in one thermal profile.

Re-exported from ``compute_block_common`` (the canonical import point). This
module has no block-kind imports, so ``kpu.py`` can use it without an import
cycle. Checks that need the grid size or the unit-class ids live in the
block that owns the domains (``kpu.KPUArchitectureBase``).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

_DOMAIN_ID_RE = r"^[a-z0-9_]+$"


class PowerDomainKind(str, Enum):
    """What a power domain groups.

    - ``cluster``: a region of the grid (``site_ranges``), e.g. a 4x4 block of
      KPU compute sites with its own rail and PLL. The DVFS / floorsweeping
      unit.
    - ``tile_class``: every instance of the listed unit classes (``members``,
      KPU ``tile_class_id`` values), wherever they are placed. E.g. one
      gateable domain for a fixed-function ISP.
    - ``uncore``: silicon outside the grid of units (NoC, shared memory cells,
      PHYs, IO, control). ``members`` optionally names the blocks it covers.
    """

    CLUSTER = "cluster"
    TILE_CLASS = "tile_class"
    UNCORE = "uncore"


class SiteRange(BaseModel):
    """An inclusive rectangle of grid sites: rows ``row_min..row_max``,
    columns ``col_min..col_max`` (0-based)."""

    row_min: int = Field(..., ge=0)
    row_max: int = Field(..., ge=0)
    col_min: int = Field(..., ge=0)
    col_max: int = Field(..., ge=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _ordered(self) -> "SiteRange":
        if self.row_max < self.row_min or self.col_max < self.col_min:
            raise ValueError(
                f"site range rows {self.row_min}..{self.row_max}, cols "
                f"{self.col_min}..{self.col_max} is empty (max < min)"
            )
        return self

    @property
    def rows(self) -> int:
        return self.row_max - self.row_min + 1

    @property
    def cols(self) -> int:
        return self.col_max - self.col_min + 1

    @property
    def sites(self) -> int:
        return self.rows * self.cols

    def cells(self) -> set[tuple[int, int]]:
        """Every (row, col) in the range."""
        return {
            (r, c)
            for r in range(self.row_min, self.row_max + 1)
            for c in range(self.col_min, self.col_max + 1)
        }

    def fits(self, rows: int, cols: int) -> bool:
        """Whether the range lies inside a ``rows x cols`` grid."""
        return self.row_max < rows and self.col_max < cols


class PowerDomain(BaseModel):
    """One power domain: a rail, a clock and a gating decision shared by a
    set of silicon.

    ``rail_id`` and ``clock_domain_id`` are labels. Domains that share a rail
    (or a clock) name the same id. A domain whose silicon can be switched off
    sets ``gateable``; only a gateable domain may be ``gated`` in a thermal
    profile's ``domain_operating_points``.
    """

    domain_id: str = Field(..., pattern=_DOMAIN_ID_RE)
    kind: PowerDomainKind
    members: list[str] = Field(
        default_factory=list,
        description="tile_class: the unit-class ids in the domain (required). "
        "cluster: optional restriction to these unit classes within the "
        "site ranges. uncore: optional names of the blocks it covers",
    )
    site_ranges: list[SiteRange] = Field(
        default_factory=list,
        description="cluster: the grid region of the domain (required); not "
        "allowed for the other kinds",
    )
    rail_id: str | None = Field(None, description="Supply rail; None = the chip core rail")
    clock_domain_id: str | None = Field(None, description="Clock; None = the chip clock")
    gateable: bool = Field(False, description="Can be power-gated (clock + leakage)")
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_kind(self) -> "PowerDomain":
        did, kind = self.domain_id, self.kind
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"power domain {did!r}: members repeat an id")
        if kind == PowerDomainKind.TILE_CLASS and not self.members:
            raise ValueError(f"power domain {did!r}: a tile_class domain needs members")
        if kind == PowerDomainKind.CLUSTER:
            if not self.site_ranges:
                raise ValueError(f"power domain {did!r}: a cluster domain needs site_ranges")
            seen: set[tuple[int, int]] = set()
            for rng in self.site_ranges:
                cells = rng.cells()
                if cells & seen:
                    raise ValueError(f"power domain {did!r}: site_ranges overlap")
                seen |= cells
        elif self.site_ranges:
            raise ValueError(
                f"power domain {did!r}: site_ranges apply only to a cluster domain, "
                f"not {kind.value}"
            )
        return self

    def sites(self) -> set[tuple[int, int]]:
        """Grid sites of a cluster domain (empty for the other kinds)."""
        out: set[tuple[int, int]] = set()
        for rng in self.site_ranges:
            out |= rng.cells()
        return out


class DomainOperatingPoint(BaseModel):
    """One power domain's operating point in one thermal profile.

    Unset fields take the profile's values: ``clock_mhz`` -> the profile
    clock, ``vdd_v`` -> the profile Vdd (or the node nominal), ``activity``
    -> the profile's activity. A ``gated`` domain is off: it has no clock,
    Vdd or activity to set.
    """

    clock_mhz: float | None = Field(None, gt=0)
    vdd_v: float | None = Field(None, gt=0)
    gated: bool = False
    activity: float | None = Field(
        None, ge=0, le=1, description="Fraction of peak switching activity, in [0, 1]"
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _gated_is_off(self) -> "DomainOperatingPoint":
        if self.gated:
            set_fields = [
                f for f in ("clock_mhz", "vdd_v", "activity") if getattr(self, f) is not None
            ]
            if set_fields:
                raise ValueError(f"a gated domain is off; it cannot also set {set_fields}")
        return self
