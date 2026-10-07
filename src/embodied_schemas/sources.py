"""Source database: the raw, cited figures behind estimated catalog values.

RFC 0001 R1.2/R1.3. Estimators (``scripts/swapc2_estimators.py``) derive
their parameters from these records instead of hard-coding them, so every
estimated number traces back to quoted source figures. The figures can be
validated locally against one another and against the catalog
(``tests/test_sources.py``).

Layout (``data/sources/``):

- ``documents.yaml``: one ``SourceDocument`` per publication, datasheet,
  product page or price listing.
- ``observations/<topic>.yaml``: ``Observation`` records, each one figure
  quoted from one document.

An observation is keyed ``<subject>.<quantity>[.<variant>]@<source_id>``.
``subject`` is what the figure describes (``tsmc_n7``,
``alpha_novatech_n30_25b``, ``natural_convection``); ``as_of`` dates the
figure itself, not the document, so ``series`` gives trends.

Query API (``SourceDB``): ``get``, ``find``, ``value``, ``series``, and
``to_sqlite`` for ad hoc SQL.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class DocumentKind(str, Enum):
    """What kind of document a figure is quoted from."""

    REPORT = "report"  # research / policy / analyst report
    ARTICLE = "article"  # trade press, magazine, news
    DATASHEET = "datasheet"  # vendor datasheet or manual
    PRODUCT_PAGE = "product_page"  # vendor or reseller product listing
    PRICE_LISTING = "price_listing"  # distributor price aggregator
    STANDARD = "standard"  # industry specification
    REFERENCE = "reference"  # materials / handbook reference


class FigureBasis(str, Enum):
    """How the source arrived at the figure."""

    DATASHEET = "datasheet"  # vendor-specified
    LIST_PRICE = "list_price"  # a price as listed for sale
    REPORTED = "reported"  # a figure the source reports from elsewhere
    MODEL_ESTIMATE = "model_estimate"  # the source's own model output
    MEASURED = "measured"  # measured by the source
    DISCLOSED = "disclosed"  # disclosed by the manufacturer (e.g. at a symposium)
    DERIVED = "derived"  # taken from other observations (``derived_from``)


# Every quantity the database holds, with its one allowed unit.
QUANTITY_UNITS: dict[str, str] = {
    "wafer_price": "usd",
    "defect_density": "per_cm2",
    "volumetric_thermal_resistance": "cm3_c_per_w",
    "thermal_resistance": "c_per_w",
    "mass": "g",
    "length": "mm",
    "width": "mm",
    "height": "mm",
    "unit_price": "usd",
    "material_density": "g_per_cm3",
    "power": "w",
    "airflow": "cfm",
    "cuda_cores": "count",
    "tensor_cores": "count",
    "cpu_cores": "count",
    "memory_capacity": "gb",
    "memory_bandwidth": "gb_per_s",
    "gpc": "count",
    "tpc": "count",
    "sm_per_tpc": "count",
    "tensor_cores_per_sm": "count",
    "cuda_cores_per_sm": "count",
    "frequency": "mhz",
    "ai_throughput": "tops",
}


# Lower bound each quantity's figures must respect: ("gt", 0) means > 0.
QUANTITY_BOUNDS: dict[str, tuple[str, float]] = {
    "wafer_price": ("gt", 0.0),
    "defect_density": ("ge", 0.0),
    "volumetric_thermal_resistance": ("gt", 0.0),
    "thermal_resistance": ("gt", 0.0),
    "mass": ("gt", 0.0),
    "length": ("gt", 0.0),
    "width": ("gt", 0.0),
    "height": ("gt", 0.0),
    "unit_price": ("ge", 0.0),
    "material_density": ("gt", 0.0),
    "power": ("ge", 0.0),
    "airflow": ("gt", 0.0),
    "cuda_cores": ("gt", 0.0),
    "tensor_cores": ("gt", 0.0),
    "cpu_cores": ("gt", 0.0),
    "memory_capacity": ("gt", 0.0),
    "memory_bandwidth": ("gt", 0.0),
    "gpc": ("gt", 0.0),
    "tpc": ("gt", 0.0),
    "sm_per_tpc": ("gt", 0.0),
    "tensor_cores_per_sm": ("gt", 0.0),
    "cuda_cores_per_sm": ("gt", 0.0),
    "frequency": ("gt", 0.0),
    "ai_throughput": ("gt", 0.0),
}

_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def check_date(value: str, field: str) -> str:
    """``value`` must be ISO ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` and a real
    date, so lexicographic order is chronological order."""
    m = _DATE_RE.match(value)
    if not m:
        raise ValueError(f"{field} {value!r} is not YYYY, YYYY-MM or YYYY-MM-DD")
    year, month, day = m.group(1), m.group(2), m.group(3)
    date(int(year), int(month or 1), int(day or 1))  # raises on e.g. 2022-13
    return value


class SourceDocument(BaseModel):
    """One document figures are quoted from."""

    id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    title: str
    publisher: str
    author: str | None = None
    kind: DocumentKind
    published: str | None = Field(None, description="YYYY, YYYY-MM or YYYY-MM-DD")
    url: str
    accessed: str = Field(..., description="YYYY-MM-DD the figures were read")
    access_note: str | None = Field(None, description="e.g. 'read via Wayback 20240602'")
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _dates(self) -> SourceDocument:
        check_date(self.accessed, "accessed")
        if len(self.accessed) != 10:
            raise ValueError(f"accessed {self.accessed!r} must be a full YYYY-MM-DD date")
        if self.published is not None:
            check_date(self.published, "published")
        return self


class Observation(BaseModel):
    """One figure, quoted from one document.

    ``value`` is the figure; a range quoted as such also sets ``value_min`` /
    ``value_max`` (``value`` is then the source's central figure, or the
    midpoint when it gives none). ``conditions`` record what the figure
    depends on (airflow, quantity break, maturity point, wafer size, ...).
    """

    subject: str = Field(..., pattern=r"^[a-z0-9_.]+$")
    category: str = Field(
        ...,
        pattern=r"^[a-z0-9_]+$",
        description="What kind of thing the subject is: heatsink, fan, m2_module, "
        "process_node, material, method, standard, ...",
    )
    quantity: str
    variant: str | None = Field(
        None, pattern=r"^[a-z0-9_]+$", description="e.g. 'typ', 'max', 'qty_1620'"
    )
    value: float
    value_min: float | None = None
    value_max: float | None = None
    unit: str
    as_of: str = Field(..., description="Date of the figure itself: YYYY, YYYY-MM or YYYY-MM-DD")
    basis: FigureBasis
    source_id: str
    quote: str = Field(..., min_length=1, description="Exact text or table row quoted")
    conditions: dict[str, str | float] = Field(default_factory=dict)
    derived_from: list[str] = Field(
        default_factory=list,
        description="Keys of the observations a 'derived' figure is taken from",
    )
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self) -> Observation:
        expected = QUANTITY_UNITS.get(self.quantity)
        if expected is None:
            raise ValueError(
                f"unknown quantity {self.quantity!r} (known: {sorted(QUANTITY_UNITS)})"
            )
        if self.unit != expected:
            raise ValueError(f"{self.quantity} must be in {expected!r}, not {self.unit!r}")
        lo = self.value_min if self.value_min is not None else self.value
        hi = self.value_max if self.value_max is not None else self.value
        if not lo <= self.value <= hi:
            raise ValueError(f"{self.key}: value {self.value} outside [{lo}, {hi}]")
        op, bound = QUANTITY_BOUNDS[self.quantity]
        if not (lo > bound if op == "gt" else lo >= bound):
            sign = ">" if op == "gt" else ">="
            raise ValueError(f"{self.key}: {self.quantity} must be {sign} {bound:g}, got {lo}")
        check_date(self.as_of, "as_of")
        if (self.basis == FigureBasis.DERIVED) != bool(self.derived_from):
            raise ValueError(f"{self.key}: basis 'derived' and derived_from go together")
        return self

    @property
    def key(self) -> str:
        """``<subject>.<quantity>[.<variant>]@<source_id>``."""
        variant = f".{self.variant}" if self.variant else ""
        return f"{self.subject}.{self.quantity}{variant}@{self.source_id}"


class SourceDB:
    """The source database, validated as a whole."""

    def __init__(self, documents: list[SourceDocument], observations: list[Observation]):
        self.documents = {d.id: d for d in documents}
        if len(self.documents) != len(documents):
            raise ValueError("duplicate document id")
        self.observations: dict[str, Observation] = {}
        for obs in observations:
            if obs.source_id not in self.documents:
                raise ValueError(f"{obs.key}: unknown source_id {obs.source_id!r}")
            if obs.key in self.observations:
                raise ValueError(f"duplicate observation {obs.key}")
            self.observations[obs.key] = obs
        categories: dict[str, set[str]] = {}
        for obs in self.observations.values():
            categories.setdefault(obs.subject, set()).add(obs.category)
        mixed = {s: c for s, c in categories.items() if len(c) > 1}
        if mixed:
            raise ValueError(f"subjects with more than one category: {mixed}")
        for obs in self.observations.values():
            for ref in obs.derived_from:
                if ref not in self.observations:
                    raise ValueError(f"{obs.key}: derived_from unknown observation {ref!r}")
                if self.observations[ref].quantity != obs.quantity:
                    raise ValueError(f"{obs.key}: derived_from {ref!r} is another quantity")

    def get(self, key: str) -> Observation:
        """One observation by key. Raises KeyError."""
        return self.observations[key]

    def value(self, key: str) -> float:
        """One observation's value by key."""
        return self.get(key).value

    def find(
        self,
        quantity: str | None = None,
        subject: str | None = None,
        variant: str | None = None,
        source_id: str | None = None,
        category: str | None = None,
        **conditions: str | float,
    ) -> list[Observation]:
        """Observations matching every given field and condition, by key."""
        out = []
        for obs in self.observations.values():
            if category is not None and obs.category != category:
                continue
            if quantity is not None and obs.quantity != quantity:
                continue
            if subject is not None and obs.subject != subject:
                continue
            if variant is not None and obs.variant != variant:
                continue
            if source_id is not None and obs.source_id != source_id:
                continue
            if any(obs.conditions.get(k) != v for k, v in conditions.items()):
                continue
            out.append(obs)
        return sorted(out, key=lambda o: o.key)

    def series(self, quantity: str, subject: str) -> list[Observation]:
        """A subject's figures for one quantity, oldest first (trends)."""
        return sorted(self.find(quantity, subject), key=lambda o: (o.as_of, o.key))

    def subjects(self, quantity: str, category: str | None = None) -> list[str]:
        """Every subject with at least one figure for ``quantity``."""
        return sorted({o.subject for o in self.find(quantity, category=category)})

    def to_sqlite(self) -> sqlite3.Connection:
        """An in-memory SQLite copy, tables ``documents`` and ``observations``
        (conditions as JSON text), for ad hoc SQL queries."""
        import json

        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, publisher TEXT, "
            "author TEXT, kind TEXT, published TEXT, url TEXT, accessed TEXT)"
        )
        con.execute(
            "CREATE TABLE observations ("
            "key TEXT PRIMARY KEY, subject TEXT, category TEXT, quantity TEXT, variant TEXT, "
            "value REAL, value_min REAL, value_max REAL, unit TEXT, as_of TEXT, basis TEXT, "
            "source_id TEXT REFERENCES documents(id), quote TEXT, conditions TEXT)"
        )
        con.executemany(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
            [
                (d.id, d.title, d.publisher, d.author, d.kind.value, d.published, d.url, d.accessed)
                for d in self.documents.values()
            ],
        )
        con.executemany(
            "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    o.key,
                    o.subject,
                    o.category,
                    o.quantity,
                    o.variant,
                    o.value,
                    o.value_min,
                    o.value_max,
                    o.unit,
                    o.as_of,
                    o.basis.value,
                    o.source_id,
                    o.quote,
                    json.dumps(o.conditions, sort_keys=True),
                )
                for o in self.observations.values()
            ],
        )
        con.commit()
        return con


def load_source_db(data_dir: Path | None = None) -> SourceDB:
    """Load and validate ``data/sources/``."""
    from embodied_schemas.loaders import get_data_dir

    root = (data_dir or get_data_dir()) / "sources"
    raw_docs = yaml.safe_load((root / "documents.yaml").read_text(encoding="utf-8")) or []
    if not raw_docs:
        raise ValueError(f"{root / 'documents.yaml'}: no source documents")
    documents = [SourceDocument.model_validate(d) for d in raw_docs]
    paths = sorted((root / "observations").glob("*.yaml"))
    if not paths:
        raise ValueError(f"{root / 'observations'}: no observation files")
    observations = []
    for path in paths:
        for i, raw in enumerate(yaml.safe_load(path.read_text(encoding="utf-8")) or []):
            try:
                observations.append(Observation.model_validate(raw))
            except Exception as exc:
                raise ValueError(f"{path.name}[{i}]: {exc}") from exc
    if not observations:
        raise ValueError(f"{root / 'observations'}: no observations")
    return SourceDB(documents, observations)
