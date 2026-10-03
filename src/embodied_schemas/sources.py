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

import sqlite3
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
}


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


class Observation(BaseModel):
    """One figure, quoted from one document.

    ``value`` is the figure; a range quoted as such also sets ``value_min`` /
    ``value_max`` (``value`` is then the source's central figure, or the
    midpoint when it gives none). ``conditions`` record what the figure
    depends on (airflow, quantity break, maturity point, wafer size, ...).
    """

    subject: str = Field(..., pattern=r"^[a-z0-9_.]+$")
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
        **conditions: str | float,
    ) -> list[Observation]:
        """Observations matching every given field and condition, by key."""
        out = []
        for obs in self.observations.values():
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

    def subjects(self, quantity: str) -> list[str]:
        """Every subject with at least one figure for ``quantity``."""
        return sorted({o.subject for o in self.find(quantity)})

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
            "CREATE TABLE observations (key TEXT PRIMARY KEY, subject TEXT, quantity TEXT, "
            "variant TEXT, value REAL, value_min REAL, value_max REAL, unit TEXT, as_of TEXT, "
            "basis TEXT, source_id TEXT REFERENCES documents(id), quote TEXT, conditions TEXT)"
        )
        con.executemany(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
            [
                (d.id, d.title, d.publisher, d.author, d.kind.value, d.published, d.url, d.accessed)
                for d in self.documents.values()
            ],
        )
        con.executemany(
            "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    o.key,
                    o.subject,
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
    documents = [SourceDocument.model_validate(d) for d in raw_docs]
    observations = []
    for path in sorted((root / "observations").glob("*.yaml")):
        for i, raw in enumerate(yaml.safe_load(path.read_text(encoding="utf-8")) or []):
            try:
                observations.append(Observation.model_validate(raw))
            except Exception as exc:
                raise ValueError(f"{path.name}[{i}]: {exc}") from exc
    return SourceDB(documents, observations)
