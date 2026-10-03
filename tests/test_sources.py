"""Local validation of the source database (``data/sources/``).

The source DB holds the quoted figures behind every estimated SWaP-C² value
(RFC 0001 R1.2/R1.3). These tests validate the figures themselves:

1. Integrity: schema, unit per quantity, references, no orphan documents.
2. Internal consistency: physical bounds and orderings within the records.
3. Cross-source agreement: independent sources for the same figure agree
   within a stated tolerance.
4. Trends: ``series`` orders a subject's figures in time.
5. Query API and the SQLite view.

The catalog is held to the DB by ``tests/test_swapc2_estimators.py``
(``cooling --check`` and ``nodes --check``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from embodied_schemas.sources import (
    QUANTITY_UNITS,
    Observation,
    SourceDB,
    SourceDocument,
    load_source_db,
)


@pytest.fixture(scope="module")
def db() -> SourceDB:
    return load_source_db()


def heatsinks(db: SourceDB) -> list[str]:
    fans = set(db.subjects("power"))
    return [s for s in db.subjects("mass") if s not in fans and db.find("length", s)]


def envelope_cm3(db: SourceDB, subject: str) -> float:
    dims = [db.find(q, subject)[0].value for q in ("length", "width", "height")]
    return dims[0] * dims[1] * dims[2] / 1000.0


# ---------------------------------------------------------------------------
# 1. Integrity
# ---------------------------------------------------------------------------


class TestIntegrity:
    def test_loads(self, db):
        assert len(db.documents) >= 20 and len(db.observations) >= 100

    def test_every_document_is_cited(self, db):
        cited = {o.source_id for o in db.observations.values()}
        assert set(db.documents) == cited

    def test_units_are_canonical(self, db):
        for obs in db.observations.values():
            assert obs.unit == QUANTITY_UNITS[obs.quantity], obs.key

    def test_every_figure_is_quoted_and_dated(self, db):
        for obs in db.observations.values():
            assert obs.quote.strip() and obs.as_of[:4].isdigit(), obs.key

    def test_dates_are_iso(self, db):
        for doc in db.documents.values():
            assert len(doc.accessed) == 10 and doc.accessed[4] == "-", doc.id

    def test_wrong_unit_rejected(self):
        with pytest.raises(ValidationError, match="must be in 'usd'"):
            Observation(
                subject="x",
                quantity="wafer_price",
                value=1,
                unit="eur",
                as_of="2020",
                basis="reported",
                source_id="d",
                quote="q",
            )

    def test_unknown_quantity_rejected(self):
        with pytest.raises(ValidationError, match="unknown quantity"):
            Observation(
                subject="x",
                quantity="price",
                value=1,
                unit="usd",
                as_of="2020",
                basis="reported",
                source_id="d",
                quote="q",
            )

    def test_value_outside_range_rejected(self):
        with pytest.raises(ValidationError, match="outside"):
            Observation(
                subject="x",
                quantity="mass",
                value=9,
                value_min=1,
                value_max=5,
                unit="g",
                as_of="2020",
                basis="datasheet",
                source_id="d",
                quote="q",
            )

    def test_unknown_source_rejected(self):
        obs = Observation(
            subject="x",
            quantity="mass",
            value=1,
            unit="g",
            as_of="2020",
            basis="datasheet",
            source_id="nope",
            quote="q",
        )
        with pytest.raises(ValueError, match="unknown source_id"):
            SourceDB([], [obs])

    def test_duplicate_rejected(self):
        doc = SourceDocument(
            id="d", title="t", publisher="p", kind="article", url="u", accessed="2026-10-02"
        )
        obs = Observation(
            subject="x",
            quantity="mass",
            value=1,
            unit="g",
            as_of="2020",
            basis="datasheet",
            source_id="d",
            quote="q",
        )
        with pytest.raises(ValueError, match="duplicate observation"):
            SourceDB([doc], [obs, obs])


# ---------------------------------------------------------------------------
# 2. Internal consistency
# ---------------------------------------------------------------------------


class TestThermalConsistency:
    def test_lee_r_vol_falls_with_airflow(self, db):
        order = [
            "natural_convection",
            "forced_air_1_0_m_s",
            "forced_air_2_5_m_s",
            "forced_air_5_0_m_s",
        ]
        mids = [db.find("volumetric_thermal_resistance", s)[0].value for s in order]
        assert mids == sorted(mids, reverse=True)

    def test_heatsinks_lighter_than_solid_aluminum(self, db):
        al = db.value("aluminum_6063_t5.material_density@quickparts_al_6063_t5")
        for s in heatsinks(db):
            density = db.find("mass", s)[0].value / envelope_cm3(db, s)
            assert 0.2 < density < al, (s, density)

    def test_effective_density_median(self, db):
        import statistics

        dens = [db.find("mass", s)[0].value / envelope_cm3(db, s) for s in heatsinks(db)]
        assert len(dens) == 13
        assert statistics.median(dens) == pytest.approx(0.95, abs=0.01)

    def test_catalog_sinks_within_lee_natural_bound(self, db):
        """R x envelope of every natural-convection catalog sink is at most
        Lee's natural-convection maximum."""
        (lee,) = db.find("volumetric_thermal_resistance", "natural_convection")
        for obs in db.find("thermal_resistance", airflow="natural"):
            assert obs.value * envelope_cm3(db, obs.subject) <= lee.value_max, obs.key

    def test_forced_air_sinks_within_lee_1_m_s_bound(self, db):
        (lee,) = db.find("volumetric_thermal_resistance", "forced_air_1_0_m_s")
        for obs in db.find("thermal_resistance", airflow_m_s=1.0):
            assert obs.value * envelope_cm3(db, obs.subject) <= lee.value_max, obs.key

    def test_resistance_falls_with_airflow_per_part(self, db):
        for s in db.subjects("thermal_resistance"):
            by_flow = {
                o.conditions.get("airflow_m_s"): o.value
                for o in db.find("thermal_resistance", s)
                if "airflow_m_s" in o.conditions
            }
            if {1.0, 2.5} <= set(by_flow):
                assert by_flow[2.5] < by_flow[1.0], s

    def test_heatsink_price_rises_with_size(self, db):
        priced = sorted(
            (envelope_cm3(db, o.subject), o.value) for o in db.find("unit_price", variant="qty_1")
        )
        assert [p for _, p in priced] == sorted(p for _, p in priced)


class TestFanConsistency:
    def test_typical_power_at_most_max(self, db):
        for s in db.subjects("power"):
            typ = db.find("power", s, variant="typ")[0].value
            mx = db.find("power", s, variant="max")[0].value
            assert 0 < typ <= mx, s

    def test_volume_discounts(self, db):
        """For one distributor, a larger quantity break is not more expensive."""
        for s in db.subjects("power"):
            by_dist: dict[str, list[tuple[float, float]]] = {}
            for o in db.find("unit_price", s):
                by_dist.setdefault(o.conditions["distributor"], []).append(
                    (o.conditions["quantity"], o.value)
                )
            for breaks in by_dist.values():
                prices = [p for _, p in sorted(breaks)]
                assert prices == sorted(prices, reverse=True), s


class TestWaferConsistency:
    CSET_ORDER = ["tsmc_n65", "tsmc_n40", "tsmc_n28hpm", "tsmc_n16", "tsmc_n7", "tsmc_n5"]

    def test_cset_price_rises_with_node(self, db):
        prices = [db.value(f"{n}.wafer_price@cset_2020_ai_chips") for n in self.CSET_ORDER]
        assert prices == sorted(prices)

    def test_cset_16_and_12_share_a_column(self, db):
        assert db.value("tsmc_n16.wafer_price@cset_2020_ai_chips") == db.value(
            "tsmc_n12.wafer_price@cset_2020_ai_chips"
        )

    def test_d0_is_hvm_scale(self, db):
        for obs in db.find("defect_density"):
            assert 0 < obs.value < 1.0, obs.key


# ---------------------------------------------------------------------------
# 3. Cross-source agreement
# ---------------------------------------------------------------------------

# Reported wafer prices (2021-2022) sit above the CSET 2020 model by TSMC's
# reported 10-20% price rises; beyond 20% a figure needs a second look.
WAFER_AGREEMENT = 0.20


def test_wafer_prices_agree_across_sources(db):
    for subject in db.subjects("wafer_price"):
        prices = [o.value for o in db.find("wafer_price", subject)]
        if len(prices) > 1:
            assert max(prices) / min(prices) - 1 <= WAFER_AGREEMENT, (subject, prices)


def test_n6_d0_equals_n7_as_disclosed(db):
    assert db.value("tsmc_n6.defect_density@tomshw_2020_08_24_tsmc_symposium") == db.value(
        "tsmc_n7.defect_density@anandtech_2020_08_25_tsmc_d0"
    )


# ---------------------------------------------------------------------------
# 4. Trends
# ---------------------------------------------------------------------------


def test_wafer_price_series_is_chronological(db):
    series = db.series("wafer_price", "tsmc_n7")
    assert [o.as_of for o in series] == ["2020", "2022-11"]
    assert series[-1].value >= series[0].value


# ---------------------------------------------------------------------------
# 5. Query API and SQLite view
# ---------------------------------------------------------------------------


class TestQuery:
    def test_find_by_condition(self, db):
        natural = db.find("thermal_resistance", airflow="natural")
        assert {o.subject for o in natural} >= {
            "alpha_novatech_n30_25b",
            "toradex_smarc_heatsink_passive",
        }

    def test_get_and_value(self, db):
        key = "delta_afb0612eh_a.power.typ@delta_afb0612eh_a"
        assert db.get(key).value == db.value(key) == 4.56

    def test_sqlite_view(self, db):
        con = db.to_sqlite()
        (n,) = con.execute("SELECT COUNT(*) FROM observations").fetchone()
        assert n == len(db.observations)
        rows = con.execute(
            "SELECT o.subject, o.value FROM observations o JOIN documents d "
            "ON o.source_id = d.id WHERE o.quantity = 'wafer_price' AND d.kind = 'report' "
            "ORDER BY o.value"
        ).fetchall()
        assert rows[0] == ("tsmc_n65", 1937.0) and rows[-1] == ("tsmc_n5", 16988.0)


# ---------------------------------------------------------------------------
# 6. Review hardening: dates, bounds, derivations, completeness
# ---------------------------------------------------------------------------


def _obs(**kw):
    base = dict(
        subject="x",
        quantity="mass",
        value=1.0,
        unit="g",
        as_of="2020",
        basis="datasheet",
        source_id="d",
        quote="q",
    )
    base.update(kw)
    return Observation(**base)


DOC = SourceDocument(
    id="d", title="t", publisher="p", kind="article", url="u", accessed="2026-10-02"
)


class TestHardening:
    @pytest.mark.parametrize("bad", ["2022-2", "2022-13", "22", "2022-02-30", "Nov 2022"])
    def test_bad_as_of_rejected(self, bad):
        with pytest.raises(ValidationError, match="as_of|YYYY|day|month"):
            _obs(as_of=bad)

    def test_bad_document_date_rejected(self):
        with pytest.raises(ValidationError, match="accessed"):
            SourceDocument(
                id="d", title="t", publisher="p", kind="article", url="u", accessed="2026-10"
            )

    def test_iso_dates_sort_chronologically(self):
        assert sorted(["2022-11", "2022", "2021-02-05", "2022-02"]) == [
            "2021-02-05",
            "2022",
            "2022-02",
            "2022-11",
        ]

    @pytest.mark.parametrize(
        "quantity,unit,value",
        [("length", "mm", 0.0), ("mass", "g", -1.0), ("unit_price", "usd", -0.01)],
    )
    def test_physical_bounds(self, quantity, unit, value):
        with pytest.raises(ValidationError, match="must be"):
            _obs(quantity=quantity, unit=unit, value=value)

    def test_free_item_price_allowed(self):
        assert _obs(quantity="unit_price", unit="usd", value=0.0).value == 0.0

    def test_derived_needs_derived_from(self):
        with pytest.raises(ValidationError, match="derived"):
            _obs(basis="derived")
        with pytest.raises(ValidationError, match="derived"):
            _obs(derived_from=["y.mass@d"])

    def test_derived_from_must_resolve_to_same_quantity(self):
        with pytest.raises(ValueError, match="unknown observation"):
            SourceDB([DOC], [_obs(basis="derived", derived_from=["nope.mass@d"])])
        other = _obs(subject="y", quantity="length", unit="mm")
        with pytest.raises(ValueError, match="another quantity"):
            SourceDB([DOC], [other, _obs(basis="derived", derived_from=[other.key])])

    def test_n6_d0_is_derived_from_n7(self, db):
        n6 = db.get("tsmc_n6.defect_density@tomshw_2020_08_24_tsmc_symposium")
        assert n6.basis.value == "derived"
        assert n6.derived_from == ["tsmc_n7.defect_density@anandtech_2020_08_25_tsmc_d0"]
        assert n6.value == db.value(n6.derived_from[0])

    def test_empty_documents_rejected(self, tmp_path):
        (tmp_path / "sources" / "observations").mkdir(parents=True)
        (tmp_path / "sources" / "documents.yaml").write_text("[]\n")
        with pytest.raises(ValueError, match="no source documents"):
            load_source_db(tmp_path)

    def test_missing_observations_rejected(self, tmp_path):
        (tmp_path / "sources").mkdir()
        (tmp_path / "sources" / "documents.yaml").write_text(
            "- {id: d, title: t, publisher: p, kind: article, url: u, accessed: '2026-10-02'}\n"
        )
        with pytest.raises(ValueError, match="no observation files"):
            load_source_db(tmp_path)
