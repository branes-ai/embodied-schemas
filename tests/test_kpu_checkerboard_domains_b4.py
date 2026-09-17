"""Phase B4 of the KPU heterogeneous-tile sprint (branes-ai/graphs#268).

Checkerboard, power domains, per-domain operating points:

- ``CheckerboardSpec`` on ``KPUArchitectureBase.checkerboard`` makes the
  compute-site grid explicit. It is checked by the site-accounting invariant
  and, for an explicit placement map, by footprint tiling.
- The generic ``PowerDomain`` (``power_domain.py``, re-exported from
  ``compute_block_common``) lives on ``KPUArchitectureBase.power_domains``.
  Tile ``power_domain_id`` references are checked there.
- ``KPUThermalProfile.domain_operating_points`` / ``tdp_scenario`` are
  checked against the architecture by ``KPUEntry`` and ``ComputeProduct``.

Pins backward compatibility: every catalog SKU loads unchanged, with the new
fields None.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

import embodied_schemas.compute_block_common as cbc
from tests.test_kpu_catalog import HETEROGENEOUS_KPU_SKU_IDS, LEGACY_KPU_SKU_IDS
from embodied_schemas import (
    SPARE_SITE,
    CheckerboardPlacement,
    CheckerboardSpec,
    ComputeProduct,
    DomainOperatingPoint,
    KPUArchitecture,
    KPUBlock,
    KPUEntry,
    KPUThermalProfile,
    PowerDomain,
    PowerDomainKind,
    SiteRange,
    KPUTileKind,
    load_compute_products,
    load_kpus,
)

T64_ID = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"
CATALOG = load_compute_products()
# The backward-compatibility contract is about the SKUs that shipped
# before the heterogeneous work; see tests/kpu_catalog.py.
KPU_PRODUCTS = {sku: CATALOG[sku] for sku in LEGACY_KPU_SKU_IDS}
T64 = KPU_PRODUCTS[T64_ID]


# ---------------------------------------------------------------------------
# Fixture: a 4x4 heterogeneous KPU
#
#   row 0:  vio  vio  isp  .
#   row 1:  vio  vio  sys  sys
#   row 2:  int8 int8 int8 int8
#   row 3:  int8 int8 int8 int8
#
# 8 pe_fabric + 2 systolic + 1 ISP + 1 VIO (2x2) = 15 sites, 1 spare.
# Figures are illustrative, not catalog data.
# ---------------------------------------------------------------------------

MAP = [
    ["vio", "vio", "isp", SPARE_SITE],
    ["vio", "vio", "systolic_int8", "systolic_int8"],
    ["int8_primary"] * 4,
    ["int8_primary"] * 4,
]


def _isp_core() -> dict:
    return {
        "function_id": "isp.raw_to_yuv",
        "contract": {"inputs": ["bayer_raw_frame"], "outputs": ["yuv420_frame"]},
        "numeric_formats": ["uint16"],
        "throughput": {"unit": "pixel", "units_per_clock": 2},
        "energy": {"pj_per_unit": 30.0, "ref_node_id": "tsmc_n16"},
    }


def _vio_core() -> dict:
    return {
        "function_id": "vio.stereo_inertial",
        "contract": {"inputs": ["stereo_gray_frame", "imu_sample"], "outputs": ["pose_6dof"]},
        "numeric_formats": ["fp32"],
        "throughput": {"unit": "frame", "cycles_per_unit": 2.0e6},
        "energy": {"pj_per_unit": 1.0e8, "ref_node_id": "tsmc_n65"},
    }


def _arch() -> dict:
    base = T64.dies[0].blocks[0].to_architecture().model_dump(mode="json")
    int8 = next(t for t in base["tiles"] if t["tile_class_id"] == "int8_primary")
    tiles = [
        {**int8, "num_tiles": 8},
        {
            "tile_kind": "systolic",
            "tile_type": "Systolic-INT8",
            "num_tiles": 2,
            "array_rows": 32,
            "array_cols": 32,
            "circuit_class": "balanced_logic",
            "mac": {"unit_id": "mac", "op": "mac", "modes": [{"operand_format": "int8",
                                                             "accumulate_format": "int32"}]},
        },
        {
            "tile_kind": "fixed_function",
            "tile_type": "ISP",
            "num_tiles": 1,
            "core": _isp_core(),
            "power_domain_id": "ff_isp",
        },
        {
            "tile_kind": "fixed_function",
            "tile_type": "VIO",
            "num_tiles": 1,
            "core": _vio_core(),
            "footprint": {"rows": 2, "cols": 2, "absorbs_memory_cells": True},
            "power_domain_id": "ff_vio",
        },
    ]
    return {
        **base,
        "total_tiles": 12,
        "tiles": tiles,
        "noc": {**base["noc"], "mesh_rows": 4, "mesh_cols": 4},
        "checkerboard": {
            "compute_sites": {"rows": 4, "cols": 4},
            "memory_cell": {"l3_kib": base["memory"]["l3_kib_per_tile"]},
            "placement": "explicit",
            "placement_map": copy.deepcopy(MAP),
            "spare_sites": 1,
        },
        "power_domains": [
            {
                "domain_id": "pe_c0",
                "kind": "cluster",
                "site_ranges": [{"row_min": 2, "row_max": 3, "col_min": 0, "col_max": 1}],
                "rail_id": "vdd_c0",
                "clock_domain_id": "clk_c0",
            },
            {
                "domain_id": "pe_c1",
                "kind": "cluster",
                "site_ranges": [{"row_min": 2, "row_max": 3, "col_min": 2, "col_max": 3}],
                "rail_id": "vdd_c1",
                "clock_domain_id": "clk_c1",
            },
            {"domain_id": "ff_isp", "kind": "tile_class", "members": ["isp"], "gateable": True},
            {"domain_id": "ff_vio", "kind": "tile_class", "members": ["vio"], "gateable": True},
            {"domain_id": "uncore", "kind": "uncore", "members": ["noc", "l3", "phy"]},
        ],
    }


def _invalid(data: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        KPUArchitecture.model_validate(data)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


#: The default partition per SKU family: (cluster edge in sites, clusters).
_DEFAULT_PARTITION = {
    "t64": (2, 16),
    "t128": (4, 8),
    "t256": (4, 16),
    "t512": (4, 32),
    "t768": (4, 48),
}


@pytest.mark.parametrize("sku", sorted(KPU_PRODUCTS))
def test_legacy_skus_carry_only_the_default_cluster_partition(sku):
    """The uniform SKUs gained the default per-cluster DVFS partition in
    graphs#268 F2, and nothing else from B4.

    Still no checkerboard (clusters resolve against the implicit mesh), and
    no operating points or TDP scenario -- so every cluster runs at its
    profile's Vdd and clock and the power model is unchanged. The partition
    itself is checked for what makes it a DVFS partition: it tiles the mesh
    exactly once, in one repeated square shape, each cluster on its own rail
    and PLL, with one uncore domain.
    """
    cp = KPU_PRODUCTS[sku]
    (block,) = [b for d in cp.dies for b in d.blocks if isinstance(b, KPUBlock)]
    assert block.checkerboard is None
    for p in cp.power.thermal_profiles:
        assert p.domain_operating_points is None and p.tdp_scenario is None

    domains = block.power_domains
    clusters = [d for d in domains if d.kind == PowerDomainKind.CLUSTER]
    uncore = [d for d in domains if d.kind == PowerDomainKind.UNCORE]
    assert len(uncore) == 1
    assert len(clusters) + len(uncore) == len(domains)  # no tile_class domains

    rows, cols = block.noc.mesh_rows, block.noc.mesh_cols
    covered = [site for d in clusters for site in d.sites()]
    assert len(covered) == len(set(covered)) == rows * cols  # exactly once

    shapes = {(r.rows, r.cols) for d in clusters for r in d.site_ranges}
    assert len(shapes) == 1
    (edge_r, edge_c), = shapes
    family = sku.split("_")[1]  # t64, t128, ...
    # Exact, per family, rather than a range: a range -- whether the design's
    # 8-64 or the shipped 8-48 -- would still accept a T256 partitioned into
    # 32 clusters. The DVFS design's table gives 2x2 for the T64 (4x4 would
    # leave only 4 clusters) and 4x4 elsewhere; the T768 is absent from the
    # table and gets 4x4 by the same rule.
    expected_edge, expected_count = _DEFAULT_PARTITION[family]
    assert edge_r == edge_c == expected_edge
    assert len(clusters) == expected_count
    assert 8 <= len(clusters) <= 64  # and inside the design's range

    assert len({d.rail_id for d in clusters}) == len(clusters)
    assert len({d.clock_domain_id for d in clusters}) == len(clusters)
    assert all(d.rail_id and d.clock_domain_id for d in domains)
    assert not any(d.gateable for d in domains)

    assert ComputeProduct.model_validate(cp.model_dump(mode="json")) == cp


def test_legacy_kpu_entries_still_load():
    """The legacy KPUEntry view still round-trips. Scoped to the SKUs that
    predate the checkerboard: kpu_h64_auto1 has one by design, and
    test_heterogeneous_kpu_entries_carry_their_checkerboard covers it."""
    entries = load_kpus()
    assert entries
    for sku in LEGACY_KPU_SKU_IDS:
        e = entries[sku]
        assert e.kpu_architecture.checkerboard is None
        assert KPUEntry.model_validate(e.model_dump(mode="json")) == e


def test_heterogeneous_kpu_entries_carry_their_checkerboard():
    """The legacy view is not lossy for a heterogeneous SKU: KPUEntry holds
    the same KPUArchitectureBase, so its checkerboard and tile kinds
    survive and round-trip."""
    entries = load_kpus()
    for sku in HETEROGENEOUS_KPU_SKU_IDS:
        e = entries[sku]
        arch = e.kpu_architecture
        assert arch.checkerboard is not None
        assert {t.tile_kind for t in arch.tiles} == {
            KPUTileKind.PE_FABRIC, KPUTileKind.SYSTOLIC,
            KPUTileKind.FIXED_FUNCTION,
        }
        assert KPUEntry.model_validate(e.model_dump(mode="json")) == e


def test_power_domain_types_are_reexported_from_compute_block_common():
    assert cbc.PowerDomain is PowerDomain
    assert cbc.PowerDomainKind is PowerDomainKind
    assert cbc.SiteRange is SiteRange
    assert cbc.DomainOperatingPoint is DomainOperatingPoint
    assert {"PowerDomain", "DomainOperatingPoint"} <= set(cbc.__all__)


# ---------------------------------------------------------------------------
# Checkerboard
# ---------------------------------------------------------------------------


def test_heterogeneous_checkerboard_round_trips():
    arch = KPUArchitecture.model_validate(_arch())
    cb = arch.checkerboard
    assert cb.placement == CheckerboardPlacement.EXPLICIT
    assert cb.compute_sites.sites == 16
    assert sum(t.total_sites for t in arch.tiles) + cb.spare_sites == 16

    block = KPUBlock.from_architecture(arch)
    again = KPUBlock.model_validate(block.model_dump(mode="json"))
    assert again == block and again.to_architecture() == arch
    keys = list(block.model_dump().keys())
    assert keys[-2:] == ["checkerboard", "power_domains"]


def test_auto_placement_needs_no_map():
    data = _arch()
    data["checkerboard"] = {"compute_sites": {"rows": 4, "cols": 4}, "spare_sites": 1}
    arch = KPUArchitecture.model_validate(data)
    assert arch.checkerboard.placement == CheckerboardPlacement.AUTO
    assert arch.checkerboard.memory_cell is None


def test_site_accounting_invariant():
    data = _arch()
    data["checkerboard"]["placement"] = "auto"
    data["checkerboard"]["placement_map"] = None
    data["checkerboard"]["spare_sites"] = 0
    _invalid(data, r"site accounting: tiles use 15 sites .* \+ 0 spare != 4x4 = 16")

    data["tiles"][0]["num_tiles"] = 9  # 16 used + 0 spare: accounting holds ...
    data["total_tiles"] = 13
    assert KPUArchitecture.model_validate(data).checkerboard.spare_sites == 0
    data["tiles"][0]["num_tiles"] = 10  # ... one more does not
    data["total_tiles"] = 14
    _invalid(data, "site accounting: tiles use 17 sites")


def test_checkerboard_cross_checks():
    data = _arch()
    data["total_tiles"] = 15  # sites, not tiles
    _invalid(data, r"total_tiles 15 != 12")

    data = _arch()
    data["noc"]["mesh_cols"] = 8
    _invalid(data, r"noc mesh 4x8 must match the 4x4 compute-site grid")

    data = _arch()
    data["checkerboard"]["memory_cell"]["l3_kib"] = 128
    _invalid(data, r"memory_cell.l3_kib 128 != memory.l3_kib_per_tile")

    data = _arch()
    data["checkerboard"].update(placement="auto", placement_map=None)
    data["tiles"][3]["footprint"] = {"rows": 1, "cols": 5}
    data["checkerboard"]["compute_sites"] = {"rows": 4, "cols": 4}
    data["tiles"][0]["num_tiles"] = 7  # keep accounting: 7 + 2 + 1 + 5 + 1 spare = 16
    data["total_tiles"] = 11
    data["power_domains"] = None
    for t in data["tiles"]:
        t.pop("power_domain_id", None)
    _invalid(data, r"1x5 footprint does not fit the 4x4 compute-site grid")


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"placement": "explicit"}, "placement 'explicit' needs a placement_map"),
        ({"placement_map": [["a"]]}, "placement_map requires placement 'explicit'"),
        ({"spare_sites": 5}, "spare_sites 5 exceeds the 2x2 grid"),
        (
            {"placement": "explicit", "placement_map": [["a", "a"]]},
            "placement_map must be 2x2",
        ),
        (
            {"placement": "explicit", "placement_map": [["a", "a"], ["a", "B-1"]]},
            r"entries \['B-1'\] are neither",
        ),
        (
            {"placement": "explicit", "placement_map": [["a", "."], ["a", "a"]]},
            "placement_map has 1 spare sites but spare_sites is 0",
        ),
    ],
)
def test_checkerboard_spec_shape(spec, match):
    with pytest.raises(ValidationError, match=match):
        CheckerboardSpec.model_validate({"compute_sites": {"rows": 2, "cols": 2}, **spec})


def test_placement_map_footprint_tiling():
    data = _arch()
    data["checkerboard"]["placement_map"][0][3] = "sgm"
    data["checkerboard"]["placement_map"][0][2] = SPARE_SITE  # keep the spare count
    data["tiles"][2]["num_tiles"] = 1
    _invalid(data, r"placement_map references unknown tile_class_id \['sgm'\]")

    # The VIO sites as an L-shape plus a stray site: not a 2x2 rectangle.
    data = _arch()
    pmap = data["checkerboard"]["placement_map"]
    pmap[1][1], pmap[0][3] = SPARE_SITE, "vio"
    _invalid(data, r"tile class 'vio' at site \(0, 0\) does not form a whole 2x2 footprint")

    # An extra ISP site steals one systolic site: counts disagree with num_tiles.
    data = _arch()
    data["checkerboard"]["placement_map"][1][3] = "isp"
    _invalid(
        data,
        r"disagrees with num_tiles \(isp: 2 placed, num_tiles 1, "
        r"systolic_int8: 1 placed, num_tiles 2\)",
    )

    # Two whole 2x2 footprints side by side are two tiles, not one.
    data = _arch()
    data["checkerboard"]["placement_map"] = [
        ["vio", "vio", "vio", "vio"],
        ["vio", "vio", "vio", "vio"],
        ["int8_primary"] * 4,
        ["int8_primary"] * 4,
    ]
    data["checkerboard"]["spare_sites"] = 0
    data["tiles"] = [data["tiles"][0], data["tiles"][3]]
    data["tiles"][1]["num_tiles"] = 2
    data["total_tiles"] = 10
    data["power_domains"] = [d for d in data["power_domains"] if d["domain_id"] != "ff_isp"]
    arch = KPUArchitecture.model_validate(data)
    assert arch.tiles[1].total_sites == 8


# ---------------------------------------------------------------------------
# Power domains
# ---------------------------------------------------------------------------


def test_power_domain_shape():
    pd = PowerDomain(domain_id="c", kind="cluster", site_ranges=[
        {"row_min": 0, "row_max": 1, "col_min": 0, "col_max": 3},
    ])
    assert pd.sites() == {(r, c) for r in range(2) for c in range(4)}
    assert pd.site_ranges[0].sites == 8 and pd.site_ranges[0].fits(2, 4)
    assert not pd.site_ranges[0].fits(2, 3)
    assert PowerDomain(domain_id="u", kind="uncore").members == []

    cases = [
        ({"domain_id": "t", "kind": "tile_class"}, "a tile_class domain needs members"),
        ({"domain_id": "c", "kind": "cluster"}, "a cluster domain needs site_ranges"),
        (
            {"domain_id": "t", "kind": "tile_class", "members": ["a"],
             "site_ranges": [{"row_min": 0, "row_max": 0, "col_min": 0, "col_max": 0}]},
            "site_ranges apply only to a cluster domain, not tile_class",
        ),
        (
            {"domain_id": "c", "kind": "cluster", "site_ranges": [
                {"row_min": 0, "row_max": 1, "col_min": 0, "col_max": 1},
                {"row_min": 1, "row_max": 2, "col_min": 1, "col_max": 2},
            ]},
            "site_ranges overlap",
        ),
        ({"domain_id": "t", "kind": "tile_class", "members": ["a", "a"]}, "members repeat"),
        ({"domain_id": "Bad-Id", "kind": "uncore"}, "String should match pattern"),
    ]
    for data, match in cases:
        with pytest.raises(ValidationError, match=match):
            PowerDomain.model_validate(data)
    with pytest.raises(ValidationError, match="is empty"):
        SiteRange(row_min=2, row_max=1, col_min=0, col_max=0)


def _domains(data: dict) -> dict[str, dict]:
    return {d["domain_id"]: d for d in data["power_domains"]}


def test_power_domain_references():
    data = _arch()
    data["power_domains"].append({"domain_id": "uncore", "kind": "uncore"})
    _invalid(data, r"duplicate power domain_id \['uncore'\]")

    data = _arch()
    _domains(data)["ff_isp"]["members"] = ["isp", "sgm"]
    _invalid(data, r"power domain 'ff_isp' references unknown tile_class_id \['sgm'\]")

    data = _arch()
    _domains(data)["ff_vio"]["members"] = ["vio", "isp"]
    _invalid(data, "tile class 'isp' is in two tile_class power domains")

    # Without a checkerboard, cluster site ranges resolve against the
    # implicit one-tile-per-site mesh (noc.mesh_rows x noc.mesh_cols), so a
    # uniform SKU can name DVFS clusters without being forced onto the
    # heterogeneous floorplan path (graphs#268 F2).
    uniform = T64.dies[0].blocks[0].to_architecture().model_dump(mode="json")
    assert uniform["checkerboard"] is None
    rows, cols = uniform["noc"]["mesh_rows"], uniform["noc"]["mesh_cols"]
    uniform["power_domains"] = [
        {"domain_id": "c_0_0", "kind": "cluster",
         "site_ranges": [{"row_min": 0, "row_max": 1, "col_min": 0, "col_max": 1}]},
    ]
    KPUArchitecture.model_validate(uniform)  # fits the 8x8 mesh
    uniform["power_domains"][0]["site_ranges"][0]["col_max"] = cols
    _invalid(uniform, f"is outside the {rows}x{cols} compute-site grid")

    data = _arch()
    _domains(data)["pe_c1"]["site_ranges"][0]["col_max"] = 4
    _invalid(data, "is outside the 4x4 compute-site grid")

    data = _arch()
    _domains(data)["pe_c1"]["site_ranges"][0]["col_min"] = 1
    _invalid(data, r"cluster power domains 'pe_c0' and 'pe_c1' overlap at site \(2, 1\)")


def test_tile_power_domain_id_references():
    data = _arch()
    data["tiles"][2]["power_domain_id"] = "ff_sgm"
    _invalid(data, "power_domain_id 'ff_sgm' is not a defined power domain")

    data = _arch()
    data["power_domains"] = None
    _invalid(data, r"power_domain_id 'ff_isp' is not a defined power domain \(defined: \[\]\)")

    data = _arch()
    data["tiles"][2]["power_domain_id"] = "uncore"
    _invalid(data, "power_domain_id 'uncore' is an uncore domain")

    data = _arch()
    data["tiles"][2]["power_domain_id"] = "ff_vio"
    _invalid(data, "power_domain_id 'ff_vio' does not list it in members")

    # A pe_fabric tile may name a cluster domain it sits in.
    data = _arch()
    data["tiles"][0]["power_domain_id"] = "pe_c0"
    assert KPUArchitecture.model_validate(data).tiles[0].power_domain_id == "pe_c0"

    # A tile listed by a tile_class domain must not name another domain.
    data = _arch()
    _domains(data)["ff_isp"]["members"] = ["isp", "systolic_int8"]
    data["tiles"][1]["power_domain_id"] = "pe_c0"
    _invalid(data, "power_domain_id 'pe_c0' disagrees with tile_class domain 'ff_isp'")


# ---------------------------------------------------------------------------
# Operating points and the TDP scenario
# ---------------------------------------------------------------------------


def test_domain_operating_point_shape():
    assert DomainOperatingPoint(gated=True).gated
    assert DomainOperatingPoint(clock_mhz=400, vdd_v=0.7, activity=0.5).activity == 0.5
    gated_msg = r"gated domain is off; it cannot also set \['clock_mhz'\]"
    with pytest.raises(ValidationError, match=gated_msg):
        DomainOperatingPoint(gated=True, clock_mhz=400)
    with pytest.raises(ValidationError):
        DomainOperatingPoint(activity=1.5)
    with pytest.raises(ValidationError):
        DomainOperatingPoint(clock_mhz=0)


def test_tdp_scenario_range():
    base = T64.power.thermal_profiles[0].model_dump(mode="json")
    ok = KPUThermalProfile.model_validate({**base, "tdp_scenario": {"a": 0.0, "b": 1.0}})
    assert ok.tdp_scenario == {"a": 0.0, "b": 1.0}
    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(ValidationError, match=r"tdp_scenario\['a'\] = .* is outside \[0, 1\]"):
            KPUThermalProfile.model_validate({**base, "tdp_scenario": {"a": bad}})


SCENARIO = {"int8_primary": 1.0, "systolic_int8": 0.5, "isp": 1.0, "vio": 0.25}


def _profiles(**over) -> list[dict]:
    out = []
    for p in T64.power.thermal_profiles:
        d = p.model_dump(mode="json")
        d.update(
            domain_operating_points={
                "pe_c1": {"clock_mhz": p.clock_mhz * 0.8, "vdd_v": 0.7},
                "ff_vio": {"gated": True},
            },
            tdp_scenario=dict(SCENARIO),
        )
        d.update(over)
        out.append(d)
    return out


def _product(**profile_over) -> dict:
    cp = T64.model_dump(mode="json")
    block = KPUBlock.from_architecture(KPUArchitecture.model_validate(_arch()))
    cp["dies"][0]["blocks"] = [block.model_dump(mode="json")]
    cp["power"]["thermal_profiles"] = _profiles(**profile_over)
    return cp


def _entry(**profile_over) -> dict:
    entry = load_kpus()[T64_ID].model_dump(mode="json")
    entry["kpu_architecture"] = _arch()
    entry["power"]["thermal_profiles"] = _profiles(**profile_over)
    return entry


@pytest.mark.parametrize("model, build", [(ComputeProduct, _product), (KPUEntry, _entry)])
def test_profile_references(model, build):
    obj = model.model_validate(build())
    prof = obj.power.thermal_profiles[0]
    assert prof.domain_operating_points["ff_vio"].gated
    assert prof.tdp_scenario == SCENARIO
    assert model.model_validate(obj.model_dump(mode="json")) == obj

    with pytest.raises(ValidationError, match="key 'pe_c9' is not a defined power domain"):
        model.model_validate(build(domain_operating_points={"pe_c9": {"clock_mhz": 100}}))
    with pytest.raises(ValidationError, match="power domain 'pe_c0' is gated but not gateable"):
        model.model_validate(build(domain_operating_points={"pe_c0": {"gated": True}}))
    with pytest.raises(ValidationError, match=r"unknown: \['sgm'\], missing: \['vio'\]"):
        scenario = {k: v for k, v in SCENARIO.items() if k != "vio"} | {"sgm": 1.0}
        model.model_validate(build(tdp_scenario=scenario))


def test_product_domain_ids_are_product_wide():
    cp = _product(domain_operating_points=None, tdp_scenario=None)
    cp["dies"][0]["blocks"].append(copy.deepcopy(cp["dies"][0]["blocks"][0]))
    with pytest.raises(ValidationError, match="domain_id 'pe_c0' is defined by two KPU blocks"):
        ComputeProduct.model_validate(cp)

    for d in cp["dies"][0]["blocks"][1]["power_domains"]:
        d["domain_id"] += "_b"
    cp["dies"][0]["blocks"][1]["tiles"][2]["power_domain_id"] = "ff_isp_b"
    cp["dies"][0]["blocks"][1]["tiles"][3]["power_domain_id"] = "ff_vio_b"
    assert len(ComputeProduct.model_validate(cp).dies[0].blocks) == 2

    cp["power"]["thermal_profiles"][0]["tdp_scenario"] = dict(SCENARIO)
    with pytest.raises(ValidationError, match="appears in more than one KPU block"):
        ComputeProduct.model_validate(cp)
