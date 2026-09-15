"""Phase B6 of the KPU heterogeneous-tile sprint (branes-ai/graphs#268).

The tile-class library:

- ``KPUTileClassEntry`` (``kpu_tile_class.py``): a one-tile template plus
  provenance. ``instantiate`` produces a self-contained SKU tile that carries
  ``tile_class_ref``.
- ``load_kpu_tile_classes()``: reads ``data/kpu-tile-classes/``, with the
  ``KPU_TILE_DATA_DIR`` private overlay (higher confidence wins, as for
  ``PROCESS_NODE_DATA_DIR``).
- The initial entries (all THEORETICAL, cited) and the ``tsmc_n65`` anchor.
- ``load_kpus`` warns about KPU products it cannot express as a KPUEntry
  instead of dropping them silently.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from embodied_schemas import (
    AbsoluteEnergy,
    DataConfidence,
    FixedFunctionTile,
    KPUArchitecture,
    KPUBlock,
    KPUTileClassEntry,
    KPUTileKind,
    KPUTileSpec,
    RelativeEnergy,
    SystolicTile,
    derive_kpu_performance,
    load_compute_products,
    load_kpu_tile_classes,
    load_kpus,
    load_process_nodes,
)
from embodied_schemas.loaders import get_data_dir, load_and_validate

LIBRARY_DIR = get_data_dir() / "kpu-tile-classes"
LIBRARY = load_kpu_tile_classes()
NODES = load_process_nodes()
T64_ID = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"
T64 = load_compute_products()[T64_ID]

EXPECTED = {
    KPUTileKind.PE_FABRIC: {
        "pe_int8_mac_i32", "pe_bf16_fma", "pe_lns16_mac", "pe_fp16_lerp", "pe_minplus_i16",
    },
    KPUTileKind.SYSTOLIC: {"systolic_int8_ws"},
    KPUTileKind.FIXED_FUNCTION: {
        "ff_isp_raw2yuv", "ff_vio_stereo_inertial", "ff_stereo_sgm",
    },
}


def _template(**over) -> dict:
    """A minimal pe_fabric library entry (not catalog data)."""
    entry = {
        "id": "pe_test_mac",
        "name": "Test MAC",
        "tile": {
            "tile_kind": "pe_fabric",
            "tile_type": "Test-MAC",
            "pe_array_rows": 4,
            "pe_array_cols": 4,
            "pe_circuit_class": "balanced_logic",
            "ops_per_tile_per_clock": {"int8": 32},
        },
        "sources": ["unit test"],
        "last_updated": "2026-09-15",
    }
    entry.update(over)
    return entry


# ---------------------------------------------------------------------------
# KPUTileClassEntry
# ---------------------------------------------------------------------------


def test_template_identity_is_filled_in():
    entry = KPUTileClassEntry.model_validate(_template())
    assert entry.tile.tile_class_id == "pe_test_mac"
    assert entry.tile.num_tiles == 1
    assert entry.tile_kind == KPUTileKind.PE_FABRIC
    assert entry.confidence == DataConfidence.THEORETICAL


@pytest.mark.parametrize(
    "tile_over, match",
    [
        ({"num_tiles": 4}, "a library template describes one tile"),
        ({"tile_class_id": "other"}, "tile.tile_class_id is 'other'"),
        ({"tile_class_ref": "pe_test_mac"}, "tile_class_ref must be unset"),
    ],
)
def test_template_rules(tile_over, match):
    data = _template()
    data["tile"].update(tile_over)
    with pytest.raises(ValidationError, match=match):
        KPUTileClassEntry.model_validate(data)
    with pytest.raises(ValidationError):
        KPUTileClassEntry.model_validate(_template(sources=[]))
    for blank in ([""], ["cited", "   "]):
        with pytest.raises(ValidationError, match="blank citations"):
            KPUTileClassEntry.model_validate(_template(sources=blank))


def test_instantiate_resolves_a_self_contained_tile():
    entry = KPUTileClassEntry.model_validate(_template())
    tile = entry.instantiate(12, power_domain_id="pe_c0", placement={"affinity": "center"})
    assert isinstance(tile, KPUTileSpec)
    assert tile.num_tiles == 12 and tile.tile_class_ref == "pe_test_mac"
    assert tile.tile_class_id == "pe_test_mac" and tile.power_domain_id == "pe_c0"
    assert entry.tile.num_tiles == 1 and entry.tile.tile_class_ref is None  # untouched
    # The same class twice in one SKU needs a second id.
    twin = entry.instantiate(2, tile_class_id="pe_test_mac_b")
    assert twin.tile_class_ref == "pe_test_mac"
    assert list(tile.model_dump().keys())[-1] == "tile_class_ref"
    with pytest.raises(ValidationError):
        entry.instantiate(0)


# ---------------------------------------------------------------------------
# Loader + private overlay
# ---------------------------------------------------------------------------


def _write(dirpath: Path, name: str, data: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / name).write_text(yaml.safe_dump(data, sort_keys=False))


def test_tile_class_overlay_confidence_wins(tmp_path, monkeypatch):
    base = tmp_path / "data"
    _write(base / "kpu-tile-classes", "pe_test_mac.yaml", _template())
    overlay = tmp_path / "private"
    _write(overlay, "pe_test_mac.yaml", _template(name="RTL", confidence="calibrated"))
    _write(overlay, "pe_secret.yaml", _template(id="pe_secret", name="Secret"))

    monkeypatch.delenv("KPU_TILE_DATA_DIR", raising=False)
    assert load_kpu_tile_classes(base)["pe_test_mac"].name == "Test MAC"

    monkeypatch.setenv("KPU_TILE_DATA_DIR", str(overlay))
    lib = load_kpu_tile_classes(base)
    assert lib["pe_test_mac"].name == "RTL"  # CALIBRATED beats THEORETICAL
    assert lib["pe_secret"].name == "Secret"  # overlay-only id is added

    # A lower-confidence overlay never downgrades the public entry.
    _write(overlay, "pe_test_mac.yaml", _template(name="stale", confidence="unknown"))
    assert load_kpu_tile_classes(base)["pe_test_mac"].name == "Test MAC"


def test_overlay_that_is_not_a_directory_is_skipped(tmp_path, monkeypatch, capsys):
    base = tmp_path / "data"
    _write(base / "kpu-tile-classes", "pe_test_mac.yaml", _template())
    monkeypatch.setenv("KPU_TILE_DATA_DIR", str(tmp_path / "missing"))
    assert set(load_kpu_tile_classes(base)) == {"pe_test_mac"}
    assert "KPU_TILE_DATA_DIR" in capsys.readouterr().out


def test_process_node_overlay_uses_the_same_rule(tmp_path, monkeypatch):
    base = tmp_path / "data"
    shutil.copytree(get_data_dir() / "process-nodes", base / "process-nodes")
    n16 = yaml.safe_load((base / "process-nodes" / "tsmc" / "n16.yaml").read_text())
    overlay = tmp_path / "pdk"
    _write(overlay, "n16.yaml", {**n16, "nominal_vdd_v": 0.75, "confidence": "calibrated"})
    monkeypatch.setenv("PROCESS_NODE_DATA_DIR", str(overlay))
    assert load_process_nodes(base)["tsmc_n16"].nominal_vdd_v == 0.75


# ---------------------------------------------------------------------------
# load_kpus reports what it cannot express
# ---------------------------------------------------------------------------


def test_load_kpus_warns_on_unrepresentable_kpu_products(tmp_path):
    base = tmp_path / "data"
    shutil.copytree(get_data_dir() / "process-nodes", base / "process-nodes")
    products = base / "compute_products"
    data = T64.model_dump(mode="json")
    _write(products, "t64.yaml", data)
    # A die with two KPU blocks is a valid ComputeProduct but not a KPUEntry.
    two_blocks = copy.deepcopy({**data, "id": "kpu_two_blocks"})
    two_blocks["dies"][0]["blocks"].append(copy.deepcopy(data["dies"][0]["blocks"][0]))
    _write(products, "two_blocks.yaml", two_blocks)
    gpu = next(
        c for c in load_compute_products().values()
        if not any(isinstance(b, KPUBlock) for d in c.dies for b in d.blocks)
    )
    _write(products, "gpu.yaml", gpu.model_dump(mode="json"))

    with pytest.warns(UserWarning, match="'kpu_two_blocks' has a KPU block but cannot be") as rec:
        entries = load_kpus(base)
    assert set(entries) == {T64_ID}
    assert len(rec) == 1  # the non-KPU product is skipped silently


def test_load_kpus_warns_on_a_missing_process_node(tmp_path):
    base = tmp_path / "data"
    shutil.copytree(get_data_dir() / "process-nodes", base / "process-nodes")
    data = T64.model_dump(mode="json")
    _write(base / "compute_products", "t64.yaml", data)
    orphan = copy.deepcopy({**data, "id": "kpu_orphan"})
    orphan["dies"][0]["process_node_id"] = "tsmc_n1"
    _write(base / "compute_products", "orphan.yaml", orphan)

    with pytest.warns(
        UserWarning, match=r"skipping 'kpu_orphan': process node 'tsmc_n1' is not in the catalog"
    ) as rec:
        entries = load_kpus(base)
    assert set(entries) == {T64_ID}
    assert len(rec) == 1


def test_catalog_load_kpus_is_warning_free(recwarn):
    assert len(load_kpus()) == 12
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


# ---------------------------------------------------------------------------
# The shipped library
# ---------------------------------------------------------------------------


def test_every_library_yaml_loads_strictly():
    # load_all_from_directory only prints a warning for a bad file; load each
    # file directly so a broken entry fails here instead of vanishing.
    paths = sorted(LIBRARY_DIR.glob("*.yaml"))
    entries = [load_and_validate(p, KPUTileClassEntry) for p in paths]
    assert {e.id for e in entries} == set(LIBRARY)
    for p, e in zip(paths, entries):
        assert p.stem == e.id
    by_kind: dict = {}
    for e in entries:
        by_kind.setdefault(e.tile_kind, set()).add(e.id)
    assert by_kind == EXPECTED


@pytest.mark.parametrize("class_id", sorted(LIBRARY))
def test_entry_provenance_and_references(class_id):
    entry = LIBRARY[class_id]
    assert entry.confidence == DataConfidence.THEORETICAL
    assert entry.sources and all(s.strip() for s in entry.sources)
    kpu_nodes = {
        d.process_node_id
        for c in load_compute_products().values()
        for d in c.dies
        if any(isinstance(b, KPUBlock) for b in d.blocks)
    }
    refs, anchors = [], []
    if entry.ref_node_id:
        refs.append(entry.ref_node_id)
    tile = entry.tile
    units = []
    if isinstance(tile, KPUTileSpec) and tile.datapath is not None:
        units = tile.datapath.functional_units
    elif isinstance(tile, SystolicTile):
        units = [tile.mac]
    for unit in units:
        for mode in unit.modes:
            assert mode.energy is not None, f"{class_id}: {unit.unit_id} has no energy"
            if isinstance(mode.energy, RelativeEnergy):
                anchors.append(mode.energy.anchor)
            elif isinstance(mode.energy, AbsoluteEnergy):
                refs.append(mode.energy.ref_node_id)
    if isinstance(tile, FixedFunctionTile):
        refs.append(tile.core.energy.ref_node_id)
        refs += [b.ref_node_id for b in tile.core.silicon or [] if b.ref_node_id]
        assert entry.ref_node_id == tile.core.energy.ref_node_id
    for ref in refs:
        assert ref in NODES, f"{class_id}: unknown process node {ref!r}"
    # Relative energies must resolve on every node a KPU SKU targets.
    for anchor in anchors:
        for node_id in kpu_nodes | {"tsmc_n40", "tsmc_n65"}:
            assert anchor in NODES[node_id].energy_per_op_pj, (class_id, anchor, node_id)


def test_int8_mac_class_is_the_legacy_int8_primary_tile():
    legacy = next(t for t in T64.dies[0].blocks[0].tiles if t.tile_class_id == "int8_primary")
    tile = LIBRARY["pe_int8_mac_i32"].instantiate(
        legacy.num_tiles, tile_type=legacy.tile_type, tile_class_id=legacy.tile_class_id
    )
    skip = {"datapath", "tile_class_ref"}
    assert tile.model_dump(exclude=skip) == legacy.model_dump(exclude=skip)
    assert tile.datapath is not None and tile.tile_class_ref == "pe_int8_mac_i32"


def test_a_heterogeneous_sku_built_from_the_library():
    base = T64.dies[0].blocks[0].to_architecture().model_dump(mode="json")
    mix = {
        "pe_int8_mac_i32": 24, "pe_bf16_fma": 8, "pe_lns16_mac": 8, "pe_fp16_lerp": 4,
        "pe_minplus_i16": 6, "systolic_int8_ws": 4, "ff_isp_raw2yuv": 1,
        "ff_vio_stereo_inertial": 1, "ff_stereo_sgm": 1,
    }
    tiles = [LIBRARY[cid].instantiate(n).model_dump(mode="json") for cid, n in mix.items()]
    arch = KPUArchitecture.model_validate(
        {**base, "tiles": tiles, "total_tiles": sum(mix.values())}
    )
    assert {t.tile_class_ref for t in arch.tiles} == set(mix)
    # 24+8+8+4+6 PE + 4 systolic + 1 ISP + 4 (2x2 VIO) + 1 SGM = 60 of 64 sites.
    assert sum(t.total_sites for t in arch.tiles) == 60

    perf = derive_kpu_performance(arch.tiles, 500.0)
    assert set(perf.fixed_function_throughput) == {
        "isp.raw_to_yuv", "vio.stereo_inertial", "stereo.sgm",
    }
    assert {"lns16", "lns8", "int8", "bf16", "fp32"} <= set(perf.peak_ops_per_sec_by_precision)
    block = KPUBlock.from_architecture(arch)
    assert KPUBlock.model_validate(block.model_dump(mode="json")) == block


def test_fixed_function_entries_reproduce_their_sources():
    # Guard the transcription of the published figures.
    vio = LIBRARY["ff_vio_stereo_inertial"].tile.core
    assert vio.energy.pj_per_unit * 1e-12 * 71 == pytest.approx(24e-3, rel=0.01)  # 24 mW @ 71 fps
    assert 62.5e6 * vio.units_per_clock == pytest.approx(71, rel=0.01)
    assert sum(b.area_mm2 for b in vio.silicon) == pytest.approx(3.54 * 4.54, rel=0.01)
    sgm = LIBRARY["ff_stereo_sgm"].tile.core
    px_per_s = 1920 * 1080 * 30
    assert sgm.energy.pj_per_unit * 1e-12 * px_per_s == pytest.approx(0.836, rel=0.01)
    assert 170e6 * sgm.units_per_clock == pytest.approx(px_per_s, rel=0.01)
    isp = LIBRARY["ff_isp_raw2yuv"].tile.core
    assert isp.energy.pj_per_unit == 224
    assert sum(b.area_mm2 for b in isp.silicon) == pytest.approx(0.36)


def test_scaling_anchor_nodes():
    for node_id, nm, vdd in (("tsmc_n40", 40, 0.9), ("tsmc_n65", 65, 1.0)):
        node = NODES[node_id]
        assert node.node_nm == nm and node.nominal_vdd_v == vdd
        assert node.confidence == DataConfidence.THEORETICAL
        assert node.transistor_topology.value == "bulk_planar"
    # Older nodes cost more energy per op and pack fewer transistors.
    chain = ["tsmc_n16", "tsmc_n28hpm", "tsmc_n40", "tsmc_n65"]
    energy = [NODES[n].energy_per_op_pj["balanced_logic:int8"] for n in chain]
    density = [NODES[n].densities["balanced_logic"].mtx_per_mm2 for n in chain]
    assert energy == sorted(energy) and density == sorted(density, reverse=True)
