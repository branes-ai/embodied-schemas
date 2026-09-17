"""The T768 Matrix class as a systolic tile (branes-ai/graphs#268 D8).

The T768 shipped with a ``Matrix`` class written as a PE-fabric tile whose
throughput figure (8192 INT8 ops per clock over an 8x8 array) carried a
systolic mechanism in a note: "128 ops/PE/cycle via systolic accumulation".
D8 moves it onto the ``systolic`` tile kind, which has to state that
mechanism -- cells x MAC lanes x ops per invocation -- instead of implying
it.

The migration is structural. These tests pin what it must not change: the
headline throughput, the energy basis (node anchor, no systolic discount),
the silicon (one per-cell block, not also carried on the tile), and the L1 /
L2 a systolic tile no longer inherits from the chip.
"""

from __future__ import annotations

import pytest

from embodied_schemas import (
    KPUTileKind,
    SystolicTile,
    derive_kpu_performance,
    load_compute_products,
)
from embodied_schemas.compute_product import KPUBlock
from embodied_schemas.local_memory import LocalMemoryLevel

T768_ID = "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc"

#: The shipped Matrix class's figures, before the migration.
LEGACY_MATRIX_OPS_PER_TILE = {"int8": 8192.0, "bf16": 4096.0, "fp16": 4096.0}
LEGACY_HEADLINE = {
    "int8_tops": 1353.8,
    "bf16_tflops": 676.9,
    "fp32_tflops": 27.1,
    "int4_tops": 756.1,
}


@pytest.fixture(scope="module")
def t768():
    return load_compute_products()[T768_ID]


@pytest.fixture(scope="module")
def block(t768) -> KPUBlock:
    (block,) = [b for d in t768.dies for b in d.blocks if isinstance(b, KPUBlock)]
    return block


@pytest.fixture(scope="module")
def matrix(block) -> SystolicTile:
    (tile,) = [t for t in block.tiles if t.tile_type == "Matrix"]
    return tile


def test_matrix_is_a_weight_stationary_systolic_array(matrix, block):
    assert isinstance(matrix, SystolicTile)
    assert (matrix.array_rows, matrix.array_cols) == (8, 8)
    assert matrix.dataflow.value == "weight_stationary"
    assert matrix.circuit_class.value == "hp_logic"
    assert (matrix.pipeline_fill_cycles, matrix.pipeline_drain_cycles) == (8, 8)
    # Only the Matrix class moved: INT8- and BF16-primary stay PE fabric.
    kinds = {t.tile_type: t.tile_kind for t in block.tiles}
    assert kinds == {
        "INT8-primary": KPUTileKind.PE_FABRIC,
        "BF16-primary": KPUTileKind.PE_FABRIC,
        "Matrix": KPUTileKind.SYSTOLIC,
    }


def test_throughput_is_the_shipped_figure_stated_as_lanes(matrix):
    """8192 INT8 ops per tile is 128 per cell: 64 lanes x 2 ops (a MAC is a
    multiply and an accumulate). The float modes run half the lanes."""
    assert matrix.ops_per_tile_per_clock == LEGACY_MATRIX_OPS_PER_TILE
    lanes = {m.operand_format: m.lanes for m in matrix.mac.modes}
    assert lanes == {"int8": 64, "bf16": 32, "fp16": 32}
    assert matrix.mac.resolved_ops_per_invocation == 2
    assert all(m.issue_interval_cycles == 1 for m in matrix.mac.modes)


def test_headline_performance_is_unchanged_and_derivable(t768, block):
    perf = t768.performance
    for field, value in LEGACY_HEADLINE.items():
        assert getattr(perf, field) == value, field
    # The declared roll-up is the derived one, split by kind.
    clock = t768.power.default_profile.clock_mhz
    assert perf.declares_rollup
    assert derive_kpu_performance(block.tiles, clock) == perf
    assert set(perf.by_tile_kind) == {KPUTileKind.PE_FABRIC, KPUTileKind.SYSTOLIC}


def test_energy_is_the_node_anchor_not_a_systolic_discount(matrix):
    """No mode declares energy, so every op is charged the node's hp_logic
    anchor exactly as the PE-fabric tile was. A discount would change TDP
    and need a Vdd re-tune: a model change of its own, not this one."""
    assert all(m.energy is None for m in matrix.mac.modes)


def test_silicon_is_counted_once(t768, matrix):
    """The per-cell silicon stays in the chip's pe_matrix block; the cell
    unit declares none, or the transistors would be counted twice."""
    assert matrix.mac.mtx is None
    (die,) = t768.dies
    (pe_matrix,) = [b for b in die.silicon_bin.blocks if b.name == "pe_matrix"]
    ts = pe_matrix.transistor_source
    assert (ts.kind.value, ts.per_unit_mtx, ts.count_ref) == ("per_pe", 0.3, "tile.Matrix")


def test_the_tile_keeps_the_l1_and_l2_it_used_to_inherit(matrix, block):
    """A systolic tile does not inherit the chip's L1 / L2 figures, so it
    states them: the same fabric-edge stream storage and reformatting buffer
    as every other compute tile of the SKU."""
    memory = {m.level: m for m in matrix.local_memory}
    assert set(memory) == {LocalMemoryLevel.L1, LocalMemoryLevel.L2}
    assert memory[LocalMemoryLevel.L1].kib == block.memory.l1_kib_per_tile
    assert memory[LocalMemoryLevel.L2].kib == block.memory.l2_kib_per_tile
    assert all(m.per.value == "tile" for m in memory.values())
