"""Fixed-function compute cores (branes-ai/graphs#268 Phase B3).

A *function core* encapsulates a whole compute segment in a custom datapath:
an ISP (raw -> YUV), a stereo SGM engine, a radar range/Doppler + CFAR
pipeline, or a full visual-inertial odometry pipeline (Navion-class). On its
function it is orders of magnitude more energy-efficient than a programmable
fabric. It needs no programmability overhead, and the segment's intermediate
data never leaves the core.

A core is characterized by its function, not by ops per clock:

- **throughput** in *work units* (pixels, frames, features, chirps, ...) per
  clock;
- **energy** per work unit, published or measured at a reference process
  node and split into logic and SRAM fractions so consumers can retarget it
  to another node;
- the **numeric formats** it computes in, so a consumer can check a
  workload's precision floor instead of silently accepting a narrower
  datapath.

``ops_equivalent_per_unit`` records how many software-equivalent ops one work
unit represents. It is for **reporting only**: it lets a report compare a
fixed-function core with a programmable engine running the same function.
It must never be added into programmable TOPS.

The same ``FunctionCore`` definition can be placed two ways, which a study
can sweep:

- inside the KPU checkerboard, as ``kpu.FixedFunctionTile.core``;
- standalone on an SoC die (a future block kind).

Architecture-neutral: no KPU imports.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.datapath import NumberFormatName
from embodied_schemas.local_memory import LocalMemory, duplicate_levels
from embodied_schemas.process_node import CircuitClass, DataConfidence

_FUNCTION_ID_RE = r"^[a-z0-9_]+(\.[a-z0-9_]+)+$"


class WorkUnit(str, Enum):
    """The unit a function core's throughput and energy are counted in."""

    PIXEL = "pixel"
    FRAME = "frame"
    FEATURE = "feature"
    POINT = "point"  # e.g. a LiDAR point or a map voxel update
    CHIRP = "chirp"  # radar
    SAMPLE = "sample"  # e.g. an IMU or audio sample
    ITERATION = "iteration"  # e.g. one solver iteration
    TOKEN = "token"


class FunctionContract(BaseModel):
    """What a core consumes and produces, and its configuration limits."""

    inputs: list[str] = Field(..., min_length=1, description="e.g. ['stereo_gray_frame', 'imu']")
    outputs: list[str] = Field(..., min_length=1, description="e.g. ['pose_6dof']")
    config_limits: dict[str, float | int | str] = Field(
        default_factory=dict,
        description="e.g. {'max_width': 1440, 'max_height': 1080, 'disparities': 128}",
    )

    model_config = {"extra": "forbid"}


class FunctionThroughput(BaseModel):
    """Throughput in work units: give exactly one of ``units_per_clock`` or
    ``cycles_per_unit``."""

    unit: WorkUnit
    units_per_clock: float | None = Field(None, gt=0)
    cycles_per_unit: float | None = Field(None, gt=0)
    fmax_mhz_ref: float | None = Field(
        None, gt=0, description="Maximum clock at the energy reference node (MHz)"
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _exactly_one_rate(self) -> "FunctionThroughput":
        if (self.units_per_clock is None) == (self.cycles_per_unit is None):
            raise ValueError("give exactly one of units_per_clock or cycles_per_unit")
        return self

    @property
    def per_clock(self) -> float:
        """Work units per clock."""
        if self.units_per_clock is not None:
            return self.units_per_clock
        return 1.0 / self.cycles_per_unit


class FunctionEnergy(BaseModel):
    """Energy per work unit at a reference node.

    ``logic_fraction`` and ``sram_fraction`` split the figure by the library
    family whose node-to-node scaling applies. They must add up to at most 1;
    any remainder (analog, IO) is taken as not scaling with the node.
    """

    pj_per_unit: float = Field(..., gt=0)
    ref_node_id: str = Field(..., min_length=1, description="ProcessNodeEntry id of the figure")
    logic_fraction: float = Field(1.0, ge=0, le=1)
    sram_fraction: float = Field(0.0, ge=0, le=1)
    confidence: DataConfidence = DataConfidence.THEORETICAL
    source: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _fractions(self) -> "FunctionEnergy":
        if self.logic_fraction + self.sram_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"logic_fraction + sram_fraction = "
                f"{self.logic_fraction + self.sram_fraction:g} exceeds 1"
            )
        return self


class FunctionIO(BaseModel):
    """Bytes moved per work unit, used to size NoC ports and stream links."""

    input_bytes_per_unit: float = Field(0.0, ge=0)
    output_bytes_per_unit: float = Field(0.0, ge=0)

    model_config = {"extra": "forbid"}


class CoreSiliconBlock(BaseModel):
    """One block of a core's silicon: a transistor count, or an area at a
    reference node (e.g. from a die shot).

    Give exactly one of ``mtx`` or ``area_mm2``. An area needs
    ``ref_node_id``, which is used to convert it to transistors through that
    node's density for ``circuit_class``.
    """

    name: str = Field(..., min_length=1)
    circuit_class: CircuitClass
    mtx: float | None = Field(None, ge=0, description="Transistors (M)")
    area_mm2: float | None = Field(None, gt=0, description="Area at ref_node_id")
    ref_node_id: str | None = Field(None, min_length=1)
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _exactly_one_size(self) -> "CoreSiliconBlock":
        if (self.mtx is None) == (self.area_mm2 is None):
            raise ValueError(f"silicon block {self.name!r}: give exactly one of mtx or area_mm2")
        if self.area_mm2 is not None and self.ref_node_id is None:
            raise ValueError(f"silicon block {self.name!r}: area_mm2 needs ref_node_id")
        return self


class FunctionCore(BaseModel):
    """A fixed-function core: one encapsulated compute segment."""

    function_id: str = Field(
        ...,
        pattern=_FUNCTION_ID_RE,
        description="Dotted id, e.g. 'isp.raw_to_yuv', 'vio.stereo_inertial', 'stereo.sgm'",
    )
    description: str = ""
    contract: FunctionContract
    numeric_formats: list[NumberFormatName] = Field(..., min_length=1)
    throughput: FunctionThroughput
    energy: FunctionEnergy
    ops_equivalent_per_unit: dict[NumberFormatName, float] = Field(
        default_factory=dict,
        description="Software-equivalent ops per work unit, by format (reporting only)",
    )
    io: FunctionIO | None = None
    local_memory: list[LocalMemory] | None = None
    silicon: list[CoreSiliconBlock] | None = None
    confidence: DataConfidence = DataConfidence.THEORETICAL
    source: str = ""
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self) -> "FunctionCore":
        dup = duplicate_levels(self.local_memory)
        if dup:
            raise ValueError(f"core {self.function_id!r}: duplicate local_memory levels {dup}")
        if any(not math.isfinite(v) or v < 0 for v in self.ops_equivalent_per_unit.values()):
            raise ValueError(
                f"core {self.function_id!r}: ops_equivalent_per_unit values must be finite and >= 0"
            )
        if self.silicon:
            names = [b.name for b in self.silicon]
            dup_names = sorted({n for n in names if names.count(n) > 1})
            if dup_names:
                raise ValueError(f"core {self.function_id!r}: duplicate silicon blocks {dup_names}")
        return self

    @property
    def units_per_clock(self) -> float:
        return self.throughput.per_clock
