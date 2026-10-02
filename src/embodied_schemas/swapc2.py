"""SWaP-C²: Size, Weight, Power, Cost and Cooling of a compute product.

Implements requirement R1 of ``docs/rfcs/0001-compute-product-unification.md``
(phase S1). SWaP-C² is the primary optimization metric for UAVs and other
weight- and endurance-bound platforms.

Two halves:

- **Facts** (``SWaPC2Spec``, attached as ``ComputeProduct.swapc2``): the
  product's own size, mass, input-power characteristics and unit cost.
  These do not depend on the operating point. Every number is a
  ``SourcedValue`` carrying its basis, confidence and source, so an
  estimate never reads as a datasheet fact.
- **Resolution** (``resolve_swapc2``): the product plus the cooling
  solution its thermal profile binds, at one thermal profile (RFC D7). The
  cooling axis is folded into the other four: cooling mass, envelope
  height, fan power and cost are added to the product's.

Decisions the code follows:

- **D7**: SWaP-C² is a function of ``(product, thermal_profile)``.
- **D9**: cost is variable unit cost at quantity 1 and 1K. Non-recurring
  engineering (masks, design, up-front IP fees, qualification, tooling)
  never enters, either stored or amortized.
- **Power boundary**: the vehicle's input rail. ``tdp_watts`` is what the
  product draws at its own input. ``conversion_efficiency`` is the
  converter between the vehicle rail and the product, when that converter
  is outside the product. Cooling ``parasitic_power_w`` is already
  input-rail power, so the conversion loss is not applied to it.
- **Size is an envelope**: the cooling solution mounts on the product's
  top face. The envelope is the larger footprint, with the heights added.
- **D5 refinement**: mass and cost of a product with ``contains`` default to
  the sum over its contents when the product states no value of its own.
  The result is ``estimated`` and excludes the product's own PCB/enclosure.

Airframe-coupled metrics (hover-power penalty, endurance) and the
verdict-first fit check against a capability tier live downstream
(graphs / Embodied-AI-Architect).
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.process_node import DataConfidence

if TYPE_CHECKING:
    from embodied_schemas.compute_product import ComputeProduct
    from embodied_schemas.cooling_solution import CoolingSolutionEntry


class ValueBasis(str, Enum):
    """How a SWaP-C² value was obtained.

    DATASHEET: published by the vendor (datasheet, product brief, price list).
    MEASURED: measured on real hardware.
    DERIVED: computed from other values without a model (e.g. L x W x H).
    ESTIMATED: produced by a model; ``source`` names the estimator and version.
    """

    DATASHEET = "datasheet"
    MEASURED = "measured"
    DERIVED = "derived"
    ESTIMATED = "estimated"


class SourcedValue(BaseModel):
    """A SWaP-C² quantity with its provenance. The unit is fixed by the field
    that holds it (``mass_g`` is grams, ``unit_price_1_usd`` is USD, ...)."""

    value: float
    basis: ValueBasis
    confidence: DataConfidence
    source: str = Field(
        ..., min_length=1, description="Citation, or estimator id + version for 'estimated'"
    )
    notes: str = ""

    model_config = {"extra": "forbid"}


class SourcedDimensions(BaseModel):
    """An L x W x H envelope in mm, with one provenance for all three."""

    length_mm: float = Field(..., gt=0)
    width_mm: float = Field(..., gt=0)
    height_mm: float = Field(..., gt=0)
    basis: ValueBasis
    confidence: DataConfidence
    source: str = Field(..., min_length=1)
    notes: str = ""

    model_config = {"extra": "forbid"}

    @property
    def footprint_mm2(self) -> float:
        return self.length_mm * self.width_mm

    @property
    def volume_cm3(self) -> float:
        return self.length_mm * self.width_mm * self.height_mm / 1000.0


def _check_range(
    sv: SourcedValue | None,
    name: str,
    *,
    gt: float | None = None,
    ge: float | None = None,
    le: float | None = None,
) -> None:
    if sv is None:
        return
    v = sv.value
    if gt is not None and not v > gt:
        raise ValueError(f"{name} must be > {gt}, got {v}")
    if ge is not None and not v >= ge:
        raise ValueError(f"{name} must be >= {ge}, got {v}")
    if le is not None and not v <= le:
        raise ValueError(f"{name} must be <= {le}, got {v}")


class SizeSpec(BaseModel):
    """Physical envelope of the product as sold."""

    dimensions_mm: SourcedDimensions | None = None
    volume_cm3: SourcedValue | None = Field(
        None,
        description="Only for a non-box shape. Defaults to the dimensions' L x W x H",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _positive(self) -> SizeSpec:
        _check_range(self.volume_cm3, "volume_cm3", gt=0)
        return self


class WeightSpec(BaseModel):
    """Mass of the product as sold, without the cooling solution."""

    mass_g: SourcedValue | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _positive(self) -> WeightSpec:
        _check_range(self.mass_g, "mass_g", gt=0)
        return self


class InputPowerSpec(BaseModel):
    """How the product is powered. The per-profile draw stays in
    ``ComputeProduct.power.thermal_profiles``."""

    input_voltage_v: tuple[float, float] | None = Field(
        None, description="Accepted input range [min, max] in volts"
    )
    conversion_efficiency: SourcedValue | None = Field(
        None,
        description=(
            "Efficiency (0, 1] of a converter between the vehicle rail and the "
            "product, when that converter is outside the product. None = none modeled"
        ),
    )
    battery_compatible: bool | None = None
    poe_support: bool | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _ranges(self) -> InputPowerSpec:
        if self.input_voltage_v is not None:
            lo, hi = self.input_voltage_v
            if not 0 < lo <= hi:
                raise ValueError(f"input_voltage_v must satisfy 0 < min <= max, got {lo}, {hi}")
        _check_range(self.conversion_efficiency, "conversion_efficiency", gt=0, le=1)
        return self


class CostSpec(BaseModel):
    """Variable unit cost (RFC D9): quantity 1 and 1K only, never NRE."""

    unit_price_1_usd: SourcedValue | None = None
    unit_price_1k_usd: SourcedValue | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _non_negative(self) -> CostSpec:
        _check_range(self.unit_price_1_usd, "unit_price_1_usd", ge=0)
        _check_range(self.unit_price_1k_usd, "unit_price_1k_usd", ge=0)
        return self


class SWaPC2Spec(BaseModel):
    """The product's own SWaP-C facts, independent of operating point. The
    cooling axis is bound per thermal profile (``cooling_solution_id``)."""

    size: SizeSpec | None = None
    weight: WeightSpec | None = None
    power: InputPowerSpec | None = None
    cost: CostSpec | None = None

    model_config = {"extra": "forbid"}

    @property
    def volume_cm3(self) -> SourcedValue | None:
        """Stated volume, else derived from the dimensions."""
        if self.size is None:
            return None
        if self.size.volume_cm3 is not None:
            return self.size.volume_cm3
        d = self.size.dimensions_mm
        if d is None:
            return None
        return SourcedValue(
            value=d.volume_cm3,
            basis=ValueBasis.DERIVED if d.basis != ValueBasis.ESTIMATED else d.basis,
            confidence=d.confidence,
            source="size.dimensions_mm",
        )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

# Strongest first. A combined value keeps the weakest confidence of its inputs.
_CONFIDENCE_ORDER = [
    DataConfidence.CALIBRATED,
    DataConfidence.INTERPOLATED,
    DataConfidence.THEORETICAL,
    DataConfidence.UNKNOWN,
]


def _weakest(confidences: list[DataConfidence]) -> DataConfidence:
    return max(confidences, key=_CONFIDENCE_ORDER.index)


def _provenance(
    inputs: list[SourcedValue | SourcedDimensions],
) -> tuple[ValueBasis, DataConfidence]:
    """Basis and confidence of a value computed from ``inputs``: ``estimated``
    if any input was estimated, else ``derived``, at the weakest confidence."""
    estimated = any(i.basis == ValueBasis.ESTIMATED for i in inputs)
    basis = ValueBasis.ESTIMATED if estimated else ValueBasis.DERIVED
    return basis, _weakest([i.confidence for i in inputs])


def combine(
    value: float, inputs: list[SourcedValue | SourcedDimensions], source: str
) -> SourcedValue:
    """A value computed from ``inputs``, with their combined provenance."""
    basis, confidence = _provenance(inputs)
    return SourcedValue(value=value, basis=basis, confidence=confidence, source=source)


class ResolvedSWaPC2(BaseModel):
    """SWaP-C² of a product at one thermal profile, cooling included.

    An axis that cannot be resolved is None, and ``unresolved`` says why.
    """

    product_id: str
    thermal_profile: str
    cooling_solution_id: str
    power_w: SourcedValue | None = Field(None, description="Draw at the vehicle input rail")
    mass_g: SourcedValue | None = None
    envelope: SourcedDimensions | None = Field(
        None, description="Product + cooling envelope (cooling on the top face)"
    )
    envelope_cm3: SourcedValue | None = None
    unit_cost_1_usd: SourcedValue | None = None
    unit_cost_1k_usd: SourcedValue | None = None
    unresolved: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


def _cooling_value(
    cooling: CoolingSolutionEntry, value: float | None, what: str
) -> SourcedValue | None:
    if value is None:
        return None
    return SourcedValue(
        value=value,
        basis=cooling.basis,
        confidence=cooling.confidence,
        source=f"cooling-solutions/{cooling.id}: {what}",
    )


def _contained_sum(
    product: ComputeProduct,
    products: Mapping[str, ComputeProduct] | None,
    getter,
    what: str,
) -> tuple[SourcedValue | None, str | None]:
    """D5 roll-up of one additive axis over ``product.contains``."""
    if not product.contains:
        return None, None
    if products is None:
        return None, f"{what}: not stated, and contains cannot be rolled up without products"
    total, inputs = 0.0, []
    for ref in product.contains:
        child = products.get(ref.id)
        if child is None:
            return None, f"{what}: contained product {ref.id!r} is not in products"
        sv = getter(child)
        if sv is None:
            return None, f"{what}: contained product {ref.id!r} states no {what}"
        total += ref.count * sv.value
        inputs.append(sv)
    rolled = combine(total, inputs, f"sum over contains ({what})")
    rolled = rolled.model_copy(
        update={
            "basis": ValueBasis.ESTIMATED,
            "notes": "D5 roll-up; excludes the product's own PCB / enclosure",
        }
    )
    return rolled, None


def _mass(p: ComputeProduct) -> SourcedValue | None:
    s = p.swapc2
    return s.weight.mass_g if s and s.weight else None


def _price(qty: int):
    def get(p: ComputeProduct) -> SourcedValue | None:
        c = p.swapc2.cost if p.swapc2 else None
        if c is None:
            return None
        return c.unit_price_1_usd if qty == 1 else c.unit_price_1k_usd

    return get


def resolve_swapc2(
    product: ComputeProduct,
    profile: str | None = None,
    cooling: Mapping[str, CoolingSolutionEntry] | None = None,
    products: Mapping[str, ComputeProduct] | None = None,
) -> ResolvedSWaPC2:
    """Resolve the SWaP-C² of ``product`` at thermal ``profile``.

    Args:
        product: The product.
        profile: Thermal-profile name. Defaults to the product's default.
        cooling: Cooling catalog by id. Defaults to ``load_cooling_solutions()``.
        products: Product catalog by id, for the D5 mass / cost roll-up over
            ``contains``. Without it, a product that states no mass or cost of
            its own leaves that axis unresolved.

    Raises:
        KeyError: ``profile`` is not one of the product's thermal profiles, or
            its ``cooling_solution_id`` is not in ``cooling``.
    """
    if cooling is None:
        from embodied_schemas.loaders import load_cooling_solutions

        cooling = load_cooling_solutions()
    name = profile or product.power.default_thermal_profile
    tp = next((p for p in product.power.thermal_profiles if p.name == name), None)
    if tp is None:
        names = [p.name for p in product.power.thermal_profiles]
        raise KeyError(f"{product.id}: no thermal profile {name!r} (available: {names})")
    if tp.cooling_solution_id not in cooling:
        raise KeyError(
            f"{product.id}: profile {name!r} binds unknown cooling solution "
            f"{tp.cooling_solution_id!r}"
        )
    cs = cooling[tp.cooling_solution_id]
    watts = tp.tdp_watts
    spec = product.swapc2
    unresolved: list[str] = []
    warnings: list[str] = []

    if watts > cs.max_total_w:
        warnings.append(
            f"profile TDP {watts} W exceeds cooling {cs.id!r} max_total_w {cs.max_total_w} W"
        )

    # Power at the vehicle input rail.
    tdp = SourcedValue(
        value=watts,
        basis=ValueBasis.DERIVED,
        confidence=product.confidence,
        source=f"{product.id}: power.thermal_profiles[{name}].tdp_watts",
    )
    eff = spec.power.conversion_efficiency if spec and spec.power else None
    parasitic = _cooling_value(cs, cs.parasitic_power_w, "parasitic_power_w")
    if parasitic is None and cs.is_active:
        warnings.append(f"active cooling {cs.id!r} states no parasitic_power_w; counted as 0 W")
    power_inputs = [tdp] + [v for v in (eff, parasitic) if v is not None]
    power_value = watts / (eff.value if eff else 1.0) + (parasitic.value if parasitic else 0.0)
    power_w = combine(power_value, power_inputs, "tdp / conversion_efficiency + parasitic")

    # Mass.
    product_mass = _mass(product)
    if product_mass is None:
        product_mass, why = _contained_sum(product, products, _mass, "mass_g")
        if why:
            unresolved.append(why)
        elif product_mass is None:
            unresolved.append("mass_g: product states no swapc2.weight.mass_g")
    cooling_mass = _cooling_value(cs, cs.mass_g_at(watts), f"mass at {watts} W")
    if cooling_mass is None:
        unresolved.append(f"mass_g: cooling {cs.id!r} states no mass")
    mass_g = None
    if product_mass is not None and cooling_mass is not None:
        mass_g = combine(
            product_mass.value + cooling_mass.value,
            [product_mass, cooling_mass],
            "product mass + cooling mass",
        )

    # Envelope: cooling on the product's top face.
    envelope = envelope_cm3 = None
    dims = spec.size.dimensions_mm if spec and spec.size else None
    if dims is None:
        unresolved.append("envelope: product states no swapc2.size.dimensions_mm")
    else:
        length, width = dims.length_mm, dims.width_mm
        cooling_height = None
        if cs.dimensions_mm is not None:
            length = max(length, cs.dimensions_mm[0])
            width = max(width, cs.dimensions_mm[1])
            cooling_height = cs.dimensions_mm[2]
        else:
            vol = cs.volume_cm3_at(watts)
            if vol is not None:
                cooling_height = vol * 1000.0 / dims.footprint_mm2
        if cooling_height is None:
            unresolved.append(f"envelope: cooling {cs.id!r} states no size")
        else:
            cooling_size = _cooling_value(cs, cooling_height, f"height at {watts} W")
            basis, confidence = _provenance([dims, cooling_size])
            envelope = SourcedDimensions(
                length_mm=length,
                width_mm=width,
                height_mm=dims.height_mm + cooling_height,
                basis=basis,
                confidence=confidence,
                source="product dimensions + cooling on top face",
            )
            envelope_cm3 = combine(envelope.volume_cm3, [envelope], "envelope L x W x H")

    # Variable unit cost (D9).
    cooling_cost = _cooling_value(cs, cs.cost_usd_at(watts), f"cost at {watts} W")
    if cooling_cost is None:
        unresolved.append(f"unit_cost: cooling {cs.id!r} states no cost")
    costs: dict[int, SourcedValue | None] = {}
    for qty in (1, 1000):
        label = "unit_price_1_usd" if qty == 1 else "unit_price_1k_usd"
        price = _price(qty)(product)
        if price is None:
            price, why = _contained_sum(product, products, _price(qty), label)
            if why:
                unresolved.append(why)
            elif price is None:
                unresolved.append(f"{label}: product states no swapc2.cost.{label}")
        costs[qty] = (
            combine(price.value + cooling_cost.value, [price, cooling_cost], f"{label} + cooling")
            if price is not None and cooling_cost is not None
            else None
        )

    return ResolvedSWaPC2(
        product_id=product.id,
        thermal_profile=name,
        cooling_solution_id=cs.id,
        power_w=power_w,
        mass_g=mass_g,
        envelope=envelope,
        envelope_cm3=envelope_cm3,
        unit_cost_1_usd=costs[1],
        unit_cost_1k_usd=costs[1000],
        unresolved=unresolved,
        warnings=warnings,
    )
