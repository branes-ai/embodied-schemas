"""Generate the NVIDIA Jetson SKU products (RFC 0001 S3e).

SKUs of a family (NVIDIA Jetson Orin, NVIDIA Jetson Thor) share the silicon
and differ by floorsweeping (units enabled) and memory configuration. Each
family has one hand-authored template, ``scripts/jetson_templates/<family>.yaml``,
which is its flagship SKU. Every SKU -- flagship included -- is the template
plus that SKU's figures from the source database
(``data/sources/observations/jetson_modules.yaml``):

All SKUs:
  ``kind: module``; the ``sku`` section (NVIDIA name, part number, floorsweep);
  module SWaP-C² (dimensions, mass, current 1KU price); a memory summary;
  ``market.launch_msrp_usd`` / ``launch_date`` from the launch-price record.

Siblings only (the flagship keeps its calibrated values):
  ``num_sms`` from the floorsweep; memory capacity, bus width and bandwidth
  (memory controllers scale with the bus); die boost clock = NVIDIA's max GPU
  clock; one thermal profile at that clock and the module's maximum power --
  NVIDIA publishes each mode's power but not its clock, so lower modes are
  left out rather than guessed; peak performance from the template's own
  fabric arithmetic at that clock.

Run:
    python scripts/generate_jetson_skus.py --check   # fail if a product YAML differs
    python scripts/generate_jetson_skus.py --write   # rewrite them
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from embodied_schemas.compute_product import ComputeProduct
from embodied_schemas.loaders import load_and_validate
from embodied_schemas.sources import SourceDB, load_source_db

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "jetson_templates"
OUT_DIR = REPO_ROOT / "src" / "embodied_schemas" / "data" / "compute_products" / "nvidia"
GENERATOR = "generate_jetson_skus.py"
LAST_UPDATED = "2026-10-07"


@dataclass(frozen=True)
class Family:
    template: str  # file in TEMPLATE_DIR
    arch: str  # source-DB subject for architecture statements
    # False when the vendor withdrew its Tensor Core counts (Thor: DS v1.4);
    # the floorsweep then omits tensor_core rather than state a withdrawn figure.
    tensor_counts_published: bool = True


@dataclass(frozen=True)
class SKU:
    catalog_id: str
    family: str
    subject: str  # source-DB subject
    nvidia_name: str  # NVIDIA SKU name
    part_number: str  # NVIDIA FAQ (nvidia_jetson_faq)
    model_tier: str
    flagship: bool = False


FAMILIES = {
    "orin": Family("nvidia_jetson_orin.yaml", "nvidia_orin_gpu"),
    "thor": Family("nvidia_jetson_thor.yaml", "nvidia_thor_gpu", tensor_counts_published=False),
}

# Part numbers: NVIDIA Jetson FAQ, https://developer.nvidia.com/embedded/faq (2026-10-04).
SKUS = [
    SKU(
        "nvidia_jetson_agx_orin_64gb",
        "orin",
        "nvidia_jetson_agx_orin_64gb",
        "Jetson AGX Orin 64GB",
        "900-13701-0050-000",
        "high",
        flagship=True,
    ),
    SKU(
        "nvidia_jetson_agx_orin_32gb",
        "orin",
        "nvidia_jetson_agx_orin_32gb",
        "Jetson AGX Orin 32GB",
        "900-13701-0040-000",
        "high",
    ),
    SKU(
        "nvidia_jetson_agx_orin_industrial",
        "orin",
        "nvidia_jetson_agx_orin_industrial",
        "Jetson AGX Orin Industrial",
        "900-13701-0080-000",
        "high",
    ),
    SKU(
        "nvidia_jetson_orin_nx_16gb",
        "orin",
        "nvidia_jetson_orin_nx_16gb",
        "Jetson Orin NX 16GB",
        "900-13767-0000-001",
        "mid",
    ),
    SKU(
        "nvidia_jetson_orin_nx_8gb",
        "orin",
        "nvidia_jetson_orin_nx_8gb",
        "Jetson Orin NX 8GB",
        "900-13767-0010-001",
        "mid",
    ),
    SKU(
        "nvidia_jetson_orin_nano_8gb",
        "orin",
        "nvidia_jetson_orin_nano_8gb",
        "Jetson Orin Nano 8GB",
        "900-13767-0030-000",
        "entry",
    ),
    SKU(
        "nvidia_jetson_orin_nano_4gb",
        "orin",
        "nvidia_jetson_orin_nano_4gb",
        "Jetson Orin Nano 4GB",
        "900-13767-0040-000",
        "entry",
    ),
    SKU(
        "nvidia_jetson_agx_thor_128gb",
        "thor",
        "nvidia_jetson_t5000",
        "Jetson T5000",
        "900-13834-0080-001",
        "high",
        flagship=True,
    ),
    SKU(
        "nvidia_jetson_t4000",
        "thor",
        "nvidia_jetson_t4000",
        "Jetson T4000",
        "900-13834-0000-001",
        "high",
    ),
]


# ---------------------------------------------------------------------------
# Source-DB access
# ---------------------------------------------------------------------------


def _obs(db: SourceDB, subject: str, quantity: str, variant: str | None = None, **cond):
    hits = db.find(quantity, subject, variant=variant, **cond)
    return hits[0] if hits else None


def _val(db: SourceDB, subject: str, quantity: str, variant: str | None = None, **cond):
    o = _obs(db, subject, quantity, variant, **cond)
    return None if o is None else o.value


def _require(db: SourceDB, subject: str, quantity: str, variant: str | None = None, **cond):
    o = _obs(db, subject, quantity, variant, **cond)
    if o is None:
        raise ValueError(
            f"{subject}: no {quantity}{'.' + variant if variant else ''} in the source DB"
        )
    return o


def fabric_peaks(gpu: dict, num_sms: int, clock_mhz: float) -> dict[str, float]:
    """The template's performance arithmetic: per precision, the sum over
    compute fabrics of units x ops/unit/clock x clock, in T-ops/s. BF16 uses
    the FP16 rate where the fabric lists no BF16 rate."""
    hz = clock_mhz * 1e6

    def total(prec: str) -> float:
        out = 0.0
        for fab in gpu["compute_fabrics"]:
            ops = fab["ops_per_unit_per_clock"]
            rate = ops.get(prec, ops.get("fp16", 0) if prec == "bf16" else 0)
            out += fab["units_per_sm"] * num_sms * rate * hz
        return out / 1e12

    return {
        "int8_tops": round(total("int8"), 2),
        "bf16_tflops": round(total("bf16"), 2),
        "fp32_tflops": round(total("fp32"), 2),
    }


def _cooling_for(watts: float) -> str:
    """Catalog cooling class for a module profile, by power."""
    if watts <= 15:
        return "passive_heatsink_small"
    if watts <= 30:
        return "passive_heatsink_large"
    return "active_fan"


# ---------------------------------------------------------------------------
# One SKU
# ---------------------------------------------------------------------------


def build(sku: SKU, db: SourceDB) -> dict:
    """The product dict for ``sku``: its family template plus its sourced figures."""
    family = FAMILIES[sku.family]
    template = yaml.safe_load((TEMPLATE_DIR / family.template).read_text(encoding="utf-8"))
    t = copy.deepcopy(template)
    s = sku.subject
    die = t["dies"][0]
    gpu = die["blocks"][0]

    # --- floorsweep -----------------------------------------------------------
    cuda = _require(db, s, "cuda_cores")
    tpc = _obs(db, s, "tpc")
    per_sm = gpu["cuda_cores_per_sm"]
    if cuda.value % per_sm:
        raise ValueError(f"{s}: {cuda.value:g} CUDA cores is not a multiple of {per_sm} per SM")
    num_sms = int(cuda.value // per_sm)
    sm_source = f"{cuda.key} / template cuda_cores_per_sm {per_sm}"
    if tpc is not None:
        sm_per_tpc = db.find("sm_per_tpc", family.arch)[0]
        if tpc.value * sm_per_tpc.value != num_sms:
            raise ValueError(f"{s}: TPC x SMs/TPC != CUDA cores / {per_sm}")
        sm_source = f"{tpc.key} x {sm_per_tpc.key}"
    floorsweep = [
        {"unit": "gpu_sm", "enabled": num_sms, "source": sm_source},
        {"unit": "cuda_core", "enabled": int(cuda.value), "source": cuda.key},
    ]
    tensor = _obs(db, s, "tensor_cores") if family.tensor_counts_published else None
    if tensor is not None:
        floorsweep.append(
            {"unit": "tensor_core", "enabled": int(tensor.value), "source": tensor.key}
        )
    cpu = _require(db, s, "cpu_cores")
    floorsweep.append({"unit": "cpu_core", "enabled": int(cpu.value), "source": cpu.key})

    # --- memory -----------------------------------------------------------------
    mem = gpu["memory"]
    cap = _require(db, s, "memory_capacity")
    bus = _require(db, s, "memory_bus_width")
    bw = _require(db, s, "memory_bandwidth", variant=None)
    if not sku.flagship:
        gpu["num_sms"] = num_sms
        mem["memory_controllers"] = max(
            1, round(mem["memory_controllers"] * bus.value / mem["memory_bus_bits"])
        )
        mem["memory_size_gb"] = cap.value
        mem["memory_bus_bits"] = int(bus.value)
        mem["memory_bandwidth_gbps"] = bw.value
        ratio = num_sms / template["dies"][0]["blocks"][0]["num_sms"]
        gpu["noc"]["unit_count"] = max(1, round(gpu["noc"]["unit_count"] * ratio))
        gpu["noc"]["bisection_bandwidth_gbps"] = round(
            gpu["noc"]["bisection_bandwidth_gbps"] * ratio, 1
        )

    # --- power and performance (siblings) ---------------------------------------
    if not sku.flagship:
        super_clock = _obs(db, s, "frequency", "gpu_max_super")
        clock = super_clock or _require(db, s, "frequency", "gpu_max")
        pmax = _require(db, s, "power", "max")
        modes = [o.value for o in db.find("power", s) if (o.variant or "").startswith("mode_")]
        if super_clock is not None:
            name = "MAXN_SUPER"
        elif modes and max(modes) == pmax.value:
            name = f"{pmax.value:g}W"
        else:
            name = "MAXN"
        die["clocks"]["boost_clock_mhz"] = clock.value
        die["clocks"]["base_clock_mhz"] = min(die["clocks"]["base_clock_mhz"], clock.value)
        t["power"] = {
            "tdp_watts": pmax.value,
            "max_power_watts": pmax.value,
            "min_power_watts": min(modes) if modes else pmax.value,
            "default_thermal_profile": name,
            "thermal_profiles": [
                {
                    "name": name,
                    "tdp_watts": pmax.value,
                    "clock_mhz": clock.value,
                    "cooling_solution_id": _cooling_for(pmax.value),
                }
            ],
        }
        t["performance"] = fabric_peaks(gpu, num_sms, clock.value)

    # --- identity and market -----------------------------------------------------
    t["id"] = sku.catalog_id
    if not sku.flagship:
        t["name"] = f"NVIDIA {sku.nvidia_name}"
    t["kind"] = "module"
    t["packaging"] = {"kind": "board", "num_dies": 1, "form_factor": "som"}
    market = t["market"]
    market["model_tier"] = sku.model_tier
    launch = _obs(db, s, "unit_price", "qty_1000", listing="launch")
    if launch is not None:
        market["launch_msrp_usd"] = launch.value
        market["launch_date"] = db.documents[launch.source_id].published
    elif not sku.flagship:
        market.pop("launch_msrp_usd", None)
        market.pop("launch_date", None)

    # --- SWaP-C² ----------------------------------------------------------------
    swapc2: dict = {}
    dims = [_obs(db, s, q) for q in ("length", "width", "height")]
    if all(dims):
        swapc2["size"] = {
            "dimensions_mm": {
                "length_mm": dims[0].value,
                "width_mm": dims[1].value,
                "height_mm": dims[2].value,
                "basis": "datasheet",
                "confidence": "calibrated",
                "source": ", ".join(d.key for d in dims),
            }
        }
    mass = _obs(db, s, "mass")
    if mass is not None:
        swapc2["weight"] = {
            "mass_g": {
                "value": mass.value,
                "basis": "datasheet",
                "confidence": "calibrated",
                "source": mass.key,
            }
        }
    price = _require(db, s, "unit_price", "qty_1000", listing="current")
    swapc2["cost"] = {
        "unit_price_1k_usd": {
            "value": price.value,
            "basis": "datasheet",
            "confidence": "calibrated",
            "source": price.key,
            "notes": f"NVIDIA 1KU list price as of {price.as_of}",
        }
    }

    out = {
        "id": t["id"],
        "name": t["name"],
        "vendor": t["vendor"],
        "kind": t["kind"],
        "packaging": t["packaging"],
        "lifecycle": t.get("lifecycle", "production"),
        "sku": {"name": sku.nvidia_name, "part_number": sku.part_number, "floorsweep": floorsweep},
        "dies": t["dies"],
        "performance": t["performance"],
        "power": t["power"],
        "market": market,
        "memory": {
            "memory_gb": cap.value,
            "memory_type": mem["memory_type"].upper(),
            "memory_bandwidth_gbps": bw.value,
        },
        "swapc2": swapc2,
        "confidence": t["confidence"],
        "notes": (
            t["notes"]
            if sku.flagship
            else (
                f"{sku.nvidia_name}, a SKU of the {market['product_family']} family: the family "
                f"silicon (template {family.template}) floorswept to {num_sms} SMs / "
                f"{int(cuda.value)} CUDA cores, with {cap.value:g} GB "
                f"{mem['memory_type'].upper()}. "
                "One thermal profile: NVIDIA's max GPU clock at the module's maximum power. NVIDIA "
                "publishes each power mode's watts but not its clock, so lower modes are omitted "
                "rather than guessed. Peaks are the template's fabric arithmetic at that clock "
                "(GPU-only, dense). Physical-die estimates (silicon_bin) are the family's."
            )
        ),
        "datasheet_url": db.documents[cuda.source_id].url,
        "last_updated": LAST_UPDATED if not sku.flagship else t["last_updated"],
    }
    return out


def render(sku: SKU, data: dict) -> str:
    header = (
        f"# GENERATED by scripts/{GENERATOR} -- do not edit.\n"
        f"# Family template: scripts/jetson_templates/{FAMILIES[sku.family].template}\n"
        f"# SKU figures: data/sources/observations/jetson_modules.yaml "
        f"(subject {sku.subject})\n\n"
    )
    return header + yaml.safe_dump(data, sort_keys=False, width=100, allow_unicode=True)


def out_path(sku: SKU) -> Path:
    return OUT_DIR / f"{sku.catalog_id.removeprefix('nvidia_')}.yaml"


def run(write: bool) -> int:
    db = load_source_db()
    status = 0
    for sku in SKUS:
        path = out_path(sku)
        new = render(sku, build(sku, db))
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if new == old:
            continue
        rel = path.relative_to(REPO_ROOT)
        if write:
            path.write_text(new, encoding="utf-8")
            load_and_validate(path, ComputeProduct)  # raises if the product is invalid
            print(f"wrote {rel}")
        else:
            print(f"FAIL: {rel} differs from {GENERATOR}")
            status = 1
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    return run(write=parser.parse_args(argv).write)


if __name__ == "__main__":
    sys.exit(main())
