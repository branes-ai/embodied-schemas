"""NVIDIA Jetson entries are SKUs of a product family (RFC 0001 D6, rev 0.21.0).

Product families are "NVIDIA Jetson Orin" and "NVIDIA Jetson Thor". SKUs
within a family share their silicon and differ by floorsweeping (the units
enabled) and memory configuration. Each entry is a SKU, identified by
NVIDIA's SKU name.
"""

import pytest

from embodied_schemas.loaders import load_compute_products, load_hardware

SKUS = {
    "nvidia_jetson_agx_orin_64gb": "NVIDIA Jetson Orin",
    "nvidia_jetson_agx_thor_128gb": "NVIDIA Jetson Thor",
}


@pytest.fixture(scope="module")
def products():
    return load_compute_products()


@pytest.mark.parametrize("sku,family", SKUS.items())
def test_sku_belongs_to_its_nvidia_family(products, sku, family):
    assert products[sku].market.product_family == family


@pytest.mark.parametrize("sku", SKUS)
def test_sku_id_is_the_nvidia_sku_name(products, sku):
    """The id and the legacy HardwareEntry id are the same SKU."""
    assert sku in load_hardware()
    assert products[sku].name == load_hardware()[sku].name


def test_orin_agx_64gb_floorsweep_and_memory(products):
    """The SKU's differentiators: units enabled and memory configuration."""
    gpu = products["nvidia_jetson_agx_orin_64gb"].dies[0].blocks[0]
    assert gpu.num_sms * gpu.cuda_cores_per_sm == 2048
    assert gpu.num_sms * gpu.tensor_cores_per_sm == 64
    assert (gpu.memory.memory_type.value, gpu.memory.memory_size_gb) == ("lpddr5", 64.0)


def test_no_family_left_unprefixed(products):
    for p in products.values():
        if p.vendor == "nvidia":
            assert p.market.product_family.startswith("NVIDIA Jetson "), p.id


# ---------------------------------------------------------------------------
# Thor floorsweep held to the source DB (corrected 2026-10-04)
# ---------------------------------------------------------------------------


def test_thor_t5000_cuda_cores_match_nvidia(products):
    """The SKU's enabled CUDA cores equal NVIDIA's published T5000 count,
    which both sources agree on."""
    from embodied_schemas.sources import load_source_db

    db = load_source_db()
    published = {o.value for o in db.find("cuda_cores", "nvidia_jetson_t5000")}
    assert published == {2560.0}
    gpu = products["nvidia_jetson_agx_thor_128gb"].dies[0].blocks[0]
    assert gpu.num_sms * gpu.cuda_cores_per_sm == 2560


def test_thor_tensor_core_gap_is_known(products):
    """NVIDIA states 96 Tensor cores; the per-SM integer model gives 80. This
    pins the known gap so it is revisited, not silently accepted."""
    from embodied_schemas.sources import load_source_db

    db = load_source_db()
    assert db.value("nvidia_jetson_t5000.tensor_cores@edom_jetson_t5000") == 96
    gpu = products["nvidia_jetson_agx_thor_128gb"].dies[0].blocks[0]
    assert gpu.num_sms * gpu.tensor_cores_per_sm == 80


def test_thor_peaks_follow_the_floorsweep(products):
    """GPU-only dense peaks at the default profile's clock from the enabled units."""
    thor = products["nvidia_jetson_agx_thor_128gb"]
    gpu = thor.dies[0].blocks[0]
    ghz = thor.power.default_profile.clock_mhz * 1e6
    cuda = gpu.num_sms * gpu.cuda_cores_per_sm
    tensor = gpu.num_sms * gpu.tensor_cores_per_sm
    assert thor.performance.fp32_tflops == pytest.approx(cuda * 2 * ghz / 1e12, abs=0.01)
    assert thor.performance.int8_tops == pytest.approx(
        (cuda * 2 + tensor * 64) * ghz / 1e12, abs=0.01
    )


def test_t4000_recorded(products):
    """The T4000 (no catalog entry yet) is recorded with its floorsweep and memory."""
    from embodied_schemas.sources import load_source_db

    db = load_source_db()
    src = "connecttech_jetson_t4000_t5000"
    assert db.value(f"nvidia_jetson_t4000.cuda_cores@{src}") == 1536
    assert db.value(f"nvidia_jetson_t4000.memory_capacity@{src}") == 64


def test_legacy_thor_entries_match_nvidia():
    """The legacy GPU, hardware and chip Thor entries carry the T5000 counts."""
    from embodied_schemas.loaders import load_chips, load_gpus, load_hardware

    gpu = next(g for g in load_gpus().values() if g.id.startswith("nvidia_thor_gpu"))
    assert (gpu.compute.cuda_cores, gpu.compute.tensor_cores) == (2560, 96)
    assert gpu.compute.streaming_multiprocessors * 128 == 2560
    hw = load_hardware()["nvidia_jetson_agx_thor_128gb"]
    # NVIDIA withdrew the Thor Tensor Core count (DS v1.4); the entry leaves it unset.
    assert (hw.capabilities.compute_units, hw.capabilities.tensor_cores) == (2560, None)
    assert load_chips()["nvidia_thor_soc"].gpu_cores == 2560


# ---------------------------------------------------------------------------
# SKUSpec / floorsweep (schema, S3e)
# ---------------------------------------------------------------------------


def _with_sku(product, floorsweep):
    from embodied_schemas import ComputeProduct

    data = product.model_dump(mode="json")
    data["sku"] = {"name": "test SKU", "floorsweep": floorsweep}
    return ComputeProduct.model_validate(data)


class TestFloorsweep:
    def test_consistent_floorsweep_accepted(self, products):
        p = _with_sku(
            products["nvidia_jetson_agx_orin_64gb"],
            [
                {"unit": "gpu_sm", "enabled": 16},
                {"unit": "cuda_core", "enabled": 2048},
                {"unit": "tensor_core", "enabled": 64},
                {"unit": "cpu_core", "enabled": 12},
            ],
        )
        assert p.sku.enabled("cuda_core") == 2048 and p.sku.enabled("dla") is None

    @pytest.mark.parametrize(
        "unit,value", [("gpu_sm", 14), ("cuda_core", 1792), ("tensor_core", 56)]
    )
    def test_mismatch_with_gpu_block_rejected(self, products, unit, value):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match=f"floorsweep {unit}"):
            _with_sku(products["nvidia_jetson_agx_orin_64gb"], [{"unit": unit, "enabled": value}])

    def test_physical_below_enabled_rejected(self):
        from pydantic import ValidationError

        from embodied_schemas import EnabledUnits

        with pytest.raises(ValidationError, match="physical 10 < enabled 20"):
            EnabledUnits(unit="gpu_sm", enabled=20, physical=10)

    def test_duplicate_unit_rejected(self):
        from pydantic import ValidationError

        from embodied_schemas import SKUSpec

        with pytest.raises(ValidationError, match="repeats a unit"):
            SKUSpec(
                name="x",
                floorsweep=[{"unit": "gpu_sm", "enabled": 1}, {"unit": "gpu_sm", "enabled": 2}],
            )

    def test_unset_sku_not_dumped(self, products):
        assert "sku" not in products["hailo_hailo_8"].model_dump()
