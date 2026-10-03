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
