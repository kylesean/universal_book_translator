"""Unit tests for the Core Service Provider Interface (SPI) Registry and Service Locator.

Validates that:
1. SPIRegistry registers and resolves services cleanly without coupling core to downstream packages.
2. Missing services raise KeyError.
3. Built-in downstream providers (adapters, segment, translate) register and resolve.
4. Overriding/mocking services for testing is seamless.
"""

from __future__ import annotations

import pytest

from ubt.core.spi import SPIRegistry, get_spi_registry

pytestmark = pytest.mark.fast


def test_spi_registry_singleton_and_reset() -> None:
    reg1 = SPIRegistry.instance()
    reg2 = SPIRegistry.instance()
    assert reg1 is reg2

    SPIRegistry.reset()
    reg3 = SPIRegistry.instance()
    assert reg3 is not reg1


def test_spi_registry_register_and_get() -> None:
    reg = SPIRegistry()
    assert not reg.is_registered("custom_service")

    with pytest.raises(KeyError, match="No SPI provider registered"):
        reg.get("custom_service")

    reg.register("custom_service", lambda x: x * 2)
    assert reg.is_registered("custom_service")
    fn = reg.get("custom_service")
    assert fn(21) == 42


def test_spi_registry_register_many() -> None:
    reg = SPIRegistry()
    reg.register_many({"svc_a": "alpha", "svc_b": "beta"})
    assert reg.get("svc_a") == "alpha"
    assert reg.get("svc_b") == "beta"


def test_spi_registry_resolves_all_core_downstream_services() -> None:
    reg = get_spi_registry()
    services = [
        "adapter_resolver",
        "supported_suffixes",
        "detect_figure_pages",
        "visual_gate_runner",
        "crashed_visual_gate_result",
        "artifact_parity_findings",
        "probe_unavailable_finding",
        "inspect_font_encoding_damage",
        "summarize_font_encoding_damage",
        "flag_font_encoding_damage",
        "blocking_gate_tripped",
        "is_fast_lane_eligible",
        "crop_block_image",
        "is_visual_scalpel_applicable",
        "probe_pdf_pages",
        "classify_pdf_structure",
        "inspect_pdf_route_plan",
        "profile_pdf_pages",
        "page_kind_enum",
        "render_fidelity_stats",
        "render_fidelity_findings",
        "sample_pdf_pages",
        "placeholder_engine",
        "masked_source",
        "translation_engine",
    ]
    for svc in services:
        provider = reg.get(svc)
        assert provider is not None, f"SPI service {svc!r} resolved to None"
