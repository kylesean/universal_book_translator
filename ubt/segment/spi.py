"""SPI provider registrations for Segment / Placeholders compiler package.

Registers segment-side capabilities with :class:`ubt.core.spi.SPIRegistry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ubt.core.spi import SPIRegistry


def _make_masked_source(
    *,
    text: str,
    code_map: dict[str, str],
    math_map: dict[str, str],
    soup_map: dict[str, str],
    cite_map: dict[str, str],
    email_map: dict[str, str] | None = None,
) -> Any:
    from ubt.segment.placeholders import MaskedSource

    return MaskedSource(
        text=text,
        email_map=email_map or {},
        code_map=code_map,
        math_map=math_map,
        soup_map=soup_map,
        cite_map=cite_map,
    )


def register_spi_providers(registry: SPIRegistry) -> None:
    """Register segment SPI services."""
    from ubt.segment.placeholders import default_placeholder_engine

    registry.register("placeholder_engine", default_placeholder_engine)
    registry.register("masked_source", _make_masked_source)
