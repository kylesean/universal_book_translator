"""SPI provider registrations for Translation compiler package.

Registers translation-side capabilities with :class:`ubt.core.spi.SPIRegistry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ubt.core.spi import SPIRegistry


def _make_translation_engine(
    *,
    placeholders: Any,
    model: str = "",
    prompt_version: str = "",
    cache: Any = None,
) -> Any:
    from ubt.translate.engine import TranslationEngine

    return TranslationEngine(
        placeholders=placeholders,
        model=model,
        prompt_version=prompt_version,
        cache=cache,
    )


def register_spi_providers(registry: SPIRegistry) -> None:
    """Register translation SPI services."""
    registry.register("translation_engine", _make_translation_engine)
