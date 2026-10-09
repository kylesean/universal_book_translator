"""``ubt.translate`` -- the translation-unit orchestration layer (translation unit segmentation layer).

:class:`~ubt.translate.engine.TranslationEngine` masks a unit's protected spans,
sends the masked source to a provider-agnostic ``translate`` coroutine, restores
under checksum verification, and marks the segment only when the restore was
clean. The draft stage drives it.
"""

from __future__ import annotations

from ubt.translate.engine import RestoreResult, TranslateFn, TranslationEngine
from ubt.translate.spi import register_spi_providers

__all__ = ["RestoreResult", "TranslateFn", "TranslationEngine", "register_spi_providers"]
