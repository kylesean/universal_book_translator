"""``ubt.translate`` -- the translation-unit orchestration layer (translation unit segmentation layer).

:class:`~ubt.translate.engine.TranslationEngine` masks a unit's protected spans,
sends the masked source to a provider-agnostic ``translate`` coroutine, restores
under checksum verification, and marks the segment only when the restore was
clean. It is the seam the draft stage will migrate onto.
"""

from __future__ import annotations

from ubt.translate.engine import RestoreResult, TranslateFn, TranslationEngine

__all__ = ["RestoreResult", "TranslateFn", "TranslationEngine"]
