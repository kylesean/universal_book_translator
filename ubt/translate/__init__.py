"""``ubt.translate`` -- the translation-unit orchestration layer (ADR-0001 Phase 2).

:class:`~ubt.translate.engine.TranslationEngine` masks a unit's protected spans,
sends the masked source to a provider-agnostic ``translate`` coroutine, restores
under checksum verification, and marks the segment only when the restore was
clean. It is the seam the draft stage will migrate onto.
"""

from __future__ import annotations

from ubt.translate.engine import TranslateFn, TranslationEngine

__all__ = ["TranslateFn", "TranslationEngine"]
