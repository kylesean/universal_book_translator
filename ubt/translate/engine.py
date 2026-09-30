"""The translation engine: a Document -> translated segments (ADR-0001 Phase 2).

This is the orchestration the draft stage used to inline. It owns the one
correct sequence for every unit:

    mask (protect spans) -> translate the masked source -> restore (verify) -> mark

and it marks a segment as translated **only when the restore was clean**. A
non-clean restore means the model dropped, renumbered, reordered or duplicated a
protected span -- a defect, not a translation -- so the segment is ``BLOCKED``
with the evidence recorded, never shipped with a silently corrupted formula or
citation. That fail-closed rule is the whole reason the engine exists; the
masking order itself lives in :class:`~ubt.segment.placeholders.PlaceholderEngine`.

The engine is provider-agnostic: it takes a ``translate`` coroutine, so it can be
driven by the router, a batch client, or an echo in a test. Wiring it into the
draft stage is a later step; this module is the seam.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ubt.model.ast import Document, TextElement
from ubt.model.segment import QA, Provenance, Segment, SegmentState
from ubt.segment.placeholders import PlaceholderEngine
from ubt.segment.xliff import xml_safe

#: A masked source in, a raw draft out. The engine never sees the provider.
TranslateFn = Callable[[str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class TranslationEngine:
    """Mask, translate and verify segments against one placeholder engine."""

    placeholders: PlaceholderEngine
    model: str = ""
    prompt_version: str = ""

    def _provenance(self) -> Provenance:
        return Provenance(source="mt", model=self.model, prompt_version=self.prompt_version)

    async def translate_text(self, element_id: str, text: str, translate: TranslateFn) -> Segment:
        """Translate one unit end to end, verifying its protected spans."""
        masked = self.placeholders.mask(xml_safe(text))
        raw = await translate(masked.text)
        outcome = self.placeholders.restore(raw, masked)
        flags = tuple(label for label, report in outcome.reports if not report.clean)
        return Segment(
            id=element_id,
            source=masked.text,
            placeholders=masked.placeholders,
            target=outcome.text if outcome.clean else None,
            state=SegmentState.TRANSLATED if outcome.clean else SegmentState.BLOCKED,
            provenance=self._provenance(),
            qa=QA(flags=flags),
        )

    async def translate_document(self, document: Document, translate: TranslateFn) -> list[Segment]:
        """Translate every translatable element of a document, in reading order."""
        results: list[Segment] = []
        for element in document.elements:
            if not isinstance(element, TextElement) or not element.text.strip():
                continue
            results.append(await self.translate_text(element.id, element.text, translate))
        return results


__all__ = ["TranslateFn", "TranslationEngine"]
