"""The translation engine: the one owner of the per-unit transform (ADR-0001 Phase 2).

A translation unit is protected text, a draft, and the restored result. Every
path that produces a draft -- the standalone reader pipeline and the draft
stage's router/batch orchestration -- must do the same three things in the same
order:

    mask (protect spans) -> [generate] -> restore (verify) -> judge

This module owns the mask, the restore and the judgement. "Generate" is the
caller's: a ``translate`` coroutine here, or the draft stage's router. The engine
never sees a provider, and it never sees an ``IRBlock`` -- it works on text, so
the draft stage can keep its own orchestration without restating the rule.

The judgement is the point: a restore that is not clean means the model dropped,
renumbered, reordered or duplicated a protected span. That is a defect, not a
translation, so :meth:`resolve` reports it and :meth:`translate_text` refuses to
mark such a segment ``TRANSLATED`` -- it is ``BLOCKED`` with the evidence
attached. Fail closed, or a silently corrupted formula ships.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ubt.core.cleaners.mask_tokens import UnmaskReport
from ubt.model.ast import Document, TextElement
from ubt.model.segment import QA, Provenance, Segment, SegmentState
from ubt.segment.placeholders import MaskedSource, PlaceholderEngine
from ubt.segment.xliff import xml_safe

#: A masked source in, a raw draft out. The engine never sees the provider.
TranslateFn = Callable[[str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class RestoreResult:
    """One draft's restored text and the protected spans it failed to keep.

    ``dirty`` is the non-clean families in :class:`~ubt.segment.placeholders.RestoreOutcome`'s
    reporting order, so a caller that formats the evidence (the draft stage) and
    one that only needs the labels (this module) read the same judgement.
    """

    text: str
    dirty: tuple[tuple[str, UnmaskReport], ...] = ()

    @property
    def clean(self) -> bool:
        """True when every protected span survived the round trip."""
        return not self.dirty


@dataclass(frozen=True, slots=True)
class TranslationEngine:
    """Mask, restore and judge translation units against one placeholder engine."""

    placeholders: PlaceholderEngine
    model: str = ""
    prompt_version: str = ""

    def mask(self, text: str) -> MaskedSource:
        """Protect a source's spans, in the engine's fixed mask order."""
        return self.placeholders.mask(text)

    def resolve(self, raw: str, masked: MaskedSource) -> RestoreResult:
        """Restore one draft and judge it against the source it came from."""
        outcome = self.placeholders.restore(raw, masked)
        return RestoreResult(
            text=outcome.text,
            dirty=tuple((label, report) for label, report in outcome.reports if not report.clean),
        )

    def _provenance(self) -> Provenance:
        return Provenance(source="mt", model=self.model, prompt_version=self.prompt_version)

    async def translate_text(self, element_id: str, text: str, translate: TranslateFn) -> Segment:
        """Translate one unit end to end, verifying its protected spans."""
        masked = self.mask(xml_safe(text))
        raw = await translate(masked.text)
        result = self.resolve(raw, masked)
        return Segment(
            id=element_id,
            source=masked.text,
            placeholders=masked.placeholders,
            target=result.text if result.clean else None,
            state=SegmentState.TRANSLATED if result.clean else SegmentState.BLOCKED,
            provenance=self._provenance(),
            qa=QA(flags=tuple(label for label, _ in result.dirty)),
        )

    async def translate_document(self, document: Document, translate: TranslateFn) -> list[Segment]:
        """Translate every translatable element of a document, in reading order."""
        results: list[Segment] = []
        for element in document.elements:
            if not isinstance(element, TextElement) or not element.text.strip():
                continue
            results.append(await self.translate_text(element.id, element.text, translate))
        return results


__all__ = ["RestoreResult", "TranslateFn", "TranslationEngine"]
