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

import contextlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ubt.core.cleaners.mask_tokens import UnmaskReport
from ubt.model.ast import Document, TextElement
from ubt.model.segment import QA, Provenance, Segment, SegmentState
from ubt.segment.placeholders import MaskedSource, PlaceholderEngine
from ubt.segment.xliff import xml_safe

if TYPE_CHECKING:
    from ubt.cache.store import CacheStore

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
    #: Content-addressed cache for the translate step (ADR-0001 Phase 4). The
    #: key is the masked source plus the model and prompt version, so a changed
    #: prompt or model cannot reuse an old draft; ``None`` calls straight through.
    cache: CacheStore | None = None

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

    async def _draft(self, masked_source: str, translate: TranslateFn) -> str:
        """Generate the raw draft, through the content cache when one is set.

        Fail-open: any cache problem (an unavailable store, a corrupt entry)
        falls back to the provider, because a cache must never break a
        translation.
        """
        if self.cache is None:
            return await translate(masked_source)
        from ubt.cache.store import step_key

        key = step_key("translate", [self.model, self.prompt_version, masked_source], {})
        cached = self._cached_draft(key)
        if cached is not None:
            return cached
        raw = await translate(masked_source)
        # A JSON envelope carrying the key makes a truncated, hand-edited or
        # split-brain entry a miss rather than a silently wrong draft.
        envelope = json.dumps({"key": key, "text": raw}, ensure_ascii=False)
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            self.cache.put(key, envelope)
        return raw

    def _cached_draft(self, key: str) -> str | None:
        """The cached draft for ``key``, or ``None`` on any miss/corruption."""
        try:
            entry = self.cache.get(key) if self.cache is not None else None
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if entry is None:
            return None
        try:
            data = json.loads(entry)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or data.get("key") != key:
            return None
        text = data.get("text")
        return text if isinstance(text, str) else None

    async def translate_text(self, element_id: str, text: str, translate: TranslateFn) -> Segment:
        """Translate one unit end to end, verifying its protected spans."""
        masked = self.mask(xml_safe(text))
        raw = await self._draft(masked.text, translate)
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
