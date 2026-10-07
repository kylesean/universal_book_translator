"""The translation engine: the one owner of the per-unit transform (translation unit segmentation layer).

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
    #: Content-addressed cache for the translate step (content-addressed cache layer). The
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

    def _cache_key(self, masked_source: str, context: str) -> str:
        """The content key for one masked source under one prompt context.

        ``context`` is the exact prompt the generate step will send. The
        standalone path's generate is a pure function of the masked source, so
        its context is empty and omitted (keeping the standalone key stable); the
        production draft path's generate also depends on glossary / neighbour /
        memory context, so that context must enter the key or a changed prompt
        would reuse a stale draft (content-addressed cache layer).
        """
        from ubt.cache.store import step_key

        parts = [self.model, self.prompt_version, masked_source]
        if context:
            parts.append(context)
        return step_key("translate", parts, {})

    def cached_draft(self, masked_source: str, *, context: str = "") -> str | None:
        """The cached raw draft for this masked source + context, or ``None``.

        For callers whose generate step is not a single coroutine of the masked
        source -- the draft stage's Batch API path, which submits many units at
        once -- so they can consult the cache before spending and record after.
        Fail-open: no store, or a corrupt entry, is a miss.
        """
        if self.cache is None:
            return None
        return self._cached_draft(self._cache_key(masked_source, context))

    def remember_draft(self, masked_source: str, raw: str, *, context: str = "") -> None:
        """Store a raw draft under its masked source + prompt context (fail-open)."""
        if self.cache is None:
            return
        # An empty (refused/filtered) draft must not be cached: it would replay
        # on every later run as a hit, and only the repair stage could ever
        # recover the block. Re-drafting is the recovery path here.
        if not raw or not raw.strip():
            return
        key = self._cache_key(masked_source, context)
        # A JSON envelope carrying the key makes a truncated, hand-edited or
        # split-brain entry a miss rather than a silently wrong draft.
        envelope = json.dumps({"key": key, "text": raw}, ensure_ascii=False)
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            self.cache.put(key, envelope)

    def _value_key(self, kind: str, context: str) -> str:
        from ubt.cache.store import step_key

        return step_key(kind, [self.model, self.prompt_version, context], {})

    def cached_value(self, context: str, *, kind: str) -> str | None:
        """The cached text for a non-draft translate step, or ``None`` (fail-open).

        A macro chunk is one provider call producing several units, so it has no
        single masked source and cannot use :meth:`cached_draft`; it keys on its
        own prompt digest under a distinct ``kind``, sharing the same store.
        """
        if self.cache is None:
            return None
        return self._cached_draft(self._value_key(kind, context))

    def remember_value(self, context: str, value: str, *, kind: str) -> None:
        """Store text for a non-draft translate step under ``kind`` (fail-open)."""
        if self.cache is None:
            return
        if not value or not value.strip():
            return
        key = self._value_key(kind, context)
        envelope = json.dumps({"key": key, "text": value}, ensure_ascii=False)
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            self.cache.put(key, envelope)

    async def draft(self, masked_source: str, translate: TranslateFn, *, context: str = "") -> str:
        """Generate the raw draft, through the content cache when one is set.

        The cache-aware generate half of a translation unit. ``context`` is the
        exact prompt the generate step will send, so a changed prompt (a new
        glossary, a new neighbour window) is a miss, not a stale draft.
        Fail-open: any cache problem (an unavailable store, a corrupt entry)
        falls back to the provider, because a cache must never break a
        translation.

        This half only *reads* the cache. It must not write: a draft is not
        cacheable until the caller has judged it (``resolve`` /
        ``finalize_draft``), and writing here cached a defective draft that then
        replayed on every later run -- a stale loss only the repair stage could
        recover. Callers record a judged-clean draft via :meth:`remember_draft`.
        """
        if self.cache is None:
            return await translate(masked_source)
        key = self._cache_key(masked_source, context)
        cached = self._cached_draft(key)
        if cached is not None:
            return cached
        return await translate(masked_source)

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
        if not isinstance(text, str) or not text.strip():
            # An empty cached draft is a refusal fossil, not a translation.
            return None
        return text

    async def translate_text(
        self, element_id: str, text: str, translate: TranslateFn, *, context: str = ""
    ) -> Segment:
        """Translate one unit end to end, verifying its protected spans.

        ``context`` is the prompt the generate step will send; the engine folds it
        into the cache key so a changed prompt is a miss (see :meth:`_cache_key`).
        """
        masked = self.mask(xml_safe(text))
        raw = await self.draft(masked.text, translate, context=context)
        result = self.resolve(raw, masked)
        # Cache only a judged-clean draft: a defective one would replay as a hit
        # on every later run, and only repair could recover the block.
        if result.clean:
            self.remember_draft(masked.text, raw, context=context)
        return Segment(
            id=element_id,
            source=masked.text,
            placeholders=masked.placeholders,
            target=result.text if result.clean else None,
            state=SegmentState.TRANSLATED if result.clean else SegmentState.BLOCKED,
            provenance=self._provenance(),
            qa=QA(flags=tuple(label for label, _ in result.dirty)),
        )


__all__ = ["RestoreResult", "TranslateFn", "TranslationEngine"]
