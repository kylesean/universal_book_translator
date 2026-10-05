"""The translation engine is lossless when honest and fail-closed when not.

The engine runs twice over the same document, once per provider:

- **echo** (a faithful provider returning the masked source unchanged): every
  segment comes back ``TRANSLATED`` with an empty flag set and a target equal
  to the original element text -- masking + restore is lossless;
- **vandal** (drops every ``⟦...⟧`` protected span): every segment that *had* a
  placeholder must come back ``BLOCKED`` with a flag; segments with nothing to
  protect stay ``TRANSLATED``. No corrupted span may ever be marked translated.

Parse-only, no LLM: the provider is a two-line function.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

import pytest

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.email_masker import EmailMasker
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker
from ubt.model.ast import Document, ElementT, Paragraph, Region, RegionKind
from ubt.model.segment import Segment, SegmentState
from ubt.segment.placeholders import PlaceholderEngine
from ubt.segment.xliff import xml_safe
from ubt.translate.engine import TranslationEngine

pytestmark = pytest.mark.fast

_TOKEN_RE = re.compile(r"⟦[^⟧]*⟧")

_ORIGINALS: tuple[tuple[str, str, bool], ...] = (
    ("p_code", "Run `make all` before you commit.", True),
    ("p_math", "Euler wrote $e^{i\\pi} + 1 = 0$ linking five constants.", True),
    ("p_cite", "As shown in [12], the score reached 98.45%.", True),
    ("p_mixed", "The call `f(x)` per [3, 7] solves $z \\in \\mathbb{R}$.", True),
    ("p_plain", "Plain prose with nothing to protect at all.", False),
)

TranslateFn = Callable[[str], Awaitable[str]]


def _document() -> Document:
    elements: tuple[ElementT, ...] = tuple(
        Paragraph(id=element_id, spine_index=index, span=None, text=text)  # type: ignore[arg-type]
        for index, (element_id, text, _) in enumerate(_ORIGINALS)
    )
    region = Region(id="r", kind=RegionKind.BODY, elements=elements)
    return Document(source=None, regions=(region,))  # type: ignore[arg-type]


def _engine() -> TranslationEngine:
    placeholders = PlaceholderEngine(
        email=EmailMasker(),
        code=CodeMasker(),
        math=MathMasker(),
        soup=SoupMathMasker(),
        citation=CitationMasker(),
    )
    return TranslationEngine(placeholders=placeholders, model="echo-test")


async def _echo(text: str) -> str:
    return text


async def _vandal(text: str) -> str:
    return _TOKEN_RE.sub("", text)


def _original_by_id(segment_id: str) -> str:
    return {element_id: text for element_id, text, _ in _ORIGINALS}[segment_id]


def _by_id(segments: list[Segment]) -> dict[str, Segment]:
    return {segment.id: segment for segment in segments}


async def test_an_honest_provider_is_lossless_end_to_end() -> None:
    segments = _by_id(await _engine().translate_document(_document(), _echo))

    assert set(segments) == {element_id for element_id, _, _ in _ORIGINALS}
    for element_id, text, has_protection in _ORIGINALS:
        segment = segments[element_id]
        assert segment.state is SegmentState.TRANSLATED, element_id
        assert not segment.qa or not segment.qa.flags, element_id
        assert segment.target == xml_safe(text), element_id
        # The mask pass is recorded on the segment either way.
        assert bool(segment.placeholders) is has_protection, element_id


async def test_a_vandal_provider_is_fail_closed() -> None:
    segments = _by_id(await _engine().translate_document(_document(), _vandal))

    for element_id, _, has_protection in _ORIGINALS:
        segment = segments[element_id]
        if has_protection:
            # A dropped span may never ship as a translation.
            assert segment.state is SegmentState.BLOCKED, element_id
            assert segment.qa is not None and segment.qa.flags, element_id
        else:
            assert segment.state is SegmentState.TRANSLATED, element_id
