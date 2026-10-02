"""The XLIFF view is a lossless bijection over segments.

The ADR's Phase-2 interop gate: the kernel's units can leave and return through
the industry format without loss. A ``Segment`` is built per text with real
protected spans (the ``PlaceholderEngine`` masking code, math, soup and
citations), serialized to XLIFF 2.1, parsed back, and every modeled field must
survive: source, target, state, the (token, kind, original) of every
placeholder -- plus the document header (languages, original file name).
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker
from ubt.model.segment import Segment, SegmentState
from ubt.segment.placeholders import PlaceholderEngine
from ubt.segment.xliff import from_xliff, to_xliff, xml_safe

pytestmark = pytest.mark.fast

_STATES = tuple(SegmentState)

_TEXTS = (
    "Run `make all` before you commit.",
    "Euler wrote $e^{i\\pi} + 1 = 0$ linking five constants.",
    "As shown in [12], the score reached 98.45%.",
    "Plain prose with nothing to protect at all.",
    "Mixed: `x = f(y)` and $z \\in \\mathbb{R}$ per [3, 7].",
    "Namespaces survive: a <b>bold</b> tag and 5 > 3 & 2 < 4 stay literal.",
)


def _engine() -> PlaceholderEngine:
    return PlaceholderEngine(
        code=CodeMasker(),
        math=MathMasker(),
        soup=SoupMathMasker(),
        citation=CitationMasker(),
    )


def _segments(texts: tuple[str, ...]) -> list[Segment]:
    engine = _engine()
    segments: list[Segment] = []
    for index, text in enumerate(texts):
        masked = engine.mask(xml_safe(text))
        segments.append(
            Segment(
                id=f"seg_{index:05d}",
                source=masked.text,
                placeholders=masked.placeholders,
                target=masked.text if index % 2 == 0 else None,
                state=_STATES[index % len(_STATES)],
            )
        )
    return segments


def _fingerprint(segment: Segment) -> tuple[object, ...]:
    return (
        segment.source,
        segment.target,
        segment.state.value,
        tuple(
            sorted(
                (placeholder.token, placeholder.kind, placeholder.original)
                for placeholder in segment.placeholders
            )
        ),
    )


def test_every_segment_field_survives_the_round_trip() -> None:
    segments = _segments(_TEXTS)
    xml = to_xliff(segments, src_lang="en", trg_lang="zh", original="book.md")
    parsed = from_xliff(xml)

    assert (parsed.src_lang, parsed.trg_lang, parsed.original) == ("en", "zh", "book.md")
    assert len(parsed.segments) == len(segments)
    by_id = {segment.id: segment for segment in parsed.segments}
    for segment in segments:
        assert by_id[segment.id] is not None
        assert _fingerprint(by_id[segment.id]) == _fingerprint(segment)


def test_the_masked_kinds_come_back_with_their_original_spans() -> None:
    segments = _segments(_TEXTS)
    xml = to_xliff(segments, src_lang="en", trg_lang="zh", original="book.md")
    parsed = from_xliff(xml)
    by_id = {segment.id: segment for segment in parsed.segments}

    code = by_id["seg_00000"]
    assert [(p.kind, p.original) for p in code.placeholders] == [("code", "`make all`")]
    math = by_id["seg_00001"]
    assert [(p.kind, p.original) for p in math.placeholders] == [("math", "$e^{i\\pi} + 1 = 0$")]
    mixed = by_id["seg_00004"]
    kinds = sorted(p.kind for p in mixed.placeholders)
    assert kinds == ["citation", "code", "math"]
    assert "[3, 7]" in [p.original for p in mixed.placeholders]


def test_a_segment_without_protection_round_trips_verbatim() -> None:
    segments = _segments(("Plain prose with nothing to protect at all.",))
    xml = to_xliff(segments, src_lang="en", trg_lang="zh", original="plain.md")
    (parsed,) = from_xliff(xml).segments
    assert parsed.source == "Plain prose with nothing to protect at all."
    assert parsed.placeholders == ()
    assert parsed.target == parsed.source


def test_an_empty_view_round_trips_as_an_empty_view() -> None:
    xml = to_xliff([], src_lang="en", trg_lang="zh", original="empty.md")
    parsed = from_xliff(xml)
    assert parsed.segments == ()
    assert (parsed.src_lang, parsed.trg_lang, parsed.original) == ("en", "zh", "empty.md")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\x00b\x0cc", "abc"),  # C0 control bytes XML 1.0 forbids are stripped
        ("keep\ttabs\nand\rnewlines", "keep\ttabs\nand\rnewlines"),  # tab/LF/CR are legal
        ("<b>bold</b> & 5 > 3", "<b>bold</b> & 5 > 3"),  # markup escaping is ET's job
        ("", ""),
    ],
)
def test_xml_safe_strips_only_what_xml_cannot_carry(raw: str, expected: str) -> None:
    assert xml_safe(raw) == expected
