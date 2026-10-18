"""The XLIFF view is a lossless bijection over segments.

The ADR's Phase-2 interop gate: the kernel's units can leave and return through
the industry format without loss. A ``Segment`` is built per text with real
protected spans (the ``PlaceholderEngine`` masking code, math, soup and
citations), serialized to XLIFF 2.1, parsed back, and every modeled field must
survive: source, target, state, the (token, kind, original) of every
placeholder -- plus the document header (languages, original file name).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.email_masker import EmailMasker
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker
from ubt.core.ir.models import BlockStatus
from ubt.model.segment import Placeholder, Segment, SegmentState
from ubt.segment.document import _block_state
from ubt.segment.placeholders import PlaceholderEngine
from ubt.segment.xliff import _read_inline, from_xliff, to_xliff, xml_safe

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
        email=EmailMasker(),
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


def test_read_inline_keeps_non_ph_inline_markup() -> None:
    # CAT tools emit paired inline markup (<pc>); treating every child as a
    # placeholder used to append an empty token and drop the wrapped text.
    element = ET.fromstring(
        '<source xmlns="urn:oasis:names:tc:xliff:document:2.1">'
        'Hello <ph dataRef="p1"/> and <pc>bold text</pc> end</source>'
    )
    placeholders: dict[str, Placeholder] = {}
    text = _read_inline(element, placeholders)
    assert text == "Hello p1 and bold text end"
    assert "p1" in placeholders


def test_read_inline_keeps_a_ph_without_dataref() -> None:
    # A third-party CAT tool may emit <ph id="1">Foo</ph> with no dataRef; the
    # visible text must survive, not be replaced by an empty token.
    element = ET.fromstring(
        '<source xmlns="urn:oasis:names:tc:xliff:document:2.1">'
        'see <ph id="1" type="bold">Foo</ph> now</source>'
    )
    placeholders: dict[str, Placeholder] = {}
    text = _read_inline(element, placeholders)
    assert text == "see Foo now"
    assert placeholders == {}


def test_block_state_fails_loud_on_unmapped_status() -> None:
    # A BlockStatus missing from the mapping is IR drift; silently labelling
    # the segment NEW would tell a reviewer to translate a block whose real
    # state is unknown.
    drafted = cast("BlockStatus", SimpleNamespace(value="drafted"))
    assert _block_state(drafted) is SegmentState.TRANSLATED
    bogus = cast("BlockStatus", SimpleNamespace(value="not_a_real_status"))
    with pytest.raises(ValueError, match="unmapped BlockStatus"):
        _block_state(bogus)


def test_to_xliff_emits_job_id_note_and_pe_import_reads_it(tmp_path: Path) -> None:
    from ubt.core.engine.pe_import import parse_xliff_revisions

    seg = Segment(
        id="block_001",
        source="Run code first",
        placeholders=(Placeholder(token="⟦CODE_MASK_0001-abc⟧", kind="code", original="make"),),
        target="运行 ⟦CODE_MASK_0001-abc⟧ 之前。",
        state=SegmentState.TRANSLATED,
    )
    xml = to_xliff([seg], src_lang="en", trg_lang="zh", original="test.md", job_id="job_xyz_123")
    assert 'category="ubt-job-id"' in xml
    assert "job_xyz_123" in xml

    xliff_file = tmp_path / "test.xliff"
    xliff_file.write_text(xml, encoding="utf-8")

    revisions = parse_xliff_revisions(xliff_file)
    assert revisions.job_ids == {"job_xyz_123"}
    assert revisions.revisions["block_001"] == "运行 ⟦CODE_MASK_0001-abc⟧ 之前。"


# --------------------------------------------------------------------------- #
# The reviewer's queue
#
# The routing was correct and the interchange format threw it away: every block
# a human needed to see was emitted in a state that hid it. These pin the three
# states a reviewer must be able to tell apart, and the evidence behind them.
# --------------------------------------------------------------------------- #


def test_the_three_review_states_are_distinguishable() -> None:
    """A reviewer must be able to answer "which of these do I work on?".

    Three blocks ship in three genuinely different conditions, and each needs a
    different XLIFF state:

    - clean: passed, nothing to do (``translated``/``reviewed``)
    - needs_human: a draft is delivered and should be post-edited
    - blocked_human: NO draft shipped (source only), so it still needs translating

    ``needs_human`` used to emit ``translated`` -- identical to a clean block, so
    a reviewer opening a 1000-block file could not find the handful that mattered.
    ``blocked_human`` emitted ``needs-review-translation``, which claims a
    translation exists to review when the target is the source under a marker.
    """
    from types import SimpleNamespace
    from typing import cast

    from ubt.core.ir.models import BlockStatus

    states = {
        status.value: _block_state(status).value
        for status in (
            BlockStatus.MTQE_PASSED,
            BlockStatus.NEEDS_HUMAN,
            BlockStatus.BLOCKED_HUMAN,
        )
    }
    assert states["mtqe_passed"] == "verified"
    assert states["needs_human"] == "needs_review"
    assert states["blocked_human"] == "blocked"
    # The whole point: three distinct values, not one collapsed pair.
    assert len(set(states.values())) == 3

    # And each survives the trip through XLIFF as its own state string.
    for status in (BlockStatus.MTQE_PASSED, BlockStatus.NEEDS_HUMAN, BlockStatus.BLOCKED_HUMAN):
        state = _block_state(status)
        xml = to_xliff(
            [Segment(id="b1", source="src", target="tgt", state=state)],
            src_lang="en",
            trg_lang="zh",
        )
        (back,) = from_xliff(xml).segments
        assert back.state is state, f"{status.value} did not round-trip"
    _ = cast("object", SimpleNamespace)


def test_needs_review_and_blocked_are_not_the_same_xliff_state() -> None:
    """The two states that used to be confused now read differently on the wire."""
    review_xml = to_xliff(
        [Segment(id="a", source="s", target="t", state=SegmentState.NEEDS_REVIEW)],
        src_lang="en",
        trg_lang="zh",
    )
    blocked_xml = to_xliff(
        [Segment(id="b", source="s", state=SegmentState.BLOCKED)],
        src_lang="en",
        trg_lang="zh",
    )
    assert 'state="needs-review-translation"' in review_xml
    assert 'state="needs-translation"' in blocked_xml
    assert "needs-review-translation" not in blocked_xml


def test_a_flagged_block_carries_its_reason_into_the_xliff() -> None:
    """The state says "look here"; the notes say "because of this".

    Without the reason a reviewer's queue is a list of opaque block ids, and the
    ledger already holds the evidence -- it was simply never projected into the
    one file the reviewer opens.
    """
    from ubt.model.segment import QA

    segment = Segment(
        id="b0042",
        source="Euler wrote $e^{i\\pi}+1=0$.",
        placeholders=(Placeholder(token="MATH_1", kind="math", original="$e^{i\\pi}+1=0$"),),
        target="欧拉写下了 MATH_1。",
        state=SegmentState.NEEDS_REVIEW,
        qa=QA(
            score=0.55,
            severity="major",
            flags=("Glossary term violation", "Omission suspected"),
        ),
    )
    xml = to_xliff([segment], src_lang="en", trg_lang="zh", original="book.md")
    assert 'category="ubt-mtqe-score"' in xml
    assert 'category="ubt-mqm-severity"' in xml
    assert 'category="ubt-error-flag"' in xml
    assert "0.55" in xml
    assert "Omission suspected" in xml

    # And it all comes back -- the verdict is not a one-way annotation.
    (back,) = from_xliff(xml).segments
    assert back.qa is not None
    assert back.qa.score == 0.55
    assert back.qa.severity == "major"
    assert back.qa.flags == ("Glossary term violation", "Omission suspected")


def test_a_clean_block_carries_no_qa_note() -> None:
    """``None`` means "never flagged", which is not the same as "scored zero".

    Emitting ``ubt-mtqe-score: 0.0`` for an unscored block would read as a
    genuine zero to anything that parses the file.
    """
    xml = to_xliff(
        [Segment(id="clean", source="s", target="t", state=SegmentState.VERIFIED)],
        src_lang="en",
        trg_lang="zh",
    )
    assert "ubt-mtqe-score" not in xml
    assert "ubt-error-flag" not in xml
    (back,) = from_xliff(xml).segments
    assert back.qa is None


def test_a_malformed_qa_note_does_not_make_the_file_unreadable() -> None:
    """QA notes are advisory metadata; a bad one must not sink the import.

    A hand-edited or third-party-tool XLIFF may carry anything in a note. The
    translation text is the payload -- it must survive regardless.
    """
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<xliff xmlns="urn:oasis:names:tc:xliff:document:2.1" version="2.1"'
        ' srcLang="en" trgLang="zh"><file id="f1">'
        '<unit id="b1"><notes><note category="ubt-mtqe-score">not-a-number</note>'
        '</notes><segment id="s1" state="translated">'
        "<source>hello</source><target>你好</target></segment></unit>"
        "</file></xliff>"
    )
    (back,) = from_xliff(xml).segments
    assert back.target == "你好"
    assert back.qa is not None
    assert back.qa.score is None  # dropped, not crashed
