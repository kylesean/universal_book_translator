"""Artifact I/O: one value names the delivered artifact's files.

The export stage carries two artifact identities at once -- ``target_output``
(the path the primary render was *asked* for, which the PE queue keys on) and
``rendered_path`` (the file the adapter *returned*, which every sidecar and
companion keys on). ``DeliveredArtifact`` holds both so a caller never picks the
wrong one; the sibling/companion naming has a single owner instead of ad-hoc
``with_name`` calls spread through the stage.

The other half pinned here is the *probe* that decides whether the artifact
carries a realization: a reflowing backend re-breaks lines, so an exact match
would report elements it actually placed -- token overlap is the measure.
``check_artifact`` itself needs a real PDF, so the pure token logic it depends
on is pinned directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.qe.fast_pass import REHEARSAL_MARKER, strip_rehearsal_marker
from ubt.pipeline.artifact import (
    ArtifactReport,
    DeliveredArtifact,
    ElementCheck,
    _expected,
    _normalize,
    _present,
    _tokenize,
    delivered_artifact,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# DeliveredArtifact: the two identities and what hangs off each.
# --------------------------------------------------------------------------- #


def test_delivered_artifact_keeps_both_identities() -> None:
    artifact = delivered_artifact("/out/book_bilingual_mono.pdf", "/out/book_bilingual.pdf")
    assert artifact.rendered_path == Path("/out/book_bilingual_mono.pdf")
    assert artifact.target_output == Path("/out/book_bilingual.pdf")


def test_sidecars_and_companions_key_on_the_returned_file() -> None:
    # The visual gate may rewrite the returned file, so sidecars follow *it*,
    # not the requested target.
    artifact = delivered_artifact("/out/book_mono.pdf", "/out/book.pdf")
    assert artifact.sidecar("contract.json") == Path("/out/book_mono_pdf_contract.json")
    assert artifact.companion(".xliff") == Path("/out/book_mono_pdf.xliff")


def test_siblings_key_on_the_requested_target() -> None:
    # A companion *render* is named beside the requested target.
    artifact = delivered_artifact("/out/book_bilingual_mono.pdf", "/out/book_bilingual.pdf")
    assert artifact.sibling("_mono") == Path("/out/book_bilingual_mono.pdf")
    assert artifact.sibling("_secondary") == Path("/out/book_bilingual_secondary.pdf")


def test_the_two_identities_differ_for_an_auto_named_primary() -> None:
    artifact = delivered_artifact("/out/book_bilingual_mono.pdf", "/out/book_bilingual.pdf")
    assert artifact.rendered_path != artifact.target_output


def test_delivered_artifact_accepts_str_or_path(tmp_path: Path) -> None:
    artifact = delivered_artifact(tmp_path / "r.pdf", str(tmp_path / "t.pdf"))
    assert isinstance(artifact, DeliveredArtifact)
    assert artifact.rendered_path == tmp_path / "r.pdf"


# --------------------------------------------------------------------------- #
# Tokenization: what counts as a probe unit.
# --------------------------------------------------------------------------- #


def test_latin_words_are_probe_units_and_hyphenated_words_stay_whole() -> None:
    assert _tokenize("Hello world foo-bar") == ["Hello", "world", "foo-bar"]


def test_cjk_is_tokenized_per_glyph() -> None:
    assert _tokenize("\u4e2d\u6587") == ["\u4e2d", "\u6587"]


def test_punctuation_and_underscores_are_not_tokens() -> None:
    assert _tokenize("a, b_c!") == ["a", "b", "c"]


def test_normalize_collapses_whitespace_including_line_breaks() -> None:
    assert _normalize("a  b\n c") == "a b c"


# --------------------------------------------------------------------------- #
# The presence probe: overlap, with a noise floor for short strings.
# --------------------------------------------------------------------------- #


def test_expected_text_shorter_than_the_probe_floor_is_always_present() -> None:
    # A bullet or lone glyph is noise; the check says nothing either way.
    assert _present("abc", frozenset()) is True


def test_empty_expected_text_is_present() -> None:
    assert _present("", frozenset()) is True


def test_full_token_overlap_is_present() -> None:
    tokens = frozenset(_tokenize("the quick brown fox jumps"))
    assert _present("the quick brown fox", tokens) is True


def test_overlap_at_the_threshold_is_present() -> None:
    # Exactly 3/5 == 0.6: the boundary is inclusive (>=).
    tokens = frozenset(_tokenize("the quick brown fox jumps"))
    assert _present("the quick brown zebra yak", tokens) is True


def test_overlap_below_the_threshold_is_absent() -> None:
    tokens = frozenset(_tokenize("the quick brown fox jumps"))
    assert _present("the quick zebra yak", tokens) is False  # 2/4 == 0.5


def test_cjk_realization_is_probed_per_glyph() -> None:
    # Four glyphs clears the probe floor, so overlap is actually measured.
    tokens = frozenset(_tokenize("\u4e2d\u6587\u5b57"))
    assert _present("\u4e2d\u6587\u5b57\u7532", tokens) is True  # 3/4
    assert _present("\u4e2d\u4e8c\u4e09\u56db", tokens) is False  # 1/4


# --------------------------------------------------------------------------- #
# ArtifactReport: the verdict.
# --------------------------------------------------------------------------- #


def _check(element_id: str, present: bool) -> ElementCheck:
    return ElementCheck(element_id, 1, present)


def test_missing_lists_only_elements_the_artifact_did_not_carry() -> None:
    report = ArtifactReport(total=2, checks=(_check("e1", True), _check("e2", False)))
    assert [check.element_id for check in report.missing] == ["e2"]


def test_report_passes_only_when_nothing_is_missing() -> None:
    passing = ArtifactReport(total=1, checks=(_check("e1", True),))
    failing = ArtifactReport(total=1, checks=(_check("e1", False),))
    assert passing.passed
    assert not failing.passed


def test_summary_line_states_how_many_realizations_were_carried() -> None:
    report = ArtifactReport(total=2, checks=(_check("e1", True), _check("e2", False)))
    assert report.summary_line() == "[FAIL] artifact carries 1/2 text realization(s)"
    passing = ArtifactReport(total=1, checks=(_check("e1", True),))
    assert passing.summary_line() == "[PASS] artifact carries 1/1 text realization(s)"


# --------------------------------------------------------------------------- #
# The rehearsal prefix: the renderer strips it, so the audit must not demand it.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "prefix",
    [REHEARSAL_MARKER, "[Mock Translation]", "[mock translation]", "[Mock  Translation]"],
)
def test_strip_rehearsal_marker_removes_every_spelling(prefix: str) -> None:
    assert strip_rehearsal_marker(f"{prefix} Hello world") == "Hello world"
    assert strip_rehearsal_marker("Hello world") == "Hello world"


def test_a_rehearsal_heading_is_found_in_an_artifact_without_the_marker() -> None:
    # Regression: the PDF renderer strips the dry-run's prefix before placing
    # text, so the artifact never carries it. Before this was shared, a 3-token
    # heading needed the marker's 4 CJK tokens inside the overlap floor
    # (3/7 < 0.6) and every rehearsal delivery reported the heading "missing".
    element = make_element(
        id="e1",
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="The Attention Machine",
    )
    block = IRBlock(element=element, target_text=f"{REHEARSAL_MARKER} The Attention Machine")
    translations = {"e1": f"{REHEARSAL_MARKER} The Attention Machine"}

    expected = _expected(block, translations)
    assert expected == "The Attention Machine"
    assert _present(expected, frozenset(_tokenize("The Attention Machine"))) is True
