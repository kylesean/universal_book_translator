"""Pre-translation skip classifier: what ships verbatim and what must not.

``classify_skip`` decides whether a block bypasses the LLM / QE / repair loop
entirely. Both error directions are costly and *asymmetric*: a false negative
sends bibliography residue and author bylines through the model (transliterated
names, broken retrievability), while a false positive silently ships real prose
untranslated. The rules are therefore precision-first, and these tests pin both
sides of that line -- the shapes that must match, and the near-misses that must
stay translatable.

The pure classifier is the unit under test; the ingest-stage wiring (status,
score, flag) lives elsewhere.
"""

from __future__ import annotations

import time

import pytest

from ubt.core.cleaners.skip_rules import _is_author_byline, classify_skip

pytestmark = pytest.mark.fast

HANDLE = "social-handle watermark / running-head chrome"
BIB = "bibliography entry (kept verbatim for retrievability)"
BYLINE = "author byline (foreign names kept in romanization per GB/T 7714)"
DEBRIS = "pure symbol/numeric debris (no translatable words)"


# --- empty input ------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\t ", "\u00ad"])
def test_blank_or_whitespace_is_never_skipped(text: str) -> None:
    assert classify_skip(text) is None


# --- social-handle watermark / running-head chrome --------------------------


@pytest.mark.parametrize(
    "text",
    ["@techNmak 2024 follow", "@a.b-c_1 hello", "@handle"],
)
def test_social_handle_at_block_start_is_skipped(text: str) -> None:
    assert classify_skip(text) == HANDLE


def test_handle_soft_hyphen_is_stripped_before_matching() -> None:
    # PDF extraction inserts soft hyphens inside handles; they are removed first.
    assert classify_skip("@techNmak\u00ad stuff") == HANDLE


def test_handle_only_matches_at_the_start_of_the_block() -> None:
    assert classify_skip("see @techNmak for details") is None


# --- arXiv identifiers ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "arXiv:2609.20519",
        "see arxiv:1234.5678",
        # arXiv is a high-precision signal and wins even inside a prose line.
        "We propose a method, see arXiv:2609.20519 for details.",
    ],
)
def test_arxiv_identifier_is_always_skipped(text: str) -> None:
    assert classify_skip(text) == BIB


# --- bracket-style reference entries ----------------------------------------


@pytest.mark.parametrize(
    "text",
    ["[12] A. Author, Journal of Things, 2019", '[12] "A Quoted Title", 2019'],
)
def test_bracket_reference_with_year_is_skipped(text: str) -> None:
    assert classify_skip(text) == BIB


@pytest.mark.parametrize(
    "text",
    [
        # lowercase continuation -> "[3] yield ..." is a numbered sentence.
        "[3] yield the following results in 2019",
        # no year -> cannot distinguish from prose that opens with "[n]".
        "[12] A. Author, Journal",
        # narrative prose overrides the bracket shape.
        "[12] We propose a method in 2019.",
    ],
)
def test_bracket_reference_requires_uppercase_start_year_and_no_prose(text: str) -> None:
    assert classify_skip(text) is None


# --- the explicit in_bibliography flag --------------------------------------


def test_bibliography_flag_skips_non_narrative_block() -> None:
    assert classify_skip("Some line without narrative", in_bibliography=True) == BIB


def test_bibliography_flag_defers_narrative_prose_to_the_llm() -> None:
    assert classify_skip("we propose a new method", in_bibliography=True) is None


def test_bibliography_flag_defers_headings() -> None:
    # A heading is structure, not a citation, even inside a bibliography region.
    assert classify_skip("Purpose, Scope", in_bibliography=True, is_heading=True) is None


# --- label-free bibliography signatures (_is_bib_entry tiers) ---------------


@pytest.mark.parametrize(
    "text",
    [
        # [Online]. Available: https://...
        "Foo. [Online]. Available: https://example.com/x",
        # quoted article title + year
        'J. Smith, "A Great Paper". 2019.',
        # documentation citation with an access year
        "See the user guide consulted 2019 for details",
        # et al. is a strong citation signal
        "Smith et al. IEEE, 2019",
        # URL + an author/venue cue
        "J. Doe. https://example.com/paper, NeurIPS 2020.",
        # >=2 initial-based authors + a venue
        "J. Smith, A. B. Jones. IEEE Access, 2019.",
        # generic journal-volume cue (needs the year gate)
        "J. Smith, Lett. 21 (5), 245-247, 2019.",
        # venue + terminal page range
        "J. Smith. IEEE, 245-247, 2019.",
        # publisher book cue + author
        "J. Smith. MIT Press, 2019.",
        # full-name author list + venue + pages
        "John Yang, Carlos Jimenez. Springer, pages 45-60, 2019.",
        # "In <Venue>" + volume/pages
        "In NeurIPS, volume 5, 2019.",
        # doi / In Proceedings structural cues
        "J. Smith. doi:10.1000/xyz, 2019.",
        "J. Smith. In Proceedings of ACL, 2019.",
    ],
)
def test_label_free_bibliography_entries_are_skipped(text: str) -> None:
    assert classify_skip(text) == BIB


@pytest.mark.parametrize(
    "text",
    [
        # No year anywhere: every label-free tier is gated on _YEAR_RE.
        "J. Smith, Lett. 21 (5), 245-247.",
        "In NeurIPS, pages 1-10",
        # A bare URL is not a citation.
        "https://example.com/paper",
        # A bare sentence naming a venue + page range, with no author signal,
        # is prose -- it must not be shipped untranslated.
        "This was published by Springer in 2019, spanning pages 45-60.",
        "Springer, pages 45-60, 2019.",
    ],
)
def test_bibliography_near_misses_stay_translatable(text: str) -> None:
    assert classify_skip(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "We propose a method. J. Smith. IEEE, 2019.",
        "This paper cites J. Smith. IEEE, 2019.",
        "Our method beats J. Smith. IEEE, 2019.",
    ],
)
def test_narrative_prose_overrides_a_bibliography_shape(text: str) -> None:
    # A citation-looking tail must not drag a real sentence out of translation.
    assert classify_skip(text) is None


# --- author byline ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Duomin Wang, Jane Doe",
        "Duomin Wang, Jane Doe, John Smith",
        "Duomin Wang and Jane Doe",
        "Duomin Wang 1, Jane Doe 2",
        "Li\u00b9\u00b2 Zhang, Wang\u00b3 Li",  # superscript affiliations
        "John Smith Jr., Jane Doe",
        "Duomin Wang, Jane Doe,",  # trailing separator
        "Duomin Wang\u2020, Jane Doe\u2021",  # dagger affiliations
    ],
)
def test_author_byline_is_skipped(text: str) -> None:
    assert classify_skip(text) == BYLINE


@pytest.mark.parametrize(
    "text",
    [
        "Duomin Wang",  # a single name is not a list
        "Smith, Jones",  # one token per segment
        "J. Smith, A. Jones",  # initials-only first tokens
        "Purpose, Scope, and Audience",  # title-case heading, one token/segment
        "Duomin Wang, Scope",  # mixed: one segment is one token
        "Duomin Wang, Jane Doe 2020",  # a year makes it a reference, not a byline
        "duomin wang, jane doe",  # lowercase -> prose
    ],
)
def test_byline_near_misses_stay_translatable(text: str) -> None:
    assert classify_skip(text) is None


def test_heading_is_never_a_byline() -> None:
    # Even a shape that would otherwise look like a name list is suppressed
    # when the block is known to be a heading.
    assert classify_skip("Duomin Wang, Jane Doe", is_heading=True) is None


def test_cjk_block_is_never_a_byline() -> None:
    # CJK names are already in the target script; romanization rules don't apply.
    assert classify_skip("\u738b\u591a\u6c11, \u674e\u56db") is None


def test_byline_scan_is_linear_on_an_adversarial_tail() -> None:
    """Regression guard for the ReDoS fix documented in ``_BYLINE_SEGMENT_RE``.

    The previous nested quantifier backtracked exponentially (11.7 s at n=18).
    ``classify_skip`` runs on every ingested block, so a crafted line must not
    stall the ingest worker. A linear scan of 200 chars is microseconds; a
    regression to exponential would not finish inside the bound.
    """
    text = "Aa Aa, Bb " + "1" * 200 + "!"
    started = time.perf_counter()
    result = _is_author_byline(text)
    elapsed = time.perf_counter() - started
    assert result is False
    assert elapsed < 1.0


# --- pure symbol / numeric debris -------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["123 --- 456 +++ ==", "K / V / x", "1 2 3 4 5", "\u2014\u2014\u2014"],
)
def test_symbol_or_numeric_debris_is_skipped(text: str) -> None:
    assert classify_skip(text) == DEBRIS


@pytest.mark.parametrize(
    "text",
    [
        "ab",  # a two-letter word counts as translatable
        "--- ab ---",
        "\u4e2d\u6587 --- 123",  # CJK content is never "debris"
        "x \u4e2d\u6587",
    ],
)
def test_debris_rule_does_not_fire_on_real_content(text: str) -> None:
    assert classify_skip(text) is None


# --- ordinary prose ---------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "This is an ordinary sentence.",
        "\u8fd9\u662f\u4e00\u53e5\u666e\u901a\u7684\u4e2d\u6587\u3002",
        # Short citation lines are explicitly NOT skipped (module docstring).
        "Research anchors: [1], [2]",
    ],
)
def test_ordinary_prose_is_translated(text: str) -> None:
    assert classify_skip(text) is None
