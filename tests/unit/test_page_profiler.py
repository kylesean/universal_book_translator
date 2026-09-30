"""Page profiler fact layer + decision layer + cache."""

from pathlib import Path

import pytest

from tests.corpus_markers import requires_synthetic_duo, requires_synthetic_mono
from ubt.adapters.pdf.page_profiler import (
    PageFacts,
    PageKind,
    classify_page,
    collect_page_facts,
    column_right_share,
    profile_pdf,
)

Rect = tuple[float, float, float, float]


def _rect(left: float, top: float, right: float, height: float = 12.0) -> Rect:
    """A pdfium-shaped rect (left, bottom, right, top) on a 612pt page."""
    return (left, top - height, right, top)


def _facts(**kwargs: object) -> PageFacts:
    base: dict[str, object] = {
        "page": 1,
        "width_pt": 595.0,
        "height_pt": 842.0,
        "n_chars": 2000,
        "n_rect_rows": 40,
        "right_row_share": 0.0,
        "formula_density": 0.02,
        "n_text_ops": 300,
        "n_path_ops": 5,
        "n_images": 0,
        "n_fonts": 2,
    }
    base.update(kwargs)
    return PageFacts(**base)  # type: ignore[arg-type]


def test_clean_prose_is_editable() -> None:
    assert classify_page(_facts()) == PageKind.EDITABLE_TEXT


def test_empty_page_with_image_is_scan() -> None:
    assert classify_page(_facts(n_chars=0, n_images=1, n_text_ops=0)) == PageKind.SCAN_IMAGE


def test_empty_page_without_content_is_scan() -> None:
    # Fail-closed: unknown/blank pages go to the vision route, never the fast path.
    assert (
        classify_page(_facts(n_chars=0, n_images=0, n_text_ops=0, n_path_ops=0))
        == PageKind.SCAN_IMAGE
    )


def test_vector_drawing_without_text_is_vector_heavy() -> None:
    p = classify_page(_facts(n_chars=0, n_images=0, n_path_ops=500))
    assert p == PageKind.VECTOR_HEAVY


def test_figure_labels_are_not_prose() -> None:
    p = classify_page(_facts(n_chars=60, n_path_ops=800, n_images=1))
    assert p == PageKind.VECTOR_HEAVY


def test_two_column_prose_is_mixed() -> None:
    assert classify_page(_facts(right_row_share=0.45)) == PageKind.MIXED_COMPLEX


def test_formula_dense_is_mixed() -> None:
    assert classify_page(_facts(formula_density=0.16)) == PageKind.MIXED_COMPLEX


def test_illustrated_prose_is_mixed() -> None:
    assert classify_page(_facts(n_images=2)) == PageKind.MIXED_COMPLEX


def test_poster_page_is_poster_fixed() -> None:
    # Image-led design page (title + art, almost no body text).
    assert (
        classify_page(_facts(n_chars=40, n_images=1, n_fonts=2, n_path_ops=50))
        == PageKind.POSTER_FIXED
    )
    # Path-heavy drawings stay vector-heavy even with an image on them.
    assert classify_page(_facts(n_chars=60, n_path_ops=800, n_images=1)) == PageKind.VECTOR_HEAVY
    # Text-led pages with an image stay mixed, not poster.
    assert classify_page(_facts(n_chars=500, n_images=1)) == PageKind.MIXED_COMPLEX


def test_dense_prose_is_resume_dense() -> None:
    # Strictly above the floor (2000-char baseline stays editable).
    assert classify_page(_facts(n_chars=2500)) == PageKind.RESUME_DENSE
    assert classify_page(_facts(n_chars=2000)) == PageKind.EDITABLE_TEXT
    # Mixed tells still win over density.
    assert classify_page(_facts(n_chars=2500, n_images=1)) == PageKind.MIXED_COMPLEX
    assert classify_page(_facts(n_chars=2500, right_row_share=0.45)) == PageKind.MIXED_COMPLEX


def test_new_kinds_prefer_overlay() -> None:
    from ubt.adapters.pdf.page_profiler import PageProfile

    for kind in (PageKind.POSTER_FIXED, PageKind.RESUME_DENSE):
        assert PageProfile(facts=_facts(), kind=kind).overlay_preferred is True
        assert PageProfile(facts=_facts(), kind=kind).needs_vision is False


def test_profile_properties() -> None:
    from ubt.adapters.pdf.page_profiler import PageProfile

    scan = PageProfile(facts=_facts(n_chars=0), kind=PageKind.SCAN_IMAGE)
    assert scan.needs_vision is True
    assert scan.overlay_preferred is True
    vec = PageProfile(facts=_facts(), kind=PageKind.VECTOR_HEAVY)
    assert vec.needs_vision is False
    assert vec.overlay_preferred is True
    edit = PageProfile(facts=_facts(), kind=PageKind.EDITABLE_TEXT)
    assert edit.overlay_preferred is False


@requires_synthetic_mono
def test_profile_real_pdf_and_cache(tmp_path: Path) -> None:
    # tests/fixtures/synthetic-mono.pdf is the live 13-page sample (generated, gitignored).
    pdf = Path("tests/fixtures/synthetic-mono.pdf")
    profiles = profile_pdf(pdf, cache_dir=tmp_path)
    assert len(profiles) == 13
    assert all(isinstance(p.kind, PageKind) for p in profiles)
    assert any(p.facts.n_chars > 100 for p in profiles)
    assert sum(1 for p in profiles if p.kind == PageKind.SCAN_IMAGE) == 0
    # Second call must be served from cache (delete source rows would fail
    # otherwise — here we just check determinism + cache file presence).
    again = profile_pdf(pdf, cache_dir=tmp_path)
    assert [p.kind for p in again] == [p.kind for p in profiles]
    assert list(tmp_path.glob("*.json")) != []


def test_empty_profile_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe must not be remembered: empty profiles are never written.

    ``collect_page_facts`` returns ``[]`` when pdfium cannot open the document;
    caching that unconditionally served one transient failure as the PDF's
    permanent profile (the cache key covers only path/size/mtime), silently
    downgrading engine routing until the file changed.
    """
    import ubt.adapters.pdf.page_profiler as profiler

    monkeypatch.setattr(profiler, "collect_page_facts", lambda _path: [])
    cache_dir = tmp_path / "cache"
    assert profiler.profile_pdf(tmp_path / "any.pdf", cache_dir) == []
    assert not list(cache_dir.glob("*.json"))


def test_column_share_merges_fragments_within_a_row() -> None:
    # A justified single-column line fragments every few glyphs; the fragments
    # must merge back into one row (small intra-row gaps) so none of them reads
    # as a right-hand column start.
    row = 700.0
    rects = [_rect(x, row, x + 90.0) for x in (72.0, 170.0, 268.0, 366.0, 464.0)]
    assert column_right_share(rects, width=612.0) == 0.0


def test_column_share_detects_a_real_second_column() -> None:
    # Every row has a genuine second-column segment (gutter-sized gap); half
    # the segments start right of the 0.45 edge ratio.
    rects: list[Rect] = []
    for row in (760.0 - i * 16.0 for i in range(10)):
        rects.append(_rect(50.0, row, 250.0))
        rects.append(_rect(300.0, row, 500.0))
    share = column_right_share(rects, width=540.0)
    assert share > 0.25


def test_column_share_counts_right_started_indented_lines() -> None:
    # A line whose only segment begins past the edge ratio still counts —
    # this is what keeps genuine two-column detection working.
    rects = [_rect(320.0, 700.0, 540.0), _rect(320.0, 684.0, 540.0)]
    assert column_right_share(rects, width=612.0) == 1.0


def test_column_share_empty_page_is_zero() -> None:
    assert column_right_share([], width=612.0) == 0.0


@requires_synthetic_mono
def test_mono_fixture_prose_pages_are_not_column_flagged() -> None:
    # The mono corpus is single-column prose: per-fragment share read ~0.55 on
    # every body page, routing the whole book down the docling mainline. After
    # the row merge, sampled prose pages must sit far below the 0.25 column bar.
    pdf = Path("tests/fixtures/synthetic-mono.pdf")
    facts = collect_page_facts(pdf)
    sampled = [1, 4, 7, 10, 13]
    for page_no in sampled:
        share = facts[page_no - 1].right_row_share
        assert share < 0.25, f"page {page_no} share={share:.3f}"


@requires_synthetic_duo
def test_duo_fixture_keeps_the_column_signal() -> None:
    # The merge must not swallow the genuine two-column tell: every duo page
    # stays above the share bar.
    pdf = Path("tests/fixtures/synthetic-duo.pdf")
    facts = collect_page_facts(pdf)
    assert facts, "duo fixture profiled empty"
    for fact in facts:
        assert fact.right_row_share > 0.25, f"page {fact.page} share={fact.right_row_share:.3f}"
