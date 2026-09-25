"""Page profiler fact layer + decision layer + cache."""

from pathlib import Path

from tests.corpus_markers import requires_synthetic_mono
from ubt.adapters.pdf.page_profiler import PageFacts, PageKind, classify_page, profile_pdf


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
    # NOTE: tests/fixtures/ removed; docs/synthetic-mono.pdf is the live 13-page sample.
    pdf = Path("docs/synthetic-mono.pdf")
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
