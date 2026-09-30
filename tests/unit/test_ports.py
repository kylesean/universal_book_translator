"""Content-probe aggregation in ubt.core.ports."""

from pathlib import Path

import pytest

from tests.pdf_builders import text_pdf
from ubt.core.ports import classify_pdf_content

pytestmark = pytest.mark.fast


def test_classify_pdf_content_majority_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """One anomalous page must not flip a whole book's route.

    The scan tell aggregates with a >=50% share so a single blank page in a
    born-digital book routes like the other pages, matching the sampled-page
    majority the engine selector applies to the same signal.
    """
    import ubt.adapters.pdf.page_profiler as profiler
    from ubt.adapters.pdf.page_profiler import PageKind

    kinds = [PageKind.SCAN_IMAGE] + [PageKind.EDITABLE_TEXT] * 7
    monkeypatch.setattr(profiler, "collect_page_facts", lambda _path: [object()] * len(kinds))
    monkeypatch.setattr(profiler, "classify_page", lambda _facts: kinds.pop(0))

    has_scan, formula_heavy = classify_pdf_content(Path("book.pdf"))
    assert has_scan is False
    assert formula_heavy is False


def test_classify_pdf_content_unanimous_scan_still_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ubt.adapters.pdf.page_profiler as profiler
    from ubt.adapters.pdf.page_profiler import PageKind

    kinds = [PageKind.SCAN_IMAGE] * 8
    monkeypatch.setattr(profiler, "collect_page_facts", lambda _path: [object()] * len(kinds))
    monkeypatch.setattr(profiler, "classify_page", lambda _facts: kinds.pop(0))

    has_scan, _ = classify_pdf_content(Path("book.pdf"))
    assert has_scan is True


def test_one_blank_page_does_not_send_the_book_to_the_long_chain(tmp_path: Path) -> None:
    # A born-digital book with a single textless page (7 prose + 1 blank) must
    # keep the fast route: the old any() aggregation flipped the whole book to
    # mode="long" off that one page.
    pdf = text_pdf(tmp_path / "mostly_text.pdf", pages=8, blank_pages=frozenset({7}))
    has_scan, formula_heavy = classify_pdf_content(pdf)
    assert has_scan is False
    assert formula_heavy is False
