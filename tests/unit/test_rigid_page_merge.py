"""Atomicity of the rigid per-page strip-and-merge."""

from pathlib import Path

import pikepdf
import pytest

from ubt.adapters.pdf.rigid.typesetter import _strip_and_merge_page

pytestmark = pytest.mark.fast


def _page_with_source_text(pdf: pikepdf.Pdf) -> pikepdf.Page:
    page = pdf.add_blank_page(page_size=(600, 800))
    page.Contents = pdf.make_stream(b"BT /F1 12 Tf 50 500 Td (Source prose) Tj ET\n")
    return page


def _carries_source_text(page: pikepdf.Page) -> bool:
    return b"Source prose" in bytes(page.Contents.read_bytes())


def _save_overlay(tmp_path: Path, pages: int) -> Path:
    overlay_path = tmp_path / "overlay.pdf"
    with pikepdf.new() as overlay:
        for _ in range(pages):
            overlay.add_blank_page(page_size=(600, 800))
        overlay.save(overlay_path)
    return overlay_path


def test_empty_overlay_leaves_source_text_intact(tmp_path: Path) -> None:
    pdf = pikepdf.new()
    page = _page_with_source_text(pdf)
    overlay_path = _save_overlay(tmp_path, pages=0)

    dropped, painted = _strip_and_merge_page(
        pdf, page, 1, str(overlay_path), [(40.0, 490.0, 300.0, 520.0)], [], set()
    )

    assert painted is False
    assert dropped == 0
    assert _carries_source_text(page)


def test_unloadable_overlay_leaves_source_text_intact(tmp_path: Path) -> None:
    pdf = pikepdf.new()
    page = _page_with_source_text(pdf)
    overlay_path = tmp_path / "missing.pdf"

    dropped, painted = _strip_and_merge_page(
        pdf, page, 1, str(overlay_path), [(40.0, 490.0, 300.0, 520.0)], [], set()
    )

    assert painted is False
    assert dropped == 0
    assert _carries_source_text(page)


def test_failed_paint_rolls_the_strip_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = pikepdf.new()
    page = _page_with_source_text(pdf)
    overlay_path = _save_overlay(tmp_path, pages=1)

    def broken_paint(*args: object, **kwargs: object) -> None:
        raise RuntimeError("paint failed")

    monkeypatch.setattr(pikepdf.Page, "add_overlay", broken_paint)
    dropped, painted = _strip_and_merge_page(
        pdf, page, 1, str(overlay_path), [(40.0, 490.0, 300.0, 520.0)], [], set()
    )

    assert painted is False
    assert dropped > 0
    assert _carries_source_text(page)


def test_successful_merge_strips_source_and_paints(tmp_path: Path) -> None:
    pdf = pikepdf.new()
    page = _page_with_source_text(pdf)
    overlay_path = _save_overlay(tmp_path, pages=1)

    dropped, painted = _strip_and_merge_page(
        pdf, page, 1, str(overlay_path), [(40.0, 490.0, 300.0, 520.0)], [], set()
    )

    assert painted is True
    assert dropped == 1
    assert not _carries_source_text(page)
