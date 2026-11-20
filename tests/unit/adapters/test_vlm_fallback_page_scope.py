from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ubt.adapters.pdf.docling_parser import vlm_fallback_missing_pages
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element

pytestmark = pytest.mark.fast


class _FakePdf:
    def __init__(self, _path: str) -> None:
        self._pages = [object(), object(), object()]

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> object:
        return self._pages[index]

    def close(self) -> None:
        return


def test_page_scope_does_not_leave_the_all_mode_profile_bounds_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="A page with text.",
            bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=100.0, y1=20.0),
        )
    )

    import pypdfium2

    import ubt.adapters.pdf.page_profiler as page_profiler
    import ubt.adapters.pdf.vlm.registry as registry

    monkeypatch.setattr(pypdfium2, "PdfDocument", _FakePdf)
    monkeypatch.setattr(
        page_profiler,
        "profile_pdf",
        lambda *_args, **_kwargs: [
            SimpleNamespace(facts=SimpleNamespace(page=1), kind=SimpleNamespace(value="figure"))
        ],
    )
    monkeypatch.setattr(
        registry,
        "probe_effective_driver",
        lambda **_kwargs: (None, None),
    )

    result = vlm_fallback_missing_pages(
        Path("scope.pdf"),
        [block],
        ocr_fallback="all",
        page_scope=[1],
    )
    assert result == [block]


def test_a_real_textless_pdf_is_refused_instead_of_delivering_an_empty_book(
    tmp_path: Path,
) -> None:
    from pdf_builders import write_text_pdf

    from ubt.core.exceptions import DocumentParseError

    path = write_text_pdf(tmp_path / "scan.pdf", [[], []], background=(1.0, 1.0, 1.0))
    with pytest.raises(DocumentParseError, match="every page lacks an extractable text layer"):
        vlm_fallback_missing_pages(path, [], ocr_fallback="off", ocr_mode="off")


def test_a_partially_textless_pdf_names_the_missing_pages(tmp_path: Path) -> None:
    from pdf_builders import write_text_pdf

    from ubt.core.exceptions import DocumentParseError

    path = write_text_pdf(
        tmp_path / "partial.pdf", [["A readable page."], [], []], background=(1.0, 1.0, 1.0)
    )
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="A readable page.",
            bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=100.0, y1=20.0),
        )
    )
    with pytest.raises(DocumentParseError, match=r"pages 2, 3"):
        vlm_fallback_missing_pages(path, [block], ocr_fallback="off", ocr_mode="off")
