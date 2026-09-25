"""5: page-kind annotation hook on the PDF adapter (best-effort)."""

from pathlib import Path

import pytest

from tests.corpus_markers import requires_synthetic_mono
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.core.ir.models import BoundingBox, FlowID, IRBlock


def _block(bid: str, page: int) -> IRBlock:
    return IRBlock(
        id=bid,
        flow_id=FlowID.MAIN_STORY,
        spine_index=page,
        bbox=BoundingBox(page=page, x0=72.0, y0=100.0, x1=540.0, y1=120.0),
        source_text=f"Block on page {page}.",
    )


@requires_synthetic_mono
def test_annotate_page_kinds_real_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = DoclingPDFAdapter()
    blocks = [_block("b1", 1), _block("b2", 13)]
    # NOTE: tests/fixtures/ removed; docs/synthetic-mono.pdf is the live 13-page sample.
    pdf = Path("docs/synthetic-mono.pdf").absolute()
    # Keep the test hermetic: profile cache goes to tmp_path's .ubt.
    monkeypatch.chdir(tmp_path)
    kinds = adapter._annotate_page_kinds(pdf, blocks)
    assert len(kinds) == 13
    assert blocks[0].provenance["page_kind"] == kinds[1]
    assert blocks[1].provenance["page_kind"] == kinds[13]


def test_annotate_page_kinds_missing_file_is_noop() -> None:
    adapter = DoclingPDFAdapter()
    blocks = [_block("b1", 1)]
    assert adapter._annotate_page_kinds(Path("/nonexistent/book.pdf"), blocks) == {}
    assert "page_kind" not in blocks[0].provenance
