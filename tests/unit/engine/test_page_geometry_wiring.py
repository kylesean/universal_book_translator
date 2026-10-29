"""Page geometry reaches the ledger from a real facsimile render.

``test_page_geometry.py`` pins the table and its accessors in isolation. This
file pins the *wiring*, which is a different claim and the one that has silently
failed in this project before: a correct storage layer with nothing writing to
it reads as done while the audited fact stays lost.

The Typst compile is stubbed for the same reason as in
``test_outline_wiring.py``: a preview compile is a subprocess, this module is
``fast``-marked, and ``test_marker_hygiene`` forbids that pairing. Nothing about
geometry depends on Typst's output — it depends on the solver's decision and the
source page's measured size, both of which run for real here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pikepdf
import pytest

from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    BoundingBox,
    FlowID,
    IRBlock,
    make_element,
)
from ubt.render.typst.facsimile_renderer import FacsimileRenderer

pytestmark = pytest.mark.fast


def _body(block_id: str, page: int) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=page - 1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="A line of source body text on this page.",
            bbox=BoundingBox(page=page, x0=40.0, y0=100.0, x1=360.0, y1=500.0),
        ),
        target_text="这一页的中文译文内容，用来测量排版。",
    )


def _source_pdf(path: Path, pages: int) -> Path:
    pdf = pikepdf.Pdf.new()
    for _ in range(pages):
        pdf.add_blank_page(page_size=(400, 600))
    pdf.save(path)
    pdf.close()
    return path


@pytest.fixture(name="stub_compile")
def _stub_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    def _render(self: FacsimileRenderer, pages, bg, out, build_dir, **kwargs):  # type: ignore[no-untyped-def]
        pdf = pikepdf.Pdf.new()
        for _ in pages:
            pdf.add_blank_page(page_size=(400, 600))
        pdf.save(out)
        pdf.close()
        return out

    monkeypatch.setattr(FacsimileRenderer, "render_document", _render)


@pytest.fixture(name="ledger")
def _ledger(tmp_path: Path) -> SQLiteJobLedger:
    return SQLiteJobLedger(tmp_path / "ledger.db")


def _render(strategy: DoclingRenderStrategy, manifest: Any, blocks: Any, out: Path) -> None:
    import asyncio

    asyncio.run(strategy.render_blocks(manifest, blocks, "zh", out))


def test_a_facsimile_render_records_one_row_per_page(tmp_path: Path, stub_compile: None) -> None:
    src = _source_pdf(tmp_path / "book.pdf", 3)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    blocks = [_body(f"b{i}", i) for i in (1, 2, 3)]
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    _render(strategy, manifest, blocks, tmp_path / "out.pdf")

    geometry = strategy.last_page_geometry
    assert [row["page_no"] for row in geometry] == [1, 2, 3]
    for row in geometry:
        assert row["page_w_pt"] == 400.0
        assert row["page_h_pt"] == 600.0
        # Both halves of the story: what the source measured and what shipped.
        assert isinstance(row["measured_size_pt"], float)
        assert isinstance(row["delivered_size_pt"], float)
        assert row["tier"] in {"L1", "L2"}


def test_the_delivered_size_is_the_solvers_decision_not_the_source_size(
    tmp_path: Path, stub_compile: None
) -> None:
    """A Chinese translation of the same page needs less room, so the solver is
    free to deliver *larger* than the source. Recording only ``measured_size_pt``
    would make the ledger describe the input, not the artifact."""
    src = _source_pdf(tmp_path / "book.pdf", 1)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    _render(strategy, manifest, [_body("b1", 1)], tmp_path / "out.pdf")

    row = strategy.last_page_geometry[0]
    assert row["delivered_size_pt"] != row["measured_size_pt"]
    assert row["delivered_size_pt"] > row["measured_size_pt"]


def test_geometry_does_not_leak_across_renders(tmp_path: Path, stub_compile: None) -> None:
    """A second render must describe its own pages; a stale count is reported as
    this page's geometry, which is worse than no record."""
    src_short = _source_pdf(tmp_path / "short.pdf", 1)
    src_long = _source_pdf(tmp_path / "long.pdf", 4)
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)

    first = BookManifest(doc_id="a", title="a", source_path=str(src_short), target_lang="zh")
    _render(strategy, first, [_body("b1", 1)], tmp_path / "a.pdf")
    assert len(strategy.last_page_geometry) == 1

    second = BookManifest(doc_id="b", title="b", source_path=str(src_long), target_lang="zh")
    _render(strategy, second, [_body(f"b{i}", i) for i in (1, 2, 3, 4)], tmp_path / "b.pdf")
    assert len(strategy.last_page_geometry) == 4


def test_the_rows_load_into_the_ledger(
    tmp_path: Path, stub_compile: None, ledger: SQLiteJobLedger
) -> None:
    """The end-to-end claim: what the renderer measured is what the ledger reads
    back, for the job that rendered it."""
    src = _source_pdf(tmp_path / "book.pdf", 2)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    _render(strategy, manifest, [_body(f"b{i}", i) for i in (1, 2)], tmp_path / "out.pdf")

    with ledger._get_conn() as conn:
        conn.execute(
            "INSERT INTO job_meta (job_id, doc_id, source_path, target_lang, "
            "total_blocks, status) VALUES ('job-1','d','/tmp/x.pdf','zh',0,'rendering')"
        )
    assert ledger.record_page_geometry("job-1", strategy.last_page_geometry) == 2

    stored = ledger.retrieve_page_geometry("job-1")
    assert set(stored) == {1, 2}
    assert stored[1]["delivered_size_pt"] == strategy.last_page_geometry[0]["delivered_size_pt"]
    assert stored[1]["tier"] == strategy.last_page_geometry[0]["tier"]


def test_a_non_facsimile_render_records_no_geometry(tmp_path: Path) -> None:
    """The reflow channel measures no page, so it must record nothing rather
    than zeroes that would read as "page 1 was 0pt"."""
    strategy = DoclingRenderStrategy(delivery_mode="reflow", enable_pdf_optimization=False)
    assert strategy.last_page_geometry == []
