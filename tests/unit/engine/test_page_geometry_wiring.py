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


def test_the_strategy_does_not_optimize_the_pdf_it_compiled(
    tmp_path: Path, stub_compile: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Optimization belongs to the export stage (one owner). A strategy that
    # also compressed its output ran the lossless rewrite twice per artifact.
    import ubt.adapters.pdf.pdf_optimizer as optimizer

    calls: list[str] = []
    monkeypatch.setattr(
        optimizer,
        "optimize_pdf_in_place",
        lambda path, profile: calls.append(str(path)),
    )
    src = _source_pdf(tmp_path / "book.pdf", 1)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=True)
    _render(strategy, manifest, [_body("b1", 1)], tmp_path / "out.pdf")

    assert calls == [], "the strategy must leave byte-level optimization to export"


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
        assert isinstance(row["measured_size_pt"], float)
        assert isinstance(row["delivered_size_pt"], float)
        assert row["tier"] in {"L1", "L2"}


def test_the_delivered_size_is_the_group_decision_not_a_per_page_search(
    tmp_path: Path, stub_compile: None
) -> None:
    src = _source_pdf(tmp_path / "book.pdf", 1)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    _render(strategy, manifest, [_body("b1", 1)], tmp_path / "out.pdf")

    row = strategy.last_page_geometry[0]
    assert row["style_group_id"] == "body:document"
    assert row["delivered_size_pt"] == row["style_group_size_pt"]
    assert row["style_exception"] is False
    group = manifest.metadata["document_style_group"]["group"]
    assert row["delivered_size_pt"] == group["size_pt"]


def test_every_L1_page_of_a_document_ships_the_same_body_size(
    tmp_path: Path, stub_compile: None
) -> None:
    src = _source_pdf(tmp_path / "book.pdf", 3)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    blocks = [_body(f"b{i}", i) for i in (1, 2, 3)]
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    _render(strategy, manifest, blocks, tmp_path / "out.pdf")

    delivered = {
        row["delivered_size_pt"] for row in strategy.last_page_geometry if row["tier"] == "L1"
    }
    assert len(delivered) == 1
    assert manifest.metadata["document_style_group"]["exception_pages"] == []


def test_geometry_does_not_leak_across_renders(tmp_path: Path, stub_compile: None) -> None:
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
    strategy = DoclingRenderStrategy(delivery_mode="reflow", enable_pdf_optimization=False)
    assert strategy.last_page_geometry == []


def test_a_reflow_render_clears_the_previous_facsimile_geometry(
    tmp_path: Path, stub_compile: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _source_pdf(tmp_path / "book.pdf", 1)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    _render(strategy, manifest, [_body("b1", 1)], tmp_path / "a.pdf")
    assert len(strategy.last_page_geometry) == 1

    def _fake_publish(
        manifest: Any,
        blocks: Any,
        target_lang: str,
        output_path: Path | str,
        bilingual: bool = False,
    ) -> Path:
        return _source_pdf(Path(output_path), 1)

    # The reflow channel is Typst-only; stub its compile to isolate the
    # geometry side channel under test.
    monkeypatch.setattr(strategy, "_render_publish_sync", _fake_publish)
    strategy.delivery_mode = "reflow"
    _render(strategy, manifest, [_body("b1", 1)], tmp_path / "b.pdf")

    assert strategy.last_page_geometry == [], (
        "a reflow render must not inherit the previous facsimile geometry"
    )


def test_render_publish_sync_clears_geometry_at_entry(
    tmp_path: Path, stub_compile: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clear belongs to the entry, not to ``render_blocks``' caller.

    ``_render_publish_sync`` is reachable directly (the companion path, resumed
    jobs, tests). An emit-both run snapshots the primary geometry before the
    secondary render; the secondary's geometry channel must be empty unless it
    measured its own pages.
    """
    from ubt.core.exceptions import DocumentParseError

    src = _source_pdf(tmp_path / "book.pdf", 1)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    _render(strategy, manifest, [_body("b1", 1)], tmp_path / "a.pdf")

    # The snapshot the export stage takes before the second render.
    primary_snapshot = list(strategy.last_page_geometry)
    assert primary_snapshot, "the facsimile render measured its page"

    def _fail_compile(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return False, "stub compile"

    monkeypatch.setattr("ubt.adapters.pdf.typst_compile.typst_compile", _fail_compile)

    with pytest.raises(DocumentParseError):
        strategy._render_publish_sync(manifest, [_body("b1", 1)], "en", tmp_path / "b.pdf")

    assert strategy.last_page_geometry == [], (
        "the entry must clear geometry even when the reflow compile fails"
    )
    assert primary_snapshot[0]["page_no"] == 1
