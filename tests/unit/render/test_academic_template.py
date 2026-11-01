from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from ubt.adapters.pdf.typst_compile import resolve_typst_binary
from ubt.core.ir.models import BlockType, BookManifest, FlowID, IRBlock, make_element
from ubt.render.typst.academic import (
    academic_preamble,
    academic_title_banner,
    is_academic_document,
    resolve_document_columns,
)
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def _compile_typst(doc_content: str) -> tuple[bool, str]:
    try:
        binary = resolve_typst_binary(os.environ.get("UBT_TYPST_BINARY", "typst"))
        if not binary:
            pytest.skip("typst compiler not found on system")
    except Exception:
        pytest.skip("typst compiler not found on system")

    with tempfile.NamedTemporaryFile("w", suffix=".typ") as f:
        f.write(doc_content)
        f.flush()
        out_pdf = Path(f.name).with_suffix(".pdf")
        try:
            res = subprocess.run(
                [binary, "compile", f.name, str(out_pdf)],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode == 0, res.stderr
        finally:
            if out_pdf.exists():
                out_pdf.unlink(missing_ok=True)


def test_is_academic_document() -> None:
    manifest_paper = BookManifest(
        doc_id="d1",
        source_path="sample.pdf",
        title="Attention",
        metadata={"doc_type": "paper"},
    )
    assert is_academic_document(manifest_paper)

    manifest_profile = BookManifest(
        doc_id="d4",
        source_path="paper.pdf",
        title="A Paper",
        metadata={"genre_profile": "academic"},
    )
    assert is_academic_document(manifest_profile)

    manifest_thesis = BookManifest(
        doc_id="d5",
        source_path="thesis.pdf",
        title="Ph.D. Thesis",
        metadata={"doc_type": "thesis"},
    )
    assert is_academic_document(manifest_thesis)

    manifest_book = BookManifest(
        doc_id="d3",
        source_path="book.epub",
        title="Novel",
        metadata={"genre_profile": "novel", "page_kinds": {1: "editable_text"}},
    )
    assert not is_academic_document(manifest_book)
    assert not is_academic_document(None)


def test_resolve_document_columns_fidelity_first() -> None:
    manifest_single = BookManifest(
        doc_id="d1",
        source_path="neurips_paper.pdf",
        title="NeurIPS Single Column",
        metadata={"genre_profile": "academic"},
    )
    assert resolve_document_columns(manifest_single) == 1

    manifest_multi_share = BookManifest(
        doc_id="d3",
        source_path="acm_paper.pdf",
        title="ACM Dual Column",
        metadata={"genre_profile": "paper"},
    )
    manifest_multi_share.run.route_decision = {
        "mode": "long",
        "multicolumn_page_share": 0.72,
    }
    assert resolve_document_columns(manifest_multi_share) == 2

    manifest_multi_flag = BookManifest(
        doc_id="d2",
        source_path="ieee_paper.pdf",
        title="IEEE Dual Column",
        metadata={"genre_profile": "academic"},
    )
    manifest_multi_flag.run.route_decision = {"has_multicolumn": True}
    assert resolve_document_columns(manifest_multi_flag) == 2

    manifest_noise = BookManifest(
        doc_id="d6",
        source_path="book.pdf",
        title="Book",
        metadata={"genre_profile": "general"},
    )
    manifest_noise.run.route_decision = {"multicolumn_page_share": 0.225}
    assert resolve_document_columns(manifest_noise) == 1

    manifest_mostly_single = BookManifest(
        doc_id="d7",
        source_path="single.pdf",
        title="Single Column",
        metadata={"genre_profile": "paper"},
    )
    manifest_mostly_single.run.route_decision = {"multicolumn_page_share": 0.05}
    assert resolve_document_columns(manifest_mostly_single) == 1

    manifest_override_dual = BookManifest(
        doc_id="d4",
        source_path="single_paper.pdf",
        title="Paper",
        metadata={"genre_profile": "academic"},
    )
    manifest_override_dual.run.layout_columns = 2
    assert resolve_document_columns(manifest_override_dual) == 2

    manifest_override_single = BookManifest(
        doc_id="d5",
        source_path="dual_paper.pdf",
        title="Paper",
        metadata={"genre_profile": "paper"},
    )
    manifest_override_single.run.route_decision = {"has_multicolumn": True}
    manifest_override_single.run.layout_columns = 1
    assert resolve_document_columns(manifest_override_single) == 1


def test_resolve_document_columns_reads_the_real_manifest_shape() -> None:
    manifest = BookManifest(
        doc_id="real",
        source_path="2101.03961v3.pdf",
        title="Switch Transformers",
        metadata={"genre_profile": "paper"},
    )
    manifest.run.route_decision = {
        "mode": "long",
        "pages": 40,
        "multicolumn_page_share": 0.0,
        "structural_page_share": 0.3,
    }
    assert resolve_document_columns(manifest) == 1, (
        "a single-column paper must stay single-column — --profile never forces two"
    )

    manifest.run.route_decision = {"mode": "long", "multicolumn_page_share": 0.83}
    assert resolve_document_columns(manifest) == 2


def _block(bid: str, block_type: BlockType, source: str, target: str) -> IRBlock:
    from ubt.core.ir.models import make_element

    return IRBlock(
        element=make_element(id=bid, spine_index=1, block_type=block_type, source_text=source),
        target_text=target,
    )


def test_single_column_academic_paper_renders_clean_single_column(tmp_path: Path) -> None:
    import asyncio

    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy

    manifest = BookManifest(
        doc_id="d",
        title="Attention Is All You Need",
        source_path="paper.pdf",
        metadata={"genre_profile": "academic"},
    )
    blocks = [
        _block("b1", BlockType.NARRATIVE, "We present the Transformer.", "我们提出 Transformer。"),
    ]
    out = tmp_path / "paper.pdf"
    strategy = DoclingRenderStrategy()
    asyncio.run(
        strategy.render_blocks(manifest=manifest, blocks=blocks, target_lang="zh", output_path=out)
    )
    assert out.exists() and out.stat().st_size > 0
    typ = (tmp_path / f".{out.stem}_typst" / "main.typ").read_text(encoding="utf-8")
    assert "columns: 1" in typ, "single-column paper was unexpectedly forced to two columns"
    assert "columns: 2" not in typ
    assert "size: 10.5pt" in typ
    assert "Attention Is All You Need" in typ


def test_dual_column_academic_paper_renders_dual_column(tmp_path: Path) -> None:
    import asyncio

    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy

    manifest = BookManifest(
        doc_id="d_ieee",
        title="IEEE Dual Column Paper",
        source_path="ieee.pdf",
        metadata={"genre_profile": "paper"},
    )
    manifest.run.route_decision = {"mode": "long", "multicolumn_page_share": 0.6}
    blocks = [
        _block("b1", BlockType.NARRATIVE, "Measurements were taken.", "进行了测量。"),
    ]
    out = tmp_path / "ieee.pdf"
    strategy = DoclingRenderStrategy()
    asyncio.run(
        strategy.render_blocks(manifest=manifest, blocks=blocks, target_lang="zh", output_path=out)
    )
    assert out.exists() and out.stat().st_size > 0
    typ = (tmp_path / f".{out.stem}_typst" / "main.typ").read_text(encoding="utf-8")
    assert "columns: 2" in typ
    assert "size: 9.5pt" in typ
    assert 'scope: "parent"' in typ


def test_academic_preamble_structure() -> None:
    preamble_single = academic_preamble(target_lang="zh", paper_title="Test Paper", columns=1)
    assert "columns: 1" in preamble_single
    assert "margin: (x: 2.5cm, y: 2.5cm)" in preamble_single
    assert "size: 10.5pt" in preamble_single
    assert "Test Paper" in preamble_single

    preamble_dual = academic_preamble(target_lang="zh", paper_title="Test Paper", columns=2)
    assert "columns: 2" in preamble_dual
    assert "margin: (x: 1.8cm, y: 2.0cm)" in preamble_dual
    assert "size: 9.5pt" in preamble_dual
    assert "Test Paper" in preamble_dual


def test_academic_title_banner() -> None:
    banner_single = academic_title_banner(
        title="Single Column Title",
        abstract="Single column abstract.",
        authors=["Author One"],
        columns=1,
    )
    assert "block(width: 100%" in banner_single
    assert "Single Column Title" in banner_single

    banner_dual = academic_title_banner(
        title="Dual Column Title",
        abstract="Dual column abstract.",
        authors=["Author One"],
        columns=2,
    )
    assert 'scope: "parent"' in banner_dual
    assert "Dual Column Title" in banner_dual


def test_typst_renderer_compiles_both_single_and_dual_columns(tmp_path: Path) -> None:
    from ubt.core.ir.models import make_element

    for cols in (1, 2):
        renderer = TypstDocumentRenderer(
            layout_template="academic",
            columns=cols,
            paper_title="Attention Is All You Need",
            abstract="The dominant sequence transduction models are based on complex neural networks.",
        )

        el1 = make_element(
            id="b1",
            spine_index=0,
            block_type=BlockType.HEADING,
            source_text="1. Introduction",
            level=1,
        )
        el2 = make_element(
            id="b2",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text="Recurrent models compute factorizations.",
        )
        el3 = make_element(
            id="b3",
            spine_index=0,
            block_type=BlockType.FORMULA,
            source_text=r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        )

        blocks = [
            IRBlock(element=el1, target_text="1. 引言"),
            IRBlock(element=el2, target_text="循环模型通常沿着序列进行计算。"),
            IRBlock(
                element=el3,
                target_text=r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
            ),
        ]

        build_dir = tmp_path / f"academic_test_col_{cols}"
        source, _ = renderer.build_source(blocks, build_dir, target_lang="zh")
        if cols == 1:
            assert "columns: 1" in source
            assert "size: 10.5pt" in source
        else:
            assert "columns: 2" in source
            assert "size: 9.5pt" in source

        ok, stderr = _compile_typst(source)
        assert ok, f"Typst compile failed for columns={cols}: {stderr}"


def _toc_block(bid: str, title: str, page: str) -> IRBlock:
    from ubt.core.ir.models import BlockProvenance, make_element

    element = make_element(id=bid, spine_index=1, block_type=BlockType.HEADING, source_text=title)
    return IRBlock(
        element=element,
        target_text=title,
        provenance=BlockProvenance(toc_entry=True, toc_page=page),
    )


def _toc_block_run(
    ids: list[str], titles: list[str], *, pages: list[str] | None = None
) -> list[IRBlock]:
    return [
        _toc_block(bid, title, pages[i] if pages else "")
        for i, (bid, title) in enumerate(zip(ids, titles, strict=True))
    ]


def _heading_block(bid: str, text: str) -> IRBlock:
    from ubt.core.ir.models import make_element

    element = make_element(id=bid, spine_index=1, block_type=BlockType.HEADING, source_text=text)
    return IRBlock(element=element, target_text=text)


def _para_block(bid: str, text: str) -> IRBlock:
    from ubt.core.ir.models import make_element

    element = make_element(id=bid, spine_index=1, block_type=BlockType.NARRATIVE, source_text=text)
    return IRBlock(element=element, target_text=text)


def test_toc_entries_group_into_runs() -> None:
    from ubt.render.typst.document_renderer import _group_toc_runs

    blocks = [
        _toc_block("t1", "1 引言", "3"),
        _toc_block("t2", "2 切换 Transformer", "4"),
        _heading_block("h1", "1 引言"),
        _toc_block("t3", "3 扩展特性", "11"),
    ]
    runs = _group_toc_runs(blocks)
    assert [[b.id for b in run] for _, run in runs] == [["t1", "t2"], ["t3"]]
    assert [start for start, _ in runs] == [0, 3]


def test_toc_introducing_heading_is_consumed(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(
        layout_template="academic", columns=1, paper_title="Paper", toc_mode="on"
    )
    blocks = [
        _heading_block("h0", "Contents"),
        _toc_block("t1", "1 引言", "3"),
        _toc_block("t2", "2 切换 Transformer", "4"),
        _heading_block("h1", "1 引言"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "#heading(level: 1)[目录]" not in source
    assert source.count("#heading(level: 1)") == 1
    assert source.count("目录") == 1


def test_a_paper_places_no_recovered_toc_by_default(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(layout_template="academic", columns=1, paper_title="Paper")
    blocks = [
        _toc_block("t1", "6 相关工作", "24"),
        _toc_block("t2", "7 讨论", "25"),
        _heading_block("h1", "1 引言"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" not in source
    assert source.count("#heading(level: 1)") == 1


def test_a_mislabelled_run_of_lines_does_not_invent_a_toc(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(layout_template="academic", columns=1, paper_title="Paper")
    blocks = [
        *_toc_block_run(
            [f"ref-{i}" for i in range(5)], [f"[{i}] Some cited work" for i in range(5)]
        ),
        _heading_block("h1", "1 Introduction"),
        _heading_block("h2", "2 Method"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" not in source


def test_a_real_contents_page_still_earns_its_place(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(layout_template="academic", columns=1, paper_title="Paper")
    titles = ["1 Introduction", "2 Method", "3 Results", "4 Discussion", "5 Conclusion"]
    blocks = [
        *_toc_block_run([f"t{i}" for i in range(5)], titles, pages=[str(i + 1) for i in range(5)]),
        *[_heading_block(f"h{i}", t) for i, t in enumerate(titles)],
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" in source


def test_a_book_places_its_recovered_toc_after_the_front_matter(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(
        layout_template="academic", columns=1, paper_title="P", toc_mode="on"
    )
    blocks = [
        _toc_block("t1", "1 引言", "3"),
        _toc_block("t2", "2 方法", "4"),
        _heading_block("h1", "1 引言"),
        _para_block("p1", "正文。"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert source.index("P") < source.index("目录")
    assert source.index("目录") < source.index("#heading(level: 1)")


def test_toc_mode_on_forces_it_for_a_paper(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(
        layout_template="academic", columns=1, paper_title="P", toc_mode="on"
    )
    blocks = [_toc_block("t1", "1 引言", "3"), _heading_block("h1", "1 引言")]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" in source


def test_toc_mode_off_suppresses_a_book_toc(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(layout_template="book", toc_mode="off")
    blocks = [_toc_block("t1", "Chapter One", "3"), _para_block("p1", "Body.")]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" not in source


def test_no_toc_is_ever_invented(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(layout_template="book", toc_mode="on")
    blocks = [_heading_block("h1", "Chapter One"), _para_block("p1", "Body.")]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" not in source


def test_toc_rows_do_not_become_section_headings(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(
        layout_template="academic", columns=1, paper_title="Paper", toc_mode="on"
    )
    blocks = [
        _toc_block("t1", "6 相关工作", "24"),
        _toc_block("t2", "7 讨论", "25"),
        _toc_block("t3", "8 未来工作", "26"),
        _heading_block("h1", "1 引言"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" in source
    assert "6 相关工作" in source and "24" in source
    assert source.count("#heading(level: 1)") == 1


def test_toc_renders_and_compiles(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(
        layout_template="academic", columns=1, paper_title="Paper", toc_mode="on"
    )
    blocks = [
        _toc_block("t1", "1 引言", "3"),
        _toc_block("t2", "2 切换 Transformer", "4"),
        _toc_block("t3", "2.1 稀疏路由的简化", "5"),
        _para_block("p1", "这是正文。"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    ok, stderr = _compile_typst(source)
    assert ok, f"Typst compile failed for a TOC document: {stderr}"


def test_a_paper_with_explicit_toc_heading_places_toc_in_auto_mode(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(layout_template="academic", columns=1, paper_title="Paper")
    blocks = [
        _heading_block("h_toc", "Contents"),
        _toc_block("t1", "1 引言", "3"),
        _toc_block("t2", "2 切换 Transformer", "4"),
        _heading_block("h1", "1 引言"),
        _para_block("p1", "这是正文。"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" in source
    assert "1 引言" in source and "3" in source


def test_a_paper_with_substantial_toc_places_toc_in_auto_mode(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(layout_template="academic", columns=1, paper_title="Paper")
    blocks = [_toc_block(f"t{i}", f"第{i}章 概念", str(i + 2)) for i in range(1, 6)] + [
        _heading_block("h1", "第1章 概念"),
        _para_block("p1", "这是正文。"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "目录" in source
    assert "第1章 概念" in source


def test_footnote_block_does_not_emit_inline_footnote_callout(tmp_path: Path) -> None:
    from ubt.core.ir.models import BlockType, make_element

    renderer = TypstDocumentRenderer(layout_template="academic", columns=1, paper_title="Paper")
    blocks = [
        _para_block("p1", "这是关键词：机器学习。"),
        IRBlock(
            element=make_element(
                id="fn1",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                flow_id=FlowID.FOOTNOTE,
                source_text="1. Equal contribution.",
            ),
            target_text="1. 贡献均等。",
        ),
        _para_block("p2", "这是后续正文。"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    assert "#footnote[" not in source
    assert "贡献均等。" in source
    assert "fill: luma(80)" in source


def test_hangul_is_measured_as_full_width() -> None:
    from ubt.render.typst.document_renderer import render_table

    def emitted_scale(text: str, width: float) -> float:
        markup = f"| A | B |\n|---|---|\n| {text} | {text} |"
        rendered = render_table(markup, content_width_pt=width)
        for line in rendered.splitlines():
            stripped = line.strip()
            if stripped.startswith("#set text(size:"):
                return float(stripped.split(":")[1].strip().removesuffix("em)"))
        return 1.0

    korean = "한국어문장입니다만"
    han = "中文句子内容测试吗"
    latin = "a" * len(korean)

    assert emitted_scale(korean, 150) == emitted_scale(han, 150)
    assert emitted_scale(korean, 150) < emitted_scale(latin, 150)


def test_a_korean_word_is_a_break_opportunity() -> None:
    from ubt.render.typst.document_renderer import render_table

    def required_scale(text: str) -> float:
        markup = f"| A | B |\n|---|---|\n| {text} | {text} |"
        rendered = render_table(markup, content_width_pt=90)
        for line in rendered.splitlines():
            stripped = line.strip()
            if stripped.startswith("#set text(size:"):
                return float(stripped.split(":")[1].strip().removesuffix("em)"))
        return 1.0

    long_korean = "한국어로작성된아주긴문장입니다정말로깁니다"
    long_latin = "abcdefghijklmnopqrstuvwxyzabcdefghij"
    assert required_scale(long_korean) >= required_scale(long_latin)


def test_an_asset_name_cannot_escape_its_typst_string(tmp_path: Path) -> None:
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    renderer = TypstDocumentRenderer()
    build = tmp_path / "build"
    build.mkdir()
    source = tmp_path / 'weird"name\\x.png'
    source.write_bytes(b"\x89PNG\r\n\x1a\n")

    copied = renderer._copy_asset(source, build)
    assert '"' not in copied.name
    assert "\\" not in copied.name
    assert copied.exists()


def test_a_table_at_the_readability_floor_is_flagged(tmp_path: Path) -> None:
    from ubt.render.typst.document_renderer import render_table

    md = (
        "| Header A | Header B | Header C | Header D |\n|---|---|---|---|\n"
        "| 很长的中文单元格 | 另一个很长的内容 | 第三个很长的 | 第四个很长的 |"
    )
    scales: list[float] = []
    render_table(md, content_width_pt=120, scale_out=scales)
    assert scales and scales[0] <= 0.60 + 1e-6

    comfortable: list[float] = []
    render_table(md, content_width_pt=520, scale_out=comfortable)
    assert comfortable and comfortable[0] > 0.60


def test_document_renderer_wires_table_continuation_by_default() -> None:
    grid = "\n".join(["| Op | Cost |", "|---|---|"] + [f"| op{i} | {i}.0 |" for i in range(6)])
    blocks = [
        IRBlock(
            element=make_element(
                id="b1",
                spine_index=1,
                block_type=BlockType.TABLE,
                source_text=grid,
            ),
            target_text=grid,
        )
    ]
    renderer = TypstDocumentRenderer()
    source, _stats = renderer.build_source(blocks, Path(tempfile.mkdtemp()))
    assert "ubt-tbl-start-tgt-b1" in source
    assert "ubt-tbl-end-tgt-b1" in source
    assert "（续下页）" in source
    assert "（续表）" in source
