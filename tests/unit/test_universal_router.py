"""Unit tests for the adaptive short/long chain router across document formats."""

import zipfile
from pathlib import Path

from ubt.core.router_mode import RouteDecision, decide


def test_markdown_short_doc_routes_short(tmp_path: Path) -> None:
    md_file = tmp_path / "article.md"
    md_file.write_text(
        "# My Article\n\n" + "This is a short article about semiconductor physics.\n\n" * 10,
        encoding="utf-8",
    )
    decision = decide(md_file)
    assert decision.mode == "short"
    assert decision.chapters == 1
    assert decision.has_scan is False
    assert decision.estimated_tokens < 25000
    assert "short born-digital" in decision.reason


def test_markdown_multi_chapter_routes_long(tmp_path: Path) -> None:
    md_file = tmp_path / "book.md"
    content = ""
    for i in range(5):
        content += f"# Chapter {i + 1}\n\nSome text content for chapter {i + 1}...\n" * 20
    md_file.write_text(content, encoding="utf-8")
    decision = decide(md_file)
    assert decision.mode == "long"
    assert decision.chapters >= 3
    assert "multi-chapter structured document" in decision.reason


def test_html_document_routing(tmp_path: Path) -> None:
    html_short = tmp_path / "page.html"
    html_short.write_text(
        "<html><body><h1>Title</h1><p>Paragraph text content...</p></body></html>", encoding="utf-8"
    )
    decision = decide(html_short)
    # Very short (<200 chars) routes long as unreadable/empty
    assert decision.mode == "long"

    html_regular = tmp_path / "article.html"
    html_regular.write_text(
        "<html><body><h1>Article Title</h1>"
        + "<p>Academic prose discussing bandgap engineering.</p>" * 20
        + "</body></html>",
        encoding="utf-8",
    )
    decision = decide(html_regular)
    assert decision.mode == "short"
    assert decision.chapters == 1


def test_epub_document_routing(tmp_path: Path) -> None:
    epub_file = tmp_path / "sample.epub"
    with zipfile.ZipFile(epub_file, "w") as zf:
        opf_xml = """<?xml version="1.0"?>
<package version="2.0" xmlns="http://www.idpf.org/2007/opf">
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
    <itemref idref="c3"/>
    <itemref idref="c4"/>
  </spine>
</package>"""
        zf.writestr("content.opf", opf_xml)
        zf.writestr("c1.xhtml", "<p>" + "Chapter 1 text. " * 50 + "</p>")
        zf.writestr("c2.xhtml", "<p>" + "Chapter 2 text. " * 50 + "</p>")
        zf.writestr("c3.xhtml", "<p>" + "Chapter 3 text. " * 50 + "</p>")
        zf.writestr("c4.xhtml", "<p>" + "Chapter 4 text. " * 50 + "</p>")

    decision = decide(epub_file)
    assert decision.mode == "long"
    assert decision.chapters >= 3
    assert "multi-chapter structured document" in decision.reason


def test_non_pdf_forced_modes(tmp_path: Path) -> None:
    md_file = tmp_path / "article.md"
    md_file.write_text("# Article\n\nContent " * 100, encoding="utf-8")
    assert decide(md_file, exec_mode="long").mode == "long"
    assert decide(md_file, exec_mode="short").mode == "short"


def test_route_decision_dict_fields() -> None:
    d = RouteDecision(
        mode="short",
        pages=5,
        chars=2000,
        has_scan=False,
        formula_heavy=False,
        reason="test",
        chapters=2,
        estimated_tokens=500,
    )
    dt = d.to_dict()
    assert dt["mode"] == "short"
    assert dt["chapters"] == 2
    assert dt["estimated_tokens"] == 500
