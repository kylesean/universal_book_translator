"""Unit tests for DOMCodec (EPUB/HTML), DOCXCodec (OOXML+OMML), MarkdownCodec & registry."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from lxml import etree

from ubt.codecs.docx import DOCXCodec
from ubt.codecs.dom import BILINGUAL_TARGET_CLASS, DOMCodec
from ubt.codecs.markdown import MarkdownCodec
from ubt.codecs.protocol import DeliveryMode
from ubt.codecs.registry import resolve_codec


def _make_sample_epub(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            '<manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>'
            '<spine><itemref idref="ch1"/></spine></package>',
        )
        zf.writestr(
            "OEBPS/ch1.xhtml",
            "<html><body>"
            '<h1 id="sec1">Introduction</h1>'
            '<p class="lead">See <strong>Fig. 1</strong> and <a href="https://example.com">link</a>'
            ' with <math><mi>x</mi></math> and <img src="a.png" alt="a"/>.</p>'
            "</body></html>",
        )
    return path


def _make_sample_docx(path: Path) -> Path:
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<w:body>"
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
        "<w:r><w:t>Chapter One</w:t></w:r></w:p>"
        "<w:p>"
        '<w:r><w:t xml:space="preserve">Value </w:t></w:r>'
        "<w:r><w:rPr><w:b/></w:rPr><w:t>alpha</w:t></w:r>"
        '<w:r><w:t xml:space="preserve"> equals </w:t></w:r>'
        "<m:oMath><m:r><m:t>E=mc^2</m:t></m:r></m:oMath>"
        "</w:p>"
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
        "<w:r><w:t>Chapter Two</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Second chapter body.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", doc_xml)
    return path


@pytest.mark.fast
def test_dom_codec_epub_preserves_inline_tags_math_and_bilingual_clones(tmp_path: Path) -> None:
    src = _make_sample_epub(tmp_path / "book.epub")
    codec = resolve_codec(src)
    assert isinstance(codec, DOMCodec)

    prof, units = codec.extract(src)
    assert prof.format_name == "epub"
    assert prof.has_math is True
    assert len(units) == 2

    para = units[1]
    assert "<b1>Fig. 1</b1>" in para.source_marked
    assert "<a2>link</a2>" in para.source_marked
    assert "<x3/>" in para.source_marked
    assert "<x4/>" in para.source_marked

    translations = {
        units[0].unit_id: "引言",
        para.unit_id: "参见<b1>图 1</b1>和<a2>链接</a2>，含 <x3/> 与 <x4/>。",
    }
    out_bi = tmp_path / "book_bi.epub"
    rep = codec.apply(
        src, units, translations, target_lang="zh", mode=DeliveryMode.BILINGUAL, output_path=out_bi
    )
    assert rep.patched_units == 2

    with zipfile.ZipFile(out_bi, "r") as zf:
        ch1 = zf.read("OEBPS/ch1.xhtml").decode("utf-8")
    assert "<strong>图 1</strong>" in ch1
    assert 'href="https://example.com"' in ch1
    assert "<math><mi>x</mi></math>" in ch1
    assert 'src="a.png"' in ch1
    assert BILINGUAL_TARGET_CLASS in ch1
    # Duplicate id="sec1" must be stripped on the bilingual clone.
    assert ch1.count('id="sec1"') == 1


@pytest.mark.fast
def test_docx_codec_preserves_rpr_omml_and_splits_chapters(tmp_path: Path) -> None:
    src = _make_sample_docx(tmp_path / "paper.docx")
    codec = resolve_codec(src)
    assert isinstance(codec, DOCXCodec)

    prof, units = codec.extract(src)
    assert prof.format_name == "docx"
    assert prof.has_math is True
    assert len(units) == 4
    # Multi-chapter partitioning on Heading1!
    assert len({u.chapter_id for u in units}) == 2

    eq_unit = units[1]
    assert "<r1>alpha</r1>" in eq_unit.source_marked
    assert "<x2/>" in eq_unit.source_marked

    translations = {
        units[0].unit_id: "第一章",
        eq_unit.unit_id: "数值 <r1>阿尔法</r1> 等于 <x2/>",
        units[2].unit_id: "第二章",
        units[3].unit_id: "第二章正文。",
    }
    out_mono = tmp_path / "paper_zh.docx"
    rep = codec.apply(
        src, units, translations, target_lang="zh", mode=DeliveryMode.MONO, output_path=out_mono
    )
    assert rep.patched_units == 4

    with zipfile.ZipFile(out_mono, "r") as zf:
        root = etree.fromstring(zf.read("word/document.xml"))
    xml_text = etree.tostring(root, encoding="unicode")
    assert "阿尔法" in xml_text
    assert "<w:b/>" in xml_text
    assert "E=mc^2" in xml_text
    assert "Noto Serif CJK SC" in xml_text


@pytest.mark.fast
def test_markdown_codec_preserves_frontmatter_code_and_math(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text(
        "---\ntitle: Demo\n---\n\n"
        "# Heading One\n\n"
        "Use `uv run` with $x_i$ and [docs](https://example.org).\n\n"
        "```python\nprint('untouched')\n```\n\n"
        "$$\n\\sum_{i=1}^n x_i\n$$\n",
        encoding="utf-8",
    )
    codec = resolve_codec(src)
    assert isinstance(codec, MarkdownCodec)

    _, units = codec.extract(src)
    assert len(units) == 2
    assert "<x1/>" in units[1].source_marked
    assert "<x2/>" in units[1].source_marked
    assert "<a3>docs</a3>" in units[1].source_marked

    translations = {
        units[0].unit_id: "一级标题",
        units[1].unit_id: "配合 <x2/> 使用 <x1/>，详见<a3>文档</a3>。",
    }
    out = tmp_path / "doc_zh.md"
    codec.apply(src, units, translations, target_lang="zh", mode=DeliveryMode.MONO, output_path=out)
    rendered = out.read_text(encoding="utf-8")
    assert rendered.startswith("---\ntitle: Demo\n---\n")
    assert "# 一级标题" in rendered
    assert "配合 $x_i$ 使用 `uv run`，详见[文档](https://example.org)。" in rendered
    assert "print('untouched')" in rendered
    assert "\\sum_{i=1}^n x_i" in rendered


@pytest.mark.fast
def test_epub_xhtml_doctype_is_stripped_before_parsing(tmp_path: Path) -> None:
    src = _make_sample_epub(tmp_path / "doctype.epub")
    bomb = (
        '<?xml version="1.0"?>\n<!DOCTYPE html [\n'
        '<!ENTITY e0 "lol9chars!">\n'
        + "".join(f'<!ENTITY e{i} "' + f"&e{i - 1};" * 10 + '">\n' for i in range(1, 8))
        + "]>\n"
    )
    with zipfile.ZipFile(src) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    members["OEBPS/ch1.xhtml"] = (
        bomb + "<html><body><h1>Safe</h1><p>Hello &e7;</p></body></html>"
    ).encode()
    with zipfile.ZipFile(src, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    prof, units = DOMCodec().extract(src)
    assert prof.format_name == "epub"
    assert any("Safe" in unit.source_plain for unit in units)
    assert all("lol9chars" not in unit.source_plain for unit in units)


@pytest.mark.fast
def test_docx_part_with_a_doctype_is_refused(tmp_path: Path) -> None:
    from ubt.core.xml_safety import UnsafeXMLError

    src = _make_sample_docx(tmp_path / "doctype.docx")
    with zipfile.ZipFile(src) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    members["word/document.xml"] = (
        b'<?xml version="1.0"?><!DOCTYPE w:document><w:document>'
        b"<w:body><w:p><w:r><w:t>x</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(src, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    with pytest.raises(UnsafeXMLError, match="DOCTYPE"):
        DOCXCodec().extract(src)
