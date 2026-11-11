"""The raw-bytes XML guard: DOCTYPEs never reach the lxml-based parsers.

F6 regression: EPUB XHTML/OPF/NCX are parsed with BeautifulSoup's lxml backend
and DOCX parts with python-docx's lxml — neither goes through defusedxml, so a
DOCTYPE carrying internal (billion-laughs) or external (``file://``) entity
declarations was expanded per the interpreter build's defaults. XHTML-ish
content gets its DOCTYPE stripped; OOXML parts fail closed.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ubt.core.xml_safety import (
    UnsafeXMLError,
    assert_no_doctype,
    strip_doctype_declarations,
)

pytestmark = pytest.mark.fast


_BILLION_LAUGHS = (
    '<?xml version="1.0"?>\n<!DOCTYPE chapter [\n<!ENTITY e0 "lol9chars!">\n'
    + "".join(f'<!ENTITY e{i} "' + f"&e{i - 1};" * 10 + '">\n' for i in range(1, 8))
    + "]>\n<chapter><p>&e7;</p></chapter>"
)

_EXTERNAL_ENTITY = (
    '<?xml version="1.0"?>'
    '<!DOCTYPE chapter [<!ENTITY stolen SYSTEM "file:///etc/hostname">]>'
    "<chapter><p>Real prose</p><p>&stolen;</p></chapter>"
)


def test_a_doctype_with_an_internal_subset_is_stripped_whole() -> None:
    raw = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "x"><html><body/></html>'
    assert "<!DOCTYPE" not in strip_doctype_declarations(raw)
    assert strip_doctype_declarations(raw).startswith("<html>")


def test_a_quoted_gt_does_not_end_the_declaration_early() -> None:
    raw = '<!DOCTYPE z [<!ENTITY a "x>y">]><z>body</z>'
    cleaned = strip_doctype_declarations(raw)
    assert "<!DOCTYPE" not in cleaned
    assert cleaned == "<z>body</z>"


def test_an_xml_stylesheet_pi_is_stripped() -> None:
    raw = '<?xml-stylesheet href="a.css" type="text/css"?><svg><rect/></svg>'
    cleaned = strip_doctype_declarations(raw)
    assert "xml-stylesheet" not in cleaned
    assert cleaned == "<svg><rect/></svg>"


def test_a_doctype_without_entities_is_stripped_too() -> None:
    raw = '<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 1.0//EN" "ncx-1.0.dtd"><ncx/>'
    assert strip_doctype_declarations(raw) == "<ncx/>"


def test_clean_markup_passes_through_unchanged() -> None:
    raw = '<?xml version="1.0" encoding="utf-8"?><z x="1"><p>a &amp; b</p></z>'
    assert strip_doctype_declarations(raw) == raw


def test_a_billion_laughs_payload_is_neutralized_before_the_parser() -> None:
    from bs4 import BeautifulSoup

    cleaned = strip_doctype_declarations(_BILLION_LAUGHS)
    # The declaration (and with it every ENTITY) is gone before any parser
    # sees the bytes, so there is nothing left to expand.
    assert "<!ENTITY" not in cleaned
    soup = BeautifulSoup(cleaned, "xml")
    # Whether the now-undefined reference survives as inert text or is dropped
    # is a parser-build detail; the security property is that nothing expanded.
    assert "lol9chars!" not in soup.get_text()
    assert "lol9chars!" not in str(soup)


def test_an_external_file_entity_is_neutralized_before_the_parser() -> None:
    from bs4 import BeautifulSoup

    cleaned = strip_doctype_declarations(_EXTERNAL_ENTITY)
    assert "<!ENTITY" not in cleaned
    assert "file:///etc/hostname" not in cleaned, (
        "the SYSTEM identifier must be stripped with the declaration"
    )
    soup = BeautifulSoup(cleaned, "xml")
    # The body survives; no host file content can appear because the entity
    # reference had nothing to expand into.
    assert "Real prose" in soup.get_text()
    assert "/etc/hostname" not in str(soup)


def test_assert_no_doctype_raises_on_a_doctype_part() -> None:
    with pytest.raises(UnsafeXMLError, match="DOCTYPE"):
        assert_no_doctype(_EXTERNAL_ENTITY, context="doc.docx:word/document.xml")


def test_assert_no_doctype_raises_on_a_stylesheet_pi() -> None:
    with pytest.raises(UnsafeXMLError, match="stylesheet"):
        assert_no_doctype(b'<?xml-stylesheet href="a.css"?><z/>', context="part")


def test_assert_no_doctype_accepts_clean_parts() -> None:
    assert_no_doctype(b"<z><w:t>plain body</w:t></z>", context="part")
    assert_no_doctype('<z xmlns:w="urn:x"/><w:t/>', context="part")


def test_the_epub_chapter_parser_neutralizes_an_entity_bearing_doctype() -> None:
    from ubt.adapters.epub.adapter import _parse_xhtml

    soup = _parse_xhtml(_EXTERNAL_ENTITY)
    text = soup.get_text()
    assert "Real prose" in text
    assert "/etc/hostname" not in text


def test_a_docx_part_carrying_a_doctype_is_refused(tmp_path: Path) -> None:
    from ubt.adapters.docx.adapter import _reject_dtd_parts
    from ubt.core.exceptions import DocumentParseError

    docx = tmp_path / "evil.docx"
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr(
            "word/document.xml",
            '<!DOCTYPE w:document [<!ENTITY a "b">]>'
            '<w:document xmlns:w="urn:x"><w:body/></w:document>',
        )
    with pytest.raises(DocumentParseError, match="DOCTYPE"):
        _reject_dtd_parts(docx)


def test_a_clean_docx_part_scan_passes(tmp_path: Path) -> None:
    from docx import Document

    from ubt.adapters.docx.adapter import _reject_dtd_parts

    docx = tmp_path / "clean.docx"
    Document().save(str(docx))
    _reject_dtd_parts(docx)  # must not raise
