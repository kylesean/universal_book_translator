from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from ubt.core.xml_safety import UnsafeXMLError, parse_xml, parse_xml_file
from ubt.core.zip_safety import ZipReadBudget, read_member

pytestmark = pytest.mark.fast


def _zip_with(names_sizes: dict[str, bytes]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in names_sizes.items():
            zf.writestr(name, data)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_a_member_over_the_member_cap_is_refused() -> None:
    zf = _zip_with({"big.bin": b"x" * 5000, "small.bin": b"ok"})
    budget = ZipReadBudget(max_member_bytes=100, max_total_bytes=10_000)
    assert read_member(zf, "big.bin", budget) is None
    assert read_member(zf, "small.bin", budget) == b"ok"


def test_members_under_the_member_cap_still_hit_the_archive_total() -> None:
    zf = _zip_with({f"m{i}.txt": b"y" * 50 for i in range(10)})
    budget = ZipReadBudget(max_member_bytes=100, max_total_bytes=200)
    read = [read_member(zf, f"m{i}.txt", budget) for i in range(10)]
    assert sum(1 for r in read if r is not None) == 4
    assert all(r is None for r in read[4:])


def test_a_missing_member_reads_as_none() -> None:
    zf = _zip_with({"a.txt": b"a"})
    assert read_member(zf, "nope.txt", ZipReadBudget()) is None


def test_the_default_budget_admits_an_ordinary_member() -> None:
    zf = _zip_with({"chapter.xhtml": b"<p>hello</p>"})
    assert read_member(zf, "chapter.xhtml") == b"<p>hello</p>"


def test_a_plain_document_parses() -> None:
    root = parse_xml('<pkg xmlns="http://www.idpf.org/2007/opf"><spine/></pkg>')
    assert root.tag.endswith("pkg")


def test_the_predefined_entities_still_parse() -> None:
    assert parse_xml("<a>x &amp; y &lt; z</a>").text == "x & y < z"


def test_a_doctype_declaration_without_entities_is_allowed() -> None:
    root = parse_xml('<?xml version="1.0"?><!DOCTYPE tmx SYSTEM "tmx11.dtd"><tmx/>')
    assert root.tag == "tmx"


def test_an_internal_entity_is_refused() -> None:
    with pytest.raises(UnsafeXMLError):
        parse_xml('<?xml version="1.0"?><!DOCTYPE z [<!ENTITY a "b">]><z>&a;</z>')


def test_a_billion_laughs_document_is_refused() -> None:
    parts = ['<!ENTITY e0 "lol">']
    for i in range(1, 8):
        parts.append(f'<!ENTITY e{i} "' + f"&e{i - 1};" * 10 + '">')
    bomb = '<?xml version="1.0"?>\n<!DOCTYPE z [\n' + "\n".join(parts) + f"\n]>\n<z>&e{7};</z>"
    with pytest.raises(UnsafeXMLError):
        parse_xml(bomb)


def test_an_external_entity_is_refused() -> None:
    xxe = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE foo [<!ENTITY x SYSTEM "file:///etc/hostname">]>'
        "<foo>&x;</foo>"
    )
    with pytest.raises(UnsafeXMLError):
        parse_xml(xxe)


def test_malformed_xml_raises_the_guard_error(tmp_path: Path) -> None:
    with pytest.raises(UnsafeXMLError):
        parse_xml("<a><unclosed>")


def test_parse_xml_file_reads_a_path(tmp_path: Path) -> None:
    doc = tmp_path / "ok.xml"
    doc.write_text("<root>hi</root>", encoding="utf-8")
    assert parse_xml_file(doc).text == "hi"


def test_parse_xml_file_refuses_an_entity_bearing_file(tmp_path: Path) -> None:
    doc = tmp_path / "evil.xml"
    doc.write_text(
        '<?xml version="1.0"?><!DOCTYPE z [<!ENTITY a "b">]><z>&a;</z>', encoding="utf-8"
    )
    with pytest.raises(UnsafeXMLError):
        parse_xml_file(doc)
