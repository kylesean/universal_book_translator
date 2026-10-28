"""Unit tests for publication colophon, CIP cataloging, and ISBN validation."""

from __future__ import annotations

import pytest

from ubt.core.colophon.generator import (
    extract_colophon_from_blocks,
    render_colophon_html,
    render_colophon_markdown,
    render_colophon_typst,
)
from ubt.core.colophon.isbn import (
    calculate_isbn10_check_digit,
    calculate_isbn13_check_digit,
    clean_isbn,
    format_isbn,
    validate_isbn,
    validate_isbn10,
    validate_isbn13,
)
from ubt.core.colophon.models import ColophonMetadata
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def test_clean_isbn() -> None:
    assert clean_isbn(" 978-7-121-39655-7 \t") == "9787121396557"
    assert clean_isbn("0-13-468599-x") == "013468599X"


def test_isbn13_validation_and_calculation() -> None:
    # 978-7-121-39655-7 is a real valid ISBN
    assert calculate_isbn13_check_digit("978712139655") == "7"
    assert validate_isbn13("978-7-121-39655-7") is True
    # Corrupt check digit
    assert validate_isbn13("978-7-121-39655-8") is False
    # Not 13 digits
    assert validate_isbn13("978-7-121") is False


def test_isbn10_validation_and_calculation() -> None:
    # 0-13-468599-7 is a real valid ISBN-10
    assert calculate_isbn10_check_digit("013468599") == "7"
    assert validate_isbn10("0-13-468599-7") is True
    # Check digit with 'X'
    # 0-8044-2957-X
    assert calculate_isbn10_check_digit("080442957") == "X"
    assert validate_isbn10("0-8044-2957-X") is True
    # Corrupt check digit
    assert validate_isbn10("0-13-468599-6") is False


def test_validate_and_format_isbn() -> None:
    assert validate_isbn("978-7-121-39655-7") is True
    assert validate_isbn("0-13-468599-7") is True
    assert validate_isbn("invalid-isbn") is False

    formatted13 = format_isbn("9787121396557")
    assert formatted13 == "978-7-121-39655-7"


def test_colophon_metadata_properties() -> None:
    meta = ColophonMetadata(
        title="深度学习实践指南",
        authors=["张三", "李四"],
        translators=["王五"],
        isbn="978-7-121-39655-7",
    )
    assert meta.has_valid_isbn is True
    assert meta.display_isbn == "978-7-121-39655-7"
    assert meta.authors_display == "张三、李四 著"
    assert meta.translators_display == "王五 译"


def test_extract_colophon_from_blocks() -> None:
    el_heading = make_element(
        id="h1",
        spine_index=0,
        block_type=BlockType.HEADING,
        source_text="深入理解分布式系统",
        level=1,
    )
    el_para = make_element(
        id="p1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Written by Martin Kleppmann. Translated by UBT Team. This is comprehensive text.",
    )
    blocks = [
        IRBlock(element=el_heading, target_text="深入理解分布式系统"),
        IRBlock(
            element=el_para,
            target_text="作者：Martin Kleppmann。译者：UBT团队。这是一本完整的专著。",
        ),
    ]

    meta = extract_colophon_from_blocks(blocks, isbn="978-7-121-39655-7")
    assert meta.title == "深入理解分布式系统"
    assert meta.authors == ["Martin Kleppmann"]
    assert meta.translators == ["UBT团队"]
    assert meta.estimated_words > 0
    assert float(meta.sheets) >= 1.0


def test_renderers_typst_html_markdown() -> None:
    meta = ColophonMetadata(
        title="系统设计精要",
        authors=["Alex Xu"],
        translators=["编译团队"],
        isbn="978-7-121-39655-7",
    )

    typst_code = render_colophon_typst(meta)
    assert "图书在版编目（CIP）数据" in typst_code
    assert "系统设计精要" in typst_code
    assert "ISBN 978-7-121-39655-7" in typst_code
    assert "#page(numbering: none" in typst_code

    html_code = render_colophon_html(meta)
    assert "图书在版编目（CIP）数据" in html_code
    assert "系统设计精要" in html_code
    assert "ISBN 978-7-121-39655-7" in html_code
    assert '<section class="ubt-colophon"' in html_code

    md_code = render_colophon_markdown(meta)
    assert "### 📖 出版版权与 CIP 编目信息" in md_code
    assert "系统设计精要" in md_code
    assert "ISBN 978-7-121-39655-7" in md_code
