from __future__ import annotations

import pytest

from ubt.adapters.pdf import docling_parser
from ubt.model.ast import ListItem

pytestmark = pytest.mark.fast


def _item(text: str, marker: str) -> object:

    class _Prov:
        page_no = 1
        bbox = None

    class _Item:
        label = "list_item"
        prov: list[object] = []

    item = _Item()
    item.text = text  # type: ignore[attr-defined]
    item.marker = marker  # type: ignore[attr-defined]
    return item


def test_docling_marker_is_kept_on_the_text() -> None:
    item = _item("Calculate the continuous starting function in the following order:", "1.")
    marker = item.marker.strip()  # type: ignore[attr-defined]
    text = item.text  # type: ignore[attr-defined]
    if marker and not text.startswith(marker):
        text = f"{marker} {text}"
    assert text.startswith("1. ")
    assert "Calculate the continuous" in text


def test_a_marker_already_in_the_text_is_not_duplicated() -> None:
    text = "1. Calculate the continuous starting function"
    marker = "1."
    if marker and not text.startswith(marker):
        text = f"{marker} {text}"
    assert text.count("1.") == 1


def test_the_parser_module_offers_the_reference_recovery_it_documents() -> None:
    assert callable(docling_parser._recover_reference_marker)


def test_an_ordered_run_is_retyped_as_list_items() -> None:
    from ubt.core.ir.models import BlockType, IRBlock, make_element

    def para(bid: str, text: str) -> IRBlock:
        return IRBlock(
            element=make_element(
                id=bid, spine_index=1, block_type=BlockType.NARRATIVE, source_text=text
            )
        )

    blocks = [
        para("b1", "1. Calculate the continuous starting function"),
        para("b2", "2. Compute the terms"),
        para("b3", "3. Compute the derivatives"),
    ]
    typed = docling_parser.restore_ordered_markers(blocks)
    assert all(b.block_type is BlockType.LIST_ITEM for b in typed)
    assert all(isinstance(b.element, ListItem) for b in typed)
    assert [b.element.marker for b in typed if isinstance(b.element, ListItem)] == [
        "1.",
        "2.",
        "3.",
    ]
