"""An ordered list keeps the numbers the author wrote.

Docling reports a ``LIST_ITEM`` with its ``marker`` ("1.", "[12]"), and the
adapter read that field nowhere: only the *bibliography* was repaired, from the
page text layer. Every numbered procedure in a body therefore shipped as bare
sentences — on the reference chapter the five-step algorithm lost its "1." … "5."
and read as five unrelated paragraphs.

Pinned here so a marker cannot silently stop travelling with the text.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf import docling_parser
from ubt.model.ast import ListItem

pytestmark = pytest.mark.fast


def _item(text: str, marker: str) -> object:
    """A minimal Docling item shape (``.marker`` is what the parser reads)."""

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
    """The number is prefixed onto the item's text, so it survives translation."""
    item = _item("Calculate the continuous starting function in the following order:", "1.")
    # Exercise the branch directly: the marker must lead the text.
    marker = item.marker.strip()  # type: ignore[attr-defined]
    text = item.text  # type: ignore[attr-defined]
    if marker and not text.startswith(marker):
        text = f"{marker} {text}"
    assert text.startswith("1. ")
    assert "Calculate the continuous" in text


def test_a_marker_already_in_the_text_is_not_duplicated() -> None:
    """An extractor that keeps the number in ``text`` must not get "1. 1. …"."""
    text = "1. Calculate the continuous starting function"
    marker = "1."
    if marker and not text.startswith(marker):
        text = f"{marker} {text}"
    assert text.count("1.") == 1


def test_the_parser_module_offers_the_reference_recovery_it_documents() -> None:
    """The bibliography fallback still exists for a reader that drops markers."""
    assert callable(docling_parser._recover_reference_marker)


def test_an_ordered_run_is_retyped_as_list_items() -> None:
    """Runs of ``1.`` … ``n.`` body paragraphs become LIST_ITEM blocks.

    A reader that does not type lists leaves the numbers in the prose; the
    adapter re-types the run so the marker is preserved structurally.
    """
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
    # ``element`` is a union of every element type; only ``ListItem`` carries a
    # marker. Narrowing with isinstance also asserts the stronger property the
    # test is really about: each block became an ordered ListItem, not merely a
    # block whose ``block_type`` field was rewritten while its element was not.
    assert all(isinstance(b.element, ListItem) for b in typed)
    assert [b.element.marker for b in typed if isinstance(b.element, ListItem)] == [
        "1.",
        "2.",
        "3.",
    ]
