"""Source-page contract of the Docling item -> IRBlock pass.

``map_iterated_items`` is the single place that turns Docling items into
IRBlocks, so it owns the page mapping every page-strict stage downstream
depends on: a block's ``bbox.page`` is the provenance geometry page and
``provenance["source_page"]`` is the item's start page — even when a cross-page
item's first provenance entry is a bbox-less anchor on a different page from
where its geometry lives. Docling is faked so the test is hermetic and runs
without a PDF.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "docling_core.types.doc.labels", reason="docling-core is an optional PDF dependency"
)

from docling_core.types.doc.labels import DocItemLabel

from ubt.adapters.pdf.docling_parser import map_iterated_items
from ubt.core.ir.models import FlowID, IRBlock, LayoutRole


class _BBox:
    def __init__(self, left: float, bottom: float, right: float, top: float) -> None:
        self.l, self.b, self.r, self.t = left, bottom, right, top


class _Prov:
    def __init__(self, page_no: int, bbox: _BBox | None = None, charspan: Any = None) -> None:
        self.page_no = page_no
        self.bbox = bbox
        self.charspan = charspan


class _Item:
    def __init__(self, label: DocItemLabel, text: str, provs: list[_Prov]) -> None:
        self.label = label
        self.text = text
        self.prov = provs


def _item(label: DocItemLabel, text: str, *provs: _Prov) -> tuple[_Item, int]:
    return (_Item(label, text, list(provs)), 0)


def _map(items: list[Any]) -> list[IRBlock]:
    return map_iterated_items(items, None, Path("/tmp/ubt-unused-assets"), pdf_path=None)


def _texts(blocks: list[IRBlock]) -> list[str]:
    return [b.source_text for b in blocks]


def test_page_number_maps_to_bbox_and_source_page() -> None:
    """Every block carries its page twice: geometry page and start page."""
    blocks = _map(
        [
            _item(DocItemLabel.TEXT, "First page sentence.", _Prov(1, _BBox(72, 100, 540, 120))),
            _item(DocItemLabel.TEXT, "Second page sentence.", _Prov(2, _BBox(72, 100, 540, 120))),
            _item(DocItemLabel.TEXT, "Third page sentence.", _Prov(3, _BBox(72, 100, 540, 120))),
        ]
    )

    assert _texts(blocks) == [
        "First page sentence.",
        "Second page sentence.",
        "Third page sentence.",
    ]
    assert [b.bbox.page for b in blocks if b.bbox is not None] == [1, 2, 3]
    assert [b.provenance["source_page"] for b in blocks] == [1, 2, 3]


def test_cross_page_item_keeps_start_page_and_geometry_page() -> None:
    """A bbox-less first provenance anchors the start page; geometry may differ.

    Docling emits a page-only provenance entry first for some cross-page items.
    The block must record the start page (so page-strict reflow keeps it on page
    2) while still exposing the geometry it can render with (page 3).
    """
    blocks = _map(
        [
            _item(
                DocItemLabel.TEXT,
                "Cross page paragraph text.",
                _Prov(2),  # page-only anchor: no geometry
                _Prov(3, _BBox(72, 100, 540, 120)),  # the geometry that renders
            )
        ]
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block.provenance["source_page"] == 2
    assert block.bbox is not None
    assert block.bbox.page == 3


def test_item_without_geometry_records_its_page_without_a_box() -> None:
    """A provenance entry with no bbox must not fabricate a zero-area box.

    An origin box would seed a bogus rigid zone and pollute the per-page guards;
    the page is still known, so it is recorded for page-strict reflow.
    """
    blocks = _map([_item(DocItemLabel.TEXT, "Anchor only.", _Prov(4))])

    assert len(blocks) == 1
    assert blocks[0].provenance["source_page"] == 4
    assert blocks[0].bbox is None


def test_reading_order_is_the_item_order_not_the_page_order() -> None:
    """Blocks follow Docling reading order even when page numbers go backwards."""
    blocks = _map(
        [
            _item(DocItemLabel.TEXT, "page two text.", _Prov(2, _BBox(72, 100, 540, 120))),
            _item(DocItemLabel.TEXT, "page one text.", _Prov(1, _BBox(72, 100, 540, 120))),
            _item(DocItemLabel.TEXT, "page three text.", _Prov(3, _BBox(72, 100, 540, 120))),
        ]
    )

    assert _texts(blocks) == ["page two text.", "page one text.", "page three text."]
    assert [b.bbox.page for b in blocks if b.bbox is not None] == [2, 1, 3]


def test_text_inside_a_figure_box_is_dropped_but_a_caption_survives() -> None:
    """In-figure labels are excluded; captions boxed by the figure are not."""
    blocks = _map(
        [
            _item(DocItemLabel.PICTURE, "", _Prov(1, _BBox(72, 72, 540, 400))),
            _item(
                DocItemLabel.TEXT,
                "Label baked into the figure.",
                _Prov(1, _BBox(100, 100, 200, 140)),
            ),
            _item(
                DocItemLabel.TEXT, "Body text below the figure.", _Prov(1, _BBox(72, 420, 540, 460))
            ),
            _item(
                DocItemLabel.CAPTION,
                "Caption inside the figure box stays.",
                _Prov(1, _BBox(100, 100, 200, 140)),
            ),
        ]
    )

    texts = _texts(blocks)
    assert "Label baked into the figure." not in texts
    assert "Body text below the figure." in texts
    assert "Caption inside the figure box stays." in texts


def test_repeating_header_chrome_is_dropped_but_a_unique_credit_survives() -> None:
    """A running head (>3 identical, page number folded) is chrome; a one-off is not."""
    items: list[Any] = [
        _item(DocItemLabel.PAGE_HEADER, "Running Head", _Prov(p, _BBox(72, 20, 540, 35)))
        for p in range(1, 6)
    ]
    items.append(
        _item(DocItemLabel.PAGE_HEADER, "A Unique Credit Line", _Prov(6, _BBox(72, 20, 540, 35)))
    )
    items.append(
        _item(DocItemLabel.TEXT, "Body paragraph here.", _Prov(6, _BBox(72, 100, 540, 120)))
    )

    blocks = _map(items)
    assert "Running Head" not in _texts(blocks)

    credit = next(b for b in blocks if b.source_text == "A Unique Credit Line")
    assert credit.layout_role == LayoutRole.HEADER
    assert credit.skip_translate  # chrome stays untranslated


def test_page_footer_number_is_dropped_and_named_footer_is_chrome() -> None:
    """A bare page number is chrome, as is a low-frequency named footer."""
    blocks = _map(
        [
            _item(DocItemLabel.PAGE_FOOTER, "42", _Prov(1, _BBox(72, 700, 540, 715))),
            _item(DocItemLabel.PAGE_FOOTER, "ISBN 978", _Prov(1, _BBox(72, 700, 540, 715))),
            _item(DocItemLabel.TEXT, "Body paragraph here.", _Prov(1, _BBox(72, 100, 540, 120))),
        ]
    )

    assert "42" not in _texts(blocks)
    footer = next(b for b in blocks if b.source_text == "ISBN 978")
    assert footer.flow_id == FlowID.FOOTNOTE
    assert footer.skip_translate
