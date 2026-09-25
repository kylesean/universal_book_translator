"""Caption-body latch: caption paragraphs flow as CAPTION, not body prose."""

from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_blocks import (
    attach_split_caption_tails,
    decouple_embedded_captions,
    defragment_narrative_blocks,
    latch_caption_bodies,
    split_prov_spans,
    unify_figure_captions,
)
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, LayoutRole


def _block(
    bid: str,
    src: str,
    page: int = 7,
    btype: BlockType = BlockType.NARRATIVE,
    flow: FlowID = FlowID.MAIN_STORY,
) -> IRBlock:
    return IRBlock(
        id=bid,
        flow_id=flow,
        spine_index=1,
        block_type=btype,
        source_text=src,
        bbox=BoundingBox(page=page, x0=80.0, y0=300.0, x1=420.0, y1=320.0),
    )


def test_caption_body_latched_single_paragraph() -> None:
    blocks = [
        _block("h", "FIG. 3.6", btype=BlockType.HEADING),
        _block("b", "Mobile electron charge density versus VG of DG FinFETs."),
        _block("n", "Body paragraph that follows the figure."),
    ]
    out = latch_caption_bodies(blocks)
    assert out[0].flow_id == FlowID.MAIN_STORY  # label untouched
    assert out[1].flow_id == FlowID.CAPTION
    assert out[2].flow_id == FlowID.MAIN_STORY  # stops after terminal punct


def test_caption_body_latched_multi_paragraph() -> None:
    blocks = [
        _block("h", "Table 2.1 Results", btype=BlockType.HEADING),
        _block("b1", "First caption sentence without terminal"),
        _block("b2", "Second caption sentence ends here."),
        _block("n", "Body text resumes."),
    ]
    out = latch_caption_bodies(blocks)
    assert out[1].flow_id == FlowID.CAPTION
    assert out[2].flow_id == FlowID.CAPTION
    assert out[3].flow_id == FlowID.MAIN_STORY


def test_latch_stops_at_page_boundary() -> None:
    blocks = [
        _block("h", "FIG. 3.6", page=7, btype=BlockType.HEADING),
        _block("b", "Caption text on the next page", page=8),
    ]
    out = latch_caption_bodies(blocks)
    assert out[1].flow_id == FlowID.MAIN_STORY


def test_non_caption_untouched() -> None:
    blocks = [
        _block("a", "Ordinary body paragraph one."),
        _block("b", "Ordinary body paragraph two."),
    ]
    out = latch_caption_bodies(blocks)
    assert all(b.flow_id == FlowID.MAIN_STORY for b in out)


def test_unify_figure_captions() -> None:
    blocks = [
        _block("h", "FIG. 3.2", page=3, btype=BlockType.NARRATIVE, flow=FlowID.CAPTION),
        _block("b", "Fin potential versus position obtained from Eq. (3.1).", page=3),
    ]
    out = unify_figure_captions(blocks)
    assert len(out) == 1
    assert out[0].source_text == "FIG. 3.2: Fin potential versus position obtained from Eq. (3.1)."
    assert out[0].flow_id == FlowID.CAPTION


def test_defragment_narrative_blocks_across_figures() -> None:
    blocks = [
        _block("b1", "which only has spatial", page=2),
        _block("fig", "assets/fig1.png", page=3, btype=BlockType.IMAGE, flow=FlowID.CAPTION),
        _block("cap", "FIG. 3.1: Schematic diagram.", page=3, flow=FlowID.CAPTION),
        _block("b2", "dependence, Nch is the channel doping.", page=3),
    ]
    out = defragment_narrative_blocks(blocks)
    assert len(out) == 3
    assert out[0].id == "b1"
    assert out[0].source_text == "which only has spatial dependence, Nch is the channel doping."
    assert out[1].id == "fig"
    assert out[2].id == "cap"


def test_decouple_embedded_caption_dotted_number() -> None:
    blocks = [
        _block(
            "b",
            "The controller was tested extensively over three months. "
            "Figure 3.1: Architecture of the control loop.",
        )
    ]
    out = decouple_embedded_captions(blocks)
    assert len(out) == 2
    assert out[0].source_text == "The controller was tested extensively over three months."
    assert out[0].flow_id == FlowID.MAIN_STORY
    assert out[1].flow_id == FlowID.CAPTION
    assert out[1].source_text == "Figure 3.1: Architecture of the control loop."
    assert out[1].layout_role == LayoutRole.CAPTION


def test_decouple_embedded_caption_fig_marker() -> None:
    blocks = [
        _block(
            "b",
            "The results were obtained as described earlier in this section. "
            "FIG. 3.2 Fin potential versus position.",
        )
    ]
    out = decouple_embedded_captions(blocks)
    assert len(out) == 2
    assert out[1].source_text == "FIG. 3.2 Fin potential versus position."


def test_decouple_embedded_caption_skips_in_text_reference() -> None:
    blocks = [
        _block(
            "b",
            "The bias was applied as shown in Fig. 3.2 and the current was measured afterwards.",
        )
    ]
    out = decouple_embedded_captions(blocks)
    assert len(out) == 1
    assert out[0].flow_id == FlowID.MAIN_STORY


def test_decouple_embedded_caption_skips_short_prose() -> None:
    blocks = [_block("b", "FIG. 3.2 Fin potential versus position.")]
    out = decouple_embedded_captions(blocks)
    assert len(out) == 1
    assert out[0].flow_id == FlowID.MAIN_STORY


def test_decoupled_caption_bands_are_disjoint_from_the_parent() -> None:
    """A caption sharing the parent's whole bbox builds no placeable rigid zone.

    Both zones then cover the same rows: the caption's zone is clipped to zero
    height and rejected, while the prose translation (which no longer contains
    the caption) is drawn over the source caption line — the caption is lost
    from the delivered PDF.
    """
    blocks = [
        _block(
            "b",
            "The controller was tested extensively over three months. "
            "Figure 3.1: Architecture of the control loop.",
        )
    ]
    out = decouple_embedded_captions(blocks)
    parent, cap = out[0], out[1]
    assert parent.bbox is not None and cap.bbox is not None
    # The bands abut without overlapping, so neither rigid zone clips the other.
    # Boxes are bottom-up, and the caption is the paragraph's last text, so it
    # owns the LOW band (the old orientation painted the translated caption over
    # the source prose lines and vice versa).
    assert cap.bbox.y1 == parent.bbox.y0
    assert cap.bbox.y0 == 300.0
    assert parent.bbox.y1 == 320.0
    assert (cap.bbox.x0, cap.bbox.x1) == (80.0, 420.0)
    assert cap.bbox.page == parent.bbox.page == 7


def _tail_block(bid: str, src: str, page: int = 7) -> IRBlock:
    b = _block(bid, src, page=page, flow=FlowID.CAPTION)
    b.provenance = {"docling_span_split_tail": True}
    b.layout_role = LayoutRole.CAPTION
    return b


def test_split_caption_tail_attached_to_label() -> None:
    blocks = [
        _block("p", "Solving Eq. (3.11) is not practical for compact modeling.", page=6),
        _tail_block("t", "Surface potential versus VG of DG FinFETs at VDS = 0.0 V."),
        _block("h", "FIG. 3.5", page=7, flow=FlowID.CAPTION),
        _block("img", "assets/fig.png", page=7, btype=BlockType.IMAGE, flow=FlowID.CAPTION),
    ]
    out = attach_split_caption_tails(blocks)
    assert [b.id for b in out] == ["p", "h", "img"]
    assert out[1].source_text == (
        "FIG. 3.5: Surface potential versus VG of DG FinFETs at VDS = 0.0 V."
    )
    assert out[1].flow_id == FlowID.CAPTION


def test_split_caption_tail_found_across_formula_blocks() -> None:
    blocks = [
        _block("p", "where psi is the potential which only has a y spatial", page=2),
        _tail_block("t", "Schematic representation of a symmetric double-gate FinFET.", page=3),
        _block("f", "x = y", page=2, btype=BlockType.FORMULA),
        _block("h", "FIG. 3.1", page=3, flow=FlowID.CAPTION),
    ]
    out = attach_split_caption_tails(blocks)
    assert [b.id for b in out] == ["p", "f", "h"]
    assert out[2].source_text == (
        "FIG. 3.1: Schematic representation of a symmetric double-gate FinFET."
    )


def test_unmarked_period_ending_paragraph_not_attached() -> None:
    blocks = [
        _block("p", "A normal paragraph that ends with a period.", page=7),
        _block("h", "FIG. 3.5", page=7, flow=FlowID.CAPTION),
    ]
    out = attach_split_caption_tails(blocks)
    assert len(out) == 2
    assert out[1].source_text == "FIG. 3.5"


def test_split_caption_tail_on_other_page_not_attached() -> None:
    blocks = [
        _tail_block("t", "Some other caption body sentence here.", page=6),
        _block("h", "FIG. 3.5", page=7, flow=FlowID.CAPTION),
    ]
    out = attach_split_caption_tails(blocks)
    assert len(out) == 2


class _FakeBBox:
    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.l, self.b, self.r, self.t = x0, y0, x1, y1


class _FakeProv:
    def __init__(self, page: int, bbox: _FakeBBox, charspan: tuple[int, int]) -> None:
        self.page_no, self.bbox, self.charspan = page, bbox, charspan


class _FakeItem:
    def __init__(self, text: str, provs: list[_FakeProv]) -> None:
        self.text, self.prov = text, provs


def test_split_prov_spans_peels_caption_sentence() -> None:
    head = "where psi is the potential which only has a y spatial"
    tail = "Schematic representation of a symmetric double-gate FinFET."
    text = f"{head} {tail}"
    item = _FakeItem(
        text,
        [
            _FakeProv(2, _FakeBBox(1, 2, 3, 4), (0, len(head))),
            _FakeProv(3, _FakeBBox(5, 6, 7, 8), (len(head) + 1, len(text))),
        ],
    )
    split = split_prov_spans(item)
    assert split is not None
    assert split[0] == head
    assert split[1] == tail
    assert split[2] is not None and split[2].page == 3


def test_split_prov_spans_ignores_same_page_and_lowercase_tail() -> None:
    text = "body sentence ends. and a lowercase tail continues"
    same_page = _FakeItem(
        "a b c d e. f g h i j.",
        [
            _FakeProv(3, _FakeBBox(1, 2, 3, 4), (0, 5)),
            _FakeProv(3, _FakeBBox(5, 6, 7, 8), (6, 12)),
        ],
    )
    assert split_prov_spans(same_page) is None
    lowercase_tail = _FakeItem(
        text,
        [
            _FakeProv(2, _FakeBBox(1, 2, 3, 4), (0, 19)),
            _FakeProv(3, _FakeBBox(5, 6, 7, 8), (20, len(text))),
        ],
    )
    assert split_prov_spans(lowercase_tail) is None


class _NoBBoxProv:
    """Provenance whose bbox Docling failed to populate."""

    def __init__(self, page: int) -> None:
        self.page_no, self.bbox, self.charspan = page, None, (0, 0)


class _LabeledItem:
    def __init__(self, text: str, label: object, prov: object) -> None:
        self.text, self.label, self.prov = text, label, prov


def test_item_without_provenance_bbox_has_no_fabricated_origin_box(tmp_path: Path) -> None:
    """An item with no prov bbox must not become a zero-area box at the page
    origin — that box seeds a bogus rigid zone and adds a phantom guard rect."""
    pytest.importorskip("docling_core")
    from docling_core.types.doc.labels import DocItemLabel

    from ubt.adapters.pdf.docling_parser import map_iterated_items

    item = _LabeledItem(
        "A normal body paragraph with enough words to survive postprocessing.",
        DocItemLabel.PARAGRAPH,
        [_NoBBoxProv(4)],
    )
    blocks = map_iterated_items([(item, 0)], doc=None, assets_dir=tmp_path / "assets")

    assert len(blocks) == 1
    assert blocks[0].bbox is None


def test_later_provenance_entry_supplies_missing_bbox(tmp_path: Path) -> None:
    """A page-only anchor must not hide geometry carried by a later provenance span."""
    pytest.importorskip("docling_core")
    from docling_core.types.doc.labels import DocItemLabel

    from ubt.adapters.pdf.docling_parser import map_iterated_items

    item = _LabeledItem(
        "A normal body paragraph with enough words to survive postprocessing.",
        DocItemLabel.PARAGRAPH,
        [_NoBBoxProv(4), _FakeProv(4, _FakeBBox(10, 20, 300, 80), (0, 10))],
    )
    blocks = map_iterated_items([(item, 0)], doc=None, assets_dir=tmp_path / "assets")

    assert len(blocks) == 1
    assert blocks[0].bbox is not None
    assert (blocks[0].bbox.x0, blocks[0].bbox.y0, blocks[0].bbox.x1, blocks[0].bbox.y1) == (
        10.0,
        20.0,
        300.0,
        80.0,
    )


def test_empty_formula_preserves_narrative_boundaries_and_bboxes(tmp_path: Path) -> None:
    """Empty formula items (from formula-enrichment=off) must not be dropped.
    
    Dropping them lets defragment_narrative_blocks falsely coalesce text across
    the equation, losing the second text segment's bounding box and leaving
    residual un-erased English text under the equation.
    """
    pytest.importorskip("docling_core")
    from docling_core.types.doc.labels import DocItemLabel
    from ubt.adapters.pdf.docling_parser import map_iterated_items

    item_before = _LabeledItem(
        "Here, psi1 is the potential contribution due to inversion carriers and is given by",
        DocItemLabel.PARAGRAPH,
        [_FakeProv(5, _FakeBBox(80, 280, 400, 304), (0, 80))],
    )
    item_formula = _LabeledItem(
        "",  # empty formula text when enrichment is bypassed
        DocItemLabel.FORMULA,
        [_FakeProv(5, _FakeBBox(190, 250, 400, 275), (0, 0))],
    )
    item_after = _LabeledItem(
        "and psi2 is the potential contribution due to the presence of ionized dopants.",
        DocItemLabel.PARAGRAPH,
        [_FakeProv(5, _FakeBBox(80, 220, 400, 245), (0, 78))],
    )

    blocks = map_iterated_items(
        [(item_before, 0), (item_formula, 0), (item_after, 0)],
        doc=None,
        assets_dir=tmp_path / "assets",
    )

    assert len(blocks) == 3
    assert blocks[0].block_type == BlockType.NARRATIVE
    assert blocks[0].bbox is not None and blocks[0].bbox.y0 == 280.0
    assert "and psi2 is" not in blocks[0].source_text

    assert blocks[1].block_type == BlockType.FORMULA
    assert blocks[1].skip_translate is True
    assert blocks[1].bbox is not None and blocks[1].bbox.y0 == 250.0

    assert blocks[2].block_type == BlockType.NARRATIVE
    assert blocks[2].bbox is not None and blocks[2].bbox.y0 == 220.0


def test_embedded_caption_takes_the_bottom_band_of_its_paragraph() -> None:
    """Boxes are bottom-up, so the paragraph's *last* text sits lowest.

    The split handed the caption the top band, which made the rigid engine paint
    the translated caption over the source prose lines and the prose over the
    caption — the collision the band split exists to prevent.
    """
    from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock

    prose = "The transformer stacks attention heads across the whole sequence."
    caption = "FIGURE 3.1: Attention weights."
    block = IRBlock(
        id="pdf_main#b001",
        spine_index=1,
        flow_id=FlowID.MAIN_STORY,
        block_type=BlockType.NARRATIVE,
        source_text=f"{prose} {caption}",
        bbox=BoundingBox(page=4, x0=60.0, y0=300.0, x1=400.0, y1=420.0),
    )

    out = decouple_embedded_captions([block])
    kept = next(b for b in out if b.id == "pdf_main#b001")
    cap = next(b for b in out if b.id.endswith("_cap"))

    assert cap.source_text == caption
    assert cap.bbox is not None and kept.bbox is not None
    # Caption: the bottom of the merged box; prose: everything above it.
    assert cap.bbox.y0 == 300.0
    assert cap.bbox.y1 == kept.bbox.y0
    assert kept.bbox.y1 == 420.0
    assert (cap.bbox.y1 - cap.bbox.y0) < (kept.bbox.y1 - kept.bbox.y0)


@pytest.mark.fast
def test_docling_figure_caption_unification_unions_bbox() -> None:
    head = IRBlock(
        id="pdf#b0001",
        flow_id=FlowID.CAPTION,
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="FIG. 3.1",
        bbox=BoundingBox(page=1, x0=50.0, y0=100.0, x1=100.0, y1=115.0),
    )
    cand = IRBlock(
        id="pdf#b0002",
        flow_id=FlowID.CAPTION,
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        source_text="A detailed description of the circuit architecture.",
        bbox=BoundingBox(page=1, x0=50.0, y0=118.0, x1=400.0, y1=180.0),
    )

    merged_blocks = unify_figure_captions([head, cand])
    assert len(merged_blocks) == 1
    m_head = merged_blocks[0]
    assert "FIG. 3.1: A detailed description" in m_head.source_text
    assert m_head.bbox is not None
    # BoundingBox must encompass cand.bbox (x1 at least 400.0, y1 at least 180.0)
    assert m_head.bbox.x1 >= 400.0
    assert m_head.bbox.y1 >= 180.0


@pytest.mark.fast
def test_docling_defragment_does_not_merge_across_formula() -> None:
    from ubt.adapters.pdf.docling_blocks import defragment_narrative_blocks

    blocks = [
        IRBlock(
            id="p1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="The kinetic energy is defined by",
        ),
        IRBlock(
            id="p2",
            spine_index=2,
            block_type=BlockType.FORMULA,
            flow_id=FlowID.MAIN_STORY,
            source_text="$E = \\frac{1}{2}mv^2$",
        ),
        IRBlock(
            id="p3",
            spine_index=3,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="where m is the mass and v is the velocity.",
        ),
    ]

    result = defragment_narrative_blocks(blocks)
    assert len(result) == 3
    assert result[0].id == "p1"
    assert result[1].id == "p2"
    assert result[2].id == "p3"
    assert "where m is the mass" not in result[0].source_text


@pytest.mark.fast
def test_docling_render_academic_figures_woven_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    from ubt.adapters.pdf import asset_extractor
    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy

    source_pdf = tmp_path / "sample.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 dummy")
    fig_png = tmp_path / "fig1.png"
    fig_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr("ubt.adapters.pdf.svg_diagram.is_svg_backend_available", lambda: False)
    monkeypatch.setattr("ubt.adapters.pdf.svg_diagram.is_svg_rendering_supported", lambda: False)
    monkeypatch.setattr("ubt.adapters.pdf.svg_diagram.detect_diagram_regions", lambda *a, **k: [])
    monkeypatch.setattr("ubt.adapters.pdf.pdf_struct.page_count", lambda _path: 1)
    monkeypatch.setattr(
        asset_extractor,
        "extract_pdf_figures",
        lambda *a, **k: {
            "fig1": asset_extractor.ExtractedFigure(
                fig_id="fig1",
                caption_en="Figure 1",
                page=1,
                image_path=fig_png,
                relative_path="assets/fig1.png",
                bbox=(50.0, 300.0, 200.0, 400.0),
            )
        },
    )

    strategy = DoclingRenderStrategy(
        reconstructor=SimpleNamespace(),  # type: ignore[arg-type]
        alternator=SimpleNamespace(),  # type: ignore[arg-type]
        diagram_localizer=cast(
            "Any",
            SimpleNamespace(
                get_page_height=lambda _src, _page: 800.0, get_page_count=lambda _src: 1
            ),
        ),
    )

    top_block = IRBlock(
        id="pdf_main#b001",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="Top text",
        bbox=BoundingBox(page=1, x0=50.0, y0=450.0, x1=200.0, y1=550.0),
    )
    bottom_block = IRBlock(
        id="pdf_main#b002",
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="Bottom text",
        bbox=BoundingBox(page=1, x0=50.0, y0=100.0, x1=200.0, y1=200.0),
    )

    blocks = [top_block, bottom_block]
    out, covered = strategy._vectorize_diagrams_sync(source_pdf, blocks, tmp_path / "assets", "zh")

    assert len(out) == 3
    assert out[0].id == "pdf_main#b001"
    assert out[1].id == "pdf_main#fig_fig1"
    assert out[2].id == "pdf_main#b002"
