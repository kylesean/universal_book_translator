"""Docling iterated/exported item mapping into UBT IR blocks."""

from __future__ import annotations

import dataclasses
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ubt.adapters.pdf import docling_outline as _docling_outline
from ubt.adapters.pdf.docling_blocks import (
    chrome_key,
    code_text_layer_is_lossy,
    crop_lossy_code_asset,
    is_inside_picture,
    is_repeat_handle,
    split_prov_spans,
    table_to_markdown,
)
from ubt.adapters.pdf.docling_cache import docling_asset_dir
from ubt.adapters.pdf.docling_outline import _toc_sort_key
from ubt.adapters.pdf.docling_structure import (
    _CHROME_REPEAT_MIN,
    _is_prose_footer,
    _iter_pdf_page_texts,
    _read_table_cells,
    _recover_reference_marker,
    _table_reader_enabled,
    annotate_layout_metadata,
    type_docling_blocks,
)
from ubt.analyze.structure import looks_like_debris, looks_like_listing
from ubt.core.cleaners.lnds_pruner import normalize_academic_pdf_math
from ubt.core.fs_perms import restrict_dir_to_owner
from ubt.core.ir.models import (
    BlockProvenance,
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    make_element,
)
from ubt.core.policy.layout_policy import CAPTION_RE, PROSE_BLOCK_TYPES
from ubt.model.ast import RegionKind
from ubt.model.span import CompositeSpan, PhysicalBox, Span

logger = logging.getLogger(__name__)

_POSITIONAL_ID_RE = re.compile(r"^pdf_main#b\d{4,}$")


def _parser_symbol(name: str, fallback: Any) -> Any:
    parser_mod = sys.modules.get("ubt.adapters.pdf.docling_parser")
    return getattr(parser_mod, name, fallback) if parser_mod is not None else fallback


def _figure_asset_dpi_tag() -> str:
    """The delivered-crop DPI tag, read lazily.

    Asset filenames carry the resolution they were cut at, so raising the DPI
    cuts new files instead of replaying old ones from the sha-keyed asset dir.
    """
    from ubt.adapters.pdf.visual_scalpel import FIGURE_ASSET_DPI_TAG

    return FIGURE_ASSET_DPI_TAG


#: The ``DocItemLabel`` members this module compares against, and the string
#: each one carries. ``DocItemLabel`` subclasses ``str``, so a plain string
#: equals the corresponding member and hashes identically — a comparison
#: against these names behaves exactly as the enum does.
_DOC_ITEM_LABEL_VALUES: dict[str, str] = {
    "EMPTY_VALUE": "empty_value",
    "MARKER": "marker",
    "CHECKBOX_SELECTED": "checkbox_selected",
    "CHECKBOX_UNSELECTED": "checkbox_unselected",
    "HANDWRITTEN_TEXT": "handwritten_text",
    "PAGE_HEADER": "page_header",
    "PAGE_FOOTER": "page_footer",
    "PICTURE": "picture",
    "TABLE": "table",
    "CODE": "code",
    "FORMULA": "formula",
    "TITLE": "title",
    "SECTION_HEADER": "section_header",
    "FOOTNOTE": "footnote",
    "CAPTION": "caption",
    "LIST_ITEM": "list_item",
    "TEXT": "text",
}


def _doc_item_labels() -> Any:
    """``DocItemLabel``, or a string-compatible stand-in when Docling is absent.

    The real enum is preferred so this module keeps matching whatever Docling
    ships. But ``docling`` lives in the ``pdf``/``academic`` extras, not in
    ``dev``, and the fast tier runs on a dev-only install — the documented
    contract in ``tests/conftest.py``. ``map_iterated_items`` is a pure
    mapper over duck-typed items (its own tests pass plain fakes whose
    ``label`` is a string), so it never needs Docling to *import*; it only
    needs the label names. Importing the enum unconditionally made this
    function unusable without the optional extra, and the whole test file
    failed to import rather than skipping.
    """
    try:
        from docling_core.types.doc.labels import DocItemLabel

        return DocItemLabel
    except ImportError:
        return SimpleNamespace(**_DOC_ITEM_LABEL_VALUES)


def _place_toc_rows(blocks: list[IRBlock], rows: list[IRBlock], insert_at: int) -> None:
    """Insert merged TOC ``rows`` at ``insert_at`` and renumber what follows.

    The rows belong where the source printed the contents — after the front
    matter, before the body — because every consumer orders by ``spine_index``
    (``analyze.bridge``, the HTML/EPUB companions, ``render_preflight``).
    Appending them at the end instead put the contents after the text it
    describes and orphaned the "Contents" heading above the first paragraph.

    Two invariants the insertion must not break:

    * **Ids stay unique.** A row landing at position ``k + 1`` takes ``#b{k+1}``,
      which the block that used to sit there already owns; the ledger's
      ``ON CONFLICT(job_id, block_id) DO UPDATE`` then resolves the clash
      silently by overwriting one block with the other. Everything from
      ``insert_at`` on is therefore renumbered.
    * **Asset prefixes survive.** ``pdf_main#img_…`` / ``code_…`` / ``formula_…``
      are not positions — the quality gate and the renderer key on them — so
      only positional ``#bNNNN`` ids are rewritten.

    Blocks *before* ``insert_at`` keep the ids they were emitted with. That is
    what stops the TOC from overwriting the title page, authors and abstract
    (the ``cb2286a`` regression).
    """
    ordered = list(blocks[:insert_at]) + list(rows) + list(blocks[insert_at:])
    for new_index, block in enumerate(ordered, start=1):
        block.set_spine_index(new_index)
        if _POSITIONAL_ID_RE.fullmatch(block.id):
            block.set_id(f"pdf_main#b{new_index:04d}")
    blocks[:] = ordered


#: A PICTURE covering at least this share of its page is a page background or a
#: full-page plate, not a figure. Treating it as a figure made every paragraph
#: on that page "in-figure text" and dropped the page's prose entirely.
_FULL_PAGE_PICTURE_SHARE = 0.7


def _bbox_area(box: Any) -> float:
    """Area of a Docling bbox, 0 for a degenerate/absent one."""
    try:
        return max(0.0, float(box.r) - float(box.l)) * max(0.0, float(box.t) - float(box.b))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _page_areas(doc: Any) -> dict[int, float]:
    """Page number -> area in points, from the DoclingDocument's page sizes."""
    areas: dict[int, float] = {}
    pages = getattr(doc, "pages", None)
    if not isinstance(pages, dict):
        return areas
    for page_no, page in pages.items():
        size = getattr(page, "size", None)
        try:
            width = float(getattr(size, "width", 0.0) or 0.0)
            height = float(getattr(size, "height", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if width > 0 and height > 0:
            areas[int(page_no)] = width * height
    return areas


def map_iterated_items(
    items: list[Any],
    doc: Any,
    assets_dir: Path,
    pdf_path: Path | None = None,
) -> list[IRBlock]:
    """Map DoclingDocument items into IRBlocks preserving document reading order.

    Advantages over the flat-dict mapping:
    - tables stay at their reading-order position (not appended at the end);
    - section eyebrows (e.g. 5 - PREFILL AND DECODE) in headers are retained;
    - table text is rendered from TableFormer grid (never a raw dict repr);
    - text fully inside a picture bbox (in-figure labels) is excluded.
    """
    DocItemLabel = _doc_item_labels()

    drop_labels = {
        DocItemLabel.EMPTY_VALUE,
        DocItemLabel.MARKER,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.HANDWRITTEN_TEXT,
    }

    # Collect header and footer text frequency to distinguish repeating book chrome from section eyebrows.
    # Keys are digit-normalized ("Handbook 02 10" == "Handbook 02 11"):
    # running heads differ per page only by their page number, and
    # without normalization every instance counts 1 and survives.
    header_counts: dict[str, int] = {}
    footer_counts: dict[str, int] = {}
    for item, _level in items:
        lbl = getattr(item, "label", None)
        t = (getattr(item, "text", "") or "").strip()
        if not t:
            continue
        key = chrome_key(t)
        if lbl == DocItemLabel.PAGE_HEADER:
            header_counts[key] = header_counts.get(key, 0) + 1
        elif lbl == DocItemLabel.PAGE_FOOTER:
            footer_counts[key] = footer_counts.get(key, 0) + 1

    # Pass 1: collect picture bounding boxes (for in-figure text exclusion).
    # A picture that covers most of the page is not a figure, it is the page:
    # a full-page background/plate makes every paragraph's centre fall "inside
    # the picture", so the whole page's prose was dropped as in-figure labels —
    # silently, at parse time, before the delivery contract or the renderer's
    # skip channel could see it.
    page_areas = _page_areas(doc)
    picture_boxes: list[tuple[int, Any]] = []
    for item, _level in items:
        if item.label == DocItemLabel.PICTURE:
            for prov in getattr(item, "prov", []) or []:
                area = page_areas.get(int(getattr(prov, "page_no", 0) or 0), 0.0)
                box = prov.bbox
                if area > 0 and _bbox_area(box) >= _FULL_PAGE_PICTURE_SHARE * area:
                    logger.debug(
                        "ignoring full-page PICTURE on page %s (%.0f%% of the page): "
                        "it is a page background, not a figure",
                        prov.page_no,
                        100.0 * _bbox_area(box) / area,
                    )
                    continue
                picture_boxes.append((prov.page_no, prov.bbox))

    blocks: list[IRBlock] = []
    seen_handles: set[str] = set()
    last_kept_label: Any = None
    last_kept_page: int | None = None
    page_lines_cache: dict[int, tuple[list[Any], float]] = {}
    #: TOC rows gathered from every DOCUMENT_INDEX item. The outline is merged
    #: into them once, after the loop (see the branch below); they are *placed*
    #: at the position of the first contents item, because the contents is read
    #: where the source printed it — after the front matter and before the body.
    pending_toc: list[IRBlock] = []
    #: ``len(blocks)`` at the first DOCUMENT_INDEX item, i.e. how many blocks
    #: precede the contents in reading order. ``None`` until one is seen.
    toc_insert_at: int | None = None
    #: Text of every page, loaded once when the first reference entry appears.
    #: Docling's ``prov.page_no`` for a reference entry is not reliable (measured
    #: on arXiv 2609.20519: entries stamped page 12 have their text on another
    #: page, and the list runs past the document's page count), so recovery
    #: searches the whole document's text layer rather than the stamped page.
    _reference_pages: list[str] | None = None

    def _reference_text() -> str:
        nonlocal _reference_pages
        if _reference_pages is None:
            if pdf_path is None:
                _reference_pages = []
            else:
                _reference_pages = _iter_pdf_page_texts(pdf_path, 1, 1 << 20)
        return "\n".join(_reference_pages)

    for item, _level in items:
        label = item.label

        if label in drop_labels:
            continue  # chrome labels dropped

        # Bare social-handle watermarks (@author) repeat on every page;
        # keep the first (cover credit) regardless of Docling's label and
        # drop the rest before label dispatch.
        item_text = (getattr(item, "text", "") or "").strip()
        if item_text and is_repeat_handle(item_text, seen_handles):
            continue

        if label == getattr(DocItemLabel, "DOCUMENT_INDEX", "document_index"):
            # Collected, not emitted: a contents section that spans pages arrives
            # as one DOCUMENT_INDEX item per page, and merging the outline into
            # each of them re-added every outline-only chapter once per page —
            # arXiv 2608.25512 (a two-page contents) delivered 128 rows for 64
            # entries, so every chapter printed twice. The merge happens once,
            # after the loop.
            #
            # Collection must not move the rows, though: they are placed back at
            # the first contents item's position below, because that is where the
            # source prints them. Appending them at the end of the book put every
            # TOC row after the body it describes and left the "Contents" heading
            # orphaned above the first paragraph.
            if toc_insert_at is None:
                toc_insert_at = len(blocks)
            toc_blocks = _parser_symbol(
                "_extract_document_index_blocks", _docling_outline._extract_document_index_blocks
            )(item, pdf_path, page_lines_cache, start_index=1)
            pending_toc.extend(toc_blocks)
            continue

        if label == DocItemLabel.PICTURE:
            prov = item.prov[0] if getattr(item, "prov", None) else None
            page_no = getattr(prov, "page_no", 0) if prov else 0
            # The DPI tag in the name makes a crop-resolution change cut new
            # files instead of replaying the old ones from the sha-keyed dir
            # (see FIGURE_ASSET_DPI_TAG in visual_scalpel).
            asset_path = (
                assets_dir / f"pic_p{page_no}_{len(blocks) + 1}.{_figure_asset_dpi_tag()}.png"
            )

            img = getattr(item, "image", None)
            if img is None and hasattr(item, "get_image"):
                try:
                    img = item.get_image(doc)
                except Exception:
                    img = None
            if (
                img is not None
                and getattr(img, "width", 0) >= 50
                and getattr(img, "height", 0) >= 40
            ):
                restrict_dir_to_owner(assets_dir)
                try:
                    img.save(asset_path)
                except Exception as exc:
                    logger.debug("Failed to extract picture asset: %s", exc)

            item_bbox = None
            if prov and getattr(prov, "bbox", None) is not None:
                b = prov.bbox
                item_bbox = BoundingBox(
                    page=page_no,
                    x0=float(getattr(b, "l", 0.0)),
                    y0=float(getattr(b, "b", 0.0)),
                    x1=float(getattr(b, "r", 0.0)),
                    y1=float(getattr(b, "t", 0.0)),
                )

            if (
                not asset_path.exists()
                and pdf_path
                and pdf_path.exists()
                and item_bbox
                and page_no >= 1
            ):
                try:
                    from ubt.adapters.pdf.visual_scalpel import (
                        FIGURE_ASSET_DPI,
                        delivered_crop_pil,
                    )

                    # A delivered figure is zoomed and printed, so its crop is
                    # cut at print resolution (300), not the VLM-read 150: the
                    # source figures here are vector drawings, and this crop is
                    # the only raster that will ever exist for them.
                    #
                    # ``delivered_crop_pil`` bounds each side's margin by the
                    # clearance to the nearest neighbouring text, so a caption
                    # sitting tight under a figure is not sliced into the picture
                    # and drawn on the page as part of it.
                    cropped = delivered_crop_pil(pdf_path, page_no, item_bbox, dpi=FIGURE_ASSET_DPI)
                    if getattr(cropped, "width", 0) >= 50 and getattr(cropped, "height", 0) >= 40:
                        restrict_dir_to_owner(assets_dir)
                        cropped.save(asset_path)
                except Exception as exc:
                    logger.debug("Fallback PDF crop failed for picture asset: %s", exc)

            if asset_path.exists():
                blocks.append(
                    IRBlock(
                        element=make_element(
                            id=f"pdf_main#img_{len(blocks) + 1:04d}",
                            spine_index=len(blocks) + 1,
                            block_type=BlockType.IMAGE,
                            flow_id=FlowID.CAPTION,
                            source_text=str(asset_path),
                            bbox=item_bbox,
                            skip_translate=True,
                        ),
                        target_text=str(asset_path),
                    )
                )
            continue

        region = None
        if label == DocItemLabel.PAGE_HEADER:
            text = (getattr(item, "text", "") or "").strip()
            if not text or header_counts.get(chrome_key(text), 0) >= _CHROME_REPEAT_MIN:
                continue
            # Survivors (e.g. title-page credit lines) stay chrome and
            # never enter translation: backfilling them garbles the
            # strip while the source remains perfectly legible.
            block_type, flow_id, skip = BlockType.HEADING, FlowID.MAIN_STORY, True
            region = RegionKind.HEADER
        elif label == DocItemLabel.PAGE_FOOTER:
            text = (getattr(item, "text", "") or "").strip()
            if (
                not text
                or footer_counts.get(chrome_key(text), 0) >= _CHROME_REPEAT_MIN
                or text.isdigit()
            ):
                continue
            # A repeating footer is chrome; a *one-off* footer that reads as a
            # sentence is a footnote paragraph the author wrote, and leaving it
            # in the source language leaks English into the deliverable
            # (measured on the PaddleOCR layout paper: the "RT-DocLayout is
            # released under the name PP-DocLayoutV3 …" footnote shipped
            # untranslated). Chrome stays skipped; prose is translated.
            if _is_prose_footer(text):
                block_type, flow_id, skip = BlockType.NARRATIVE, FlowID.FOOTNOTE, False
            else:
                block_type, flow_id, skip = BlockType.NARRATIVE, FlowID.FOOTNOTE, True
            region = RegionKind.FOOTER
        elif label == DocItemLabel.TABLE:
            text = table_to_markdown(item, doc)
            # A vision reader can recover the data cells Docling fuses (measured:
            # 144/144 exact, against a column of stacked names returned as one
            # blob). It is opt-in, needs a running service, and can only upgrade:
            # any failure, or any disagreement about the shape, keeps Docling's
            # table untouched. See ``table_reader`` for why the header stays
            # Docling's.
            if _table_reader_enabled():
                text = _read_table_cells(item, pdf_path, text) or text
            block_type, flow_id, skip = BlockType.TABLE, FlowID.TABLE_GRID, False
        elif label == DocItemLabel.CODE:
            text = (getattr(item, "text", "") or "").strip()
            # A listing whose text layer lost its line structure (Docling
            # collapses listing lines; the source PDF may draw the identifier
            # underscores as vector rules) is unreadable as code when typeset.
            # Crop the region — the same recovery the unreadable-formula branch
            # below already applies — instead of shipping a wrapped prose blob.
            if code_text_layer_is_lossy(text):
                code_asset = crop_lossy_code_asset(item, assets_dir, pdf_path, len(blocks) + 1)
                if code_asset is not None:
                    blocks.append(code_asset)
                    continue
            block_type, flow_id, skip = BlockType.CODE, FlowID.MAIN_STORY, True
        elif label == DocItemLabel.FORMULA:
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                # Docling read no LaTeX for this formula. The semantic reflow
                # renderer has no source canvas to fall back on, so emitting a
                # bare ``$$`` would silently drop an equation from the page.
                # Crop the region instead and emit it as an IMAGE block: the
                # reader sees the original equation, and the quality report can
                # count what was recovered rather than lost.
                formula_asset = _parser_symbol(
                    "_crop_formula_asset", _docling_outline._crop_formula_asset
                )(item, assets_dir, pdf_path, len(blocks) + 1)
                if formula_asset is not None:
                    blocks.append(formula_asset)
                    continue
                text = "$$"
            block_type, flow_id, skip = BlockType.FORMULA, FlowID.MAIN_STORY, True
        elif label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
            text = (getattr(item, "text", "") or "").strip()
            if re.match(r"^(?:fig(?:ure)?|table|图|表)\.?\s*\d+", text, re.IGNORECASE):
                block_type, flow_id, skip = BlockType.NARRATIVE, FlowID.CAPTION, False
            else:
                block_type, flow_id, skip = BlockType.HEADING, FlowID.MAIN_STORY, False
        elif label == DocItemLabel.FOOTNOTE:
            text = (getattr(item, "text", "") or "").strip()
            block_type, flow_id, skip = BlockType.NARRATIVE, FlowID.FOOTNOTE, False
        elif label == DocItemLabel.CAPTION:
            text = (getattr(item, "text", "") or "").strip()
            block_type, flow_id, skip = BlockType.NARRATIVE, FlowID.CAPTION, False
        elif label == DocItemLabel.LIST_ITEM:
            text = (getattr(item, "text", "") or "").strip()
            block_type, flow_id, skip = BlockType.LIST_ITEM, FlowID.MAIN_STORY, False
            # Docling's own ``marker`` is authoritative when present: on the
            # reference chapter it is correct for all 39 list items ("1." … "5."
            # for the numbered procedure, "[1]" … "[34]" for the bibliography),
            # and it was being read nowhere. Only the bibliography used to be
            # repaired, so every *ordered* step list in the body shipped without
            # its number — "Calculate the continuous starting function in the
            # following order:" became a bare sentence and its five steps lost the
            # 1..5 the author wrote.
            #
            # The marker goes into the text, not onto the element, because the
            # translator rewrites the text and the renderer typesets it: a marker
            # kept only on the element would have to be re-inserted from two
            # sides, and the ledger already carries it for audit.
            marker = (getattr(item, "marker", "") or "").strip()
            if marker and not text.startswith(marker):
                text = f"{marker} {text}"
            elif pdf_path is not None:
                # Docling leaves ``marker`` empty and strips the ``[n]`` off a
                # reference entry, so the delivered bibliography loses its
                # numbering (measured on arXiv 2609.20519: 108 markers in the
                # source, none in the ledger). Put it back from the page's own
                # text layer before the entry is typed, so the number travels
                # with the text through translation and rendering.
                recovered = _recover_reference_marker(text, _reference_text())
                if recovered:
                    text = f"{recovered} {text}"
        else:
            # text / paragraph / reference / form fields → narrative
            text = (getattr(item, "text", "") or "").strip()
            block_type, flow_id, skip = BlockType.NARRATIVE, FlowID.MAIN_STORY, False

        if not text:
            continue

        # Docling sometimes glues a cross-page paragraph fragment and the
        # next page's figure-caption body into one item; the item's prov
        # charspans expose both regions. Peel the caption tail out so it
        # can be re-attached to its "FIG. N" label instead of riding
        # inside a body paragraph (where it gets no caption rendering and
        # the label below the figure renders bare).
        span_split = None
        if label not in (
            DocItemLabel.PICTURE,
            DocItemLabel.TABLE,
            DocItemLabel.FORMULA,
            DocItemLabel.CODE,
        ):
            span_split = split_prov_spans(item)
        if span_split is not None:
            text = span_split[0]

        text = normalize_academic_pdf_math(text)

        # In-figure text exclusion: keep captions/footnotes even if boxed.
        prov_list = getattr(item, "prov", []) or []
        item_page = prov_list[0].page_no if prov_list else None
        is_caption_cand = (
            label == DocItemLabel.CAPTION
            or (last_kept_label == DocItemLabel.CAPTION and last_kept_page == item_page)
            or bool(CAPTION_RE.match(text))
        )
        if not is_caption_cand and label != DocItemLabel.FOOTNOTE:
            for prov in prov_list:
                if is_inside_picture(prov.page_no, prov.bbox, picture_boxes):
                    # Figure-internal text (an axis title, a legend label): the
                    # figure is carried as an image asset, so its own text stays
                    # inside that graphic rather than being translated separately.
                    logger.debug(
                        "Docling item on page %s dropped as picture text: %r",
                        prov.page_no,
                        text[:60],
                    )
                    text = ""
                    break
        if not text:
            continue

        # The analyzer types math debris and algorithm listings itself, from the
        # shared text-content rules (native analyzer AST type production: the analyzer produces the
        # types, not a later repair pass). A short math/algorithm token is a
        # FORMULA/CODE held byte-identical; Docling's own label is only a prior.
        if not skip and block_type in PROSE_BLOCK_TYPES:
            if looks_like_debris(text):
                block_type, skip = BlockType.FORMULA, True
            elif looks_like_listing(text):
                block_type, skip = BlockType.CODE, True

        last_kept_label = label
        last_kept_page = item_page

        # Extract page numbers and bounding boxes from provenance metadata.
        # A provenance entry without a bbox leaves ``item_bbox`` None rather
        # than fabricating a zero-area box at the page origin: the zone
        # builder skips bbox-less blocks, but an origin box seeds a bogus zone
        # and pollutes the per-page guards with a phantom rect.
        boxes: list[PhysicalBox] = []
        for prov in getattr(item, "prov", []) or []:
            page_no = int(getattr(prov, "page_no", 0))
            prov_bbox = getattr(prov, "bbox", None)
            if prov_bbox is None or page_no < 1:
                continue
            bx0 = float(getattr(prov_bbox, "l", 0.0))
            by0 = float(getattr(prov_bbox, "b", 0.0))
            bx1 = float(getattr(prov_bbox, "r", 0.0))
            by1 = float(getattr(prov_bbox, "t", 0.0))
            if bx1 > bx0 and by1 > by0:
                boxes.append(PhysicalBox.of(page_no, (bx0, by0, bx1, by1)))

        item_bbox = None
        composite_span: CompositeSpan | None = None
        if boxes:
            first_box = boxes[0]
            item_bbox = BoundingBox(
                page=first_box.page,
                x0=first_box.bbox[0],
                y0=first_box.bbox[1],
                x1=first_box.bbox[2],
                y1=first_box.bbox[3],
            )
            if len(boxes) > 1:
                composite_span = CompositeSpan(boxes=tuple(boxes))

        # The page is known even when the provenance entry carries no bbox.
        # Recording it lets page-strict reflow keep the block on its own page
        # instead of the nearest *preceding* block's page, which broke the 1:1
        # source/target page alignment that page-strict mode exists to hold.
        block_provenance: dict[str, Any] = {}
        if item_page is not None:
            block_provenance["source_page"] = int(item_page)
        if len(boxes) > 1:
            block_provenance["physical_boxes"] = [
                {"page": b.page, "bbox": list(b.bbox)} for b in boxes
            ]

        blocks.append(
            IRBlock(
                element=make_element(
                    id=f"pdf_main#b{len(blocks) + 1:04d}",
                    spine_index=len(blocks) + 1,
                    block_type=block_type,
                    flow_id=flow_id,
                    source_text=text,
                    bbox=item_bbox,
                    span=composite_span,
                    skip_translate=skip,
                    region=region,
                ),
                provenance=BlockProvenance(**block_provenance),
            )
        )
        if span_split is not None:
            tail_text, tail_bbox = span_split[1], span_split[2]
            blocks.append(
                IRBlock(
                    element=make_element(
                        id=f"pdf_main#b{len(blocks) + 1:04d}",
                        spine_index=len(blocks) + 1,
                        block_type=BlockType.NARRATIVE,
                        flow_id=FlowID.CAPTION,
                        source_text=normalize_academic_pdf_math(tail_text),
                        bbox=tail_bbox,
                        region=RegionKind.CAPTION,
                    ),
                    provenance=BlockProvenance(docling_span_split_tail=True),
                )
            )

    if pending_toc:
        # The outline supplies the top-level chapters Docling's TOC pass drops
        # (see ``read_pdf_outline_entries``); the merge is additive and runs
        # once, over the rows of *every* DOCUMENT_INDEX item.
        pending_toc.extend(
            _parser_symbol(
                "_extract_outline_only_blocks", _docling_outline._extract_outline_only_blocks
            )(pdf_path, pending_toc, start_index=len(pending_toc) + 1)
        )
        # Stable: rows sharing a destination keep the order their source
        # produced them in, so a sub-section never jumps above its chapter when
        # both name the same page. Appending the outline rows put every chapter
        # *after* every section, so the delivered contents read "2.1, 2.2, … 1
        # Introduction, 2 Switch Transformer" — the chapter that contains a
        # section printed below it (measured on the gold corpus,
        # switch-transformers: the TOC opened on "2.1").
        ordered_toc = sorted(pending_toc, key=_toc_sort_key)
        # The rows are inserted where the contents was read, not appended: the
        # TOC is part of the reading order at its own position (page 2 of a
        # paper, after the title/authors/abstract and before the body), and
        # every consumer sorts by ``spine_index`` — ``analyze.bridge``, the
        # HTML/EPUB companions, ``render_preflight``. Appending them put every
        # row after the body it describes and left the "Contents" heading
        # orphaned at the top of the first paragraph.
        #
        # ``insert_at`` is ``len(blocks)`` at the *first* contents item, so the
        # front matter keeps the ids it was emitted with — renumbering it from 1
        # made the ledger's ``ON CONFLICT(job_id, block_id) DO UPDATE`` overwrite
        # the title page, authors and abstract block-for-block, and the delivered
        # PDF opened on the contents with no front matter anywhere in it.
        insert_at = len(blocks) if toc_insert_at is None else min(toc_insert_at, len(blocks))
        _place_toc_rows(blocks, ordered_toc, insert_at)

    typed = type_docling_blocks(
        annotate_layout_metadata(drop_phantom_continuation_boxes(blocks), pdf_path)
    )
    # Docling reports every heading at level 1 (verified on four cached
    # conversions: 26, 53 and 73 headings, all level 1), so a rebuilt book
    # typesets a chapter title and its 4.2.2 sub-subsection identically and the
    # PDF outline comes out flat. The source PDF's bookmark tree declares the
    # real depth; restore it here, before the blocks leave the adapter.
    restored = _parser_symbol(
        "assign_heading_levels_from_outline", _docling_outline.assign_heading_levels_from_outline
    )(pdf_path, typed)
    if restored:
        logger.info(
            "Restored heading depth for %d heading(s) of '%s' from its PDF outline",
            restored,
            pdf_path.name if pdf_path is not None else "document",
        )
    return typed


def drop_phantom_continuation_boxes(blocks: list[IRBlock]) -> list[IRBlock]:
    """Drop a spurious cross-page box Docling attached to a single element.

    Docling occasionally gives one text item two ``prov`` boxes on different
    pages, the second of which is *another* element's box (a caption, a figure
    label). The parser fused them into a ``CompositeSpan`` and the compositor
    then flowed the paragraph's text across the page boundary into the caption's
    box -- overlapping, unreadable text on arXiv 2610.02185 p6.

    The tell is exact duplication: a real cross-page continuation's second box is
    a distinct region at the top of the next page, never the *same* box another
    block already owns. A chain box that coincides with a single-box block's box
    on that page is that block's, not this one's, so it is removed. When the
    first box alone remains, the block is a normal single-box element.
    """
    owned: dict[tuple[int, tuple[float, float, float, float]], str] = {}
    for block in blocks:
        boxes = block.provenance.physical_boxes
        if len(boxes) <= 1 and block.bbox is not None and block.bbox.page > 0:
            key = (
                block.bbox.page,
                (
                    round(block.bbox.x0, 1),
                    round(block.bbox.y0, 1),
                    round(block.bbox.x1, 1),
                    round(block.bbox.y1, 1),
                ),
            )
            owned.setdefault(key, block.id)

    for block in blocks:
        boxes = block.provenance.physical_boxes
        if len(boxes) <= 1:
            continue
        kept = []
        for entry in boxes:
            key = (entry["page"], tuple(round(v, 1) for v in entry["bbox"]))
            owner = owned.get(key)
            if owner is not None and owner != block.id:
                continue  # another block owns this box: not this element's
            kept.append(entry)
        if len(kept) == len(boxes):
            continue
        if len(kept) <= 1:
            # Back to a single-box element: drop the chain on both carriers --
            # the renderer reads ``element.span`` (the CompositeSpan), the ledger
            # round-trip reads ``provenance.physical_boxes``; both must agree.
            block.provenance.physical_boxes = []
            if kept:
                single = BoundingBox(
                    page=kept[0]["page"],
                    x0=kept[0]["bbox"][0],
                    y0=kept[0]["bbox"][1],
                    x1=kept[0]["bbox"][2],
                    y1=kept[0]["bbox"][3],
                )
                block.set_bbox(single)
                if isinstance(block.element.span, CompositeSpan):
                    block.element = dataclasses.replace(
                        block.element,
                        span=Span(
                            page=single.page,
                            bbox=(single.x0, single.y0, single.x1, single.y1),
                        ),
                    )
        else:
            block.provenance.physical_boxes = kept
            block.element = dataclasses.replace(
                block.element,
                span=CompositeSpan(
                    boxes=tuple(
                        PhysicalBox.of(entry["page"], tuple(entry["bbox"])) for entry in kept
                    )
                ),
            )
    return blocks


def map_export_dict(data: dict[str, Any]) -> list[IRBlock]:
    """Flat-dict fallback mapping for non-standard DoclingDocument shapes.

    Reading-order parity is impossible across the parallel texts/tables
    lists here, so this path is only a compatibility shim (unit mocks,
    legacy docling-core) — real pipelines must use ``map_iterated_items``.
    """
    blocks: list[IRBlock] = []

    def _table_text_from_dict(tbl: dict[str, Any]) -> str:
        data_obj = tbl.get("data", {})
        markdown = data_obj.get("markdown")
        if markdown:
            return str(markdown)
        grid = data_obj.get("grid") or []
        rows = [
            [
                (
                    str(
                        getattr(cell, "text", "") or cell.get("text", "")
                        if isinstance(cell, dict)
                        else getattr(cell, "text", "") or ""
                    ).strip()
                )
                for cell in row
            ]
            for row in grid
        ]
        if not rows:
            return ""
        lines = ["| " + " | ".join(row) + " |" for row in rows]
        lines.insert(1, "|" + "---|" * len(rows[0]))
        return "\n".join(lines)

    for item in data.get("texts", []):
        text = (item.get("text") or "").strip()
        if not text:
            continue

        label = str(item.get("label", "")).lower()
        block_type = BlockType.NARRATIVE
        flow_id = FlowID.MAIN_STORY
        skip_translate = False

        if label in ("page_header", "page_footer"):
            continue
        elif label in ("title", "section_header"):
            if re.match(r"^(?:fig(?:ure)?|table|图|表)\.?\s*\d+", text, re.IGNORECASE):
                block_type = BlockType.NARRATIVE
                flow_id = FlowID.CAPTION
            else:
                block_type = BlockType.HEADING
        elif label == "code":
            block_type = BlockType.CODE
            skip_translate = True
        elif label in ("formula", "equation"):
            block_type = BlockType.FORMULA
            skip_translate = True
        elif label == "footnote":
            flow_id = FlowID.FOOTNOTE
        elif label == "caption":
            flow_id = FlowID.CAPTION
        elif label in ("list_item", "list-item"):
            block_type = BlockType.LIST_ITEM

        blocks.append(
            IRBlock(
                element=make_element(
                    id=f"pdf_main#b{len(blocks) + 1:04d}",
                    spine_index=len(blocks) + 1,
                    block_type=block_type,
                    flow_id=flow_id,
                    source_text=text,
                    skip_translate=skip_translate,
                )
            )
        )

    for tbl in data.get("tables", []):
        tbl_text = _table_text_from_dict(tbl)
        if not tbl_text:
            continue
        blocks.append(
            IRBlock(
                element=make_element(
                    id=f"pdf_main#b{len(blocks) + 1:04d}",
                    spine_index=len(blocks) + 1,
                    block_type=BlockType.TABLE,
                    flow_id=FlowID.TABLE_GRID,
                    source_text=tbl_text,
                )
            )
        )

    return blocks


def map_docling_document_to_blocks(
    doc: Any,
    path: Path,
    file_sha256: str,
) -> list[IRBlock]:
    """Map a parsed DoclingDocument to a list of IRBlocks."""
    assets_dir = docling_asset_dir(file_sha256)

    try:
        from docling_core.types.doc.common.content_layer import ContentLayer

        items = list(
            doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE})
        )
    except Exception:
        try:
            items = list(doc.iterate_items())
        except Exception as exc:
            logger.warning(
                "Docling iterate_items failed on %s, falling back to export dict: %s", path, exc
            )
            items = []
    if not items:
        # Fall back to flat export-dict mapping if iterate_items produces no elements.
        return map_export_dict(doc.export_to_dict())

    return map_iterated_items(items, doc, assets_dir, pdf_path=path)


def annotate_page_kinds(path: Path, blocks: list[IRBlock]) -> dict[int, str]:
    """Stamp ``provenance["page_kind"]`` per block; return the page→kind map."""
    try:
        from ubt.adapters.pdf.page_profiler import profile_pdf
        from ubt.adapters.pdf.pdf_profiler import PROFILE_CACHE_DIR

        profiles = profile_pdf(path, cache_dir=PROFILE_CACHE_DIR)
    except Exception as exc:
        logger.debug("page profiling skipped for '%s': %s", path.name, exc)
        return {}
    kinds = {p.facts.page: p.kind.value for p in profiles}
    for b in blocks:
        if b.bbox is not None and b.bbox.page in kinds:
            b.provenance.page_kind = kinds[b.bbox.page]
    logger.debug("page kinds for '%s': %s", path.name, kinds)
    return kinds
