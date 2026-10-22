"""Docling/oxide extraction leg of the PDF adapter.

Everything that turns a PDF into ``IRBlock``s: Docling pipeline construction,
item→block mapping, the textless-page VLM fallback, page-kind profiling, and
the pdf_oxide fast fallback. No delivery state (reconstructor/alternator) leaks
in, so these are plain functions; the adapter injects the two import seams
(``symbols``, ``has_accelerator``) explicitly so its test hooks keep working.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from ubt.adapters.pdf.docling_blocks import (
    chrome_key,
    code_text_layer_is_lossy,
    crop_lossy_code_asset,
    is_inside_picture,
    is_repeat_handle,
    merge_table_continuation_fragments,
    resolve_overlapping_formula_blocks,
    split_prov_spans,
    table_to_markdown,
)
from ubt.adapters.pdf.docling_cache import (  # noqa: F401 -- re-exported
    DOCLING_ASSET_SUBDIR,
    DOCLING_CACHE_DIR,
    DOCLING_CACHE_VERSION,
    _cleanup_docling_converter,
    _ensure_docling_pdfium_lock,
    _package_version,
    _strip_picture_payloads,
    docling_asset_dir,
    docling_cache_key,
    read_docling_document,
    write_docling_document,
)
from ubt.adapters.pdf.docling_outline import (  # noqa: F401 -- re-exported
    _crop_formula_asset,
    _extract_document_index_blocks,
    _extract_outline_only_blocks,
    _toc_sort_key,
    assign_heading_levels_from_outline,
    read_pdf_outline_entries,
)
from ubt.adapters.pdf.encryption import reject_encrypted_pdf
from ubt.adapters.pdf.page_chunking import compute_pdf_page_chunks, get_pdf_page_chunk_size
from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK
from ubt.adapters.pdf.plain_text_extractor import pages_to_blocks
from ubt.analyze.structure import looks_like_debris, looks_like_listing, pdf_list_marker
from ubt.core.cleaners.lnds_pruner import normalize_academic_pdf_math
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockProvenance,
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterMeta,
    FlowID,
    IRBlock,
    StyleMeta,
    make_element,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.core.job_options import clean_source_stem
from ubt.core.policy.layout_policy import (
    CAPTION_RE,
    PROSE_BLOCK_TYPES,
    VLM_CIRCUIT_FAIL_PCT,
    VLM_CIRCUIT_MIN_TRIES,
)
from ubt.model.ast import RegionKind
from ubt.model.span import CompositeSpan, PhysicalBox, Span, boxes_to_provenance

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

#: Headroom Docling's layout/table models need before its first converter can be
#: built. The models themselves are small; the cost is a CUDA context plus
#: fragmentation, and a co-resident LLM server grows its own allocation at load
#: time. Measured on this project's reference machine: with 0.06 GiB free,
#: Docling OOM'd on a 20 MiB request *after* paying for converter
#: initialisation, then re-ran the whole conversion on CPU.
_DOCLING_MIN_FREE_VRAM_GIB: float = 2.0


def _figure_asset_dpi_tag() -> str:
    """The delivered-crop DPI tag, read lazily.

    Asset filenames carry the resolution they were cut at, so raising the DPI
    cuts new files instead of replaying old ones from the sha-keyed asset dir.
    """
    from ubt.adapters.pdf.visual_scalpel import FIGURE_ASSET_DPI_TAG

    return FIGURE_ASSET_DPI_TAG


def extract_manifest(path: Path, *, is_docling_installed: bool) -> BookManifest:
    """Extract lightweight PDF book manifest."""
    path = Path(path)
    if not path.exists():
        raise DocumentParseError(f"PDF file not found: {path}")

    # Both PDF engines reach the manifest before parsing; intercepting a
    # password-protected file here keeps Docling's ConversionError and the
    # pdf_oxide decryption noise out of the operator-facing error.
    reject_encrypted_pdf(path)

    doc_id = compute_file_sha256_cached(path)
    clean_title = clean_source_stem(path)

    total_pages = 0
    try:
        from ubt.adapters.pdf.short_doc import probe_pdf_pages

        total_pages, _ = probe_pdf_pages(path)
    except Exception:
        total_pages = 0

    chunk_size = get_pdf_page_chunk_size()
    if total_pages > chunk_size:
        chunks = compute_pdf_page_chunks(total_pages, chunk_size=chunk_size)
        chapters = [
            ChapterMeta(
                chapter_id=f"c{i + 1:04d}",
                title=f"Pages {chunk[0]}-{chunk[-1]}",
                spine_index=i + 1,
                source_file=path.name,
            )
            for i, chunk in enumerate(chunks)
        ]
    else:
        chapters = [
            ChapterMeta(
                chapter_id="pdf_main",
                title=clean_title,
                spine_index=1,
                source_file=path.name,
            )
        ]

    parser_engine = "docling" if is_docling_installed else "oxide_fallback"
    return BookManifest(
        doc_id=doc_id,
        title=clean_title,
        source_path=str(path),
        chapters=chapters,
        metadata={"pdf_parser_engine": parser_engine, "page_count": total_pages},
    )


def resolve_formula_enrichment(
    formula_enrichment: str,
    has_accelerator: Callable[[], bool],
    path: Path | None = None,
) -> bool:
    """Resolve effective formula enrichment policy (on-demand + hardware probe)."""
    mode = str(formula_enrichment).lower().strip()
    if mode in ("off", "false", "0"):
        return False
    if mode in ("on", "true", "1"):
        return True

    # Mode == "auto": resolve from what the *renderer* can do with each option.
    #
    # The renderer rebuilds the page from the IR, so a formula it
    # cannot read as markup has to be recovered some other way. Two ways exist:
    #
    # * crop the source region (``_crop_formula_asset``) — instant, and the
    #   reader gets the author's own equation, glyph for glyph;
    # * run the VLM math OCR — 15-60s *per formula* on CPU, and its LaTeX is a
    #   model's transcription, which can be wrong in ways a crop cannot. The
    #   crop margin is now tuned (``formula_crop_margin``), which removes the
    #   neighbouring-text pollution, but the model still mis-reads some symbols
    #   (e.g. "expert" -> ``\exp``), so a faithful crop stays the safe default.
    #
    # So ``auto`` takes the crop: faithful and affordable. ``on`` forces the VLM
    # for a caller who wants scalable vector math and accepts the cost.
    logger.info(
        "Docling formula enrichment set to False (auto: unreadable formulas are "
        "recovered by cropping the source region, which is exact and instant; "
        "set UBT_FORMULA_ENRICHMENT=on to transcribe them with the VLM instead)"
    )
    return False


def configure_hf_environment(*, enrich: bool = False) -> None:
    """Optimize Hugging Face environment for Docling model loading.

    1. If required models are already cached locally, set HF_HUB_OFFLINE=1 so
       snapshot_download does not hang for 135s * 3 attempting HEAD requests
       to huggingface.co without an international proxy.
    2. If models need downloading and no proxy is configured, set
       HF_ENDPOINT='https://hf-mirror.com' to avoid network dropouts/hangs.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    required_repos = [
        "models--docling-project--docling-layout-heron",
        "models--docling-project--docling-models",
    ]
    if enrich:
        required_repos.append("models--docling-project--CodeFormulaV2")

    all_cached = hf_home.is_dir() and all(
        (hf_home / repo / "snapshots").is_dir() and any((hf_home / repo / "snapshots").iterdir())
        for repo in required_repos
    )
    if all_cached:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        logger.debug(
            "Configured HF_HUB_OFFLINE=1 (all %d Docling models verified in local cache)",
            len(required_repos),
        )
    else:
        has_proxy = any(
            os.environ.get(k)
            for k in (
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
            )
        )
        mirror_opted_in = os.environ.get("UBT_ALLOW_HF_MIRROR", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if not has_proxy and "HF_ENDPOINT" not in os.environ:
            if mirror_opted_in:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                logger.info(
                    "HF_ENDPOINT=https://hf-mirror.com for Hugging Face model downloads "
                    "(UBT_ALLOW_HF_MIRROR is set)."
                )
            else:
                # Repointing model downloads at a third party is an egress
                # decision the operator must make: same posture as the offline
                # retry ladder below, which only switches with the opt-in.
                logger.info(
                    "Uncached Docling models will download from huggingface.co; set "
                    "UBT_ALLOW_HF_MIRROR=1 to use https://hf-mirror.com instead."
                )


#: An ordered-marker run must be at least this long to be re-typed as a list:
#: a lone ``(2024) ...`` citation opens like a marker but is not a list item.
_ORDERED_MARKER_MIN_RUN = 2

#: How many times the *same* text must appear in a page-header/footer position
#: before it is treated as running chrome rather than one-off content.
#:
#: The comparison is ``>=``, not ``>``: a running head recurs once per page it
#: spans, and three occurrences is already repetition — a title-page credit line
#: appears once, a real section heading is never re-emitted by the *layout*
#: model in the header band at all. With the old strict ``> 3`` a head that
#: spanned exactly three pages survived as a ``HEADING`` block in the main
#: story, so "3.2 Unified FinFET compact model" and "Appendix Explicit surface
#: potential model" were typeset as chapter-level headings mid-chapter, in the
#: source language, three times each.
_CHROME_REPEAT_MIN = 3
#: Ordered-marker shapes (numbered ``(1)`` / ``1.`` / ``1)``, lettered ``a.`` /
#: ``(a)``, CJK ``一、``). Bullets are deliberately excluded: Docling already
#: types real bullet lists, so a bullet match here is more likely a symbol.
#: Floor for a one-off page footer to count as a footnote paragraph rather than
#: chrome (see ``_is_prose_footer``). A page number or running title is far below.
_FOOTER_PROSE_MIN_CHARS = 80
_FOOTER_PROSE_MIN_WORDS = 12

#: How much of a reference entry's text to match against the page's own text
#: layer when recovering the ``[n]`` Docling dropped. 40 characters is well past
#: the point where two entries in one bibliography can share a prefix (they start
#: with different author names) and short enough to tolerate a re-wrapped line.
_REFERENCE_PREFIX_CHARS = 40

_ORDERED_MARKER_RE = re.compile(
    r"^(?:"
    r"[（(]?[0-9０-９]{1,3}(?:[.．][0-9０-９]{1,3})*[.．、)）]"
    r"|[（(]?[a-zA-ZＡ-Ｚａ-ｚ][.．)）]"
    r"|[一二三四五六七八九十百]+[、.．)）]"
    r")$"
)
#: A first-line indent is this many em beyond the body margin (matches
#: ``reader_pdf.INDENT_FACTOR``); the two readers must agree on what an indent is.
_INDENT_FACTOR = 1.0
#: A single-line block carries no body margin to compare against, so an indent
#: is only measurable on a block of at least this many source lines.
_INDENT_MIN_LINES = 2


def _ordered_marker(block: IRBlock) -> tuple[str, str] | None:
    """``(marker, text)`` when a body paragraph opens with an ordered marker.

    Docling's own list typing misses parenthesized enumerations like ``(1)`` that
    it reads as one paragraph, so the marker survives only if the LLM reproduces
    it -- which it does inconsistently. Re-typing the run here lets the
    compositor's marker restorer put the number back deterministically.
    """
    if block.skip_translate or block.block_type is not BlockType.NARRATIVE:
        return None
    if block.region is not RegionKind.BODY:
        return None
    parsed = pdf_list_marker(block.source_text or "")
    if parsed is None:
        return None
    marker, text = parsed
    if not _ORDERED_MARKER_RE.match(marker):
        return None
    return marker, text


def _as_list_item(block: IRBlock, marker: str, text: str) -> IRBlock:
    element = make_element(
        id=block.id,
        spine_index=block.spine_index,
        block_type=BlockType.LIST_ITEM,
        flow_id=block.flow_id,
        region=block.region,
        source_text=text,
        bbox=block.bbox,
        span=block.element.span,
        skip_translate=block.skip_translate,
        confidence=block.element.confidence,
        decorative=block.element.decorative,
        marker=marker,
    )
    return block.model_copy(update={"element": element})


def restore_ordered_markers(blocks: list[IRBlock]) -> list[IRBlock]:
    """Re-type runs of marker-led body paragraphs as list items.

    Only a *run* (``_ORDERED_MARKER_MIN_RUN`` or more consecutive marker-led
    paragraphs) is converted, so a paragraph that merely starts with a bracketed
    year or a reference stays prose.
    """
    out = list(blocks)
    index = 0
    while index < len(out):
        if _ordered_marker(out[index]) is None:
            index += 1
            continue
        run: list[tuple[int, str, str]] = []
        cursor = index
        while cursor < len(out):
            parsed = _ordered_marker(out[cursor])
            if parsed is None:
                break
            run.append((cursor, parsed[0], parsed[1]))
            cursor += 1
        if len(run) >= _ORDERED_MARKER_MIN_RUN:
            for position, marker, text in run:
                out[position] = _as_list_item(out[position], marker, text)
        index = cursor
    return out


def _is_prose_footer(text: str) -> bool:
    """Whether a one-off page footer is a footnote paragraph, not chrome.

    A footer is prose the author wrote when it reads like a sentence: several
    words AND sentence punctuation/length. A page number, a running title, a
    copyright strip or an arXiv stamp is short and carries none of that, so it
    stays skipped. Kept precision-first — a false positive would translate a
    watermark into the wrong language.
    """
    body = (text or "").strip()
    if len(body) < _FOOTER_PROSE_MIN_CHARS:
        return False
    if len(body.split()) < _FOOTER_PROSE_MIN_WORDS:
        return False
    # A footnote reads as one or more sentences: it ends in sentence punctuation
    # OR contains an internal sentence boundary (". " / "。"). A bare label, DOI
    # or copyright strip carries neither.
    if body.endswith((".", "。", "：", ":")):
        return True
    return ". " in body or "。" in body


def _table_reader_enabled() -> bool:
    """Whether the opt-in vision table reader is configured."""
    from ubt.adapters.pdf import table_reader

    return table_reader.enabled()


def _read_table_cells(item: Any, pdf_path: Path | None, docling_markdown: str) -> str | None:
    """Docling's table with its data cells re-read from the page image, or ``None``.

    The crop is Docling's own table bbox — the reader is a region reader, and
    feeding it a page collapses several tables into one (measured). The header
    row and the column count stay Docling's: the reader misreads header text,
    and the header is what translation and the leak metric key on.
    """
    if pdf_path is None or not pdf_path.exists():
        return None
    prov = item.prov[0] if getattr(item, "prov", None) else None
    if prov is None or getattr(prov, "bbox", None) is None:
        return None
    page_no = int(getattr(prov, "page_no", 0) or 0)
    if page_no < 1:
        return None
    b = prov.bbox
    bbox = BoundingBox(
        page=page_no,
        x0=float(getattr(b, "l", 0.0)),
        y0=float(getattr(b, "b", 0.0)),
        x1=float(getattr(b, "r", 0.0)),
        y1=float(getattr(b, "t", 0.0)),
    )
    if bbox.x1 - bbox.x0 < 24 or bbox.y1 - bbox.y0 < 16:
        # Too small to be a table; cropping it would send noise to the model.
        return None
    try:
        from ubt.adapters.pdf import table_reader
        from ubt.adapters.pdf.visual_scalpel import FIGURE_ASSET_DPI, crop_block_pil

        image = crop_block_pil(pdf_path, page_no, bbox, dpi=FIGURE_ASSET_DPI)
    except Exception as exc:  # noqa: BLE001 - no crop means no upgrade
        logger.debug("table crop failed on page %s: %s", page_no, exc)
        return None
    read = table_reader.read_table_image_sync(image)
    if read is None:
        return None
    docling_rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in docling_markdown.splitlines()
        if line.strip().startswith("|") and "---" not in line
    ]
    merged = table_reader.merge_with_docling(docling_rows, read)
    if merged is None:
        return None
    header, _, _ = docling_markdown.partition("\n|---")
    separator = "|" + "|".join("---" for _ in merged[0]) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in merged)
    if header:
        return f"{header}\n{separator}\n{body}"
    return f"{separator}\n{body}"


def _recover_reference_marker(text: str, page_text: str) -> str | None:
    """The bracketed number Docling dropped from a reference entry, or ``None``.

    Docling types a reference as a ``list_item`` but leaves ``marker`` empty and
    strips the number out of ``text``: on arXiv 2609.20519 the source carries 108
    citation markers, every ``[n]`` prefix survives in the page's own text layer,
    and the extraction kept none of them — the delivered bibliography lost its
    numbering.

    Recovered by *text prefix*, not by coordinates: the page text holds
    ``[1] Mark Chen, Jerry Tworek, …`` while Docling's item holds
    ``Mark Chen, Jerry Tworek, …``. A prefix match cannot confuse two entries
    (unlike a bbox match, whose coordinate systems did not line up in testing and
    would have silently numbered the wrong paragraphs).
    """
    head = " ".join((text or "").split())[:_REFERENCE_PREFIX_CHARS]
    if not head or not page_text:
        return None
    # Compare on a whitespace-collapsed haystack so a line break *inside a word*
    # ("Da\nvid") still matches: the page's text layer wraps mid-word where
    # Docling's text has no break at all, and a pure regex over the raw text
    # misses those entries (measured: 52/54 recovered with this, vs 1/57 for a
    # raw-text regex).
    collapsed = " ".join(page_text.split())
    pattern = r"(\[\d{1,3}\])\s*" + re.escape(head)
    match = re.search(pattern, collapsed)
    if match:
        return match.group(1)
    # Second failure mode: Docling drops a space *inside* an organisation name —
    # "[38] Axiom Math. AxiomProver…" arrives as "AxiomMath. AxiomProver…" — so
    # retry with both sides space-stripped. This recovers the two organisation
    # entries the whitespace pass missed (measured: 54/54 on the reference list).
    compact_head = head.replace(" ", "")
    compact_page = collapsed.replace(" ", "")
    match = re.search(
        r"(\[\d{1,3}\])\s*" + re.escape(compact_head),
        compact_page,
    )
    if match:
        return match.group(1)
    # Third: the page layer uses typographic punctuation the extractor flattens
    # ("Anthropic’s" arrives as "Anthropic's"), so compare with apostrophes and
    # dashes folded to their ASCII forms.
    folded_head = re.sub(r"[\u2018\u2019\u02bc]", "'", compact_head)
    folded_page = re.sub(r"[\u2018\u2019\u02bc]", "'", compact_page)
    match = re.search(r"(\[\d{1,3}\])\s*" + re.escape(folded_head), folded_page)
    return match.group(1) if match else None


def _iter_pdf_page_texts(pdf_path: Path, first: int, last: int) -> list[str]:
    """Text of pages ``first..last`` (1-based, clamped), unreadable pages as ``""``."""
    out: list[str] = []
    try:
        import pypdfium2 as pdfium

        with PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(pdf_path))
            try:
                total = len(pdf)
                for page_no in range(max(1, first), min(total, last) + 1):
                    try:
                        out.append(pdf[page_no - 1].get_textpage().get_text_range())
                    except Exception:  # noqa: BLE001 - one bad page must not lose the rest
                        out.append("")
            finally:
                pdf.close()
    except Exception as exc:  # noqa: BLE001 - recovery is best-effort
        logger.debug("reference-marker recovery: %s unreadable (%s)", pdf_path, exc)
    return out


def _line_in_box(rect: tuple[float, float, float, float], box: BoundingBox) -> bool:
    center_y = (rect[1] + rect[3]) / 2.0
    if not (box.y0 - 1.0 <= center_y <= box.y1 + 1.0):
        return False
    return min(rect[2], box.x1) - max(rect[0], box.x0) > 1.0


def _first_line_indent(box: BoundingBox, lines: list[Any]) -> float | None:
    """Indent (pt) of the block's first line past its own body margin, or ``None``."""
    inside = [line for line in lines if line.text.strip() and _line_in_box(line.rect, box)]
    if len(inside) < _INDENT_MIN_LINES:
        return None
    inside.sort(key=lambda line: -line.rect[3])
    body_x0 = min(float(line.rect[0]) for line in inside[1:])
    first = inside[0]
    indent = float(first.rect[0]) - body_x0
    font = float(first.font_size) if first.font_size and first.font_size > 0 else (box.y1 - box.y0)
    if indent > _INDENT_FACTOR * max(font, 1.0) and indent < (box.x1 - box.x0) * 0.5:
        return indent
    return None


def _is_centered(box: BoundingBox, lines: list[Any]) -> bool:
    """Whether any line of the block hangs symmetrically inside its own box.

    A centered title's short lines sit evenly between the margins; a
    left-aligned heading's lines start at the box edge (zero left gap). One
    qualifying line is enough — a multi-line centered title usually has its
    widest line touching both margins, so the evidence lives in the others.
    """
    inside = [line for line in lines if line.text.strip() and _line_in_box(line.rect, box)]
    width = box.x1 - box.x0
    if width <= 0:
        return False
    gap_min = max(6.0, 0.02 * width)
    tolerance = max(6.0, 0.05 * width)
    for line in inside:
        left = float(line.rect[0]) - box.x0
        right = box.x1 - float(line.rect[2])
        if left > gap_min and right > gap_min and abs(left - right) <= tolerance:
            return True
    return False


def _line_boxes(box: BoundingBox, lines: list[Any]) -> tuple[PhysicalBox, ...]:
    """The block's own lines as a reading-order box chain (top to bottom)."""
    inside = [line for line in lines if line.text.strip() and _line_in_box(line.rect, box)]
    inside.sort(key=lambda line: -line.rect[3])
    return tuple(
        PhysicalBox.of(
            int(box.page),
            (float(line.rect[0]), float(line.rect[1]), float(line.rect[2]), float(line.rect[3])),
        )
        for line in inside
    )


def annotate_layout_metadata(blocks: list[IRBlock], pdf_path: Path | None) -> list[IRBlock]:
    """Record per-block layout facts the block box alone cannot carry.

    Docling gives one bounding box per block (the margin), so three facts are
    invisible in its output and must come from the page's own line rects
    (textgeom):

    - a body paragraph's first-line indent (``first_line_indent_pt``);
    - a list item's marker-column indent — the block box starts at the
      *wrapped* lines' margin, so without the recorded indent the compositor
      draws the marker flush with the body margin where the source hangs it
      to the right;
    - a heading's centering (``alignment="center"``) — a centered title's
      short lines hang symmetrically inside the box; without the flag the
      fragment is drawn left-aligned and the title hugs the margin.

    A page whose lines cannot be read (a scan with no text layer) simply gets
    no metadata.
    """
    if pdf_path is None:
        return blocks
    from ubt.adapters.pdf.textgeom import extract_lines

    lines_by_page: dict[int, list[Any]] = {}

    def _page_lines(page: int) -> list[Any]:
        if page not in lines_by_page:
            try:
                lines_by_page[page] = list(extract_lines(pdf_path, page)[0])
            except Exception:
                lines_by_page[page] = []
        return lines_by_page[page]

    for block in blocks:
        if block.skip_translate:
            continue
        box = block.bbox
        if box is None or box.page <= 0:
            continue
        if block.block_type in (BlockType.NARRATIVE, BlockType.LIST_ITEM):
            if block.region is not RegionKind.BODY:
                continue
            indent = _first_line_indent(box, _page_lines(int(box.page)))
            if indent is not None:
                block.style = (block.style or StyleMeta()).model_copy(
                    update={"first_line_indent_pt": indent}
                )
        elif block.block_type is BlockType.HEADING:
            if block.region not in (RegionKind.BODY, RegionKind.TITLE):
                continue
            page_lines = _page_lines(int(box.page))
            if _is_centered(box, page_lines):
                block.style = (block.style or StyleMeta()).model_copy(
                    update={"alignment": "center"}
                )
            # A multi-line heading also records its own line boxes: the
            # compositor flows the translation across the source's line
            # structure, so the title breaks where the source title breaks
            # (usually at the subtitle colon) instead of at an arbitrary
            # width-fill point, which can strand a particle ("的") at a line
            # head and mangle the phrasing.
            line_boxes = _line_boxes(box, page_lines)
            if len(line_boxes) >= 2:
                block.element = dataclasses.replace(
                    block.element, span=CompositeSpan(boxes=line_boxes)
                )
                # Mirror the chain into the one provenance key the ledger can
                # rebuild a CompositeSpan from (``ledger_base._row_to_block``).
                # Without it the export stage -- which always reloads its blocks
                # from the ledger -- reads the heading back as a single span
                # holding the *first* line's box: only that line is masked and
                # the whole translation is squeezed into it, while the rest of
                # the source heading stays in the source language on the page.
                block.provenance.physical_boxes = boxes_to_provenance(line_boxes)
    return blocks


def type_docling_blocks(blocks: list[IRBlock]) -> list[IRBlock]:
    """The analyzer's own typing stage: raw Docling blocks -> final typed blocks.

    This is the Docling analyzer producing its own types (native analyzer AST type production). It
    trusts Docling's labels end to end -- reading order, chrome (PAGE_HEADER /
    PAGE_FOOTER), headings (TITLE / SECTION_HEADER), captions (CAPTION and the
    ``FIG. N`` title shape) and segmentation -- and the emission loop types
    debris/listing directly from the shared rules. There is no caption fuse /
    decouple / latch pass re-guessing boundaries afterwards.

    Rendering-correctness merges:
    1. Vertically overlapping FORMULA boxes are unioned so the shared region is cropped once;
    2. Loose code/formula fragments stranded immediately below a table are merged into the table.
    """
    blocks = resolve_overlapping_formula_blocks(blocks)
    blocks = merge_table_continuation_fragments(blocks)
    blocks = restore_ordered_markers(blocks)
    return blocks


def _free_vram_gib() -> float | None:
    """Free GPU memory in GiB, or ``None`` when no usable CUDA device exists.

    Deliberately uses the driver's figure (``torch.cuda.mem_get_info``) rather
    than the allocator's: the question is how much room the whole card has, and
    an LLM server in another process holds most of it.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return float(free) / (1024**3)
    except Exception:
        return None


def _docling_vram_verdict(device: str) -> tuple[str, str | None]:
    """Decide where Docling should run, and why if it is not the default.

    Returns ``(device, note)``; ``note`` is ``None`` when the caller's request
    stands. Docling's layout/table models need well under 2 GiB, but the process
    also pays for CUDA context and fragmentation, and a co-resident LLM server
    expands its own allocation at load time. On this machine a local
    ``llama-server`` held 9.28 GiB while swapping models, so Docling's first
    attempt OOM'd after initialising a converter -- then fell back to CPU and
    re-ran the conversion, discarding everything that had been set up.

    Only ``auto`` is overridden. An operator who pinned ``cpu`` or ``cuda`` has
    made a decision this probe must not second-guess.
    """
    if device != "auto":
        return device, None
    free = _free_vram_gib()
    if free is None:
        return device, None
    if free < _DOCLING_MIN_FREE_VRAM_GIB:
        return (
            "cpu",
            f"only {free:.2f} GiB of GPU memory is free (Docling needs about "
            f"{_DOCLING_MIN_FREE_VRAM_GIB:.1f} GiB of headroom); using CPU for extraction"
            + _vram_hint(),
        )
    return device, None


def _vram_hint() -> str:
    """Name the processes holding the card, when the query tool can tell us.

    A number alone ("0.06 GiB free") leaves the operator guessing what to close;
    ``nvidia-smi`` already knows.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if not rows:
        return ""
    # Largest holders first: that is the one worth closing.
    #
    # Every row is parsed defensively. ``nvidia-smi`` exits 0 on some hosts while
    # printing a driver-failure banner ("NVIDIA-SMI has failed because it
    # couldn't communicate with the NVIDIA driver…") to *stdout*; that line has
    # no commas, so an unpacking assignment raised ``ValueError`` and crashed
    # the extraction path on a machine whose only sin was having a broken
    # driver. A hint is best-effort text for a log line — a row this function
    # cannot read is skipped, never fatal.
    parsed: list[tuple[int, str]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",", 2)]
        if len(parts) != 3:
            continue
        pid, name, usage = parts
        with suppress(ValueError, IndexError):
            parsed.append((int(usage.split()[0]), f"pid {pid} {name.split('/')[-1]}"))
    if not parsed:
        return ""
    top = sorted(parsed, reverse=True)[:2]
    return "; largest holders: " + ", ".join(f"{label} ({mb} MiB)" for mb, label in top)


def extract_with_docling(
    path: Path,
    page_range: tuple[int, int] | None,
    *,
    symbols: Callable[[], tuple[Any, Any, Any, Any, Any]],
    enrich: bool,
    device: str = "auto",
    formula_crop_margin: float = 0.08,
) -> list[IRBlock]:
    """Extract structured blocks using IBM Docling in strict reading order."""
    configure_hf_environment(enrich=enrich)
    (
        input_format,
        pipeline_options_cls,
        converter_cls,
        format_option_cls,
        code_formula_options_cls,
    ) = symbols()
    # Docling is now imported: make its pypdfium2 access share PDFIUM_LOCK.
    _ensure_docling_pdfium_lock()

    options = pipeline_options_cls()
    # Born-digital academic PDFs: OCR off (fast, no false positives);
    options.do_ocr = False
    options.do_formula_enrichment = enrich
    if enrich:
        # Docling crops each formula with an ``expansion_factor`` margin before
        # the VLM reads it, defaulting to 0.18 of the bbox on every side. On a
        # tight academic layout that reaches into the equation number and the
        # neighbouring prose line, so the model transcribes *that* text into the
        # LaTeX. Docling's own guidance is to lower the margin (and stop
        # repetition loops); both are first-class preset options (docling#4548).
        # ``stop_on_repetition`` also trims the filler a too-tight crop can
        # provoke, so a smaller margin does not trade one artifact for another.
        options.code_formula_options = code_formula_options_cls.from_preset(
            "codeformulav2",
            expansion_factor=formula_crop_margin,
            stop_on_repetition=True,
        )
    options.generate_picture_images = True
    options.images_scale = 2.0
    # Delivered figure/formula/code assets do NOT come from this scale: the
    # conversion cache strips picture payloads on write, so on every cached
    # re-ingest ``item.image`` is None and the asset is re-cut from the source
    # page by the visual scalpel (``FIGURE_ASSET_DPI``, 300). Raising this scale
    # instead was measured on the reference corpus: it multiplied the page
    # rasters Docling retains across a conversion (``keep_images`` stays on
    # whenever picture images are requested) from ~0.8 GB to ~4 GB peak RSS
    # while changing no delivered pixel — the scalpel fallback dominated.
    # Whether Docling's layout/table/formula models touch the GPU is a separate
    # decision from whether Docling runs. ``auto`` (the default) leaves Docling's
    # own probe in charge; ``cpu`` keeps it off the card so a local LLM sharing
    # the GPU can still load its weights (see
    # docs/guides/PDF_ENGINE_ROUTING_REDESIGN.md §L4). The device is part of the
    # conversion cache key below, so a CPU and a CUDA conversion never alias.
    device = (device or "auto").strip().lower()
    # Decide the device BEFORE any converter is built. Without this the code
    # below discovers the shortage the expensive way: it initialises Docling on
    # cuda, hits OOM on the first allocation, tears the converter down, and
    # re-runs the conversion on CPU -- paying twice and dropping whatever the
    # first attempt had produced.
    device, vram_note = _docling_vram_verdict(device)
    if vram_note:
        # Say explicitly that earlier "Accelerator device: 'cuda:0'" lines from
        # Docling's own import-time probe do not apply: the converter is pinned
        # to cpu below, so a reader scanning for the device would otherwise
        # conclude the run used the GPU.
        logger.warning(
            "Docling device auto-selected 'cpu' (overrides Docling's own probe): %s", vram_note
        )
    if device in ("cpu", "cuda"):
        options.accelerator_options.device = device
        logger.info("Docling accelerator device pinned to '%s' for '%s'", device, path.name)

    if enrich:
        logger.info(
            "Docling formula enrichment active (VLM on GPU). "
            "Transcribing mathematical equations in '%s' to LaTeX...",
            path.name,
        )
    else:
        logger.info(
            "Docling formula enrichment bypassed for '%s' (fast ingest active, 0 GPU VLM inference overhead).",
            path.name,
        )

    def _make_converter(opt: PdfPipelineOptions) -> DocumentConverter:
        return cast(
            "DocumentConverter",
            converter_cls(
                format_options={input_format.PDF: format_option_cls(pipeline_options=opt)}
            ),
        )

    converter = _make_converter(options)
    # Docling takes one inclusive (start, end) range; a page-ranged job
    # pays only for those pages instead of converting and discarding the
    # rest of the PDF.
    convert_kwargs: dict[str, Any] = {"page_range": page_range} if page_range is not None else {}
    file_sha256 = compute_file_sha256_cached(path)

    def _convert() -> Any:
        """Convert with the on-disk cache in front of it.

        Read lazily through the closure so every retry of the ladder below —
        which rebinds ``converter`` and mutates ``options`` — is keyed on the
        configuration it is actually about to run with.
        """
        key = docling_cache_key(file_sha256, page_range, options)
        cached = read_docling_document(key)
        if cached is not None:
            logger.info(
                "Reusing the cached Docling conversion of '%s' instead of re-running it.",
                path.name,
            )
            return cached
        document = converter.convert(path, **convert_kwargs).document
        write_docling_document(key, document)
        return document

    try:
        try:
            doc = _convert()
        except Exception as exc:
            # configure_hf_environment set HF_HUB_OFFLINE process-wide; the online
            # retry below pops it, so it must be restored once the ladder is done.
            # Leaving it popped silently disabled offline mode for every in-flight
            # job in a long-lived server.
            # Known tradeoff: the pop spans the whole retry conversion, so a
            # concurrently-running job sees the mutated env in that window.
            # The mutation only happens on the rare offline-retry path, and
            # serializing conversions behind a lock for it would stall every
            # other job — accepted and documented rather than locked.
            offline_before = os.environ.get("HF_HUB_OFFLINE")
            # The retry below may point HF at the third-party hf-mirror.com — but
            # only with the operator's explicit opt-in (UBT_ALLOW_HF_MIRROR=1, the
            # same posture as allow_page_upload in config.py): silently repointing
            # model downloads at a host the operator never chose is an egress
            # decision, not a performance default, so the default retry goes to
            # huggingface.co. Whichever way HF_ENDPOINT ends up, it must not
            # outlive the ladder: every later HF download in the process
            # (including the DeepSeek worker's snapshot_download, which inherits
            # subprocess_env()) would otherwise use it silently.
            endpoint_before = os.environ.get("HF_ENDPOINT")
            try:
                retry_succeeded = False
                if "OfflineMode" in str(type(exc)) or "offline" in str(exc).lower():
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    proxied = any(
                        os.environ.get(k)
                        for k in (
                            "http_proxy",
                            "https_proxy",
                            "all_proxy",
                            "HTTP_PROXY",
                            "HTTPS_PROXY",
                            "ALL_PROXY",
                        )
                    )
                    mirror_opted_in = os.environ.get("UBT_ALLOW_HF_MIRROR", "").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                    )
                    if proxied:
                        logger.warning(
                            "Docling requested an uncached model in offline mode (%s); retrying online through the configured proxy...",
                            exc,
                        )
                    elif mirror_opted_in:
                        logger.warning(
                            "Docling requested an uncached model in offline mode (%s); retrying with online mirror...",
                            exc,
                        )
                        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                    else:
                        logger.warning(
                            "Docling requested an uncached model in offline mode (%s); retrying against huggingface.co. "
                            "If that host is unreachable from this network, set UBT_ALLOW_HF_MIRROR=1 to retry via "
                            "https://hf-mirror.com instead.",
                            exc,
                        )
                    _cleanup_docling_converter(converter)
                    converter = _make_converter(options)
                    try:
                        doc = _convert()
                        retry_succeeded = True
                    except Exception as retry_exc:
                        exc = retry_exc
                # A successful offline retry already produced the enriched
                # document. Falling into the enrichment fallback below (keyed
                # on the stale offline `exc`) discarded it and re-converted with
                # the VLM disabled — untested because every offline-ladder case
                # used enrich=False.
                if retry_succeeded:
                    pass
                # Graceful degradation: if formula enrichment caused failure/OOM, fallback to plain extraction
                elif options.do_formula_enrichment:
                    logger.warning(
                        "Docling formula enrichment failed (%s); gracefully falling back to extraction without VLM (do_formula_enrichment=False)",
                        exc,
                    )
                    options.do_formula_enrichment = False
                    _cleanup_docling_converter(converter)
                    converter = _make_converter(options)
                    try:
                        doc = _convert()
                    except Exception as inner_exc:
                        if (
                            "cuda" in str(inner_exc).lower()
                            or "out of memory" in str(inner_exc).lower()
                        ):
                            logger.warning(
                                "Docling accelerator failed (%s); falling back to CPU for this "
                                "conversion, which will be several times slower. %s",
                                inner_exc,
                                _vram_hint() or "Set UBT_DOCLING_DEVICE=cpu to skip this probe.",
                            )
                            options.accelerator_options.device = "cpu"
                            _cleanup_docling_converter(converter)
                            converter = _make_converter(options)
                            doc = _convert()
                        else:
                            raise
                elif "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                    logger.warning(
                        "Docling accelerator failed (%s); falling back to CPU for this "
                        "conversion, which will be several times slower. %s",
                        exc,
                        _vram_hint() or "Set UBT_DOCLING_DEVICE=cpu to skip this probe.",
                    )
                    options.accelerator_options.device = "cpu"
                    _cleanup_docling_converter(converter)
                    converter = _make_converter(options)
                    doc = _convert()
                else:
                    raise
            finally:
                if offline_before is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = offline_before
                if endpoint_before is None:
                    os.environ.pop("HF_ENDPOINT", None)
                else:
                    os.environ["HF_ENDPOINT"] = endpoint_before
    finally:
        _cleanup_docling_converter(converter)

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
    picture_boxes: list[tuple[int, Any]] = []
    for item, _level in items:
        if item.label == DocItemLabel.PICTURE:
            for prov in getattr(item, "prov", []) or []:
                picture_boxes.append((prov.page_no, prov.bbox))

    blocks: list[IRBlock] = []
    seen_handles: set[str] = set()
    last_kept_label: Any = None
    last_kept_page: int | None = None
    page_lines_cache: dict[int, tuple[list[Any], float]] = {}
    #: TOC rows gathered from every DOCUMENT_INDEX item, merged with the PDF
    #: outline and placed once after the loop (see that branch).
    pending_toc: list[IRBlock] = []
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
            toc_blocks = _extract_document_index_blocks(
                item, pdf_path, page_lines_cache, start_index=1
            )
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
                assets_dir.mkdir(parents=True, exist_ok=True)
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
                        assets_dir.mkdir(parents=True, exist_ok=True)
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
                code_asset = crop_lossy_code_asset(
                    item, assets_dir, pdf_path, len(blocks) + 1, text
                )
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
                formula_asset = _crop_formula_asset(item, assets_dir, pdf_path, len(blocks) + 1)
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
            _extract_outline_only_blocks(pdf_path, pending_toc, start_index=len(pending_toc) + 1)
        )
        # Stable: rows sharing a destination keep the order their source
        # produced them in, so a sub-section never jumps above its chapter when
        # both name the same page. Appending the outline rows put every chapter
        # *after* every section, so the delivered contents read "2.1, 2.2, … 1
        # Introduction, 2 Switch Transformer" — the chapter that contains a
        # section printed below it (measured on the gold corpus,
        # switch-transformers: the TOC opened on "2.1").
        #
        # The renumbering is not cosmetic: every downstream consumer orders
        # blocks by ``spine_index`` (``analyze.bridge``, ``render_preflight``,
        # the ledger), so reordering the list alone is undone on the next sort
        # and the delivered contents still opened on "2.1". Ids are rebuilt from
        # the new position too, so they stay consistent with the index they
        # encode.
        #
        # The renumber starts *after* the blocks already emitted (the title
        # page, authors, abstract and footnotes), never at 1: those slots are
        # taken, and the ledger's ``ON CONFLICT(job_id, block_id) DO UPDATE``
        # made the clash silent — the TOC rows *overwrote* the front matter, so
        # the delivered PDF opened on the contents page with no title, authors
        # or abstract anywhere in it.
        base = len(blocks)
        ordered_toc = sorted(pending_toc, key=_toc_sort_key)
        for new_index, block in enumerate(ordered_toc, start=base + 1):
            block.set_spine_index(new_index)
            block.set_id(f"pdf_main#b{new_index:04d}")
        blocks.extend(ordered_toc)

    typed = type_docling_blocks(
        annotate_layout_metadata(drop_phantom_continuation_boxes(blocks), pdf_path)
    )
    # Docling reports every heading at level 1 (verified on four cached
    # conversions: 26, 53 and 73 headings, all level 1), so a rebuilt book
    # typesets a chapter title and its 4.2.2 sub-subsection identically and the
    # PDF outline comes out flat. The source PDF's bookmark tree declares the
    # real depth; restore it here, before the blocks leave the adapter.
    restored = assign_heading_levels_from_outline(pdf_path, typed)
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


def extract_with_oxide(path: Path) -> list[IRBlock]:
    """Fallback lightweight text extractor using MIT/Apache pdf_oxide.

    The analyzer types directly from the text content (debris/listing/heading/
    list -- the shared rules in :mod:`ubt.analyze.structure`); there is no flow
    repair pass. With no geometry to measure, that is the strongest honest
    typing this fallback can make.
    """
    from ubt.adapters.pdf import oxide_render

    texts = oxide_render.extract_page_texts(path)
    if not texts:
        raise DocumentParseError(f"Failed to extract text from PDF with pdf_oxide: {path.name}")
    return pages_to_blocks((page_num, text) for page_num, text in enumerate(texts, start=1))


def annotate_page_kinds(path: Path, blocks: list[IRBlock]) -> dict[int, str]:
    """Stamp ``provenance["page_kind"]`` per block; return the page→kind map."""
    try:
        from ubt.adapters.pdf.engine_selector import PROFILE_CACHE_DIR
        from ubt.adapters.pdf.page_profiler import profile_pdf

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


def _has_text_content(blocks: list[IRBlock]) -> bool:
    """Return True if the parsed blocks contain any actual readable/translatable text."""
    for b in blocks:
        if b.block_type == BlockType.IMAGE:
            continue
        if b.source_text and b.source_text.strip():
            return True
    return False


def _reject_empty_book(
    path: Path,
    blocks: list[IRBlock],
    ocr_mode: str | None = None,
    *,
    missing_pages: Sequence[int] = (),
    total_pages: int | None = None,
) -> list[IRBlock]:
    """Fail loudly when a parse produced zero content blocks, or dropped pages.

    A PDF whose pages are all scanned images yields zero blocks whenever OCR
    is off (the default) or no driver is available; letting it through would
    make downstream stages "succeed" and export an empty book with every
    counter at zero — the classic silent-data-loss shape.

    The *partial* case is the same failure in smaller proportion, and it used to
    pass: ``_has_text_content`` returns True as soon as **one** block carries
    text, so a 100-page book with 30 scanned pages and 70 text pages shipped with
    30 pages silently absent whenever no OCR driver was available. The reader got
    a shorter book than the one they asked for and every gate reported success.
    ``missing_pages`` is the caller's own measurement (it already computes which
    pages have no text layer), so this refuses instead of guessing, and the
    message names the pages so the operator can enable OCR or drop them
    deliberately.
    """
    if not _has_text_content(blocks):
        mode_note = f" (ocr_mode='{ocr_mode}')" if ocr_mode else ""
        raise DocumentParseError(
            f"No content could be parsed from '{path.name}': every page lacks an "
            f"extractable text layer{mode_note}. Enable OCR (--ocr rapidocr, "
            "a Docker OCR sidecar, or --ocr cloud) or set UBT_VLM_SCAN_FALLBACK=missing."
        )
    if missing_pages:
        page_list = ", ".join(str(p) for p in sorted(missing_pages)[:12])
        more = "" if len(missing_pages) <= 12 else f" (+{len(missing_pages) - 12} more)"
        of_total = f" of {total_pages}" if total_pages else ""
        mode_note = f" (ocr_mode='{ocr_mode}')" if ocr_mode else ""
        raise DocumentParseError(
            f"Only part of '{path.name}' could be read: {len(missing_pages)}{of_total} "
            f"page(s) have no extractable text layer{mode_note} (pages {page_list}{more}). "
            "Refusing to deliver a book with those pages missing. Enable OCR "
            "(--ocr rapidocr, a Docker OCR sidecar, or --ocr cloud) to transcribe "
            "them, or restrict the run to the pages that do have text with --pages."
        )
    return blocks


# Circuit-breaker knobs for the paid transcription loop live in the policy
# registry (VLM_CIRCUIT_MIN_TRIES / VLM_CIRCUIT_FAIL_PCT): stop after this many
# pages with (nearly) no success. A dead endpoint or revoked key would
# otherwise burn its provider timeout once per page across a whole scanned
# book.

# Block types a VLM page transcription can actually re-produce: narrative
# paragraphs from measured lines. A proofread upgrade must not delete the
# original blocks it cannot re-create (IMAGE, TABLE, FORMULA, CODE);
# dialogue/list items are transcribable text whose type narrows
# to narrative on upgraded pages (content preserved, typing lost is accepted).
_VLM_TEXTUAL_TYPES = frozenset(
    {BlockType.NARRATIVE, BlockType.HEADING, BlockType.DIALOGUE, BlockType.LIST_ITEM}
)


def _driver_can_transcribe_scans(driver: Any) -> bool:
    """Whether ``driver`` can supply geometry for a page with no text layer.

    A vision-LLM driver returns text with no boxes (``measured_boxes=False``);
    :func:`ubt.adapters.pdf.vlm.anchor.anchor_transcript` fails closed on such a
    transcript in recognition mode, so it can *proofread* a page that already
    has a text layer but cannot transcribe a true scan. A measured-box engine
    (rapidocr, sidecar, cloud REST) can.
    """
    return bool(getattr(driver, "measured_boxes", True))


def _ocr_unavailable_hint(mode: str) -> str:
    """Accurate remediation when no OCR driver is available for scanned pages.

    The vision-LLM route (``--ocr vlm``) is deliberately NOT offered: it yields
    no measured boxes, so it cannot supply geometry for a page that has none.
    """
    return (
        "To enable OCR for scanned pages:\n"
        "  1. Local sidecar (recommended): "
        "docker build -t ubt-ocr-sidecar deploy/docker/ocr-sidecar && "
        "docker run -d -p 8765:8765 ubt-ocr-sidecar:latest\n"
        "  2. Or install local rapidocr: pip install rapidocr-onnxruntime\n"
        "  3. Or a measured-box cloud OCR endpoint: "
        "--ocr cloud --ocr-endpoint <url> --ocr-api-key <key>\n"
        "Note: --ocr vlm (vision LLM) returns text without geometry, so it can "
        "proofread a page that already has a text layer but cannot transcribe a "
        "scanned page.\n"
        "Original scanned pages will be preserved."
    )


def _pages_covered_by_blocks(blocks: Sequence[IRBlock]) -> set[int]:
    """Every page number a textual block actually came from.

    Three sources, and all three are needed:

    1. ``provenance.physical_boxes`` -- the full page chain of a block that spans
       a page break. Docling emits ONE ``TextItem`` for a paragraph that runs
       across pages, carrying one ``prov`` entry per page; the IR keeps that
       chain in ``physical_boxes`` (and mirrors it in ``CompositeSpan``).
    2. ``bbox.page`` -- a single-box block.
    3. ``provenance.source_page`` -- a block Docling placed without a bbox.

    Reading only (2) and (3) was the bug: ``bbox.page`` is the *first* page of a
    fused continuation, so a four-page paragraph whose blocks Docling merged
    covered page 1 and nothing else. ``_pages_without_text`` then reported pages
    2-4 as having no text layer and the partial-read guard refused a book it had
    in fact read completely — every crash/resume e2e failed for that reason.

    IMAGE blocks are excluded on purpose: an image-only page has no text layer to
    translate, which is exactly what OCR exists for.
    """
    covered: set[int] = set()
    for block in blocks:
        if block.block_type == BlockType.IMAGE:
            continue
        if not (block.source_text and block.source_text.strip()):
            continue
        for entry in block.provenance.physical_boxes or ():
            page = entry.get("page") if isinstance(entry, dict) else None
            if page:
                covered.add(int(page))
        if block.bbox is not None and block.bbox.page:
            covered.add(block.bbox.page)
        source_page = block.provenance.source_page
        if source_page is not None:
            covered.add(int(source_page))
    return covered


def _pages_without_text(
    path: Path,
    blocks: Sequence[IRBlock],
    page_range: tuple[int, int] | None,
    page_scope: Sequence[int] | None = None,
) -> tuple[list[int], int]:
    """Pages in scope that carry no textual block, and the document's page count.

    "Textual" excludes IMAGE blocks on purpose: an image-only page has no text
    layer to translate, which is exactly the page OCR exists for. Returns
    ``([], 0)`` when the PDF cannot be opened — the caller then has no evidence
    of a gap and must not invent one.

    ``page_scope`` is the *actual* set of pages the run asked for, and it wins
    over ``page_range``. The range is only the enclosing window of one extraction
    chunk, so a non-contiguous selection (``--pages 1,50``) arrives as
    ``(1, 50)``; treating that interval as the requirement reported pages 2-49
    as unreadable and refused a run whose pages were all fine. ``None`` keeps
    the range (or the whole document) as the requirement.
    """
    import pypdfium2 as pdfium

    text_covered = _pages_covered_by_blocks(blocks)
    with PDFIUM_LOCK:
        try:
            probe_doc = pdfium.PdfDocument(str(path))
            try:
                total = len(probe_doc)
            finally:
                probe_doc.close()
        except Exception as exc:
            logger.warning("Page-coverage probe: cannot open '%s': %s", path.name, exc)
            return [], 0
    if page_scope is not None:
        required = sorted({p for p in page_scope if 1 <= p <= total})
    else:
        first = page_range[0] if page_range else 1
        last = min(page_range[1], total) if page_range else total
        required = list(range(first, last + 1))
    return [p for p in required if p not in text_covered], total


def vlm_fallback_missing_pages(
    path: Path,
    blocks: list[IRBlock],
    ocr_mode: str | None = None,
    ocr_endpoint: str | None = None,
    ocr_api_key: str | None = None,
    ocr_model: str | None = None,
    page_range: tuple[int, int] | None = None,
    allow_page_upload: bool | None = None,
    *,
    page_scope: Sequence[int] | None = None,
) -> list[IRBlock]:
    """Transcribe missing pages or weak/mixed pages via vlm/ core.

    ``page_scope`` is the run's actual page selection (``--pages``), passed
    through so a non-contiguous choice is not widened to its enclosing range;
    see :func:`_pages_without_text`.

    Four-tier fallback controlled by ``UBT_VLM_SCAN_FALLBACK``:
    - ``off`` (default): untouched blocks, 0 behavior change.
    - ``missing``: only transcribe 0-block pages (scanned/textless).
    - ``weak``: also upgrade pages with high formula debris (>= 0.15) or sparse text to VLM proofread.
    - ``all``: upgrade all non-editable_text pages to VLM proofread/recognition.
    """
    from ubt.adapters.pdf.vlm.transcribe import (
        VlmFallbackMode,
        get_fallback_mode,
        transcribe_page_to_blocks,
    )

    mode = get_fallback_mode()
    if mode == VlmFallbackMode.OFF:
        clean_ocr_mode = (ocr_mode or "").strip().lower()
        if (clean_ocr_mode and clean_ocr_mode not in ("off", "auto")) or (
            not _has_text_content(blocks) and clean_ocr_mode != "off"
        ):
            mode = VlmFallbackMode.MISSING
        else:
            # No OCR channel will run. Refuse a *partial* read too: a book with
            # some scanned pages used to pass this branch because one text block
            # anywhere satisfied ``_has_text_content``, so those pages shipped
            # silently absent from the delivered book.
            missing, total = _pages_without_text(path, blocks, page_range, page_scope=page_scope)
            return _reject_empty_book(
                path,
                blocks,
                ocr_mode=ocr_mode,
                missing_pages=missing,
                total_pages=total,
            )

    import pypdfium2 as pdfium

    # A page with only image blocks has no extractable text layer; pages with
    # genuine textual blocks are covered, while image-only or empty pages require
    # OCR. Shared with the no-driver branch above so both measure coverage the
    # same way -- including the full ``physical_boxes`` page chain of a block
    # that spans a page break (see ``_pages_covered_by_blocks``).
    text_covered = _pages_covered_by_blocks(blocks)
    with PDFIUM_LOCK:
        try:
            probe_doc = pdfium.PdfDocument(str(path))
            try:
                total = len(probe_doc)
            finally:
                probe_doc.close()
        except Exception as exc:
            logger.warning("VLM fallback: cannot open '%s': %s", path.name, exc)
            return blocks

    if page_scope is not None:
        required = sorted({p for p in page_scope if 1 <= p <= total})
    else:
        first = page_range[0] if page_range else 1
        last = min(page_range[1], total) if page_range else total
        required = list(range(first, last + 1))
    missing = [p for p in required if p not in text_covered]
    proofread_pages: set[int] = set()

    if mode in (VlmFallbackMode.WEAK, VlmFallbackMode.ALL):
        page_blocks: dict[int, list[IRBlock]] = {}
        for b in blocks:
            if b.bbox is not None:
                page_blocks.setdefault(b.bbox.page, []).append(b)

        if mode == VlmFallbackMode.ALL:
            try:
                from ubt.adapters.pdf.engine_selector import PROFILE_CACHE_DIR
                from ubt.adapters.pdf.page_profiler import profile_pdf

                for p in profile_pdf(path, cache_dir=PROFILE_CACHE_DIR):
                    if (
                        p.facts.page in text_covered
                        and p.kind.value != "editable_text"
                        and first <= p.facts.page <= last
                    ):
                        proofread_pages.add(p.facts.page)
            except Exception as exc:
                logger.debug("page profiling for all-mode proofread failed: %s", exc)
        elif mode == VlmFallbackMode.WEAK:
            from ubt.core.policy.layout_policy import (
                PROBE_FORMULA_SHARE,
                PROBE_MIN_CHARS,
                formula_debris_share,
            )

            for p_num, p_blks in page_blocks.items():
                p_text = " ".join(b.source_text for b in p_blks)
                debris = formula_debris_share(p_text)
                if debris >= PROBE_FORMULA_SHARE or len(p_text.strip()) < PROBE_MIN_CHARS:
                    proofread_pages.add(p_num)

    if not missing and not proofread_pages:
        return blocks

    effective_mode = ocr_mode or os.environ.get("UBT_OCR_MODE", "auto")
    effective_endpoint = ocr_endpoint or os.environ.get("UBT_OCR_ENDPOINT")
    # OCR credentials resolve strictly from explicit config or UBT_OCR_API_KEY,
    # never ambient credentials from other programs.
    effective_api_key = ocr_api_key or os.environ.get("UBT_OCR_API_KEY")
    # The resolved config wins over the environment for the same reason it does
    # for the endpoint: the assessor and the spend pre-flight price this exact
    # model, so the channel that bills must use it.
    effective_model = ocr_model or os.environ.get("UBT_OCR_MODEL")

    from ubt.adapters.pdf.vlm.registry import probe_effective_driver

    # The egress gate is owned by ``UBTConfig.allow_page_upload`` and reaches
    # here through ``AdapterRuntimeConfig``. ``None`` means a direct caller with
    # no config handle (tests, library use): fall back to the environment so the
    # default stays aligned with the config field — a "true" here would
    # resurrect the cloud OCR hop the gate exists to forbid.
    if allow_page_upload is None:
        allow_upload = os.environ.get("UBT_ALLOW_PAGE_UPLOAD", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    else:
        allow_upload = allow_page_upload
    driver_type, driver = probe_effective_driver(
        mode=effective_mode,
        endpoint=effective_endpoint,
        api_key=effective_api_key,
        model=effective_model,
        allow_page_upload=allow_upload,
    )

    if driver is None:
        logger.warning(
            "PDF page(s) %s have no extracted text (scanned/image pages). "
            "Pluggable OCR is currently unavailable (mode='%s').\n%s",
            sorted(set(missing) | proofread_pages),
            effective_mode,
            _ocr_unavailable_hint(effective_mode),
        )
        return _reject_empty_book(path, blocks, f"{effective_mode}: no OCR driver available")

    if missing and not _driver_can_transcribe_scans(driver):
        # Selecting `--ocr vlm` for a document with textless pages is a dead
        # end: recognition fails closed without measured boxes, so say it now
        # instead of letting every page fail one at a time.
        logger.warning(
            "OCR driver '%s' (mode='%s') returns text without measured boxes, so "
            "it cannot transcribe the %d page(s) with no text layer; those pages "
            "will be left empty. Use a measured-box engine (rapidocr / sidecar / "
            "--ocr cloud) to transcribe scans.",
            driver_type,
            effective_mode,
            len(missing),
        )

    out: list[IRBlock] = []
    replaced_pages: set[int] = set()
    #: Pages that had no text layer and that this pass did NOT succeed in
    #: transcribing. Declared before the ``try`` so the completion guard below
    #: can see it even when the pass raises mid-way.
    still_missing: set[int] = set(missing)

    try:
        # Handle proofread replacements first. A VLM transcription only
        # re-creates narrative text, so an upgrade page must keep every original
        # block outside _VLM_TEXTUAL_TYPES (tables, images, formulas, code) with
        # their skip flags, and a page with no transcribable text at all skips
        # the paid call entirely.
        attempts = 0
        failures = 0
        sorted_proofread = sorted(proofread_pages)
        for idx, page_no in enumerate(sorted_proofread):
            # Document-level circuit breaker on the PAID loop: once enough pages
            # have been attempted, stop when most attempts failed — a dead
            # endpoint must not be retried once per page for a whole book.
            if (
                attempts >= VLM_CIRCUIT_MIN_TRIES
                and 100 * failures // attempts >= VLM_CIRCUIT_FAIL_PCT
            ):
                logger.error(
                    "VLM proofread circuit breaker: %d of %d attempted upgrade(s) "
                    "failed on '%s'; retaining original blocks for the remaining "
                    "%d page(s) instead of billing for them.",
                    failures,
                    attempts,
                    path.name,
                    len(sorted_proofread) - idx,
                )
                break
            page_originals = [b for b in blocks if b.bbox is not None and b.bbox.page == page_no]
            if not any(b.block_type in _VLM_TEXTUAL_TYPES for b in page_originals):
                logger.warning(
                    "VLM proofread upgrade: page %d of '%s' has no transcribable "
                    "text blocks; skipping the paid transcription",
                    page_no,
                    path.name,
                )
                continue
            # Count the paid attempt before the call, so a wholly dead endpoint
            # still advances `attempts` and the breaker can trip; incrementing
            # only on success would leave attempts at 0 and bill every page anyway.
            attempts += 1
            try:
                fresh, stats = transcribe_page_to_blocks(
                    path, page_no, start_index=len(out) + 1, driver=driver
                )
            except Exception as exc:
                failures += 1
                logger.warning(
                    "VLM proofread upgrade: page %d of '%s' failed (%s); retaining original blocks",
                    page_no,
                    path.name,
                    exc,
                )
                continue
            if fresh:
                logger.info(
                    "VLM tiering: page %d of '%s' proofread-upgraded with %d block(s)",
                    page_no,
                    path.name,
                    len(fresh),
                )
                replaced_pages.add(page_no)
                # The upgrade page is excluded from the not-replaced retention
                # loop below, so the originals the transcription cannot re-create
                # (tables, images, formulas, code) must be re-attached here.
                # Splice the transcribed prose in at the first textual block's
                # slot and keep every non-textual original in place: the final
                # sort is stable per page, so appending the fresh text first
                # shoved a table/figure that sat above the prose to the page tail.
                inserted = False
                for orig in page_originals:
                    if orig.block_type in _VLM_TEXTUAL_TYPES:
                        if not inserted:
                            out.extend(fresh)
                            inserted = True
                    else:
                        out.append(orig)

        # Retain original blocks that weren't replaced
        for b in blocks:
            if b.bbox is None or b.bbox.page not in replaced_pages:
                out.append(b)

        # Handle missing (zero-block) pages
        for idx, page_no in enumerate(missing):
            if (
                attempts >= VLM_CIRCUIT_MIN_TRIES
                and 100 * failures // attempts >= VLM_CIRCUIT_FAIL_PCT
            ):
                logger.error(
                    "VLM fallback circuit breaker: %d of %d attempted page transcription(s) "
                    "failed on '%s'; refusing to bill for the remaining %d page(s). "
                    "Check the OCR endpoint/credentials and re-run.",
                    failures,
                    attempts,
                    path.name,
                    len(missing) - idx,
                )
                break
            attempts += 1
            try:
                fresh, stats = transcribe_page_to_blocks(
                    path, page_no, start_index=len(out) + 1, driver=driver
                )
            except Exception as exc:
                failures += 1
                logger.warning(
                    "VLM fallback: page %d of '%s' failed (%s); left empty",
                    page_no,
                    path.name,
                    exc,
                )
                continue
            if fresh:
                still_missing.discard(page_no)
                logger.info(
                    "VLM fallback: page %d of '%s' gained %d block(s)",
                    page_no,
                    path.name,
                    len(fresh),
                )
                out.extend(fresh)
    finally:
        close_fn = getattr(driver, "close", None)
        if callable(close_fn):
            close_fn()

    # Slot pages into physical page order with a *stable* page-only key. The
    # old key also sorted by ``-y1``, which re-derived every page's order from
    # geometry and interleaved the two columns of every multi-column page in the
    # book — including the pages this pass never touched. Docling's own order is
    # the column-aware reading order, so keeping it inside each page is the
    # point; bbox-less blocks inherit the last page seen in the source order
    # rather than floating to the front of the book.
    page_keys: dict[int, int] = {}
    _carry = 0
    for _b in blocks:
        if _b.bbox is not None:
            _carry = _b.bbox.page
        page_keys[id(_b)] = _carry
    out.sort(key=lambda b: page_keys.get(id(b), b.bbox.page if b.bbox else _carry))
    for i, block in enumerate(out):
        block.set_spine_index(i + 1)
    # The OCR pass is over: refuse to hand back a book with pages it could not
    # read. Without this the missing-page set was computed, used to decide what
    # to bill, and then dropped — so a mixed text+scan book paid for every page
    # the driver could not transcribe and still cleared every downstream gate as
    # a complete delivery. The no-OCR branch above already passes the same
    # evidence; both paths must reach the guard with it.
    #
    # Circuit-breaker bailout and per-page failures both land here: whatever is
    # still in ``still_missing`` is content the reader will not get.
    return _reject_empty_book(
        path, out, ocr_mode=effective_mode, missing_pages=sorted(still_missing)
    )


#: Names this module still exports for its consumers: the cache seam moved to
#: :mod:`ubt.adapters.pdf.docling_cache` and is re-exported here so existing
#: imports (the render stack, the identity tests) keep resolving.
__all__ = [
    "DOCLING_ASSET_SUBDIR",
    "DOCLING_CACHE_DIR",
    "DOCLING_CACHE_VERSION",
    "annotate_page_kinds",
    "docling_asset_dir",
    "docling_cache_key",
    "drop_phantom_continuation_boxes",
    "extract_manifest",
    "read_pdf_outline_entries",
    "_toc_sort_key",
    "_crop_formula_asset",
    "_extract_document_index_blocks",
    "_extract_outline_only_blocks",
    "extract_with_docling",
    "extract_with_oxide",
    "read_docling_document",
    "resolve_formula_enrichment",
    "restore_ordered_markers",
    "annotate_layout_metadata",
    "vlm_fallback_missing_pages",
    "write_docling_document",
]
