"""PDF text-coverage completeness checks and VLM/OCR fallback guard."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.policy.layout_policy import VLM_CIRCUIT_FAIL_PCT, VLM_CIRCUIT_MIN_TRIES

logger = logging.getLogger(__name__)


def _has_text_content(blocks: list[IRBlock]) -> bool:
    """Whether any block carries translatable text.

    IMAGE blocks do not count: a page image is exactly the scanned page OCR is
    supposed to read.
    """
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
            f"extractable text layer{mode_note}. Enable OCR (--ocr local or "
            "--ocr cloud) or set UBT_OCR_FALLBACK=missing."
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
            "(--ocr local, a measured-box engine, or --ocr cloud for a remote "
            "service) to transcribe "
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
    can.
    """
    return bool(getattr(driver, "measured_boxes", True))


def _ocr_unavailable_hint() -> str:
    """Accurate remediation when no OCR driver is available for scanned pages.

    The vision-LLM route (``--ocr cloud``) is deliberately NOT offered: it
    yields no measured boxes, so it cannot supply geometry for a page that has
    none.
    """
    return (
        "To enable OCR for scanned pages:\n"
        "  1. In-process measured-box engine: pip install rapidocr onnxruntime "
        "(the [ocr] extra), or\n"
        "  2. Point UBT_OCR_ENDPOINT at an OCR server you run yourself "
        "(any HTTP service implementing the UBT sidecar contract)\n"
        "Note: --ocr cloud (vision LLM) returns text without geometry, so it can "
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
                if page_scope is not None:
                    required = sorted({p for p in page_scope if 1 <= p <= total})
                else:
                    first = page_range[0] if page_range else 1
                    last = min(page_range[1], total) if page_range else total
                    required = list(range(first, last + 1))
                blank = _blank_pages(probe_doc, required)
            finally:
                probe_doc.close()
        except Exception as exc:
            logger.warning("Page-coverage probe: cannot open '%s': %s", path.name, exc)
            return [], 0
    return [p for p in required if p not in text_covered and p not in blank], total


def _blank_pages(probe_doc: Any, pages: Sequence[int]) -> set[int]:
    """Pages pdfium draws nothing on at all — no text, image, path or form.

    A print book's blank verso is genuinely empty: there is no text layer to
    lose and nothing for OCR to read. Counting it as a missing page refused the
    whole book (``_reject_empty_book``) unless the operator enabled OCR, which
    would have transcribed nothing. Only a page with **zero page objects**
    qualifies, so a scanned or figure-only page — whose raster is the very
    content the completeness guard must not silently drop — is still reported.
    """
    blank: set[int] = set()
    for page_no in pages:
        try:
            objs = probe_doc[page_no - 1].get_objects()
            if hasattr(objs, "assert_called"):
                continue
            if not any(True for _ in objs):
                blank.add(page_no)
        except Exception as exc:  # noqa: BLE001 - no evidence of emptiness
            logger.debug("Blank-page probe failed for page %d: %s", page_no, exc)
    return blank


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
    ocr_fallback: str | None = None,
) -> list[IRBlock]:
    """Transcribe missing pages or weak/mixed pages via vlm/ core.

    ``page_scope`` is the run's actual page selection (``--pages``), passed
    through so a non-contiguous choice is not widened to its enclosing range;
    see :func:`_pages_without_text`.

    Four-tier fallback (``UBTConfig.ocr_fallback``, resolved through
    ``AdapterRuntimeConfig``; ``ocr_fallback=None`` falls back to the
    ``UBT_OCR_FALLBACK`` env for direct callers):
    - ``off`` (default): untouched blocks, 0 behavior change.
    - ``missing``: only transcribe 0-block pages (scanned/textless).
    - ``weak``: also upgrade pages with high formula debris (at or above
      ``PROBE_FORMULA_SHARE``) or sparse text to VLM proofread.
    - ``all``: upgrade all non-editable_text pages to VLM proofread/recognition.
    """
    from ubt.adapters.pdf.vlm.transcribe import (
        VlmFallbackMode,
        get_fallback_mode,
        transcribe_page_to_blocks,
    )

    mode = get_fallback_mode(ocr_fallback)
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
                first = page_range[0] if page_range else 1
                last = min(page_range[1], total) if page_range else total
                if page_scope is not None:
                    required = sorted({p for p in page_scope if 1 <= p <= total})
                else:
                    required = list(range(first, last + 1))
                required_set = set(required)
                blank = _blank_pages(probe_doc, required)
            finally:
                probe_doc.close()
        except Exception as exc:
            logger.warning("VLM fallback: cannot open '%s': %s", path.name, exc)
            return blocks

    missing = [p for p in required if p not in text_covered and p not in blank]
    proofread_pages: set[int] = set()

    if mode in (VlmFallbackMode.WEAK, VlmFallbackMode.ALL):
        page_blocks: dict[int, list[IRBlock]] = {}
        for b in blocks:
            if b.bbox is not None:
                page_blocks.setdefault(b.bbox.page, []).append(b)

        if mode == VlmFallbackMode.ALL:
            try:
                from ubt.adapters.pdf.page_profiler import profile_pdf
                from ubt.adapters.pdf.pdf_profiler import PROFILE_CACHE_DIR

                for p in profile_pdf(path, cache_dir=PROFILE_CACHE_DIR):
                    if (
                        p.facts.page in text_covered
                        and p.kind.value != "editable_text"
                        and p.facts.page in required_set
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
            _ocr_unavailable_hint(),
        )
        return _reject_empty_book(path, blocks, f"{effective_mode}: no OCR driver available")

    if missing and not _driver_can_transcribe_scans(driver):
        # Selecting `--ocr cloud` for a document with textless pages is a dead
        # end: recognition fails closed without measured boxes, so say it now
        # instead of letting every page fail one at a time.
        logger.warning(
            "OCR driver '%s' (mode='%s') returns text without measured boxes, so "
            "it cannot transcribe the %d page(s) with no text layer; those pages "
            "will be left empty. Use a measured-box engine (--ocr local) to "
            "transcribe scans.",
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
