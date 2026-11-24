"""Docling/oxide extraction leg of the PDF adapter.

Everything that turns a PDF into ``IRBlock``s: Docling pipeline construction,
item→block mapping, the textless-page VLM fallback, page-kind profiling, and
the pdf_oxide fast fallback. No delivery state (reconstructor/alternator) leaks
in, so these are plain functions; the adapter injects the two import seams
(``symbols``, ``has_accelerator``) explicitly so its test hooks keep working.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ubt.adapters.pdf.docling_cache import (  # noqa: F401 -- re-exported
    DOCLING_ASSET_SUBDIR,
    DOCLING_CACHE_DIR,
    DOCLING_CACHE_VERSION,
    _cleanup_docling_converter,
    _ensure_docling_pdfium_lock,
    docling_asset_dir,
    docling_cache_key,
    read_docling_document,
    write_docling_document,
)
from ubt.adapters.pdf.docling_mapping import (
    _bbox_area,
    _doc_item_labels,
    _figure_asset_dpi_tag,
    _page_areas,
    _place_toc_rows,
    annotate_page_kinds,
    drop_phantom_continuation_boxes,
    map_docling_document_to_blocks,
    map_export_dict,
    map_iterated_items,
)
from ubt.adapters.pdf.docling_ocr_guard import (
    _blank_pages,
    _driver_can_transcribe_scans,
    _has_text_content,
    _ocr_unavailable_hint,
    _pages_covered_by_blocks,
    _pages_without_text,
    _reject_empty_book,
    vlm_fallback_missing_pages,
)
from ubt.adapters.pdf.docling_outline import (  # noqa: F401 -- re-exported
    _crop_formula_asset,
    _extract_document_index_blocks,
    _extract_outline_only_blocks,
    _toc_sort_key,
    assign_heading_levels_from_outline,
    read_pdf_outline_entries,
)
from ubt.adapters.pdf.docling_structure import (
    _as_list_item,
    _first_line_indent,
    _is_centered,
    _is_prose_footer,
    _iter_pdf_page_texts,
    _line_boxes,
    _line_in_box,
    _ordered_marker,
    _read_table_cells,
    _recover_reference_marker,
    _table_reader_enabled,
    annotate_layout_metadata,
    reanchor_figure_captions,
    restore_appendix_headings,
    restore_ordered_markers,
    type_docling_blocks,
)
from ubt.adapters.pdf.encryption import reject_encrypted_pdf
from ubt.adapters.pdf.page_chunking import compute_pdf_page_chunks, get_pdf_page_chunk_size
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BookManifest,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.core.job_options import clean_source_stem

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

#: Headroom Docling's layout/table models need before its first converter can be
#: built. The models themselves are small; the cost is a CUDA context plus
#: fragmentation, and a co-resident LLM server grows its own allocation at load
#: time. Measured on this project's reference machine: with 0.06 GiB free,
#: Docling OOM'd on a 20 MiB request *after* paying for converter
#: initialisation, then re-ran the whole conversion on CPU.
_DOCLING_MIN_FREE_VRAM_GIB: float = 2.0


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


def resolve_formula_enrichment(formula_enrichment: str) -> bool:
    """Resolve the effective formula-enrichment policy."""
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
    2. If models need downloading, no proxy is configured and the operator set
       ``UBT_ALLOW_HF_MIRROR=1``, set HF_ENDPOINT='https://hf-mirror.com' to
       avoid network dropouts/hangs. Otherwise log the opt-in and leave the
       endpoint alone (uploading to a third-party mirror is an egress decision).
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
        from ubt.core.env import subprocess_env

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
            env=subprocess_env(),
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


def default_docling_symbols() -> tuple[Any, Any, Any, Any, Any]:
    """Import hook for Docling converter symbols."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        CodeFormulaVlmOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    return (
        InputFormat,
        PdfPipelineOptions,
        DocumentConverter,
        PdfFormatOption,
        CodeFormulaVlmOptions,
    )


_WARM_IN_PROCESS_CONVERTER: Any = None
_WARM_IN_PROCESS_KEY: tuple[Any, ...] | None = None


def cleanup_in_process_docling_converter() -> None:
    """Explicitly clean up in-process cached Docling converter and free memory."""
    global _WARM_IN_PROCESS_CONVERTER, _WARM_IN_PROCESS_KEY
    if _WARM_IN_PROCESS_CONVERTER is not None:
        _cleanup_docling_converter(_WARM_IN_PROCESS_CONVERTER)
        _WARM_IN_PROCESS_CONVERTER = None
        _WARM_IN_PROCESS_KEY = None


def convert_docling_document(
    path: Path,
    page_range: tuple[int, int] | None = None,
    *,
    enrich: bool,
    device: str = "auto",
    formula_crop_margin: float = 0.08,
    file_sha256: str | None = None,
    existing_converter: Any = None,
    symbols: Callable[[], tuple[Any, Any, Any, Any, Any]] | None = None,
) -> tuple[Any, Any, str]:
    """Execute Docling document conversion with retry ladder and disk caching.

    Returns:
        (docling_document, converter, cache_key)
    """
    configure_hf_environment(enrich=enrich)
    if symbols is None:
        symbols = default_docling_symbols
    (
        input_format,
        pipeline_options_cls,
        converter_cls,
        format_option_cls,
        code_formula_options_cls,
    ) = symbols()
    _ensure_docling_pdfium_lock()

    options = pipeline_options_cls()
    options.do_ocr = False
    options.do_formula_enrichment = enrich
    if enrich:
        options.code_formula_options = code_formula_options_cls.from_preset(
            "codeformulav2",
            expansion_factor=formula_crop_margin,
            stop_on_repetition=True,
        )
    options.generate_picture_images = True
    options.images_scale = 2.0
    device = (device or "auto").strip().lower()
    device, vram_note = _docling_vram_verdict(device)
    if vram_note:
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

    if file_sha256 is None:
        file_sha256 = compute_file_sha256_cached(path)

    key = docling_cache_key(file_sha256, page_range, options)
    expected_pages = page_range[1] - page_range[0] + 1 if page_range is not None else None
    cached = read_docling_document(key, expected_pages=expected_pages)
    if cached is not None:
        logger.info(
            "Reusing the cached Docling conversion of '%s' instead of re-running it.",
            path.name,
        )
        return cached, existing_converter, key

    def _make_converter(opt: Any) -> Any:
        return cast(
            "Any",
            converter_cls(
                format_options={input_format.PDF: format_option_cls(pipeline_options=opt)}
            ),
        )

    converter = existing_converter or _make_converter(options)
    convert_kwargs: dict[str, Any] = {"page_range": page_range} if page_range is not None else {}

    def _convert() -> Any:
        nonlocal key
        key = docling_cache_key(file_sha256, page_range, options)
        c = read_docling_document(key, expected_pages=expected_pages)
        if c is not None:
            logger.info(
                "Reusing the cached Docling conversion of '%s' instead of re-running it.",
                path.name,
            )
            return c
        document = converter.convert(path, **convert_kwargs).document
        write_docling_document(key, document)
        return document

    try:
        try:
            doc = _convert()
        except Exception as exc:
            offline_before = os.environ.get("HF_HUB_OFFLINE")
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
                if retry_succeeded:
                    pass
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
    except Exception:
        _cleanup_docling_converter(converter)
        raise

    return doc, converter, key


def extract_with_docling(
    path: Path,
    page_range: tuple[int, int] | None,
    *,
    symbols: Callable[[], tuple[Any, Any, Any, Any, Any]] | None = None,
    enrich: bool,
    device: str = "auto",
    formula_crop_margin: float = 0.08,
    isolate_process: bool = False,
) -> list[IRBlock]:
    """Extract structured blocks using IBM Docling in strict reading order."""
    global _WARM_IN_PROCESS_CONVERTER, _WARM_IN_PROCESS_KEY

    file_sha256 = compute_file_sha256_cached(path)

    if isolate_process:
        from ubt.adapters.pdf.docling_worker import get_docling_worker_pool

        pool = get_docling_worker_pool()
        try:
            key = pool.convert(
                path=path,
                page_range=page_range,
                enrich=enrich,
                device=device,
                formula_crop_margin=formula_crop_margin,
                file_sha256=file_sha256,
            )
            cached_doc = read_docling_document(
                key,
                expected_pages=(
                    page_range[1] - page_range[0] + 1 if page_range is not None else None
                ),
            )
            if cached_doc is not None:
                return map_docling_document_to_blocks(cached_doc, path, file_sha256)
            raise DocumentParseError(
                f"Isolated Docling worker produced no usable text for '{path.name}' "
                f"(cache key {key}); refusing to load the Docling model into the host process."
            )
        except DocumentParseError:
            raise
        except TimeoutError as exc:
            # Do NOT fall back to in-process conversion here. The isolated
            # worker exists so a docling segfault or OOM cannot take the server
            # down with it; falling back on a timeout loaded the multi-GB model
            # into the API process and made every later crash fatal to all jobs.
            # A timed-out slice fails this chunk, and ``--job-id`` resumes it.
            raise DocumentParseError(str(exc)) from exc
        except BrokenProcessPool as exc:
            # Same rule as the timeout above, and for the same reason: a
            # ``BrokenProcessPool`` means the worker died natively (segfault or
            # OOM — exactly the failure isolation exists for), so falling back
            # here would load the multi-GB model into the API process and make
            # every later crash fatal to all jobs. The pool closed itself on
            # re-raise; the next conversion starts a fresh worker.
            raise DocumentParseError(str(exc)) from exc
        except Exception as exc:
            logger.warning(
                "Docling isolated worker pool conversion failed (%s); falling back to in-process conversion.",
                exc,
            )

    conv_key = (enrich, device, formula_crop_margin)
    if _WARM_IN_PROCESS_CONVERTER is not None and conv_key != _WARM_IN_PROCESS_KEY:
        cleanup_in_process_docling_converter()

    doc, converter, _ = convert_docling_document(
        path=path,
        page_range=page_range,
        enrich=enrich,
        device=device,
        formula_crop_margin=formula_crop_margin,
        file_sha256=file_sha256,
        existing_converter=_WARM_IN_PROCESS_CONVERTER,
        symbols=symbols,
    )
    _WARM_IN_PROCESS_CONVERTER = converter
    _WARM_IN_PROCESS_KEY = conv_key

    return map_docling_document_to_blocks(doc, path, file_sha256)


#: :mod:`ubt.adapters.pdf.docling_cache` and is re-exported here so existing
#: imports (the render stack, the identity tests) keep resolving.
__all__ = [
    "DOCLING_ASSET_SUBDIR",
    "DOCLING_CACHE_DIR",
    "DOCLING_CACHE_VERSION",
    "_as_list_item",
    "_bbox_area",
    "_blank_pages",
    "_crop_formula_asset",
    "_doc_item_labels",
    "_driver_can_transcribe_scans",
    "_extract_document_index_blocks",
    "_extract_outline_only_blocks",
    "_figure_asset_dpi_tag",
    "_first_line_indent",
    "_has_text_content",
    "_is_centered",
    "_is_prose_footer",
    "_iter_pdf_page_texts",
    "_line_boxes",
    "_line_in_box",
    "_ocr_unavailable_hint",
    "_ordered_marker",
    "_page_areas",
    "_pages_covered_by_blocks",
    "_pages_without_text",
    "_place_toc_rows",
    "_read_table_cells",
    "_recover_reference_marker",
    "_reject_empty_book",
    "_table_reader_enabled",
    "_toc_sort_key",
    "annotate_layout_metadata",
    "annotate_page_kinds",
    "cleanup_in_process_docling_converter",
    "convert_docling_document",
    "default_docling_symbols",
    "docling_asset_dir",
    "docling_cache_key",
    "drop_phantom_continuation_boxes",
    "extract_manifest",
    "extract_with_docling",
    "map_docling_document_to_blocks",
    "map_export_dict",
    "map_iterated_items",
    "read_docling_document",
    "read_pdf_outline_entries",
    "reanchor_figure_captions",
    "resolve_formula_enrichment",
    "restore_appendix_headings",
    "restore_ordered_markers",
    "type_docling_blocks",
    "vlm_fallback_missing_pages",
    "write_docling_document",
]
