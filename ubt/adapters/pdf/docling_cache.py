"""The Docling conversion cache and its asset directory.

Split out of :mod:`ubt.adapters.pdf.docling_parser` (see that module's
docstring): the cache is a self-contained seam -- content-addressed conversion
records plus the extracted figure assets -- that the parser, the render stack
(``docling_asset_dir``) and the identity tests all read.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.pdfium_gate import unify_docling_pdfium_lock
from ubt.cache.dirs import cache_root
from ubt.core.fs_perms import restrict_dir_to_owner, restrict_file_to_owner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Docling conversion cache
# ---------------------------------------------------------------------------

# The enriched Docling pass dominates ingest: measured at 759 s of VLM equation
# transcription out of 769 s total for a 26-page chapter. Without this cache,
# re-ingest pays it in full — ``--fresh``, a second target language over the
# same file, or a crash before the entry is written — even though the layout
# result depends only on (file bytes, pipeline options, Docling version).
# Picture payloads are not serialized with the document (see
# :func:`_strip_picture_payloads`), which is safe here because figures are
# re-rendered from the source PDF and nothing reads ``picture.image``.
#: Where the Docling conversion cache lives. Under the shared cache root, not a
#: CWD-relative ``.ubt/docling_cache``: a relative path forked the cache per
#: working directory (re-converting the same PDF in each), and landing inside the
#: repository made the cached manuscript text world-readable in a shared checkout.
DOCLING_CACHE_DIR = cache_root() / "docling_cache"
DOCLING_CACHE_VERSION = 1

#: Subdirectory of the cache above holding extracted figure assets, per file SHA.
#:
#: Assets live under the persistent cache root so they survive reboots and are
#: reclaimed alongside the cached conversion when clearing the cache directory.
DOCLING_ASSET_SUBDIR = "assets"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def docling_cache_key(
    pdf_sha256: str,
    page_range: tuple[int, int] | None,
    options: Any,
) -> str:
    """Identify a conversion by everything that can change its output.

    ``options`` is Docling's ``PdfPipelineOptions`` — anything with a
    ``model_dump()`` of the settings that shaped the conversion will do.

    The option half of the key is that dump, read at call time: the degradation
    ladder mutates precisely those fields (``do_formula_enrichment``, accelerator
    device) before retrying, so a degraded retry can never be served the entry the
    healthy attempt would have written. Docling's version is included because its
    layout and formula models define the result — an upgrade must re-convert, not
    replay.
    """
    digest = hashlib.sha256()
    digest.update(f"v{DOCLING_CACHE_VERSION}|{pdf_sha256}|{page_range}|".encode())
    digest.update(json.dumps(options.model_dump(mode="json"), sort_keys=True, default=str).encode())
    digest.update(
        f"|docling={_package_version('docling')}|docling-core={_package_version('docling-core')}".encode()
    )
    return digest.hexdigest()[:32]


def _strip_picture_payloads(node: Any) -> Any:
    """Blank out embedded base64 picture data in a Docling dump before caching.

    ``model_dump`` carries every ``PictureItem.image.uri`` data-URI, so the
    "cheap re-ingest" this cache exists for was storing 31 MB of picture bytes
    per chapter (measured on a real entry) and ``read_docling_document`` then
    held the raw bytes, the decompressed string, the dict and the validated model
    at once — peak RSS scaled about 5x with the book. The geometry stays; only
    the payload goes.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "image" and isinstance(value, dict):
                uri = value.get("uri")
                if isinstance(uri, str) and uri.startswith("data:"):
                    value["uri"] = ""
            else:
                _strip_picture_payloads(value)
    elif isinstance(node, list):
        for item in node:
            _strip_picture_payloads(item)
    return node


def read_docling_document(key: str, *, cache_dir: Path | None = None) -> Any | None:
    """Return the cached DoclingDocument for ``key``, or None on any miss."""
    target = (cache_dir or DOCLING_CACHE_DIR) / f"{key}.json.gz"
    try:
        # Stream the decompression into the parser: ``read_bytes`` +
        # ``gzip.decompress`` held the compressed file and its full
        # decompressed form in memory before ``json.loads`` even built the dict.
        with gzip.open(target, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        from docling_core.types.doc.document import DoclingDocument

        return DoclingDocument.model_validate(payload)
    except (OSError, ValueError, TypeError) as exc:
        if target.exists():
            logger.warning("Docling cache entry %s is unreadable (%s); re-converting", target, exc)
        return None


def write_docling_document(key: str, document: Any, *, cache_dir: Path | None = None) -> None:
    """Persist a conversion so the next ingest of the same file skips it.

    A write failure never fails the run, and a document with no pages at all is
    not stored: that shape means the conversion itself went wrong, and caching it
    would make the mistake permanent. A scanned book with zero extracted *text*
    still has pages, so it does get cached.
    """
    if not getattr(document, "pages", None):
        return
    cache = cache_dir or DOCLING_CACHE_DIR
    target = cache / f"{key}.json.gz"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        # The cache holds the extracted full manuscript text: restrict to owner.
        restrict_dir_to_owner(cache)
        # Stream the dump straight into the gzip file. Building the JSON as a
        # string (``json.dumps``), then its UTF-8 bytes, then a compressed blob
        # held three full copies of the manuscript in memory at once, on top of
        # the model dict -- the peak this cache exists to keep down.
        staging = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        with gzip.open(staging, "wt", encoding="utf-8") as fh:
            json.dump(
                _strip_picture_payloads(document.model_dump(mode="json")),
                fh,
                ensure_ascii=False,
            )
        staging.replace(target)
        restrict_file_to_owner(target)
        logger.info("Cached Docling conversion for reuse: %s", target)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Docling cache write failed for %s: %s", target, exc)


def docling_asset_dir(pdf_sha256: str) -> Path:
    """Where the figure assets extracted from ``pdf_sha256`` live.

    Read from ``DOCLING_CACHE_DIR`` at call time (not captured at import), so a
    caller that redirects the cache — tests, or a future ``--cache-dir`` knob —
    redirects the assets that belong to it as well.
    """
    return DOCLING_CACHE_DIR / DOCLING_ASSET_SUBDIR / pdf_sha256


def _cleanup_docling_converter(converter: Any) -> None:
    """Deterministically release VLM engines held inside a Docling DocumentConverter.

    Docling's ``CodeFormulaVlmModel.__del__`` calls ``self.engine.cleanup()``
    and logs a warning on failure. When left to Python interpreter shutdown
    (``Py_FinalizeEx``), ``sys.meta_path`` is already ``None`` and Rich's log
    renderer raises ``ImportError: sys.meta_path is None, Python is likely
    shutting down``. Cleaning up and nulling ``stage.engine`` immediately after
    conversion frees GPU VRAM early and turns ``__del__`` into a safe no-op.
    """
    if converter is None:
        return
    try:
        pipelines = getattr(converter, "initialized_pipelines", None)
        if isinstance(pipelines, dict):
            for pipe in list(pipelines.values()):
                for attr in ("enrichment_pipe", "build_pipe", "models"):
                    stages = getattr(pipe, attr, None)
                    if not isinstance(stages, (list, tuple)):
                        continue
                    for stage in stages:
                        engine = getattr(stage, "engine", None)
                        if engine is not None:
                            cleanup = getattr(engine, "cleanup", None)
                            if callable(cleanup):
                                try:
                                    cleanup()
                                except Exception as exc:
                                    logger.debug("Docling VLM engine cleanup ignored: %s", exc)
                            with contextlib.suppress(Exception):
                                stage.engine = None
            pipelines.clear()
    except Exception as exc:
        logger.debug("Docling converter cleanup skipped: %s", exc)


_DOCLING_LOCK_CHECKED = False


def _ensure_docling_pdfium_lock() -> None:
    """Unify docling's pypdfium2 lock with ``PDFIUM_LOCK`` once, after import.

    Docling's PDF backends render through the same ``libpdfium.so`` pypdfium2
    loads but hold their own lock (see ``pdfium_gate``). The first real
    conversion — after ``symbols()`` has imported docling — is the earliest
    point the lock bindings exist, so the rebind happens here and is cached only
    on success: while docling is not yet imported nothing is latched, so a later
    import still gets unified.
    """
    global _DOCLING_LOCK_CHECKED
    if _DOCLING_LOCK_CHECKED or "docling" not in sys.modules:
        return
    if unify_docling_pdfium_lock():
        _DOCLING_LOCK_CHECKED = True
        return
    logger.warning(
        "docling is imported but none of its pypdfium2_lock bindings were "
        "found; a concurrent docling render could race UBT pdfium work. "
        "Update ubt.adapters.pdf.pdfium_gate._DOCLING_LOCK_MODULES."
    )


__all__ = [
    "DOCLING_ASSET_SUBDIR",
    "DOCLING_CACHE_DIR",
    "DOCLING_CACHE_VERSION",
    "_cleanup_docling_converter",
    "_ensure_docling_pdfium_lock",
    "_package_version",
    "_strip_picture_payloads",
    "docling_asset_dir",
    "docling_cache_key",
    "read_docling_document",
    "write_docling_document",
]
