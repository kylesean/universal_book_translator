"""Out-of-core page chunking and eviction utilities for long PDF documents."""

from __future__ import annotations

import os
from collections.abc import Sequence

DEFAULT_PDF_PAGE_CHUNK_SIZE: int = 50


def get_pdf_page_chunk_size() -> int:
    """Return page chunk size for out-of-core streaming of large PDFs.

    Controlled via environment variable ``UBT_PDF_STREAM_CHUNK_SIZE``.
    Defaults to 50 pages per chunk.
    """
    raw = os.environ.get("UBT_PDF_STREAM_CHUNK_SIZE", "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_PDF_PAGE_CHUNK_SIZE


def compute_pdf_page_chunks(
    total_pages: int,
    pages: Sequence[int] | set[int] | None = None,
    chunk_size: int | None = None,
) -> list[list[int]]:
    """Partition a PDF's page numbers into sequential chunks for bounded memory streaming.

    If ``pages`` is given, only the requested pages are chunked in ascending order.
    Otherwise, pages ``1..total_pages`` are chunked.
    If ``total_pages <= 0`` and ``pages`` is None, returns ``[[]]`` indicating a single
    unconstrained fallback pass.
    """
    if chunk_size is None or chunk_size <= 0:
        chunk_size = get_pdf_page_chunk_size()

    if pages is not None:
        sorted_pages = sorted(pages)
        if not sorted_pages:
            return []
        return [sorted_pages[i : i + chunk_size] for i in range(0, len(sorted_pages), chunk_size)]

    if total_pages <= 0:
        return [[]]

    return [
        list(range(start_p, min(start_p + chunk_size, total_pages + 1)))
        for start_p in range(1, total_pages + 1, chunk_size)
    ]
