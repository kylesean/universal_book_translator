"""Isolated worker process pool for Docling PDF conversions.

Runs Docling conversions in a separate spawned process to:
1. Isolate the main orchestrator process from native PyTorch/CUDA C++ crashes
   (SIGSEGV, SIGBUS, OOM).
2. Keep a warm ``DocumentConverter`` loaded across multiple 50-page chunks of a
   book, eliminating 5-15s cold-start weight loads per chunk.
3. Fully return 100% of GPU VRAM and system RAM to the OS kernel upon
   ``close()`` when ingestion finishes.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
import os
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.docling_cache import (
    _cleanup_docling_converter,
)

logger = logging.getLogger(__name__)

# Process-local warm converter inside the worker process
_WORKER_CONVERTER: Any = None
_WORKER_CONVERTER_KEY: tuple[Any, ...] | None = None


def _worker_convert_task(
    path_str: str,
    page_range: tuple[int, int] | None,
    enrich: bool,
    device: str,
    formula_crop_margin: float,
    file_sha256: str,
) -> str:
    """Execute PDF conversion inside the isolated worker process."""
    global _WORKER_CONVERTER, _WORKER_CONVERTER_KEY

    path = Path(path_str)
    current_key = (enrich, device, formula_crop_margin)

    # Invalidate converter if runtime options changed
    if _WORKER_CONVERTER is not None and current_key != _WORKER_CONVERTER_KEY:
        _cleanup_docling_converter(_WORKER_CONVERTER)
        _WORKER_CONVERTER = None
        _WORKER_CONVERTER_KEY = None

    from ubt.adapters.pdf.docling_parser import convert_docling_document

    doc, converter, key = convert_docling_document(
        path=path,
        page_range=page_range,
        enrich=enrich,
        device=device,
        formula_crop_margin=formula_crop_margin,
        file_sha256=file_sha256,
        existing_converter=_WORKER_CONVERTER,
    )
    _WORKER_CONVERTER = converter
    _WORKER_CONVERTER_KEY = current_key
    return key


class DoclingWorkerPool:
    """Manages an isolated worker process for Docling conversions."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._closed = False

    def _ensure_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._closed:
            raise RuntimeError("DoclingWorkerPool is closed")
        if self._executor is None:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=self._ctx,
            )
        return self._executor

    def convert(
        self,
        path: Path,
        page_range: tuple[int, int] | None,
        enrich: bool,
        device: str,
        formula_crop_margin: float,
        file_sha256: str,
        timeout: float = 600.0,
    ) -> str:
        """Submit conversion to the isolated worker process, returning the cache key."""
        executor = self._ensure_executor()
        future = executor.submit(
            _worker_convert_task,
            str(path),
            page_range,
            enrich,
            device,
            formula_crop_margin,
            file_sha256,
        )
        try:
            return future.result(timeout=timeout)
        except BrokenProcessPool as exc:
            logger.error("Docling worker process died (native segfault or OOM): %s", exc)
            self.close()
            raise
        except Exception as exc:
            logger.warning("Docling worker conversion failed: %s", exc)
            raise

    def close(self) -> None:
        """Terminate the worker process and release all memory."""
        self._closed = True
        executor = self._executor
        self._executor = None
        if executor is not None:
            try:
                processes = getattr(executor, "_processes", None)
                if processes:
                    for p in list(processes.values()):
                        try:
                            if p.is_alive():
                                p.terminate()
                        except Exception:
                            pass
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception as exc:
                logger.debug("Error shutting down DoclingWorkerPool: %s", exc)


_POOL_INSTANCE: DoclingWorkerPool | None = None


def get_docling_worker_pool() -> DoclingWorkerPool:
    """Return the active DoclingWorkerPool singleton, creating one if needed."""
    global _POOL_INSTANCE
    if _POOL_INSTANCE is None or _POOL_INSTANCE._closed:
        _POOL_INSTANCE = DoclingWorkerPool()
    return _POOL_INSTANCE


def close_docling_worker_pool() -> None:
    """Close and terminate the DoclingWorkerPool singleton."""
    global _POOL_INSTANCE
    if _POOL_INSTANCE is not None:
        _POOL_INSTANCE.close()
        _POOL_INSTANCE = None


def is_docling_process_isolation_enabled() -> bool:
    """Whether Docling conversions should run in an isolated worker process."""
    val = os.environ.get("UBT_DOCLING_ISOLATE_PROCESS", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


__all__ = [
    "DoclingWorkerPool",
    "close_docling_worker_pool",
    "get_docling_worker_pool",
    "is_docling_process_isolation_enabled",
]
