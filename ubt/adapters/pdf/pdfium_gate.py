"""PdfiumGateway: process-wide serialization + pinned font policy for libpdfium.

pdfium mutates process-global state without internal locking: the
CFX_FolderFontInfo font-substitution table is built lazily on the first
font match (``FPDF_LoadPage`` -> ``LoadSubstFont`` -> ``EnumFontList``)
and page-content parsing walks shared object trees. Two threads entering
libpdfium concurrently (UBT dispatches pdfium work through many
``asyncio.to_thread`` call sites on the shared default executor) corrupt
the native heap — observed as SIGSEGV in ``__tree_balance_after_insert``
and as glibc ``double free or corruption (!prev)`` aborts, both during
the export stage of a multi-page paper (see
``docs/PDFIUM_THREAD_SAFETY_2026-09-20.md``).

Two defenses, matching the production pdfium paradigm:

1. **Single serial entry.** Every code path that calls into pypdfium2
   (directly or transitively, e.g. through docling-parse) must hold
   ``PDFIUM_LOCK`` for the duration of the native work. Use
   ``@pdfium_serialized`` for whole-function pdfium work, or
   ``with PDFIUM_LOCK:`` when only part of the function is pdfium work
   (never hold the lock across LLM/network/subprocess calls).
   ``threading.RLock`` makes same-thread re-entry safe for nested pdfium
   calls (e.g. ``rigid.extract_pages`` -> ``textgeom.extract_lines``).
   The lazy font-table initialization happens under this lock, so it is
   warm-once and race-free without any separate warmup step.

2. **Pinned substitution font set.** On Linux pdfium's default font
   provider walks whole system font directories (800+ faces on desktop
   machines), which is slow, machine-dependent, and widens the race
   window on first touch. This module installs a ctypes hook before
   pypdfium2 is first imported so that ``FPDF_InitLibraryWithConfig``
   receives ``m_pUserFontPaths`` pointing at a small pinned set
   (Liberation/Nimbus/DejaVu + Noto CJK on Linux, or whatever
   ``UBT_PDFIUM_FONT_DIRS`` lists). Deterministic faces, a fraction of
   the scan, and no host-font leakage. If pypdfium2 was already imported
   without this gate we log a warning and rely on defense 1 alone.

``open_document`` is the canonical entry point for new code; existing
call sites are individually locked and may migrate to it incrementally.

Longer term (crash isolation): move rasterization behind a subprocess
boundary like ``vlm/drivers/sidecar_driver.py`` — a pdfium segfault then
kills only the child. Tracked in the design doc.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PDFIUM_LOCK = threading.RLock()

#: ``os.pathsep``-separated override for the pinned substitution dirs.
FONT_DIRS_ENV = "UBT_PDFIUM_FONT_DIRS"

#: Production substitution faces per the standard-14 mapping:
#: Liberation stands in for Arial/Helvetica, Nimbus (gsfonts) for Times,
#: DejaVu for Courier, Noto CJK for CJK. Only dirs that exist are used.
_DEFAULT_FONT_DIRS: tuple[str, ...] = (
    "/usr/share/fonts/liberation",
    "/usr/share/fonts/gsfonts",
    "/usr/share/fonts/dejavu",
    "/usr/share/fonts/noto-cjk",
)


def _resolve_font_dirs() -> list[str]:
    raw = os.environ.get(FONT_DIRS_ENV, "")
    candidates = raw.split(os.pathsep) if raw else list(_DEFAULT_FONT_DIRS)
    found = [d for d in candidates if d and Path(d).is_dir()]
    if not found:
        # Keep pdfium's default behaviour rather than starving it of faces.
        logger.debug(
            "pdfium_gate: no pinned font dirs found (env %s, defaults %s); "
            "falling back to system font scan",
            FONT_DIRS_ENV,
            _DEFAULT_FONT_DIRS,
        )
    return found


def install_font_policy() -> bool:
    """Wrap ``FPDF_InitLibraryWithConfig`` so pdfium sees pinned font dirs.

    Must run before the ``pypdfium2`` high-level package is imported
    (it initializes the library at import time). Returns True when the
    policy is (or was already) installed. Safe to call repeatedly.
    """
    global _POLICY_INSTALLED
    if _POLICY_INSTALLED:
        return True
    if "pypdfium2" in sys.modules:
        logger.warning(
            "pdfium_gate: pypdfium2 was imported before the font policy could "
            "install; falling back to host font scan (serialization still active)"
        )
        return False
    try:
        import pypdfium2_raw.bindings as _b
    except ImportError:
        return False

    dirs = _resolve_font_dirs()
    if not dirs:
        return False

    # pdfium's FPDF_LIBRARY_CONFIG.m_pUserFontPaths is char* const*
    # (POINTER(POINTER(c_char))), NULL-terminated. Keep the array alive for
    # the process lifetime: pdfium stores the pointer past init. The closure
    # below retains it.
    _CharPtr = ctypes.POINTER(ctypes.c_char)
    # The c_char_p wrappers own the bytes buffers; keep them alive next to
    # the array (pdfium stores the pointers past init). The closure retains
    # paths_array, which anchors the buffer list.
    _bufs = [ctypes.c_char_p(os.fsencode(d)) for d in dirs]
    paths_array = (_CharPtr * (len(dirs) + 1))(
        *(ctypes.cast(buf, _CharPtr) for buf in _bufs),
        _CharPtr(),
    )
    paths_array._ubt_buf_keepalive = _bufs  # type: ignore[attr-defined]

    _orig_init = _b.FPDF_InitLibraryWithConfig

    def _init_with_fonts(config: Any) -> Any:
        # pypdfium2's _library_scope passes a bare struct instance; the
        # _FuncPtr converts it per argtypes inside __call__, so mutating
        # the same object we received is visible to the real init.
        try:
            cfg = config.contents if hasattr(config, "contents") else config
            # A fresh ctypes POINTER field reads as a NULL LP_ object, not
            # None — test truthiness (NULL pointers are falsy).
            if cfg is not None and not bool(cfg.m_pUserFontPaths):
                cfg.m_pUserFontPaths = ctypes.cast(
                    paths_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
                )
                _record("applied", len(dirs))
        except Exception as exc:  # pragma: no cover - defensive
            _record("error", repr(exc))
            logger.debug("pdfium_gate: font policy skipped", exc_info=True)
        return _orig_init(config)

    _b.FPDF_InitLibraryWithConfig = _init_with_fonts
    _POLICY_INSTALLED = True
    logger.debug("pdfium_gate: pinned font dirs for pdfium: %s", dirs)
    return True


_POLICY_INSTALLED = False
#: Observability for tests/logs: ("applied", n_dirs) once the pinned paths
#: reach a real FPDF_InitLibraryWithConfig call.
_POLICY_STATE: tuple[str, Any] = ("pending", None)


def _record(state: str, detail: Any) -> None:
    global _POLICY_STATE
    _POLICY_STATE = (state, detail)


# Installing at import time is the contract: every gated module imports this
# gate before it imports pypdfium2.
install_font_policy()


def pdfium_serialized[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Run ``fn`` while holding the process-wide pdfium lock."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kw: P.kwargs) -> R:
        with PDFIUM_LOCK:
            return fn(*args, **kw)

    return wrapper


@contextlib.contextmanager
def open_document(pdf_path: Path | str) -> Iterator[Any]:
    """Canonical entry for new pdfium code: locked open/use/close of a document.

    ``with open_document(p) as doc: ...`` — the lock is held for the whole
    body, so keep LLM/network/subprocess calls out of it.
    """
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            yield doc
        finally:
            doc.close()
