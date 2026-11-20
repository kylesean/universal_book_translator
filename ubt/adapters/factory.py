"""Factory helpers to instantiate document adapters based on file extensions."""

import importlib.util
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from ubt.adapters.base import BaseDocumentAdapter, BasePDFEngineAdapter
from ubt.adapters.codec_adapter import DocumentCodecAdapter
from ubt.adapters.docx.adapter import DOCXAdapter
from ubt.adapters.epub.adapter import EPUBAdapter
from ubt.adapters.html.adapter import HTMLAdapter
from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.adapters.pdf.docling_adapter import ScanPDFToTypstCodec
from ubt.adapters.pdf.encryption import reject_encrypted_pdf
from ubt.adapters.pdf.native_adapter import NativePDFAdapter
from ubt.adapters.pdf.pdf_profiler import inspect_pdf_profile
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.codecs.registry import CODEC_SUFFIXES
from ubt.core.exceptions import (
    DocumentParseError,
    OptionalDependencyError,
    UnsupportedDocumentFormatError,
)
from ubt.core.fs_perms import VerifiedOpenError, open_verified

logger = logging.getLogger(__name__)

# Extensible registry mapping file extensions to adapter factory callables.
# Factories receive ``(pdf_engine, path)``; most adapters ignore both.
_ADAPTER_REGISTRY: dict[str, Callable[[str, Path], BaseDocumentAdapter]] = {}

# Extensible registry mapping PDF engine keys to PDF adapter factory callables
_PDF_ENGINE_REGISTRY: dict[str, Callable[[], BasePDFEngineAdapter]] = {}


def register_adapter(
    extensions: list[str] | tuple[str, ...],
) -> Callable[
    [Callable[[str, Path], BaseDocumentAdapter]], Callable[[str, Path], BaseDocumentAdapter]
]:
    """Decorator to register a custom adapter factory for given file extensions.

    In-process extension: call it from module code running in the UBT process
    before the factory resolves a path.
    """

    def decorator(
        fn: Callable[[str, Path], BaseDocumentAdapter],
    ) -> Callable[[str, Path], BaseDocumentAdapter]:
        for ext in extensions:
            _ADAPTER_REGISTRY[ext.lower()] = fn
        return fn

    return decorator


def register_pdf_engine(
    names: list[str] | tuple[str, ...],
) -> Callable[[Callable[[], BasePDFEngineAdapter]], Callable[[], BasePDFEngineAdapter]]:
    """Decorator to register a custom PDF engine adapter factory.

    Same in-process contract as :func:`register_adapter`.
    """

    def decorator(fn: Callable[[], BasePDFEngineAdapter]) -> Callable[[], BasePDFEngineAdapter]:
        for name in names:
            _PDF_ENGINE_REGISTRY[name.lower()] = fn
        return fn

    return decorator


def is_pdf_engine_registered(name: str) -> bool:
    """Whether ``name`` resolves to a registered PDF engine.

    A membership check against the registry, which is open to in-process
    :func:`register_pdf_engine` calls and so cannot be a static ``Literal``.
    """
    return name.lower() in _PDF_ENGINE_REGISTRY


def _resolve_pdf_adapter(pdf_engine: str, path: Path | None = None) -> BasePDFEngineAdapter:
    # The routing probe opens the PDF with pdf_oxide; on an encrypted file that
    # emits a burst of "Decryption failed ... Not authenticated" before the
    # manifest guard can run. Refuse here, before anything touches the file.
    if path is not None:
        reject_encrypted_pdf(path)

    engine_key = pdf_engine.lower()
    if engine_key == "babeldoc":
        raise UnsupportedDocumentFormatError(
            "BabelDOC engine is deprecated and removed due to licensing constraints "
            "and upstream CLI contract incompatibility. Please use pdf_engine='native' or 'docling'."
        )
    if engine_key in ("auto", "pdfium"):
        if path is None:
            engine_key = "native" if engine_key == "auto" else "pdfium"
        else:
            plan = inspect_pdf_profile(path)
            if plan.has_scan:
                logger.warning(
                    "'%s' looks scanned (pages without a text layer); routing it to docling for OCR-grade ingest.",
                    path.name,
                )
                engine_key = "docling"
            elif engine_key == "auto":
                engine_key = "native"
            else:
                logger.warning(
                    "pdfium requested for '%s': its reader cannot represent tables, "
                    "figures or formulas, so any the document contains are omitted "
                    "from the rebuilt PDF (probe signals — multicolumn=%s, "
                    "formulas=%s, vector-heavy=%s — are only a lower bound). "
                    "Use --pdf-engine native (default) for 1:1 in-place surgical delivery.",
                    path.name,
                    plan.has_multicolumn,
                    plan.has_formulas,
                    plan.has_vector_diagrams,
                )
    factory_fn = _PDF_ENGINE_REGISTRY.get(engine_key)
    if factory_fn is None:
        # Read the registry rather than a hardcoded list: this is the one place
        # that can tell the truth about what is selectable, custom entries included.
        available = ", ".join(["auto", *sorted(_PDF_ENGINE_REGISTRY)])
        raise UnsupportedDocumentFormatError(
            f"Unsupported or unregistered PDF engine: '{pdf_engine}'. Available: {available}."
        )
    if engine_key == "docling" and importlib.util.find_spec("docling") is None:
        # docling IS the engine every PDF route resolves to (``auto`` included,
        # and a scanned page escapes to it even under an explicit ``pdfium``),
        # so a missing package cannot run the product's primary scenario.
        # Falling back to the geometric pdfium reader would deliver a book with
        # every table, figure and formula dropped, under a job id that later
        # runs resume into — a degraded run must be refused before ingest, not
        # warned about in server logs the web console never shows. (docling now
        # ships in the base dependency set; reaching this branch means the
        # environment is broken, and the message says how to repair it.)
        raise OptionalDependencyError(
            "PDF ingest needs the 'docling' package, which is not installed "
            f"(pdf_engine={pdf_engine.lower()!r} resolves to the docling semantic "
            "engine for every PDF). Repair the environment with `uv sync --all-extras` "
            "(or `pip install 'universal-book-translator[pdf]'`), or pass "
            "--pdf-engine pdfium to accept the geometric fast path, which cannot "
            "represent tables, figures or formulas."
        )
    return factory_fn()


# Standard built-in engine & adapter bindings. Only the canonical names live
# here: ``docling`` (the Typst-reflow mainline) and ``pdfium`` (the born-digital
# fast path), plus the ``auto`` probe handled above. Each engine has exactly one
# name; pass ``docling`` (or ``auto``) rather than an alias.
_PDF_ENGINE_REGISTRY.update(
    {
        "docling": ScanPDFToTypstCodec,
        "native": NativePDFAdapter,
        "pdfium": PDFiumAdapter,
    }
)

_ADAPTER_REGISTRY.update(
    {
        ".epub": lambda _engine, _path: EPUBAdapter(),
        ".md": lambda _engine, _path: MarkdownAdapter(),
        ".markdown": lambda _engine, _path: MarkdownAdapter(),
        ".txt": lambda _engine, _path: MarkdownAdapter(plain_text=True),
        ".html": lambda _engine, _path: HTMLAdapter(),
        ".htm": lambda _engine, _path: HTMLAdapter(),
        ".docx": lambda _engine, _path: DOCXAdapter(),
        ".pdf": lambda engine, path: _resolve_pdf_adapter(engine, path),
    }
)


def supported_suffixes() -> list[str]:
    """Input suffixes the adapter registry can open, custom entries included.

    Surfaces print this instead of restating the list: ``ubt version`` claimed
    three formats while the registry had eight.
    """
    return sorted(set(_ADAPTER_REGISTRY) | CODEC_SUFFIXES)


def get_adapter_for_path(
    path: Path | str,
    pdf_engine: str = "auto",
    *,
    sandbox_roots: Sequence[Path] | None = None,
    prefer_codec: bool = True,
    inspection: bool = False,
) -> BaseDocumentAdapter:
    """Factory helper to resolve the appropriate document adapter by file extension.

    The default matches the config's ``pdf_engine`` default so a direct call
    resolves the same engine the pipeline would: ``auto`` is the native
    in-place codec for a text-layer-clean PDF and Docling for a scan; ``pdfium``
    is an explicit operator opt-in that carries a loud multicolumn/scanned
    warning.

    Before an adapter is handed out for an existing file, the file itself is
    re-opened through :func:`ubt.core.fs_perms.open_verified` — the point where
    the pipeline is about to open the submitted input document. The API and MCP
    surfaces validate (and fully resolve) a submitted path at *submit* time and
    the stored path is only opened here, so a local writer replacing the
    validated file with a symlink in between would otherwise be followed at
    open. The gate refuses that swap fail-closed with a
    :class:`~ubt.core.exceptions.DocumentParseError`. ``sandbox_roots``, when a
    caller knows the sandbox that validated the path, additionally pins the
    final resolution inside it (see ``open_verified``); without it the
    anti-symlink verification still applies.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if prefer_codec and not inspection and suffix != ".pdf" and suffix in _ADAPTER_REGISTRY:
        _verify_input_open(file_path, sandbox_roots)
        return DocumentCodecAdapter(file_path)

    factory = _ADAPTER_REGISTRY.get(suffix)
    if factory is not None:
        _verify_input_open(file_path, sandbox_roots)
        return factory(pdf_engine, file_path)

    raise UnsupportedDocumentFormatError(
        f"No adapter registered for document extension '{suffix}' ({file_path.name})"
    )


def get_inspection_adapter(
    path: Path | str,
    pdf_engine: str = "auto",
    *,
    sandbox_roots: Sequence[Path] | None = None,
) -> BaseDocumentAdapter:
    """Resolve the typed-reader adapter used by read-only inspection surfaces.

    Translation always goes through :class:`DocumentCodecAdapter`; the CLI
    audits (table/math/figure/styler/backmatter) need the reader's semantic
    blocks (``Table``, ``Formula``, ``Figure``) that the translation codec does
    not project. Keeping the inspection adapter explicit stops the legacy
    render path from being reachable by accident.
    """
    return get_adapter_for_path(
        path,
        pdf_engine=pdf_engine,
        sandbox_roots=sandbox_roots,
        inspection=True,
    )


def _verify_input_open(file_path: Path, sandbox_roots: Sequence[Path] | None) -> None:
    """Re-open the input through the anti-swap gate before an adapter is built.

    Missing files are left alone: every adapter already answers them with its
    own "not found" error, and the swap this gate exists for requires a file a
    validator once saw. A refusal names the swap, so an operator whose CLI
    convenience symlink is refused gets an actionable message instead of a
    bare ``ELOOP``.
    """
    if not file_path.exists():
        return
    try:
        open_verified(file_path, sandbox_roots).close()
    except VerifiedOpenError as exc:
        raise DocumentParseError(
            f"Refusing to open {file_path.name}: {exc}. Pass the real file path "
            "(the path a validator resolved, not a symlink standing in for it)."
        ) from exc
