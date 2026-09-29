"""Factory helpers to instantiate document adapters based on file extensions."""

import logging
from collections.abc import Callable
from pathlib import Path

from ubt.adapters.base import BaseDocumentAdapter, BasePDFEngineAdapter
from ubt.adapters.docx.adapter import DOCXAdapter
from ubt.adapters.epub.adapter import EPUBAdapter
from ubt.adapters.html.adapter import HTMLAdapter
from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.engine_selector import select_pdf_engine
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.core.exceptions import UnsupportedDocumentFormatError

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

    Replaces what the old ``PdfEngine`` Literal checked statically, now that
    the set is open to in-process :func:`register_pdf_engine` calls.
    """
    return name.lower() in _PDF_ENGINE_REGISTRY


def _resolve_pdf_adapter(pdf_engine: str, path: Path | None = None) -> BasePDFEngineAdapter:
    engine_key = pdf_engine.lower()
    if engine_key == "babeldoc":
        raise UnsupportedDocumentFormatError(
            "BabelDOC engine is deprecated and removed due to licensing constraints "
            "and upstream CLI contract incompatibility. Please use pdf_engine='docling' (MIT/Apache-2.0)."
        )
    if engine_key == "auto":
        # First-page heuristic routing — born-digital single-column
        # PDFs take the pypdfium2 fast path; scans and multi-column layouts
        # stay on the Docling mainline (best reading-order / OCR quality).
        engine_key = select_pdf_engine(path) if path is not None else "docling"
    elif engine_key == "pdfium" and path is not None and path.exists():
        # Gate 1b (explicit routing): a forced pdfium fast path on a
        # formula-dense/scan/multicolumn document reproduces the KV-handbook
        # failure (shattered equations translated as prose). The force is
        # honored — speed is a legitimate choice — but never silently.
        try:
            probe = select_pdf_engine(path)
        except Exception:
            probe = "pdfium"
        if probe == "docling":
            logger.warning(
                "UBT_PDF_ENGINE='pdfium' forced on '%s', but the content probe "
                "recommends 'docling' (formula-dense/scan/multicolumn): "
                "expect math-debris blocks guarded as verbatim FORMULA skips "
                "and degraded equation fidelity. Use 'auto' for probe routing.",
                path.name,
            )
    factory_fn = _PDF_ENGINE_REGISTRY.get(engine_key)
    if factory_fn is None:
        # Read the registry rather than a hardcoded list: this is the one place
        # that can tell the truth about what is selectable, custom entries included.
        available = ", ".join(["auto", *sorted(_PDF_ENGINE_REGISTRY)])
        raise UnsupportedDocumentFormatError(
            f"Unsupported or unregistered PDF engine: '{pdf_engine}'. Available: {available}."
        )
    return factory_fn()


# Standard built-in engine & adapter bindings. Only the canonical names live
# here: ``docling`` (the Typst-reflow mainline) and ``pdfium`` (the born-digital
# fast path), plus the ``auto`` probe handled above. The old ``typst`` /
# ``modern`` aliases all pointed at ``DoclingPDFAdapter`` and made one engine
# look like three; pass ``docling`` (or ``auto``) instead.
_PDF_ENGINE_REGISTRY.update(
    {
        "docling": DoclingPDFAdapter,
        "pdfium": PDFiumAdapter,
    }
)

_ADAPTER_REGISTRY.update(
    {
        ".epub": lambda _engine, _path: EPUBAdapter(),
        ".md": lambda _engine, _path: MarkdownAdapter(),
        ".markdown": lambda _engine, _path: MarkdownAdapter(),
        ".txt": lambda _engine, _path: MarkdownAdapter(
            plain_text=True
        ),  # flat text: blank-line paragraphs, no Markdown syntax
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
    return sorted(_ADAPTER_REGISTRY)


def get_adapter_for_path(
    path: Path | str,
    pdf_engine: str = "docling",
) -> BaseDocumentAdapter:
    """Factory helper to resolve the appropriate document adapter by file extension.

    Uses Docling + Typst + pdf_oxide for 100% permissive Zero-AGPL modern delivery.
    ``pdf_engine='auto'`` routes PDFs through the first-page heuristic
    (born-digital single-column → pypdfium2 fast path, else Docling).
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    factory = _ADAPTER_REGISTRY.get(suffix)
    if factory is not None:
        return factory(pdf_engine, file_path)

    raise UnsupportedDocumentFormatError(
        f"No adapter registered for document extension '{suffix}' ({file_path.name})"
    )
