"""Factory helpers to instantiate document adapters based on file extensions."""

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path

from ubt.adapters.base import BaseDocumentAdapter, BasePDFEngineAdapter
from ubt.adapters.docx.adapter import DOCXAdapter
from ubt.adapters.epub.adapter import EPUBAdapter
from ubt.adapters.html.adapter import HTMLAdapter
from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.encryption import reject_encrypted_pdf
from ubt.adapters.pdf.pdf_profiler import inspect_pdf_profile
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.core.exceptions import OptionalDependencyError, UnsupportedDocumentFormatError

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
            "and upstream CLI contract incompatibility. Please use pdf_engine='docling' (MIT/Apache-2.0)."
        )
    if engine_key in ("auto", "pdfium"):
        # ``auto`` is the lossless docling engine, unconditionally: the probe
        # reports what the document contains, it does not vote on the engine.
        # ``pdfium`` is an explicit fast-path opt-in: it stays on the fast path
        # for every fuzzy signal, but
        # a scanned (text-less) document still escapes to docling — pdfium
        # extracts zero blocks from one, which is a capability boundary, not a
        # routing preference. See docs/guides/PDF_ENGINE_ROUTING_REDESIGN.md.
        if path is None:
            # No path to probe (a registry lookup): ``auto`` is conservative.
            engine_key = "docling" if engine_key == "auto" else "pdfium"
        else:
            try:
                plan = inspect_pdf_profile(path)
            except Exception as exc:
                logger.warning(
                    "Content probe failed on '%s' (%s); defaulting engine to 'pdfium'.",
                    path.name,
                    exc,
                    exc_info=True,
                )
                plan = None
            if plan is None:
                engine_key = "pdfium"
            elif engine_key == "auto":
                engine_key = plan.primary_engine
            elif plan.has_scan:
                logger.warning(
                    "'%s' looks scanned (pages without a text layer); pdfium would "
                    "extract 0 blocks, so routing it to docling for OCR-grade ingest.",
                    path.name,
                )
                engine_key = "docling"
            else:
                # An explicit ``pdfium`` request is honoured, but never silently:
                # pdfium's reader represents only headings, paragraphs, list items
                # and code, so every table, figure and formula in the document is
                # dropped from the rebuilt PDF. The probe's signals say how likely
                # that is, but they cannot rule it out (a vector-drawn figure reads
                # as zero on all of them).
                logger.warning(
                    "pdfium requested for '%s': its reader cannot represent tables, "
                    "figures or formulas, so any the document contains are omitted "
                    "from the rebuilt PDF (probe signals — multicolumn=%s, "
                    "formulas=%s, vector-heavy=%s — are only a lower bound). "
                    "Use --pdf-engine docling (or --preset standard) for lossless "
                    "ingest.",
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
        # An explicit ``--pdf-engine docling`` names an engine that cannot run.
        # Falling back would silently deliver a book with every table, figure
        # and formula dropped, under a job id that later runs resume into, so
        # this is refused before ingest rather than degraded. ``auto`` keeps its
        # documented fallback (see ``DoclingPDFAdapter._extract_blocks_sync``),
        # but never silently: the warning names what is lost, and the job id
        # records the engine that actually ran.
        if pdf_engine.lower() == "docling":
            raise OptionalDependencyError(
                "pdf_engine='docling' was requested but the 'docling' package is not "
                "installed. Install it with `pip install 'universal-book-translator[pdf]'` "
                "(or `uv sync --extra pdf`), or pass --pdf-engine pdfium to accept the "
                "fast path, which cannot represent tables, figures or formulas."
            )
        if path is not None:
            logger.warning(
                "pdf_engine='auto' selected docling for '%s' but the 'docling' package "
                "is not installed: falling back to the geometric pdfium reader, which "
                "cannot represent tables, figures or formulas — those are omitted from "
                "the rebuilt PDF. Install the [pdf] extra for lossless ingest.",
                path.name,
            )
    return factory_fn()


# Standard built-in engine & adapter bindings. Only the canonical names live
# here: ``docling`` (the Typst-reflow mainline) and ``pdfium`` (the born-digital
# fast path), plus the ``auto`` probe handled above. Each engine has exactly one
# name; pass ``docling`` (or ``auto``) rather than an alias.
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
    pdf_engine: str = "auto",
) -> BaseDocumentAdapter:
    """Factory helper to resolve the appropriate document adapter by file extension.

    The default matches the config's ``pdf_engine`` default so a direct call
    resolves the same engine the pipeline would: ``auto`` is Docling,
    unconditionally; ``pdfium`` is an explicit operator opt-in that carries a
    loud multicolumn/scanned warning.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    factory = _ADAPTER_REGISTRY.get(suffix)
    if factory is not None:
        return factory(pdf_engine, file_path)

    raise UnsupportedDocumentFormatError(
        f"No adapter registered for document extension '{suffix}' ({file_path.name})"
    )
