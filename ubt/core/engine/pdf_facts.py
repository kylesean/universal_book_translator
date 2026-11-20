"""Core-side PDF fact queries, resolved through an application-installed provider.

The delivery/router code needs a page census and a cheap page probe before it
can choose a contract. Those live in the PDF adapter stack, which core must not
name at module or function scope (the architecture gate checks both). The
application composition root installs an implementation here; core calls only
this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ubt.core.engine import _pdf_facts_registry as _registry


@dataclass(frozen=True)
class PdfStructureFacts:
    """Page-level structure census feeding render-engine routing.

    All ratios are over *all* pages (not a sample) and are page-level, so the
    signal does not drift when the parser re-chunks text into more IR blocks.
    ``has_scan`` / ``formula_heavy`` keep their historical majority semantics.
    """

    has_scan: bool
    formula_heavy: bool
    multicolumn_page_share: float
    structural_page_share: float
    confirmed_font_damage_pages: int = 0


class PdfFactsProvider(Protocol):
    """The PDF adapter operations the core planner is allowed to call."""

    def classify_structure(self, input_path: Path) -> PdfStructureFacts: ...

    def probe_pages(self, input_path: Path) -> tuple[int, int]: ...

    def sample_pages(self, input_path: Path) -> tuple[int, bool, str]: ...

    def probe_feasibility(self, input_path: Path) -> Any: ...

    def page_count(self, input_path: Path) -> int: ...

    def inspect_profile(self, input_path: Path) -> Any: ...

    def profile_pages(self, input_path: Path) -> list[Any]: ...

    def page_kind(self) -> Any: ...


def configure_pdf_facts(provider: PdfFactsProvider) -> None:
    """Install the app's implementation (the composition root calls this once)."""
    _registry.provider = provider


def _provider() -> PdfFactsProvider:
    provider = _registry.provider
    if provider is None:
        raise RuntimeError(
            "PDF facts provider is not configured; import the application composition root "
            "(ubt.app.bootstrap) before running a job"
        )
    return provider


def classify_pdf_structure(input_path: Path) -> PdfStructureFacts:
    """Census the document's pages once and return the routing signals."""
    return _provider().classify_structure(input_path)


def probe_pdf_pages(input_path: Path) -> tuple[int, int]:
    """Return (page_count, char_count) via the short-doc probe."""
    return _provider().probe_pages(input_path)


def sample_pdf_pages(input_path: Path) -> tuple[int, bool, str]:
    """Sample a PDF into ``(page_count, is_scanned, text_preview)``."""
    return _provider().sample_pages(input_path)


def probe_pdf_feasibility(input_path: Path) -> Any:
    """Run the native in-place feasibility preflight."""
    return _provider().probe_feasibility(input_path)


def pdf_page_count(input_path: Path) -> int:
    """Return the source PDF's page count."""
    return _provider().page_count(input_path)


def inspect_pdf_profile(input_path: Path) -> Any:
    """Return the doc-wide PDFDocumentProfile."""
    return _provider().inspect_profile(input_path)


def profile_pdf_pages(input_path: Path) -> list[Any]:
    """Return the per-page profile list."""
    return _provider().profile_pages(input_path)


def page_kind_enum() -> Any:
    """Return the adapter-side PageKind enum."""
    return _provider().page_kind()


__all__ = [
    "PdfFactsProvider",
    "PdfStructureFacts",
    "classify_pdf_structure",
    "configure_pdf_facts",
    "probe_pdf_feasibility",
    "page_kind_enum",
    "pdf_page_count",
    "probe_pdf_pages",
    "inspect_pdf_profile",
    "profile_pdf_pages",
    "sample_pdf_pages",
]
