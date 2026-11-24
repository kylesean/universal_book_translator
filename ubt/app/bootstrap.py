"""Composition root: install the application services core resolves through."""

from __future__ import annotations

import threading

from ubt.app.artifact_services import AdapterArtifactServices
from ubt.app.document_services import AdapterDocumentServices
from ubt.app.pdf_facts import AdapterPdfFacts
from ubt.core.engine.artifact_services import configure_artifact_services
from ubt.core.engine.document_services import configure_document_services
from ubt.core.engine.pdf_facts import configure_pdf_facts

_installed = False
_lock = threading.Lock()


def install_services() -> None:
    """Install the adapter-backed services core calls through its protocols.

    Idempotent: the CLI, the API app factory, the MCP server and the test
    session all call it; the first call owns the process-wide provider.
    """
    global _installed
    if _installed:
        return
    with _lock:
        if _installed:
            return
        configure_pdf_facts(AdapterPdfFacts())
        configure_artifact_services(AdapterArtifactServices())
        configure_document_services(AdapterDocumentServices())
        _installed = True


def reset_services_for_testing() -> None:
    """Allow tests to re-install default services after swapping fake providers."""
    global _installed
    with _lock:
        _installed = False


__all__ = ["install_services", "reset_services_for_testing"]
