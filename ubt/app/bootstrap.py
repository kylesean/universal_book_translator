"""Composition root: install the application services core resolves through."""

from __future__ import annotations

from ubt.app.pdf_facts import AdapterPdfFacts
from ubt.core.engine.pdf_facts import configure_pdf_facts

_installed = False


def install_services() -> None:
    """Install the adapter-backed services core calls through its protocols.

    Idempotent: the CLI, the API app factory, the MCP server and the test
    session all call it; the first call owns the process-wide provider.
    """
    global _installed
    if _installed:
        return
    configure_pdf_facts(AdapterPdfFacts())
    _installed = True


__all__ = ["install_services"]
