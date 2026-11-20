"""Mutable slot for the app-installed PDF facts provider (one owner).

Separated from :mod:`ubt.core.engine.pdf_facts` so tests can reset the slot
without importing the protocol module's public surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ubt.core.engine.pdf_facts import PdfFactsProvider

provider: PdfFactsProvider | None = None
