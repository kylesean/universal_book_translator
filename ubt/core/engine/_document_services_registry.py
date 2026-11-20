"""Mutable slot for the app-installed document services (one owner)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ubt.core.engine.document_services import DocumentServices

provider: DocumentServices | None = None
