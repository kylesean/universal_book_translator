"""Mutable slot for the app-installed artifact services (one owner)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ubt.core.engine.artifact_services import ArtifactServices

provider: ArtifactServices | None = None
