"""Compatibility re-export of :mod:`ubt.core.engine.services`.

The per-run collaborator contract moved into the core engine so the stages no
longer import a compiler package; this shim keeps the old import path working
for external callers.
"""

from __future__ import annotations

from ubt.core.engine.services import RunServices

__all__ = ["RunServices"]
