"""Compatibility re-export of :mod:`ubt.core.engine.blocks`.

The revision-guarded block view moved into the core engine so the stages no
longer import a compiler package; this shim keeps the old import path working
for external callers.
"""

from __future__ import annotations

from ubt.core.engine.blocks import BlockReader

__all__ = ["BlockReader"]
