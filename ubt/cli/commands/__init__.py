"""CLI command implementations for Universal Book Translator."""

from __future__ import annotations

from typing import Any


def resolve_db_dir() -> Any:
    """The single owner of ``--db-dir`` resolution, ``main._resolve_db_dir``.

    Imported lazily: ``ubt.cli.__init__`` imports ``main``, so a module-level
    ``from ubt.cli.main import ...`` here would be a cycle. Every command that
    needs the resolver calls this instead of re-declaring the forwarder.
    """
    from ubt.cli.main import _resolve_db_dir

    return _resolve_db_dir


__all__ = ["resolve_db_dir"]
