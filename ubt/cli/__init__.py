"""Universal Book Translator CLI package.

``app`` is resolved lazily. Importing it eagerly pulled ``ubt.cli.main`` — and
with it ``typer`` and every command module — into anything that touched this
package, including `scripts/build_console_assets.py`, which wants only the
JavaScript toolchain probe in ``ubt.cli.jstoolchain``. That made a pure
build helper unusable from an interpreter without the project's dependencies
installed (CI called it with a bare ``python``), and it slowed every
``import ubt.cli.*`` by the whole CLI import graph.

``from ubt.cli import app`` still works — it is the same object, just built on
first use — so the public surface is unchanged.
"""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from ubt.cli.main import app as _app

        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
