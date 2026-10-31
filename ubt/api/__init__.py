"""Lazy (PEP 562) re-exports: importing ``ubt.api`` must not build the app or
configure logging as a side effect. ``create_app`` is the supported library
entry point; ``app`` is resolved only when actually named (uvicorn's
``ubt.api.app:app`` target).

``app`` is in ``__all__`` even though it is not a module-level binding: the
star-import and the tooling that reads ``__all__`` should see the same surface
``__getattr__`` actually answers for, and one documented spelling of ``ubt.api``
is exactly the uvicorn target.
"""

from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from ubt.api.app import create_app

        return create_app
    if name == "app":
        from ubt.api.app import _get_app

        return _get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
