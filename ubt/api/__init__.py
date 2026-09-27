"""Universal Book Translator FastAPI Microservice package.

Lazy (PEP 562) re-exports: importing ``ubt.api`` must not build the app or
configure logging as a side effect. ``create_app`` is the supported library
entry point; ``app`` is resolved only when actually named (uvicorn's
``ubt.api.app:app`` target).
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from ubt.api.app import create_app

        return create_app
    if name == "app":
        from ubt.api.app import _get_app

        return _get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
