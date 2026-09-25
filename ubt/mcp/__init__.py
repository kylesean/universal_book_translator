"""UBT MCP server package (stdio-first, local-trust)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ubt.mcp.server import main, mcp

__all__ = ["main", "mcp"]


def __getattr__(name: str) -> Any:
    # Lazy on purpose: ``ubt.mcp.server`` imports the optional ``mcp`` extra at
    # module scope, so an eager re-export here made ``import ubt.mcp`` itself
    # fail without the extra. ``from ubt.mcp import main, mcp`` still resolves.
    if name in __all__:
        from ubt.mcp import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Advertise the lazy names so pydoc/IDE/tool discovery can see them.
    return sorted(set(globals()) | set(__all__))
