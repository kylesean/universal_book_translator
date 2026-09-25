"""CLI entrypoint wrapper for ``ubt-mcp``.

``ubt.mcp.server`` imports the optional ``mcp`` package at module scope, so a
missing extra raises :class:`OptionalDependencyError` on import — the right
shape for library callers, which must be able to catch it. A console script,
though, should print the actionable install hint and exit non-zero instead of
dumping a traceback. This entrypoint converts the exception into that clean exit.
"""

from __future__ import annotations

from ubt.core.exceptions import OptionalDependencyError


def main() -> None:
    """Run the stdio server, or exit cleanly when the ``mcp`` extra is absent."""
    try:
        from ubt.mcp.server import main as _serve
    except OptionalDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    _serve()


if __name__ == "__main__":
    main()
