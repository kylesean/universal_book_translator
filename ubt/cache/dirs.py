"""Shared cache-directory resolution for every disk cache the engine keeps."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """The per-user UBT cache root: ``UBT_CACHE_DIR``, else XDG, else ``~/.cache/ubt``.

    All of the engine's disk caches (page-profile facts, rendered math
    SVG/PNG, the typst SVG probe) resolve to this one root so a cache stays
    put no matter which working directory a run starts from, and
    ``UBT_CACHE_DIR`` relocates all of them at once. Never a predictable
    world-writable /tmp path, which other users on a shared machine could
    pre-create or symlink.
    """
    env = os.environ.get("UBT_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "ubt"
    return Path.home() / ".cache" / "ubt"
