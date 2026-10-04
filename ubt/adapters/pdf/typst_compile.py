"""Typst compilation primitives shared by the PDF typesetting paths.

Subprocess probe used by the rigid typesetter (per-page overlay compile)
and the math probe. The in-place statement-bisect fallback was retired with
its engine.

Both helpers are total: a missing binary, a timeout, or an OS error becomes a
``(False, message)`` verdict (or a clean availability check) instead of an
unhandled ``FileNotFoundError`` / ``TimeoutExpired``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ubt.core.env import subprocess_env


def resolve_typst_binary(binary: str = "typst") -> str | None:
    """Return the executable path, or None when it cannot be found."""
    found = shutil.which(binary)
    if found:
        return found
    # An explicit path (not on PATH) still counts when it exists.
    candidate = Path(binary)
    return str(candidate) if candidate.is_file() else None


def typst_available(binary: str = "typst") -> bool:
    return resolve_typst_binary(binary) is not None


def typst_compile(
    typ_path: str, pdf_path: str, binary: str = "typst", timeout: float = 120.0
) -> tuple[bool, str]:
    """Compile one Typst file; return ``(ok, stderr_tail)``. Never raises."""
    resolved = resolve_typst_binary(binary)
    if resolved is None:
        return False, f"Typst compiler binary '{binary}' not found on PATH"
    try:
        proc = subprocess.run(
            [resolved, "compile", typ_path, pdf_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            # typst keeps PATH/HOME/XDG_*/fontconfig; keys are not its business.
            env=subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"Typst compilation timed out after {int(timeout)}s"
    except OSError as exc:
        return False, f"Typst compiler could not be executed: {exc}"
    return proc.returncode == 0, proc.stderr[-1500:]


def typst_version(binary: str = "typst") -> str | None:
    """Return the installed Typst version string, or None if unavailable."""
    resolved = resolve_typst_binary(binary)
    if resolved is None:
        return None
    try:
        proc = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            env=subprocess_env(),
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return None


__all__ = ["resolve_typst_binary", "typst_available", "typst_compile", "typst_version"]
