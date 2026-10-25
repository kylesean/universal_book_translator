"""Typst compilation primitives shared by the PDF typesetting paths.

Subprocess helpers used by the document render and the math compile probe.

Both are total: a missing binary, a timeout, or an OS error becomes a
``(False, message)`` verdict (or a clean availability check) instead of an
unhandled ``FileNotFoundError`` / ``TimeoutExpired``.

**Compiles are byte-reproducible.** Typst stamps ``/CreationDate`` and
``/ModDate`` with the wall clock by default, so two compiles of the same source
produced different bytes — which contradicted ``ARCHITECTURE.md``'s "same blocks
in, same bytes out" and made every artifact diff a false positive. The compile
now pins the timestamp (see :data:`REPRODUCIBLE_TIMESTAMP`).
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

from ubt.core.env import subprocess_env

try:
    _typst_mod = importlib.import_module("typst")
    _HAS_TYPST_PY = hasattr(_typst_mod, "compile")
except Exception:
    _HAS_TYPST_PY = False

#: The timestamp every compile stamps into the PDF's document info.
#:
#: A fixed instant, not "now": the artifact is a function of the IR, and a
#: wall-clock field makes two renders of one document differ in bytes while
#: being identical in content. 2000-01-01T00:00:00Z is arbitrary but stable and
#: unambiguous — a reader can tell the value is synthetic at a glance, which a
#: plausible-looking real date would hide.
REPRODUCIBLE_TIMESTAMP = 946_684_800


def resolve_typst_binary(binary: str = "typst") -> str | None:
    """Return the executable path, or None when it cannot be found."""
    found = shutil.which(binary)
    if found:
        return found
    # An explicit path (not on PATH) still counts when it exists.
    candidate = Path(binary)
    return str(candidate) if candidate.is_file() else None


def typst_available(binary: str = "typst") -> bool:
    if _HAS_TYPST_PY:
        return True
    return resolve_typst_binary(binary) is not None


def _bundled_font_dir() -> Path | None:
    """The bundled font directory, or ``None`` when the set is not installed.

    Imported lazily: this module sits on the compile hot path and must not pull
    the render package's import graph at load time. A font lookup that fails
    must never stop a compile — the caller falls back to system fonts, which is
    the previous behaviour.
    """
    try:
        from ubt.render.typst.fonts import bundled_font_dir

        return bundled_font_dir()
    except Exception:  # noqa: BLE001 - a probe failure degrades, never blocks
        return None


def typst_compile(
    typ_path: str, pdf_path: str, binary: str = "typst", timeout: float = 120.0
) -> tuple[bool, str]:
    """Compile one Typst file; return ``(ok, stderr_tail)``. Never raises.

    The compile is pinned to :data:`REPRODUCIBLE_TIMESTAMP` so the produced PDF
    is a function of the Typst source alone. See the module docstring.
    """
    last_err: str | None = None
    if _HAS_TYPST_PY:
        try:
            typst = importlib.import_module("typst")
            typst.compile(typ_path, output=pdf_path, creation_timestamp=REPRODUCIBLE_TIMESTAMP)
            return True, ""
        except TypeError:
            # A ``typst-py`` that predates the keyword. Falling through to the
            # CLI is deliberate: the binding would otherwise hand back a
            # wall-clock artifact while the CLI path is reproducible.
            pass
        except Exception as exc:
            last_err = str(exc)

    resolved = resolve_typst_binary(binary)
    if resolved is None:
        if last_err is not None:
            return False, f"Typst compilation failed: {last_err}"
        return False, f"Typst compiler binary '{binary}' not found on PATH"
    # A bundled font set makes the PDF a function of the source alone: the same
    # book renders identically on a workstation and in an Alpine container.
    # Without it the output depends on whatever the host happened to install.
    bundled = _bundled_font_dir()
    font_args = ["--font-path", str(bundled)] if bundled else []
    try:
        proc = subprocess.run(
            [
                resolved,
                "compile",
                "--creation-timestamp",
                str(REPRODUCIBLE_TIMESTAMP),
                *font_args,
                typ_path,
                pdf_path,
            ],
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
    if _HAS_TYPST_PY:
        try:
            typst = importlib.import_module("typst")
            v = getattr(typst, "__version__", None)
            if v:
                return f"typst-py {v}"
        except Exception:
            pass
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
