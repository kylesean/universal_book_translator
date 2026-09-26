"""Owner-only permissions for the on-disk artifacts that carry manuscript text.

Ledgers, translation memory and API/CLI logs store the full source *and* target of
every paragraph of an unpublished book, and SQLite plus ``RotatingFileHandler``
both create their files with ``0666 & ~umask`` — i.e. ``0644`` under a normal
umask, world-readable on any shared host, CI runner or multi-user GPU box. The
writer lock already asks for ``0o600`` (``engine/writer_lock.py``); this module
applies the same rule to the files that actually hold the text.

Everything here is best-effort and never raises: a filesystem that refuses the
chmod (foreign mount, read-only ACL) must not fail a translation run.

Second, unrelated job: this module also carries the project's *sensitive path
name* list (``SENSITIVE_FILENAME_PARTS``). It lives in core rather than only in
``ubt.api.security`` so consumers that must not import the API package — the
stdio MCP server, whose import of ``ubt.api`` would build the whole FastAPI
app — can share the same deny list instead of keeping a second copy that
drifts. ``ubt.api.security`` re-exports it under its historical name.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from ubt.core.policy.layout_policy import BOOK_TEXT_DIR_MODE, BOOK_TEXT_FILE_MODE

logger = logging.getLogger(__name__)

#: Path components naming a credential store, shell configuration or secret
#: file. Matched (casefolded) against every component of a resolved path, so
#: ``/home/u/.ssh/id_rsa`` is refused because of ``.ssh`` — and so is
#: ``/home/u/.SSH/id_rsa``: the check is casefolded because Windows and macOS
#: resolve both spellings to the same file (review L1).
SENSITIVE_FILENAME_PARTS = (
    ".ssh",
    ".aws",
    ".docker",
    ".kube",
    ".bashrc",
    ".bash_profile",
    ".profile",
    ".zshrc",
    ".zsh_history",
    # Shell history files record command lines verbatim — API keys and
    # ``cat ~/.ssh/id_rsa`` included.
    ".bash_history",
    ".config",
    ".git",
    ".git-credentials",
    # GnuPG keyrings and the pass password store: the secrets themselves.
    ".gnupg",
    ".password-store",
    ".env",
    # Per-environment dotenv variants; their whole purpose is to hold secrets.
    ".env.local",
    "secrets.env",
    "credentials.json",
    ".secrets",
    ".netrc",
    "auth.json",
    "id_rsa",
    "id_ed25519",
)

#: Precomputed casefolded form of :data:`SENSITIVE_FILENAME_PARTS`, so the
#: per-request path check does not rebuild it each time.
SENSITIVE_PARTS_CASEFOLD = frozenset(part.casefold() for part in SENSITIVE_FILENAME_PARTS)


def is_sensitive_path_part(part: str) -> bool:
    """Whether a resolved path component names a credential/secret location."""
    return part.casefold() in SENSITIVE_PARTS_CASEFOLD


def restrict_dir_to_owner(path: Path) -> Path:
    """Create ``path`` (parents included) and limit it to its owner."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(BOOK_TEXT_DIR_MODE)
    except OSError as exc:
        logger.debug("Could not restrict %s to owner: %s", path, exc)
    return path


def restrict_file_to_owner(path: Path) -> None:
    """Limit an existing file to its owner; a no-op when it is absent."""
    try:
        if path.exists():
            path.chmod(BOOK_TEXT_FILE_MODE)
    except OSError as exc:
        logger.debug("Could not restrict %s to owner: %s", path, exc)


def restrict_sqlite_family(path: Path) -> None:
    """Restrict a database file plus the WAL sidecars SQLite may hold open."""
    for candidate in (path, path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm")):
        restrict_file_to_owner(candidate)


def restrict_env_file(path: Path | None = None) -> Path | None:
    """Converge the dotenv file that carries credentials to ``0600``.

    ``.env`` is where ``UBT_LLM_API_KEY`` / ``UBT_API_KEY`` /
    ``UBT_OCR_SIDECAR_TOKEN`` live, and editors, ``cp`` and ``git checkout``
    create it as ``0666 & ~umask`` — ``0644``, i.e. world-readable on any
    shared host, CI runner or multi-user GPU box (the same incident class this
    module exists for). The process
    entry points call this once at boot, before the config layer reads the file.

    ``path`` defaults to ``.env`` in the working directory, which is where
    pydantic-settings resolves ``env_file=".env"``. An absent file returns
    ``None``; a refused chmod warns instead of raising — a read-only or foreign
    mount must not stop a translation run or a server boot.
    """
    target = Path.cwd() / ".env" if path is None else Path(path)
    try:
        if not target.is_file():
            return None
        previous = target.stat().st_mode & 0o777
        target.chmod(BOOK_TEXT_FILE_MODE)
    except OSError as exc:
        logger.warning("Could not restrict %s to owner-only permissions: %s", target, exc)
        return None
    if previous != BOOK_TEXT_FILE_MODE:
        # Warning, once: the old mode was the exposure (credentials readable by
        # every account on the host) and the fix is not durable across an
        # editor that rewrites the file with its own mode.
        logger.warning(
            "Tightened %s from %04o to %04o: .env carries API credentials and must stay owner-only",
            target,
            previous,
            BOOK_TEXT_FILE_MODE,
        )
    return target


def world_readable_files(directory: Path, extra_dirs: Iterable[Path] = ()) -> list[Path]:
    """Files under ``directory`` — plus any ``extra_dirs`` — that group or other users could still read.

    ``extra_dirs`` extends the scan to artifact trees the same caller owns but
    the default check never walked: the pipeline's export report only scans
    ``db_dir`` (see :func:`world_readable_files`'s caller in
    ``ubt/core/engine/pipeline.py``), so rendered
    PDFs, quality reports and the Docling conversion cache — all of which hold
    the manuscript text — were invisible to it. Passing them here keeps one
    definition of "exposed"; the empty default preserves the original
    single-directory behaviour for existing callers.
    """
    found: dict[Path, None] = {}

    def _onerror(err: OSError) -> None:
        # ``rglob`` silently skipped an unreadable directory, so a subtree of
        # world-readable manuscript files could hide below the very report that
        # exists to find them. Surface the gap instead of an all-clear.
        logger.warning(
            "world-readable permission scan could not enter %s: %s",
            err.filename or "an unreadable path",
            err,
        )

    for root in (directory, *extra_dirs):
        try:
            if not root.is_dir():
                continue
            for dirpath, _dirnames, filenames in os.walk(root, onerror=_onerror):
                for name in filenames:
                    child = Path(dirpath) / name
                    try:
                        if child.is_file() and child.stat().st_mode & 0o077:
                            found[child] = None
                    except OSError:
                        continue
        except OSError:
            # Unreadable root itself: report what was scanned rather than
            # failing the permission report.
            continue
    return sorted(found)


def warn_world_readable(directory: Path, extra_dirs: Iterable[Path] = ()) -> None:
    """Warn once about book-text files that group/other users can still read.

    The scan is recursive (``rglob`` + ``stat``), so callers must run it off the
    event loop. The warning names each exposed file's real parent — the exposed
    files may live outside ``directory`` (deliverables, caches), and pointing the
    operator at the already-owner-only ``directory`` changed nothing.
    """
    exposed = world_readable_files(directory, extra_dirs=extra_dirs)
    if not exposed:
        return
    parents = sorted({str(path.parent) for path in exposed})
    target = " ".join(f"'{parent}'" for parent in parents)
    logger.warning(
        "%d file(s) carry book text but are readable by group/other "
        "(first: %s). Tighten them with: chmod -R go-rwx %s",
        len(exposed),
        exposed[0],
        target,
    )
