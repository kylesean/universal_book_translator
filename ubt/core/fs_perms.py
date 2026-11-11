"""Owner-only permissions for the on-disk artifacts that carry manuscript text.

Ledgers, translation memory and API/CLI logs store the full source *and* target of
every paragraph of an unpublished book, and SQLite plus ``RotatingFileHandler``
both create their files with ``0666 & ~umask`` — i.e. ``0644`` under a normal
umask, world-readable on any shared host, CI runner or multi-user GPU box. The
single-writer guard now lives in the ledger's own SQLite file (``job_locks``); this
module applies the owner-only rule to the files that actually hold the text.

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
import stat
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import IO

from ubt.core.policy.layout_policy import BOOK_TEXT_DIR_MODE, BOOK_TEXT_FILE_MODE

logger = logging.getLogger(__name__)

#: Path components naming a credential store, shell configuration or secret
#: file. Matched (casefolded) against every component of a resolved path, so
#: ``/home/u/.ssh/id_rsa`` is refused because of ``.ssh`` — and so is
#: ``/home/u/.SSH/id_rsa``: the check is casefolded because Windows and macOS
#: resolve both spellings to the same file.
SENSITIVE_FILENAME_PARTS = (
    ".ssh",
    ".aws",
    ".azure",
    ".docker",
    ".gcloud",
    ".kube",
    ".bashrc",
    ".bash_profile",
    ".profile",
    ".zshrc",
    ".zsh_history",
    # Shell history files record command lines verbatim — API keys and
    # ``cat ~/.ssh/id_rsa`` included.
    ".bash_history",
    # The whole ``.config`` tree is denied, which already covers the per-tool
    # credential stores that live under it (``.config/gcloud``,
    # ``.config/chromium``, ``.config/google-chrome``): they are listed here as
    # a comment only, so nobody "fixes" their absence by re-adding a subset.
    ".config",
    ".mozilla",
    ".git",
    ".git-credentials",
    # GitHub's authenticated ``git https://...@github.com/...`` cookie jar.
    ".gitcookies",
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
    # Tool-specific credential stores: npm publish tokens, PyPI upload tokens,
    # MySQL client logins, and the rc files wget/curl read for stored
    # credentials or ``.netrc`` pointers. ``.htpasswd`` is a password file by
    # construction.
    ".npmrc",
    ".pypirc",
    ".my.cnf",
    ".wgetrc",
    ".curlrc",
    ".htpasswd",
    "auth.json",
    # SSH private keys (the individual files, for callers whose paths carry
    # them outside a ``.ssh`` directory). The ``.pub`` siblings are not
    # secrets, but they are listed so the whole key pair is treated uniformly:
    # a caller that can read ``id_ed25519.pub`` has confirmed key material
    # lives beside it, which is reconnaissance a denial should not hand over.
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa.pub",
    "id_dsa.pub",
    "id_ecdsa.pub",
    "id_ed25519.pub",
)

#: Precomputed casefolded form of :data:`SENSITIVE_FILENAME_PARTS`, so the
#: per-request path check does not rebuild it each time.
SENSITIVE_PARTS_CASEFOLD = frozenset(part.casefold() for part in SENSITIVE_FILENAME_PARTS)


def is_sensitive_path_part(part: str) -> bool:
    """Whether a resolved path component names a credential/secret location."""
    folded = part.casefold()
    if folded in SENSITIVE_PARTS_CASEFOLD:
        return True
    # Per-environment dotenv variants (``.env.production``) and ``.envrc`` hold
    # the same secrets as ``.env``; exact matching let them through.
    return folded == ".envrc" or folded.startswith(".env.")


class VerifiedOpenError(OSError):
    """A file could not be opened under the anti-swap contract of :func:`open_verified`.

    Subclasses :class:`OSError` so callers that already handle filesystem
    failures catch it unchanged; the message names what refused, because a
    bare ``ELOOP`` from ``os.open`` would read as "file broken" instead of
    "the validated path was replaced by a symlink".
    """


def open_verified(
    path: Path | str,
    allowed_roots: Iterable[Path] | None = None,
) -> IO[bytes]:
    """Open ``path`` for reading while refusing a symlink swapped in after validation.

    The API resolves and sandboxes a submitted input path at *submit* time, but
    the pipeline opens the stored path later. A local writer can replace the
    validated file with a symlink in between, and the open would follow it to
    any file on the host — the classic validation/use TOCTOU. This helper closes
    that window at the moment of the open:

    * the file is opened with ``O_NOFOLLOW`` (where the platform has it — on
      Windows, where it does not, the ``lstat``/``fstat`` identity check below
      carries the refusal), so a final-component symlink never opens;
    * the opened descriptor must be a regular file, so a swap to a FIFO, a
      device or a directory is refused instead of read (or hung on);
    * the opened file's identity (``st_dev``/``st_ino`` from ``fstat``) must
      match ``lstat`` of the path, so a swap between the two calls fails;
    * when ``allowed_roots`` is given, the path's final resolution must still
      land inside one of them, so a swapped directory component cannot steer
      the read outside the sandbox.

    Returns a binary file object (the caller owns closing it). Raises
    :class:`VerifiedOpenError` — an :class:`OSError` — with a message that says
    *why* the open was refused. Without ``allowed_roots`` the containment check
    is skipped (callers that do not know a sandbox still get the anti-symlink
    verification).
    """
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    # ``O_NONBLOCK`` so a swap to a FIFO (a regular-file-shaped denial of
    # service whose ``os.open`` would block forever waiting for a writer)
    # returns immediately and the regular-file check below refuses it. On
    # regular files it is a no-op; on Windows the constant does not exist.
    flags |= getattr(os, "O_NONBLOCK", 0)
    if sys.platform == "win32":
        flags |= getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise VerifiedOpenError(
            f"refusing to open {target.name}: the validated file could not be "
            f"opened directly ({exc.strerror or exc}); if it was replaced by a "
            "symlink, this refusal is the sandbox working as intended"
        ) from exc
    try:
        st_fd = os.fstat(fd)
        if not stat.S_ISREG(st_fd.st_mode):
            raise VerifiedOpenError(
                f"refusing to open {target.name}: it is not a regular file "
                "(it was replaced after validation?)"
            )
        st_path = os.lstat(target)
        if (st_fd.st_dev, st_fd.st_ino) != (st_path.st_dev, st_path.st_ino):
            raise VerifiedOpenError(
                f"refusing to open {target.name}: the file on disk changed while "
                "being opened (a symlink swapped in after validation?)"
            )
        if allowed_roots is not None:
            roots = [root.resolve() for root in allowed_roots]
            resolved = target.resolve()
            if not any(resolved == root or root in resolved.parents for root in roots):
                raise VerifiedOpenError(
                    f"refusing to open {target.name}: it no longer resolves inside "
                    "the sandbox that validated it"
                )
        return os.fdopen(fd, "rb")
    except BaseException:
        # Every refusal above must not leak the descriptor it is abandoning.
        os.close(fd)
        raise


def system_disallowed_prefixes() -> tuple[Path, ...]:
    """Resolved system directories no sandbox may expose (shared by REST + MCP)."""
    prefixes: list[Path] = [
        Path("/etc"),
        Path("/root"),
        Path("/proc"),
        Path("/sys"),
        Path("/dev"),
        Path("/boot"),
        Path("/var"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/lib"),
        Path("/opt"),
    ]
    if sys.platform == "win32":
        for var in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
            val = os.environ.get(var)
            if val:
                prefixes.append(Path(val))
        for fallback in ("C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)"):
            p = Path(fallback)
            if p.exists():
                prefixes.append(p)
    # Resolve once: compared against the *resolved* request path, and on macOS
    # /etc, /var and /tmp are symlinks into /private.
    return tuple(p.resolve() for p in prefixes)


#: Precomputed so the per-request path check does not rebuild it.
SYSTEM_DISALLOWED_PREFIXES: tuple[Path, ...] = system_disallowed_prefixes()


def restrict_dir_to_owner(path: Path) -> Path:
    """Create ``path`` (parents included) and limit it to its owner."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(BOOK_TEXT_DIR_MODE)
    except OSError as exc:
        logger.debug("Could not restrict %s to owner: %s", path, exc)
    return path


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` if absent and restrict it to its owner; leave an existing
    directory's mode untouched.

    The ledger's parent is the operator-configurable ``db_dir``. Chmod'ing a
    directory UBT did not create (a shared workspace, a team cache, ``/tmp``) is
    a surprising side effect on someone else's path, so only a directory UBT
    itself creates is tightened here. An existing world-readable ``db_dir`` is
    reported by :func:`warn_world_readable` instead of silently rewritten.
    """
    try:
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        if not existed:
            path.chmod(BOOK_TEXT_DIR_MODE)
    except OSError as exc:
        logger.debug("Could not ensure private dir %s: %s", path, exc)
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


def world_readable_files(directory: Path, extra_dirs: Iterable[Path] = ()) -> list[Path]:
    """Files under ``directory`` — plus any ``extra_dirs`` — that group or other users could still read.

    ``extra_dirs`` extends the scan to artifact trees the same caller owns but
    the pre-existing check never walked: before it, the scan covered only
    ``db_dir``, so rendered PDFs, quality reports and the Docling conversion
    cache — all of which hold the manuscript text — were invisible to it.
    The pipeline now passes the trees it wrote. Passing them here keeps one
    definition of "exposed"; the empty default preserves the original
    single-directory behaviour for existing callers.
    """
    found: dict[Path, None] = {}

    def _onerror(err: OSError) -> None:
        # ``os.walk`` without ``onerror`` silently skipped an unreadable
        # directory, so a subtree of world-readable manuscript files could hide
        # below the very report that exists to find them. Surface the gap
        # instead of an all-clear.
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
                # Exposure needs traversal: a directory that grants no search
                # permission (x) to group or other walls off everything below
                # it, no matter the files' own modes -- UBT's own build dirs
                # are born 0700 exactly for this. Walking into them would
                # list files no other user can open.
                try:
                    dir_mode = Path(dirpath).stat().st_mode
                except OSError:
                    continue
                if not dir_mode & 0o011:
                    continue
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

    The scan is recursive (``os.walk`` + ``stat``), so callers must run it off
    the event loop. The warning names each exposed file's real parent — the
    exposed files may live outside ``directory`` (deliverables, caches), and
    pointing the operator at the already-owner-only ``directory`` changed
    nothing.
    """
    if sys.platform == "win32":
        # Windows synthesizes POSIX mode bits (a writable file reports 0666),
        # so the scan would flag every file and the chmod remedy it prints is
        # meaningless there. Privacy on Windows is an NTFS-ACL question; the
        # doctor reports this honestly as a SKIP instead of a standing warn.
        return
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


def sanitize_display_path(path: Path | str | None) -> str | None:
    """Mask user home directory with '~' for external status/info endpoints.

    Avoids leaking host user names and absolute home directory trees while
    preserving the recognizable relative path structure.
    """
    if path is None:
        return None
    try:
        p = Path(path).expanduser().resolve()
        home = Path.home().resolve()
        if p == home:
            return "~"
        if p.is_relative_to(home):
            return f"~/{p.relative_to(home)}"
        return str(p)
    except (ValueError, RuntimeError):
        return str(path)
