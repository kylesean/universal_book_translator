"""Job-level single-writer lock over a ledger file (stdlib-only, cross-platform).

There is no per-block worker lease in the ledger schema, so a job-level
guard is the only thing standing between two processes running the same
``job_id`` simultaneously: both fetch the same pending blocks, both pay the
provider, and their checkpoints overwrite one another. The
orchestrator holds an exclusive, non-blocking lock next to the ledger DB
for the lifetime of a run; read-only tools (status/inspect/MCP fallback
reads) never take it.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from ubt.core.exceptions import LedgerWriterLockConflictError

logger = logging.getLogger(__name__)

#: errno values that mean "the lock is held by someone else", as opposed to a
#: real failure to lock at all. ``fcntl.flock`` raises ``EAGAIN``/``EWOULDBLOCK``
#: on contention; ``msvcrt.locking`` raises ``EACCES`` (and sometimes
#: ``EDEADLK``). Anything else (``EOPNOTSUPP`` on NFS/overlayfs, ``EBADF``,
#: ``EINVAL``) is not contention and must not be reported as "already being
#: written": doing so made the worker requeue the same job forever and
#: ``run_until_idle`` never return.
_LOCK_BUSY_ERRNOS = frozenset(
    {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK, errno.EBUSY}
)

#: Byte offset the Windows lock is taken at, deliberately past the holder pid.
#:
#: ``msvcrt.locking`` locks ``count`` bytes from the handle's *current position*,
#: and the CRT enforces that against other handles in the same process. Locking at
#: byte 0 therefore blocked the one thing the lock file is there for: reading the
#: holder pid (both the test's assertion and ``_read_holder_pid``, which names the
#: competing process in the busy message) raised PermissionError and was swallowed,
#: so a locked job on Windows reported "already being written" without the pid.
#: ``fcntl.flock`` locks the whole file and ignores position, so POSIX is
#: unaffected by where the byte sits.
_LOCK_BYTE_OFFSET = 1024


def _try_exclusive_lock(fd: int) -> None:
    """Take the advisory exclusive lock non-blockingly; raise OSError if busy."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        with suppress(OSError):
            os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


class LedgerWriterLock:
    """Exclusive advisory lock on ``<db_path>.writer.lock`` for one job.

    ``acquire()`` raises :class:`LedgerError` when another process already
    holds it; ``release()`` is idempotent.
    """

    def __init__(self, db_path: Path, job_id: str) -> None:
        self.lock_path = Path(str(db_path) + ".writer.lock")
        self.job_id = job_id
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _try_exclusive_lock(fd)
        except OSError as exc:
            os.close(fd)
            if exc.errno not in _LOCK_BUSY_ERRNOS:
                # Not contention: the filesystem cannot lock at all (NFS /
                # overlayfs EOPNOTSUPP, a bad fd, ...). Report it as a hard
                # failure so the job fails loudly instead of being requeued
                # forever behind a lock that can never be taken.
                raise RuntimeError(
                    f"Cannot acquire writer lock for job '{self.job_id}' at "
                    f"{self.lock_path}: {exc.strerror or exc} (errno {exc.errno}). "
                    "The filesystem does not support advisory file locking, so "
                    "single-writer safety cannot be guaranteed."
                ) from exc
            holder = self._read_holder_pid()
            raise LedgerWriterLockConflictError(
                f"Job {self.job_id} is already being written by another process"
                + (f" (pid {holder})" if holder else "")
                + ": refusing to run it twice — a second writer would pay the "
                "provider again for blocks the first is drafting and overwrite "
                "its checkpoints. Wait for the running job, or pick a distinct "
                "--job-id.",
                details={"job_id": self.job_id, "holder_pid": holder},
            ) from exc
        except ImportError as exc:  # pragma: no cover - exotic platforms
            os.close(fd)
            raise RuntimeError(
                f"Cannot acquire writer lock for job '{self.job_id}': platform file-locking "
                "module is unavailable. Single-writer safety cannot be guaranteed."
            ) from exc
        with suppress(OSError):
            # Back to byte 0: _try_exclusive_lock left the handle positioned at the
            # lock byte, and the pid has to land where _read_holder_pid looks.
            os.lseek(fd, 0, os.SEEK_SET)
            os.truncate(fd, 0)
            os.write(fd, str(os.getpid()).encode("ascii"))
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        with suppress(OSError):
            _unlock(fd)
        with suppress(OSError):
            os.close(fd)

    def __enter__(self) -> LedgerWriterLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()

    def _read_holder_pid(self) -> int | None:
        with suppress(OSError, ValueError):
            text = self.lock_path.read_text(encoding="ascii").strip()
            return int(text) if text else None
        return None
