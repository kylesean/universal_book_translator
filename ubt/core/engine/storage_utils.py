"""Filesystem detection for SQLite storage paths.

SQLite in WAL mode depends on POSIX advisory locks and shared memory (mmap)
backed by the `-shm` file. On network/distributed file systems (NFS, SMB, CIFS,
Ceph, GlusterFS, 9P), file locking semantics are either unreliable or fail
completely, leading to database lockups, silent corruption, or I/O errors.

This module only detects and warns. There is no shared PRAGMA helper: every
storage-backed component (Ledger, Queue, TM, WriterLock, RateLimiter) applies
its own PRAGMAs where it opens its own connection, so a central one would be a
second, unused source of truth.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

#: Known network and distributed filesystem types prone to POSIX mmap / lock failures.
NETWORK_FILESYSTEMS: Final[frozenset[str]] = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb",
        "smb3",
        "smbfs",
        "9p",
        "glusterfs",
        "ceph",
        "fuse.sshfs",
        "afpfs",
        "gpfs",
        "lustre",
    }
)


def detect_filesystem_type(path: Path) -> tuple[str, bool]:
    """The filesystem type backing ``path``, and whether it is a network mount.

    Detection that is unsupported or fails folds to ``("unknown", False)``: an
    unreadable ``/proc/mounts`` never blocks a run on a local disk.
    """
    resolved = path.resolve() if path.exists() else path.parent.resolve()

    if sys.platform.startswith("linux"):
        try:
            mounts: list[tuple[str, str]] = []
            with Path("/proc/mounts").open(encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        mounts.append((parts[1], parts[2].lower()))
            resolved_str = str(resolved)
            matching = [
                m
                for m in mounts
                if resolved_str == m[0] or resolved_str.startswith(m[0].rstrip("/") + "/")
            ]
            if matching:
                best = max(matching, key=lambda m: len(m[0]))
                fs_type = best[1]
                return fs_type, fs_type in NETWORK_FILESYSTEMS
        except (OSError, ValueError) as exc:
            logger.debug("Failed to read /proc/mounts for %s: %s", resolved, exc)

    return "unknown", False


def warn_if_network_filesystem(path: Path, purpose: str = "SQLite WAL database") -> str | None:
    """Warn once when a database path sits on a network filesystem.

    The return value exists for callers that want to branch on it (a health
    check records the type); the storage constructors ignore it and only want
    the log line.
    """
    fs_type, is_net = detect_filesystem_type(path)
    if is_net:
        logger.warning(
            "%s at '%s' is located on network filesystem '%s'. "
            "SQLite WAL relies on local POSIX advisory locking and shared memory (mmap); "
            "running on network storage risks database corruption, lockups, and I/O errors. "
            "Mounting on a local SSD or block device is strongly recommended.",
            purpose,
            path,
            fs_type,
        )
        return fs_type
    return None
