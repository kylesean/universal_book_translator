"""Filesystem detection and SQLite WAL storage configuration.

SQLite in WAL mode depends on POSIX advisory locks and shared memory (mmap)
backed by the `-shm` file. On network/distributed file systems (NFS, SMB, CIFS,
Ceph, GlusterFS, 9P), file locking semantics are either unreliable or fail
completely, leading to database lockups, silent corruption, or I/O errors.

This module provides deterministic filesystem type detection and unified
PRAGMA configuration (WAL, synchronous=NORMAL, busy_timeout=30s) across all
storage-backed components (Ledger, Queue, TM, WriterLock, RateLimiter).
"""

from __future__ import annotations

import logging
import sqlite3
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

DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS: Final[float] = 30.0


def detect_filesystem_type(path: Path) -> tuple[str, bool]:
    """Detect the filesystem type for a path and whether it is a network filesystem.

    Returns:
        tuple of (fs_type_string, is_network_filesystem).
        If detection is unsupported or fails, returns ("unknown", False).
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
            # Find the longest matching mount prefix
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
    """Log a warning if the path resides on a network filesystem.

    Returns:
        The detected network filesystem type if detected, otherwise None.
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


def configure_sqlite_wal(
    conn: sqlite3.Connection,
    timeout_sec: float = DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS,
    *,
    wal_mode: bool = True,
) -> None:
    """Apply standard PRAGMAs to an active SQLite connection.

    Ensures consistent row_factory, busy_timeout (30s), journal_mode=WAL, and synchronous=NORMAL.
    """
    conn.row_factory = sqlite3.Row
    timeout_ms = max(1000, int(timeout_sec * 1000))
    conn.execute(f"PRAGMA busy_timeout = {timeout_ms};")
    if wal_mode:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
