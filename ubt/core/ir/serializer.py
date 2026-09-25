"""Deterministic file fingerprinting for document adapters.

Note: the former JSON IR serialization functions
(``serialize_ir`` / ``deserialize_ir`` / ``save_ir_atomic`` /
``load_ir_file`` / ``compute_block_hash``) were removed — the production
persistence contract is the SQLite ledger schema, and no production code
serialized DocumentIR to JSON.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

from ubt.core.exceptions import DocumentParseError


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Calculate SHA-256 fingerprint of a file safely without loading it all into memory."""
    hasher = hashlib.sha256()
    try:
        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
    except OSError as err:
        raise DocumentParseError(
            f"Failed to read file for fingerprint: {file_path}",
            details={"error": str(err)},
        ) from err
    return hasher.hexdigest()


@lru_cache(maxsize=8)
def _sha256_for_stat(resolved_path: str, mtime_ns: int, size: int) -> str:
    """Memoized digest keyed on the file identity that matters (see below)."""
    del mtime_ns, size  # part of the cache key only
    return compute_file_sha256(Path(resolved_path))


def compute_file_sha256_cached(file_path: Path) -> str:
    """SHA-256 of *file_path*, memoized on ``(path, mtime_ns, size)``.

    The same PDF is fingerprinted at more than one layer (manifest ``doc_id``
    and the Docling cache key), so a scan would otherwise be read end to end
    twice per job. Keying on the stat tuple means an edited file still
    re-hashes — a stale digest can never be served.
    """
    path = Path(file_path)
    stat = path.stat()
    return _sha256_for_stat(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
