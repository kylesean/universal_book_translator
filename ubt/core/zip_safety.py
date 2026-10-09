"""Bounded reads from untrusted zip archives (decompression-bomb defense).

An EPUB/DOCX is a zip whose central directory may *declare* a small compressed
member that expands to gigabytes. ``zipfile`` decompresses only up to the
declared uncompressed size and then verifies the CRC, so the declared size is
the amount that would actually be materialised — capping it *before* ``read``
is what keeps one crafted member from exhausting memory. A per-member cap alone
is not enough: an archive can hold thousands of members each just under the cap,
so a running total bounds what a single uploaded archive can make the process
hold.

Reads that exceed either cap are refused (``None``) rather than raising, so a
caller can skip the offending member and still deliver the rest of the book.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Refuse to materialise a single member beyond this. This is the memory bound:
#: members are read one at a time, so peak resident size is one member.
MAX_MEMBER_BYTES = 300 * 1024 * 1024
#: Refuse to materialise more than this in total from one archive. This is the
#: *work* bound: a bomb of many just-under-member-cap members (1000 x 250 MB)
#: costs no more memory than one, but unbounded decompression. Set far above any
#: legitimate book — a 512 MB upload of text compresses ~3-5x, so even an
#: illustrated one stays under ~2 GB — while still turning the classic bomb
#: (a few KB that declares petabytes) into a bounded, logged refusal.
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(slots=True)
class ZipReadBudget:
    """A per-archive read budget, shared across every member read.

    One instance per opened ``ZipFile``; ``spent`` accumulates the declared
    sizes of the members actually returned so far.
    """

    max_member_bytes: int = MAX_MEMBER_BYTES
    max_total_bytes: int = MAX_TOTAL_BYTES
    spent: int = field(default=0)

    def allow(self, name: str, size: int) -> bool:
        """Whether a *size*-byte member may be read; charge it when allowed."""
        if size > self.max_member_bytes:
            logger.warning(
                "zip: skipping oversized member %r (%d bytes > %d cap)",
                name,
                size,
                self.max_member_bytes,
            )
            return False
        if self.spent + size > self.max_total_bytes:
            logger.warning(
                "zip: skipping member %r (%d bytes) — archive read total would reach %d "
                "over the %d-byte cap",
                name,
                size,
                self.spent + size,
                self.max_total_bytes,
            )
            return False
        self.spent += size
        return True


def read_member(
    zf: zipfile.ZipFile, name: str, budget: ZipReadBudget | None = None
) -> bytes | None:
    """Read one archive member unless it is missing or bomb-scale.

    Returns ``None`` (after logging) for a missing or over-budget member instead
    of raising, so the caller can skip it and still deliver the rest.
    """
    if budget is None:
        budget = ZipReadBudget()
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    if not budget.allow(name, info.file_size):
        return None
    return zf.read(name)


__all__ = ["MAX_MEMBER_BYTES", "MAX_TOTAL_BYTES", "ZipReadBudget", "read_member"]
