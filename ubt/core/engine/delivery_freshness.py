"""Whether a job's delivered file still matches the text it was rendered from.

A human edit lands in the ledger; the delivered PDF/EPUB/HTML/XLIFF are files
written by an earlier export. Nothing forces the two back together, so a job
can sit in a state where the ledger says one thing and every downloadable file
says another — and the download endpoint happily serves the file.

The console used to track this in a React `useState` flag. That made the answer
depend on whether the operator's tab had been open when the edit happened:
reload the page, open the job on another machine, or come back tomorrow, and
the "re-render pending" marker was gone while the staleness was not. The
download then handed a customer the pre-edit draft with no warning at all.

This module is the single owner of that question, and it answers it from the
ledger rather than from any surface's memory. Two values are involved:

``rendered_text_digest``
    A digest of the text the delivered file was last built from. The export
    stage writes it after a successful render.

``text_digest``
    A digest of the text the ledger holds *now*, recomputed on demand.

A job is stale when the two differ. Living in metadata means the answer
survives a restart, is shared by every surface (CLI, REST, MCP), and cannot be
forgotten by a client.

Why a digest rather than a timestamp: a re-render that fails halfway must not
look fresh, and two saves that revert each other must not look stale. Both fall
out of comparing content. Timestamps also break when the machine's clock jumps,
and a resumed run rewrites `updated_at` on rows that did not change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ubt.core.engine.ledger import SQLiteJobLedger

#: Job-metadata key holding the digest of the text the last export rendered.
RENDERED_TEXT_DIGEST_KEY = "rendered_text_digest"

#: Text fields that reach a delivered file. A change to any of them makes the
#: artifact stale; a change to a score or a flag does not (those go to the
#: quality report, which a re-export rebuilds anyway).
_ARTIFACT_TEXT_FIELDS = ("target_text", "status")


def text_digest(blocks: Sequence[object]) -> str:
    """Digest of the text a render would emit for ``blocks``.

    Ordered by block id so two reads of the same ledger agree regardless of the
    order SQLite returns rows in, and so an unrelated reordering (a re-ingest
    that renumbers) is a real change rather than noise.

    ``status`` is included because it decides *which* text ships: a block that
    moves to NEEDS_HUMAN after a failed repair renders its source instead of
    its draft, so a status flip changes the output even when ``target_text``
    does not.
    """
    hasher = hashlib.sha256()
    for block in sorted(blocks, key=lambda b: str(getattr(b, "id", ""))):
        payload = json.dumps(
            [
                str(getattr(block, "id", "")),
                *[getattr(block, f, None) for f in _ARTIFACT_TEXT_FIELDS],
            ],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        hasher.update(payload.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def record_rendered_digest(ledger: SQLiteJobLedger, job_id: str, digest: str | None = None) -> str:
    """Record the current ledger text as rendered. Returns the digest written.

    Called by the export stage *after* the artifact is on disk, so a render that
    raises leaves the previous digest in place and the job still reads stale.

    A normal run does **not** come through here: it passes the digest to
    :meth:`SQLiteJobLedger.finalize_job` so the status flip and the digest share
    one transaction. This function is the re-export path, which deliberately
    leaves the job's terminal status alone.
    """
    if digest is None:
        digest = text_digest(ledger.get_all_blocks(job_id))
    ledger.set_job_metadata_value(job_id, RENDERED_TEXT_DIGEST_KEY, digest)
    return digest


def rendered_digest_for(ledger: SQLiteJobLedger, job_id: str) -> str:
    """The digest of the text the ledger would render right now."""
    return text_digest(ledger.get_all_blocks(job_id))


def is_delivery_stale(ledger: SQLiteJobLedger, job_id: str) -> bool:
    """True when the delivered files predate the ledger's current text.

    A ledger with no recorded digest is *not* reported stale. Jobs that finished
    before this marker existed have no baseline, and calling them stale would
    block every download of work that is in fact fine — the marker only earns
    its keep going forward, once a render has recorded one.
    """
    rendered = ledger.get_job_metadata_value(job_id, RENDERED_TEXT_DIGEST_KEY)
    if not rendered:
        return False
    return str(rendered) != text_digest(ledger.get_all_blocks(job_id))
