"""Test-only document seeding helper.

Production used to expose ``DocumentIR`` + ``LedgerJobsMixin.init_job`` purely
as a convenient way to seed a ledger with job metadata and a block set; the
pipeline itself initializes jobs from a ``BookManifest``
(``init_job_from_manifest``). Both were removed to stop carrying an unused
production data path, so tests seed through :func:`seed_job` and the light
:class:`SeedDoc` holder instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ubt.core.engine.ledger_base import _upsert_blocks_batch
from ubt.core.ir.models import FlowID, IRBlock


@dataclass
class SeedDoc:
    """Minimal stand-in for the removed ``DocumentIR`` (doc_id + blocks)."""

    doc_id: str
    source_path: str
    format_type: str = "md"
    metadata: dict[str, Any] = field(default_factory=dict)
    blocks: list[IRBlock] = field(default_factory=list)

    def get_blocks_by_flow(self, flow_id: FlowID) -> list[IRBlock]:
        return [b for b in self.blocks if b.flow_id == flow_id]

    @property
    def total_blocks(self) -> int:
        return len(self.blocks)


def seed_job(
    ledger: Any,
    job_id: str,
    doc: SeedDoc,
    target_lang: str = "zh",
) -> None:
    """Insert ``job_meta`` + the doc's blocks, mirroring the removed ``init_job``."""
    with ledger._get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        cursor = conn.cursor()
        cursor.execute("SELECT job_id FROM job_meta WHERE job_id = ?", (job_id,))
        if cursor.fetchone() is not None:
            conn.execute("COMMIT;")
            return
        cursor.execute(
            """
            INSERT INTO job_meta (
                job_id, doc_id, source_path, target_lang, total_blocks, status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                doc.doc_id,
                doc.source_path,
                target_lang,
                len(doc.blocks),
                "initialized",
                json.dumps(doc.metadata, ensure_ascii=False),
            ),
        )
        _upsert_blocks_batch(cursor, job_id, doc.blocks)
        conn.execute("COMMIT;")
    ledger._mark_blocks_changed()
