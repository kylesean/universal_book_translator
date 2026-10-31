"""Rebuild a run's manifest, glossary and events from its ledger.

Three surfaces read a job back out of its ledger and render it: the provider
pipeline (through the run's own objects), the Agent workflow (``finish``) and
the human-edit re-export. The last two carried private copies of the same
readers, and the copies drifted: the Agent manifest whitelisted only eight
artifact keys, so an Agent-rendered job never saw the ``delivery_mode`` /
``delivery_plan`` the ingest stage had persisted for it, and a re-export's
event factory reported "0 completed" where the Agent's reported the real
counts. Reading the ledger is one operation; these helpers keep it one.

The readers are deliberately tolerant about *values* (a missing title is an
empty string, a malformed chapter row is skipped) and strict about
*existence*: a job id with no snapshot row is not a job, and every caller
needs to hear that instead of rendering an empty book.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Terminology
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stage_context import EventFactory
from ubt.core.engine.stages.bible import _BIBLE_CACHE_KEY
from ubt.core.exceptions import UBTError, UnsupportedDocumentFormatError
from ubt.core.ir.models import BookManifest, ChapterMeta
from ubt.core.ir.run_metadata import RunMetadata

#: Ledger metadata a render reads back from the manifest. This is a whitelist,
#: not a pass-through: ``get_job_metadata`` also returns operational rows
#: (runs, session state, request fingerprints) that must not travel into a
#: manifest snapshot.
_ARTIFACT_METADATA_KEYS = frozenset(
    {
        "author",
        "chapter_translations",
        "opf_path",
        "pdf_parser_engine",
        "boilerplate_footers",
        "is_page_slice_epub",
        "page_kinds",
        "typst_version",
        "delivery_mode",
        "delivery_plan",
        "delivery_mode_requested",
        "delivery_mode_effective",
        "delivery_mode_legacy",
        "document_mode",
        "has_scan",
        "page_count",
        "confirmed_font_damage_pages",
        "font_damage_pages",
        "document_style_group",
        "document_style_profile",
    }
)


def _chapters_from(ledger: SQLiteJobLedger, job_id: str) -> list[ChapterMeta]:
    raw = ledger.get_job_metadata_value(job_id, "chapters") or []
    chapters: list[ChapterMeta] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                try:
                    chapters.append(ChapterMeta.model_validate(entry))
                except Exception:  # a malformed chapter row must not sink the render
                    continue
    return chapters


def _run_metadata_from(metadata: Mapping[str, Any]) -> RunMetadata:
    run = RunMetadata()
    for field in RunMetadata.model_fields:
        if field in metadata:
            setattr(run, field, metadata[field])
    return run


def manifest_from_ledger(ledger: SQLiteJobLedger, job_id: str) -> BookManifest:
    """The manifest the original run was built with, read back from the ledger.

    ``init_job_from_manifest`` persisted doc id, source path, languages, title,
    chapters and run metadata, so this is the same manifest — not a
    reconstruction that could disagree with it.

    An Agent job's ``agent_config`` travels as ``metadata["_agent_config"]``:
    the finish path reads the prepare-time settings from there, and no other
    reader looks at the key.
    """
    snapshot = ledger.get_job_snapshot(job_id)
    if snapshot is None:
        raise UBTError(f"no such job: {job_id}")
    metadata = ledger.get_job_metadata(job_id)
    manifest = BookManifest(
        doc_id=str(snapshot["doc_id"]),
        title=str(ledger.get_job_metadata_value(job_id, "title") or ""),
        source_path=str(snapshot["source_path"]),
        source_lang=str(ledger.get_job_metadata_value(job_id, "source_lang") or "en"),
        target_lang=str(snapshot["target_lang"]),
        chapters=_chapters_from(ledger, job_id),
        metadata={key: value for key, value in metadata.items() if key in _ARTIFACT_METADATA_KEYS},
        run=_run_metadata_from(metadata),
    )
    agent_config = metadata.get("agent_config")
    if isinstance(agent_config, dict):
        manifest.metadata["_agent_config"] = agent_config
    return manifest


def terminology_from_ledger(ledger: SQLiteJobLedger, job_id: str) -> Terminology:
    """The glossary the run enforced, from its persisted bible cache.

    Read from the cache rather than re-reading the config's glossary file, so a
    later render enforces exactly the terms the quality gate judged — a fix
    cannot disagree with the flag the reviewer saw.
    """
    payload = ledger.get_job_metadata_value(job_id, _BIBLE_CACHE_KEY)
    if not isinstance(payload, dict):
        return Terminology()
    return Terminology(
        glossary_dicts=[
            dict(entry) for entry in payload.get("glossary_dicts", []) if isinstance(entry, dict)
        ],
        abbreviation_entries=[
            dict(entry)
            for entry in payload.get("abbreviation_entries", [])
            if isinstance(entry, dict)
        ],
    )


def ledger_event_factory(ledger: SQLiteJobLedger, job_id: str) -> EventFactory:
    """A stage event factory that fills the counters from the ledger.

    Used by the render-only paths (Agent finish, re-export). The provider
    orchestrator has its own factory because it additionally folds live spend
    and billing, but everything these paths emit is a count over ledger rows, so
    one factory covers both.
    """

    async def _create(
        event_type: EventType,
        job_id_arg: str,
        ledger_arg: SQLiteJobLedger,
        message: str = "",
        active_block_id: str | None = None,
        artifact_path: str | None = None,
    ) -> TranslationProgressEvent:
        stats = await asyncio.to_thread(ledger_arg.get_job_stats, job_id_arg)
        return TranslationProgressEvent(
            event_type=event_type,
            job_id=job_id_arg,
            total_blocks=int(stats.get("total", 0) or 0),
            skipped_blocks=int(stats.get("skipped", 0) or 0),
            completed_blocks=int(stats.get("completed", 0) or 0),
            drafted_blocks=int(stats.get("drafted", 0) or 0),
            repair_pending_blocks=int(stats.get("repair_pending", 0) or 0),
            repaired_blocks=int(stats.get("repaired", 0) or 0),
            failed_blocks=int(stats.get("failed", 0) or 0),
            needs_human_blocks=int(stats.get("needs_human", 0) or 0),
            blocked_human_blocks=int(stats.get("blocked_human", 0) or 0),
            message=message,
            active_block_id=active_block_id,
            artifact_path=artifact_path,
        )

    return _create


def require_output_suffix(
    adapter: object, output_path: Path, *, subject: str, remedy: str = ""
) -> None:
    """Refuse an output name whose suffix the adapter cannot write.

    The adapter decides the artifact format, so a contradicting ``--output``
    must fail before a single token is spent or a broken file is written (a PDF
    engine into ``.md`` would also leak quarantine ``<mark>`` wrappers into the
    deliverable). A suffix-less name stays allowed: the caller gets the format
    without having said so.

    ``subject`` and ``remedy`` let each surface keep its own actionable advice
    ("--output ...", "re-exporting to ...") while the rule lives in one place.
    """
    allowed: frozenset[str] = getattr(type(adapter), "output_suffixes", frozenset())
    if not allowed:
        return
    suffix = output_path.suffix.lower()
    if not suffix or suffix in allowed:
        return
    wanted = "/".join(sorted(allowed))
    raise UnsupportedDocumentFormatError(
        f"{type(adapter).__name__} writes {wanted}, so {subject} {output_path.name} "
        f"would not be a readable {suffix.lstrip('.')}." + (f" {remedy}" if remedy else "")
    )


__all__ = [
    "ledger_event_factory",
    "manifest_from_ledger",
    "require_output_suffix",
    "terminology_from_ledger",
]
