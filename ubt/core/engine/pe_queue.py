"""Human PE (HITL) queue exporters: CSV (default) and XLIFF 2.1.

Segments with status ``NEEDS_HUMAN`` (MQM Major / low-score Minor) or
``BLOCKED_HUMAN`` (MQM Critical unresolved after escalated repair) are exported
for human post-editing. The CSV variant is a flat round-trip table (edit the
``revised_translation`` column and re-import); the XLIFF 2.1 variant targets
enterprise TMS interoperability (Smartcat / Smartling style), one ``<unit>``
per block with the exported target in ``<segment>`` and MQM context in
``<notes>``.

Re-import (``ubt/core/engine/pe_import.py``) detects revisions, updates the
ledger, and feeds the translation memory with ``provenance='human_pe'``.
"""

from __future__ import annotations

import csv
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ubt.core.ir.models import BlockStatus, IRBlock

logger = logging.getLogger(__name__)

PE_QUEUE_STATUSES: frozenset[BlockStatus] = frozenset(
    {BlockStatus.NEEDS_HUMAN, BlockStatus.BLOCKED_HUMAN}
)

VALID_PE_EXPORT_FORMATS: tuple[str, ...] = ("csv", "xliff", "none")

_XLIFF_NS = "urn:oasis:names:tc:xliff:document:2.1"

CSV_COLUMNS: tuple[str, ...] = (
    "block_id",
    "job_id",
    "status",
    "mqm_severity",
    "mtqe_score",
    "source_text",
    "draft_text",
    "current_target",
    "mqm_spans",
    "suggested_correction",
    "revised_translation",
)


@dataclass(frozen=True)
class PEQueueResult:
    """Outcome of a PE queue export."""

    path: Path
    fmt: str
    segment_count: int


def select_pe_queue_blocks(blocks: list[IRBlock]) -> list[IRBlock]:
    """Return blocks queued for human post-editing, in reading order."""
    return [b for b in blocks if b.status in PE_QUEUE_STATUSES]


def _suggested_correction(spans: list[dict[str, Any]]) -> str:
    """Compact human-readable suggestion string derived from MQM spans."""
    parts: list[str] = []
    for span in spans:
        error_type = str(span.get("error_type", ""))
        erroneous = str(span.get("erroneous_text", ""))
        expected = str(span.get("expected", ""))
        if expected:
            parts.append(f"{error_type}: '{erroneous}' -> '{expected}'")
        else:
            parts.append(f"{error_type}: {span.get('reason', '')}")
    return " | ".join(parts)


def export_pe_queue_csv(
    blocks: list[IRBlock],
    output_path: Path,
    job_id: str,
) -> PEQueueResult:
    """Export the PE queue as a flat CSV round-trip table.

    Every row carries ``job_id``: block ids are structural names that repeat
    across the ledgers of the same book, so the re-import refuses files whose
    binding does not match the ledger it is applied to.
    """
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for b in blocks:
            writer.writerow(
                {
                    "block_id": b.id,
                    "job_id": job_id,
                    "status": b.status.value,
                    "mqm_severity": b.mqm_severity or "",
                    "mtqe_score": "" if b.mtqe_score is None else f"{b.mtqe_score:.4f}",
                    "source_text": b.source_text,
                    "draft_text": b.draft_text or "",
                    "current_target": b.target_text or "",
                    "mqm_spans": json.dumps(b.mqm_spans, ensure_ascii=False),
                    "suggested_correction": _suggested_correction(b.mqm_spans),
                    # Human fills this column in; empty rows are skipped on import.
                    "revised_translation": "",
                }
            )
    return PEQueueResult(path=output_path, fmt="csv", segment_count=len(blocks))


def export_pe_queue_xliff(
    blocks: list[IRBlock],
    output_path: Path,
    source_lang: str,
    target_lang: str,
    job_id: str,
    original_name: str = "ubt-job",
) -> PEQueueResult:
    """Export the PE queue as an XLIFF 2.1 document (enterprise TMS interchange)."""
    xliff = ET.Element(
        "xliff",
        {
            "xmlns": _XLIFF_NS,
            "version": "2.1",
            "srcLang": source_lang,
            "trgLang": target_lang,
        },
    )
    file_el = ET.SubElement(xliff, "file", {"id": "f1", "original": original_name})
    for idx, b in enumerate(blocks, start=1):
        unit = ET.SubElement(file_el, "unit", {"id": b.id})
        notes = ET.SubElement(unit, "notes")
        ET.SubElement(notes, "note", {"category": "ubt-job-id"}).text = job_id
        ET.SubElement(notes, "note", {"category": "ubt-status"}).text = b.status.value
        ET.SubElement(notes, "note", {"category": "ubt-mqm-severity"}).text = b.mqm_severity or ""
        ET.SubElement(notes, "note", {"category": "ubt-mqm-spans"}).text = json.dumps(
            b.mqm_spans, ensure_ascii=False
        )
        ET.SubElement(notes, "note", {"category": "ubt-qe-score"}).text = (
            "" if b.mtqe_score is None else f"{b.mtqe_score:.4f}"
        )

        segment = ET.SubElement(unit, "segment", {"id": f"s{idx}"})
        # XLIFF 2.1 segment state: 'initial' = untouched (Critical quarantine),
        # 'translated' = machine draft awaiting review (Major / low-score).
        segment.set("state", "initial" if b.status == BlockStatus.BLOCKED_HUMAN else "translated")
        ET.SubElement(segment, "source").text = b.source_text
        ET.SubElement(segment, "target").text = b.target_text or ""

    tree = ET.ElementTree(xliff)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return PEQueueResult(path=output_path, fmt="xliff", segment_count=len(blocks))


def export_pe_queue(
    blocks: list[IRBlock],
    output_stem_path: Path,
    fmt: str,
    source_lang: str,
    target_lang: str,
    job_id: str,
) -> PEQueueResult | None:
    """Export the PE queue in the configured format next to the rendered output.

    ``fmt`` accepts ``csv`` | ``xliff`` | ``none`` (unrecognized values fall
    back to ``csv`` with a warning). Returns ``None`` when the queue is empty
    or the format is ``none``.
    """
    normalized = (fmt or "csv").lower()
    if normalized not in VALID_PE_EXPORT_FORMATS:
        logger.warning("Unknown pe_export_format '%s'; falling back to 'csv'", fmt)
        normalized = "csv"
    if normalized == "none":
        return None

    queue_blocks = select_pe_queue_blocks(blocks)
    if not queue_blocks:
        return None

    if normalized == "csv":
        path = output_stem_path.with_name(f"{output_stem_path.stem}_pe_queue.csv")
        return export_pe_queue_csv(queue_blocks, path, job_id=job_id)
    path = output_stem_path.with_name(f"{output_stem_path.stem}_pe_queue.xliff")
    return export_pe_queue_xliff(
        queue_blocks,
        path,
        source_lang=source_lang,
        target_lang=target_lang,
        job_id=job_id,
        original_name=output_stem_path.name,
    )
