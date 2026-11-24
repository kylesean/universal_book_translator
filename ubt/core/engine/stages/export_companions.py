"""Companion artifact writers and delivery text projection for export stage."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.core.atomic import atomic_write_text
from ubt.core.engine.artifact_services import check_artifact as _default_check_artifact
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.job_options import CompanionArtifactResult

if TYPE_CHECKING:
    from ubt.adapters.pdf.artifact_text import ArtifactReport
    from ubt.core.content.contract import ReconciliationReport

logger = logging.getLogger(__name__)


def _delivery_translations(blocks: Sequence[IRBlock]) -> dict[str, str]:
    """The ``element id -> placed target`` map the delivered blocks describe.

    Only a translation the renderer actually placed is in the map: a block kept
    in the source has none, so the views show its source text instead.

    The value is :func:`ubt.render.delivered_text.delivered_text`'s, not
    ``target_text`` verbatim. The companion views treat a map entry as an
    override that wins over their own ``delivered_text`` fallback (an
    interactive PE edit is plain text, so that ordering is right for *them*),
    which made this map the one path where the pipeline's
    ``<mark class="ubt-failed-draft" title="Repair error: …">`` wrapper still
    reached the reader: the PDF showed the clean sentence while the HTML and
    EPUB companions shipped the wrapper class and the provider's exception text
    escaped onto the page. Stripping here keeps the PDF and the views agreeing.
    """
    from ubt.core.content.adapt import kept_in_source
    from ubt.core.validators.inline_tags import strip_inline_tags
    from ubt.render.delivered_text import delivered_text

    placed: dict[str, str] = {}
    for block in blocks:
        if not (block.target_text or "").strip():
            continue
        if block.block_type in (BlockType.FORMULA, BlockType.TABLE, BlockType.IMAGE):
            continue
        if kept_in_source(block):
            continue
        placed[block.id] = strip_inline_tags(delivered_text(block)).strip()
    return placed


def _write_spill_appendix(
    ctx: StageContext,
    rendered_path: Path,
    blocks: list[IRBlock],
    contract: ReconciliationReport,
) -> Path | None:
    """Write an overflow/spill appendix companion when blocks could not fit into layout boxes."""
    from ubt.core.content.contract import _is_space_failure
    from ubt.core.job_options import companion_path
    from ubt.core.target_labels import appendix_labels

    spill_violations = [v for v in contract.violations if _is_space_failure(v.detail)]
    if not spill_violations:
        return None
    try:
        by_id = {b.id: b for b in blocks}
        labels = appendix_labels(ctx.target_lang)
        items: list[dict[str, Any]] = []
        md_lines: list[str] = [
            labels.spill_title,
            "",
            f"- **Job ID**: `{ctx.job_id}`",
            f"- **Document**: `{rendered_path.name}`",
            f"- **{labels.spill_count}**: {len(spill_violations)}",
            "",
            labels.spill_intro,
            "",
            "| " + " | ".join(labels.spill_headers) + " |",
            "| :--- | :---: | :--- | :--- | :--- |",
        ]
        for v in spill_violations:
            block = by_id.get(v.node_id)
            page = block.bbox.page if block and block.bbox else "—"
            src = (block.source_text or "") if block else ""
            tgt = (block.target_text or "") if block else ""
            items.append(
                {
                    "block_id": v.node_id,
                    "page": page,
                    "reason": v.detail,
                    "source_text": src,
                    "target_text": tgt,
                }
            )
            src_escaped = src.replace("\n", " ").replace("|", "\\|")[:80]
            tgt_escaped = tgt.replace("\n", " ").replace("|", "\\|")[:80]
            md_lines.append(
                f"| `{v.node_id}` | P{page} | `{v.detail}` | {src_escaped} | {tgt_escaped} |"
            )

        json_path = companion_path(rendered_path, ".spill_appendix.json")
        atomic_write_text(
            json_path,
            json.dumps({"job_id": ctx.job_id, "spills": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_path = companion_path(rendered_path, ".spill_appendix.md")
        atomic_write_text(md_path, "\n".join(md_lines), encoding="utf-8")
        logger.info("Spill appendix generated: %s (%d items)", md_path.name, len(items))
        return md_path
    except Exception as exc:
        logger.warning("Spill appendix generation skipped for job %s: %s", ctx.job_id, exc)
        return None


def _write_table_appendix(
    ctx: StageContext,
    rendered_path: Path,
    blocks: list[IRBlock],
) -> Path | None:
    """Write translated table appendix companion (.md / .json) and append pages to PDF when appropriate.

    The renderer re-typesets every table inline
    (``ubt/render/typst/document_renderer.py``), so appending the same tables
    again would deliver each complex table twice. This remains only for the
    ``--table-render-mode`` companion exports; no PDF page is appended.
    """
    from ubt.core.job_options import companion_path
    from ubt.core.target_labels import appendix_labels
    from ubt.render.table_compiler import probe_table_complexity

    table_mode = getattr(ctx.config, "table_render_mode", "auto")
    if table_mode == "skip":
        return None

    table_blocks = [
        b
        for b in blocks
        if b.block_type is BlockType.TABLE
        and not b.skip_translate
        and bool((b.target_text or "").strip())
    ]
    if not table_blocks:
        return None

    appendix_tables: list[tuple[int, str, str]] = []
    for b in table_blocks:
        target = (b.target_text or "").strip()
        page = b.bbox.page if b.bbox and b.bbox.page > 0 else 1
        label = b.id
        if table_mode == "appendix":
            appendix_tables.append((page, label, target))
        elif table_mode == "auto":
            comp = probe_table_complexity(target, b.bbox)
            if comp.is_complex:
                appendix_tables.append((page, label, target))

    if not appendix_tables:
        return None

    try:
        items = [
            {
                "block_id": label,
                "page": page,
                "target_text": target,
            }
            for page, label, target in appendix_tables
        ]
        json_path = companion_path(rendered_path, ".table_appendix.json")
        atomic_write_text(
            json_path,
            json.dumps({"job_id": ctx.job_id, "tables": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        labels = appendix_labels(ctx.target_lang)
        md_lines: list[str] = [
            labels.table_title,
            "",
            f"- **Job ID**: `{ctx.job_id}`",
            f"- **Document**: `{rendered_path.name}`",
            f"- **{labels.table_count}**: {len(appendix_tables)}",
            "",
            labels.table_intro,
            "",
        ]
        for page, label, target in appendix_tables:
            md_lines.append(labels.table_page.format(page=page, label=label))
            md_lines.append("")
            md_lines.append(target)
            md_lines.append("")

        md_path = companion_path(rendered_path, ".table_appendix.md")
        atomic_write_text(md_path, "\n".join(md_lines), encoding="utf-8")
        logger.info("Table appendix companion generated: %s (%d items)", md_path.name, len(items))
        return md_path

    except Exception as exc:
        logger.warning("Table appendix generation skipped for job %s: %s", ctx.job_id, exc)
        return None


def _write_xliff_companion(
    ctx: StageContext, rendered_path: Path, blocks: list[IRBlock]
) -> CompanionArtifactResult:
    """Write a bilingual XLIFF beside the artifact for review (best-effort).

    Read-only by construction: it serializes the *delivered* blocks -- each
    source with its protected spans masked into inline codes, and the target that
    actually shipped -- so an editor opens source and target side by side in any
    CAT tool or XLIFF viewer. It never touches the rendered artifact, so any
    failure is logged and skipped rather than allowed to sink the delivery.
    """
    if not ctx.config.emit_xliff_companion:
        return CompanionArtifactResult(
            kind="xliff",
            status="skipped",
            reason="config.emit_xliff_companion is disabled",
        )
    t0 = time.perf_counter()
    try:
        from ubt.core.engine.document_services import export_xliff
        from ubt.core.job_options import companion_path

        res = export_xliff(
            blocks,
            src_lang=ctx.source_lang or "en",
            trg_lang=ctx.target_lang,
            original=Path(ctx.input_path).name,
            job_id=ctx.job_id,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        if not res:
            return CompanionArtifactResult(
                kind="xliff",
                status="skipped",
                duration_ms=duration_ms,
                reason="no extractable text segments in delivered blocks",
            )
        xml, segment_count = res
        path = companion_path(rendered_path, ".xliff")
        atomic_write_text(path, xml, encoding="utf-8")
        size_bytes = path.stat().st_size if path.exists() else None
        logger.info(
            "XLIFF companion for job %s: %s (%d segment(s), %s bytes)",
            ctx.job_id,
            path.name,
            segment_count,
            size_bytes,
        )
        return CompanionArtifactResult(
            kind="xliff",
            status="generated",
            path=path,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
        )
    except Exception as exc:  # a companion must never sink the delivery
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.warning("XLIFF companion skipped for job %s: %s", ctx.job_id, exc, exc_info=True)
        return CompanionArtifactResult(
            kind="xliff",
            status="failed",
            duration_ms=duration_ms,
            error=str(exc),
        )


def _write_html_view(
    ctx: StageContext,
    rendered_path: Path,
    translations: dict[str, str],
    blocks: list[IRBlock],
) -> CompanionArtifactResult:
    """Write a semantic HTML view of the delivery beside the artifact.

    A *view*, so it is read-only and best-effort: a failure is logged and
    skipped, never allowed to sink the delivery.
    """
    if not ctx.config.emit_html_companion:
        return CompanionArtifactResult(
            kind="html",
            status="skipped",
            reason="config.emit_html_companion is disabled",
        )
    t0 = time.perf_counter()
    try:
        from ubt.core.job_options import companion_path
        from ubt.layout.theme import direction_for
        from ubt.render.html_view import compose_html

        path = companion_path(rendered_path, ".html")
        compose_html(
            blocks,
            translations,
            path,
            lang=ctx.target_lang or "en",
            direction=direction_for(ctx.target_lang or "en").value,
            # Bilingual by default: the companion's whole point is the
            # three-mode reading console (target-only / stacked / side-by-side),
            # and without this the toolbar and paired rows were never emitted.
            bilingual=True,
        )
        size_bytes = path.stat().st_size if path.exists() else None
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("HTML view for job %s: %s (%s bytes)", ctx.job_id, path.name, size_bytes)
        return CompanionArtifactResult(
            kind="html",
            status="generated",
            path=path,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
        )
    except Exception as exc:  # a view must never sink the delivery
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.warning("HTML view skipped for job %s: %s", ctx.job_id, exc, exc_info=True)
        return CompanionArtifactResult(
            kind="html",
            status="failed",
            duration_ms=duration_ms,
            error=str(exc),
        )


def _write_epub_view(
    ctx: StageContext,
    rendered_path: Path,
    translations: dict[str, str],
    blocks: list[IRBlock],
) -> CompanionArtifactResult:
    """Write an EPUB 3 view of the delivery beside the artifact."""
    if not ctx.config.emit_epub_companion:
        return CompanionArtifactResult(
            kind="epub",
            status="skipped",
            reason="config.emit_epub_companion is disabled",
        )
    t0 = time.perf_counter()
    try:
        from ubt.core.job_options import companion_path
        from ubt.layout.theme import direction_for
        from ubt.render.epub_view import compose_epub

        path = companion_path(rendered_path, ".epub")
        compose_epub(
            blocks,
            translations,
            path,
            title=str(getattr(ctx.manifest, "title", "") or "UBT translation"),
            lang=ctx.target_lang or "en",
            direction=direction_for(ctx.target_lang or "en").value,
            doc_id=str(getattr(ctx.manifest, "doc_id", "") or ""),
        )
        size_bytes = path.stat().st_size if path.exists() else None
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("EPUB view for job %s: %s (%s bytes)", ctx.job_id, path.name, size_bytes)
        return CompanionArtifactResult(
            kind="epub",
            status="generated",
            path=path,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
        )
    except Exception as exc:  # a view must never sink the delivery
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.warning("EPUB view skipped for job %s: %s", ctx.job_id, exc, exc_info=True)
        return CompanionArtifactResult(
            kind="epub",
            status="failed",
            duration_ms=duration_ms,
            error=str(exc),
        )


def _artifact_payload(report: ArtifactReport) -> dict[str, Any]:
    """Serialize an :class:`~ubt.adapters.pdf.artifact_text.ArtifactReport` for the sidecar."""
    return {
        "total": report.total,
        "missing": [check.element_id for check in report.missing],
        "summary": report.summary_line(),
    }


def _write_artifact_check(
    ctx: StageContext,
    rendered_path: Path,
    translations: Mapping[str, str],
    blocks: list[IRBlock],
) -> CompanionArtifactResult:
    """Write the artifact check beside the artifact.

    Records whether the delivered artifact actually carries each text block's
    realization. Read-only and off-loop; a failure is logged and skipped and can
    never sink the delivery.
    """
    if not ctx.config.emit_attestation_shadow:
        return CompanionArtifactResult(
            kind="attestation",
            status="skipped",
            reason="config.emit_attestation_shadow is disabled",
        )
    if rendered_path.suffix.lower() != ".pdf":
        return CompanionArtifactResult(
            kind="attestation",
            status="skipped",
            reason=f"Attestation artifact check only applies to PDF deliverables, got {rendered_path.suffix}",
        )
    t0 = time.perf_counter()
    try:
        from ubt.core.job_options import companion_path

        export_mod = sys.modules.get("ubt.core.engine.stages.export")
        check_fn = getattr(export_mod, "check_artifact", _default_check_artifact)
        artifact = check_fn(blocks, translations, rendered_path)
        payload: dict[str, Any] = {"artifact": _artifact_payload(artifact)}
        if not artifact.passed:
            logger.warning(
                "Artifact check for job %s: %s (missing: %s)",
                ctx.job_id,
                artifact.summary_line(),
                ", ".join(check.element_id for check in artifact.missing[:5]),
            )
        path = companion_path(rendered_path, "_attestations.json")
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        size_bytes = path.stat().st_size if path.exists() else None
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        return CompanionArtifactResult(
            kind="attestation",
            status="generated",
            path=path,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
        )
    except Exception as exc:  # an audit companion must never sink the delivery
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.warning("Artifact check skipped for job %s: %s", ctx.job_id, exc, exc_info=True)
        return CompanionArtifactResult(
            kind="attestation",
            status="failed",
            duration_ms=duration_ms,
            error=str(exc),
        )
