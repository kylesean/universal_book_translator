"""Stage 1: Manifest Registration & Streaming Chapter Preprocessing."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Sequence

from ubt.core.cleaners.lnds_pruner import LNDSPageCleaner, dedup_duplicate_blocks
from ubt.core.cleaners.skip_rules import classify_skip
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stage_context import StageContext
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock, LayoutRole
from ubt.core.ir.serializer import compute_file_sha256
from ubt.core.policy.verdict import apply_verdict, judge_block

logger = logging.getLogger(__name__)

# Recorded as the source fingerprint when hashing the input failed at ingest.
# A real fingerprint enables change-detection on resume; this sentinel means
# "we could not compare", so resume must neither clear blocks (a fully
# translated book would be destroyed and re-billed) nor raise a mismatch.
_FINGERPRINT_UNAVAILABLE = "fingerprint_unavailable"

_BIB_SECTION_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\s*[\.\-–—:]?\s*)?"
    r"(?:references|bibliography|works\s+cited|literature\s+cited|参考(?:文献|资料)|引用文献)\b",
    re.IGNORECASE,
)


def update_bibliography_section_state(
    blocks: Sequence[IRBlock],
    idx: int,
    in_bibliography: bool,
) -> bool:
    """Update the bibliography section state machine at block index ``idx``.

    - A bibliography heading (``References``, ``7. References``, ``Bibliography``, etc.)
      opens ``in_bibliography=True`` when at least one of the next 4 blocks is a
      standalone bibliography entry (preventing Table-of-Contents lines from firing).
    - Once ``in_bibliography=True`` is active, an internal sub-heading (e.g.
      ``Primary papers and current implementation sources`` immediately under
      ``REFERENCES``) preserves ``in_bibliography=True`` as long as at least one of
      the next 4 blocks after that sub-heading is still a bibliography entry.
    - A real post-bibliography section heading (e.g. ``Appendix A``) whose upcoming
      blocks are normal prose cleanly resets ``in_bibliography=False``.
    """
    b = blocks[idx]
    if b.block_type is not BlockType.HEADING:
        return in_bibliography
    h_txt = (b.source_text or "").strip()
    next_slice = blocks[idx + 1 : idx + 5]
    has_upcoming_bib = any(classify_skip(nb.source_text or "") is not None for nb in next_slice)
    if _BIB_SECTION_HEADING_RE.match(h_txt):
        return has_upcoming_bib
    return bool(in_bibliography and has_upcoming_bib)


def _guard_chapter_window(
    ledger: SQLiteJobLedger,
    job_id: str,
    start_chapter: int,
    max_chapters: int | None,
) -> None:
    """Record this job's chapter window, or refuse a resume that asks for another.

    Derived job ids already namespace the window, so this mainly protects an
    explicit ``--job-id``. It matters because a windowed ledger holds no blocks
    outside its range: resuming it as a full-book run drafts nothing new, then
    exports only the windowed subset while ``finalize_job`` still reports
    "completed" — a silent truncation with a success status.
    """
    requested: list[int | None] = [start_chapter, max_chapters]
    stored = ledger.get_job_metadata_value(job_id, "chapter_window")
    if stored is None:
        ledger.set_job_metadata_value(job_id, "chapter_window", requested)
        return
    # Stored rows or external ledger inputs may hold an unexpected shape; only a
    # well-formed list is comparable, and anything else is treated as unknown
    # rather than as a mismatch that would block a legitimate resume.
    if not isinstance(stored, list):
        return
    if list(stored) != requested:
        raise DocumentParseError(
            f"Job {job_id} was ingested for chapter window {stored} but is being "
            f"resumed with {requested}. Its ledger holds no blocks outside that "
            "range, so the exported document would be silently truncated. Re-run "
            "with the original window, use a different --job-id, or add --fresh "
            "to discard the stale blocks and re-ingest (--fresh restarts the job, "
            "so a different window is allowed with it)."
        )


def _guard_selected_pages(
    ledger: SQLiteJobLedger,
    job_id: str,
    selected_pages: set[int] | None,
) -> None:
    """Record this job's page selection, or refuse a resume that asks for another.

    Mirrors :func:`_guard_chapter_window` for the ``--pages`` dimension: a
    resumed run skips parsing whenever the ledger already holds
    blocks, so changing the selection would silently export the OLD selection
    while ``finalize_job`` still reports "completed". A whole-document run is
    recorded as the EMPTY LIST, never ``None``: ``None`` is what
    ``get_job_metadata_value`` returns while the key is absent, so using it as
    the recorded value would make a recorded whole-book run indistinguishable
    from a first call and the job would refuse to ingest itself. ``[]`` is
    unambiguous because ``parse_page_ranges`` never yields an empty selection.
    """
    requested: list[int] = sorted(selected_pages) if selected_pages else []
    stored = ledger.get_job_metadata_value(job_id, "selected_pages")
    if stored is None:
        ledger.set_job_metadata_value(job_id, "selected_pages", requested)
        return
    # Stored rows or external ledger inputs may hold an unexpected shape; only
    # a list is comparable, anything else counts as unknown and must not block
    # a legitimate resume.
    if not isinstance(stored, list):
        return
    if stored != requested:
        raise DocumentParseError(
            f"Job {job_id} was ingested with the page selection {stored} but is "
            f"being resumed with {requested}. Its ledger holds no blocks outside "
            "that selection, so the exported document would be silently "
            "truncated. Re-run with the original selection, use a different "
            "--job-id, or add --fresh to discard the stale blocks and re-ingest "
            "(--fresh restarts the job, so a different selection is allowed "
            "with it)."
        )


#: One metadata key holding the identity a *derived* job id already namespaces
#: (genre profile / engine knobs / rehearsal mode). An explicit ``--job-id``
#: skips that namespacing, so the guard below records and compares it instead.
_RUN_IDENTITY_KEY = "run_identity"


def _run_identity(profile_name: str, engine_sig: str, mock_run: bool) -> dict[str, object]:
    return {"profile_name": profile_name, "engine_signature": engine_sig, "mock_run": mock_run}


def _record_run_identity(
    ledger: SQLiteJobLedger,
    job_id: str,
    profile_name: str,
    engine_sig: str,
    mock_run: bool,
) -> None:
    """Overwrite the recorded identity (used by the ``--fresh`` restart path)."""
    ledger.set_job_metadata_value(
        job_id, _RUN_IDENTITY_KEY, _run_identity(profile_name, engine_sig, mock_run)
    )


def _guard_run_identity(
    ledger: SQLiteJobLedger,
    job_id: str,
    profile_name: str,
    engine_sig: str,
    mock_run: bool,
) -> None:
    """Record this job's profile/engine/rehearsal identity, or refuse a resume
    that asks for another.

    Derived job ids already namespace all three, so this mainly protects an
    explicit ``--job-id`` (mirrors :func:`_guard_chapter_window`). Without it a
    resume under a different ``--profile`` keeps the earlier run's translations
    in place, and a rehearsal (``--dry-run``) ledger resumed as a real run ships
    the echo text as the translation. A ledger written before this guard has no
    key: the current values are recorded rather than blocked, so older jobs stay
    resumable (the same forward-compatible rule as the window guard).
    """
    requested = _run_identity(profile_name, engine_sig, mock_run)
    stored = ledger.get_job_metadata_value(job_id, _RUN_IDENTITY_KEY)
    if stored is None:
        ledger.set_job_metadata_value(job_id, _RUN_IDENTITY_KEY, requested)
        return
    # Stored rows or external ledger inputs may hold an unexpected shape; only a
    # dict is comparable, anything else counts as unknown rather than a mismatch
    # that would block a legitimate resume.
    if not isinstance(stored, dict):
        return
    if stored != requested:
        raise DocumentParseError(
            f"Job {job_id} was created with {stored} but is being resumed with "
            f"{requested}. A different profile, engine preset, or rehearsal mode "
            "would mix its output into this ledger. Use a different --job-id, "
            "re-run with the original settings, or add --fresh to restart the job."
        )


def _reset_job_for_fresh_run(
    ledger: SQLiteJobLedger,
    job_id: str,
    start_chapter: int,
    max_chapters: int | None,
    selected_pages: set[int] | None,
) -> None:
    """Re-record the ingest bounds of a ``--fresh`` run, and start a new bill.

    ``--fresh`` exists to discard the ledger's blocks and re-ingest, so a changed
    window or page selection is the intent rather than the corruption the guards
    refuse — and the lifetime usage total, which prices exactly the blocks being
    thrown away, resets along with them. Uses the same empty-list encoding of
    "whole document" as :func:`_guard_selected_pages`. Blocking SQLite, so
    callers go through ``asyncio.to_thread``.
    """
    ledger.set_job_metadata_value(job_id, "chapter_window", [start_chapter, max_chapters])
    ledger.set_job_metadata_value(
        job_id, "selected_pages", sorted(selected_pages) if selected_pages else []
    )
    ledger.record_job_usage(job_id, {})


async def run_ingest_stage(
    ctx: StageContext,
) -> AsyncIterator[TranslationProgressEvent]:
    """Register manifest and stream chapters into SQLite ledger with LNDS noise cleaning."""
    adapter = ctx.require_adapter()
    start_chapter = ctx.start_chapter
    max_chapters = ctx.max_chapters
    input_path = ctx.input_path
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    manifest = ctx.manifest
    source_lang = ctx.source_lang
    create_event_fn = ctx.create_event
    fresh = ctx.config.fresh
    selected_pages = ctx.config.get_selected_pages()
    if selected_pages:
        manifest.run.selected_pages = sorted(selected_pages)
    target_lang = ctx.target_lang
    translate_chrome = ctx.config.translate_chrome
    # Identity a derived job id already namespaces; recorded so an explicit
    # ``--job-id`` resume cannot silently change it. Imported lazily because the
    # pipeline module imports this stage (a module-level import would cycle).
    from ubt.core.engine.pipeline import engine_signature

    # Coerce to JSON-safe scalars: the identity is persisted, and a test (or a
    # duck-typed caller) may hand in a non-serializable stand-in.
    profile_name = str(ctx.profile_name)
    engine_sig = engine_signature(ctx.config)
    mock_run = bool(ctx.is_mock_run)
    if selected_pages and not getattr(adapter, "supports_page_selection", True):
        # A page-ranged request against an adapter that declares no page
        # geometry would be silently ignored (its blocks have no ``bbox.page``),
        # translating — and billing — the whole document. Refuse before spend.
        # Duck-typed adapters without the flag default to "supported".
        raise DocumentParseError(
            f"--pages is only supported for PDF input; {type(adapter).__name__} "
            f"has no page selection and would translate the entire document. "
            "Drop --pages, or use a PDF input."
        )
    try:
        await asyncio.to_thread(ledger.init_job_from_manifest, actual_job_id, manifest)
    except Exception as exc:
        # Manifest registration failure is fatal: continuing without job_meta
        # leaves every later finalize as a silent 0-row update.
        raise DocumentParseError(
            f"Failed to register job {actual_job_id} in the ledger: {exc}"
        ) from exc

    # After the row exists, so the writes land. The two resume guards refuse a
    # window/selection-changing resume, whose export would be silently
    # truncated; --fresh discards the stale blocks on purpose, so it re-records
    # both instead of refusing (its error text advertises exactly that escape).
    if not fresh:
        await asyncio.to_thread(
            _guard_chapter_window, ledger, actual_job_id, start_chapter, max_chapters
        )
        await asyncio.to_thread(_guard_selected_pages, ledger, actual_job_id, selected_pages)
        await asyncio.to_thread(
            _guard_run_identity, ledger, actual_job_id, profile_name, engine_sig, mock_run
        )
    else:
        await asyncio.to_thread(
            _reset_job_for_fresh_run,
            ledger,
            actual_job_id,
            start_chapter,
            max_chapters,
            selected_pages,
        )
        await asyncio.to_thread(
            _record_run_identity, ledger, actual_job_id, profile_name, engine_sig, mock_run
        )

    if create_event_fn:
        yield await create_event_fn(
            EventType.JOB_STARTED,
            actual_job_id,
            ledger,
            message=f"Started job {actual_job_id}",
        )

    lnds_cleaner = LNDSPageCleaner(source_lang=source_lang)
    stats = await asyncio.to_thread(ledger.get_job_stats, actual_job_id)
    try:
        current_fingerprint = await asyncio.to_thread(compute_file_sha256, input_path)
    except DocumentParseError:
        current_fingerprint = ""
    if int(stats.get("total", 0)) != 0:
        if fresh:
            removed = await asyncio.to_thread(ledger.clear_job_blocks, actual_job_id)
            logger.info("Fresh re-ingest for %s: cleared %d stale blocks", actual_job_id, removed)
        else:
            stored = await asyncio.to_thread(ledger.get_job_fingerprint, actual_job_id)
            # Resume language guard: an explicit --job-id reuse
            # towards a different target language must not mix translations
            # into the existing ledger. Default-derived job ids already carry
            # the language; this catches the explicit-id path.
            stored_lang = await asyncio.to_thread(ledger.get_job_target_lang, actual_job_id)
            if stored_lang and target_lang and stored_lang != target_lang:
                raise DocumentParseError(
                    f"Job {actual_job_id} was ingested with target language "
                    f"'{stored_lang}' but is being resumed towards '{target_lang}'. "
                    "Use a different --job-id, re-run with the original target "
                    "language, or add --fresh to restart the job."
                )
            # Change-detection is only meaningful when both sides carry a real
            # hash. The "unavailable" sentinel (hash failed at ingest) means we
            # cannot compare, so resume proceeds without clearing or raising —
            # clearing here would destroy a fully translated book and re-bill it.
            if (
                stored
                and stored != _FINGERPRINT_UNAVAILABLE
                and current_fingerprint
                and stored != current_fingerprint
            ):
                raise DocumentParseError(
                    f"Source file changed since job {actual_job_id} was ingested "
                    f"(stored fingerprint {stored[:12]}…, current {current_fingerprint[:12]}…). "
                    "Re-run with --fresh to discard the stale blocks and re-ingest, "
                    "or restore the original file to resume."
                )
            if not stored:
                removed = await asyncio.to_thread(ledger.clear_job_blocks, actual_job_id)
                logger.warning(
                    "Incomplete ingest detected for %s (no fingerprint); cleared %d partial blocks to ensure clean re-ingest",
                    actual_job_id,
                    removed,
                )
    parsed_count = 0
    stats = await asyncio.to_thread(ledger.get_job_stats, actual_job_id)
    if int(stats.get("total", 0)) == 0:
        loaded_count = 0
        async for chapter in adapter.parse_stream(input_path, selected_pages):
            parsed_count += 1
            if parsed_count < start_chapter:
                continue

            # Clean noise and monotonic page numbers across whole chapter
            cleaned_blocks = lnds_cleaner.clean_chapter_blocks(chapter.blocks)
            # Drop physically duplicated extraction fragments (same page,
            # same text) before they trip the visual gate's overlap finding.
            cleaned_blocks = dedup_duplicate_blocks(cleaned_blocks)
            in_bibliography = False
            for idx_b, b in enumerate(cleaned_blocks):
                in_bibliography = update_bibliography_section_state(
                    cleaned_blocks, idx_b, in_bibliography
                )
                # Role layers (explicit FlowID/BlockType derivation only).
                b.derive_roles()
                # Chrome opt-in: the adapter marks running heads and footers
                # skip-at-parse; with the flag on they get a real translation
                # (page numbers never do) and the rigid gate paints them.
                if translate_chrome and b.layout_role in (
                    LayoutRole.HEADER,
                    LayoutRole.FOOTER,
                ):
                    b.skip_translate = False
                    b.policy_translate = None
                    b.policy_reason = None
                # Zero-LLM-cost verdict runs before any model spend. A keep
                # ships verbatim on the same path as rule-based skips below.
                verdict = judge_block(b, translate_chrome=translate_chrome)
                apply_verdict(b, verdict)
                if not verdict.translate:
                    logger.debug("Verdict keep %s: %s", b.id, verdict.reason)
                    b.skip_translate = True
                    b.status = BlockStatus.MTQE_PASSED
                    b.target_text = b.source_text
                    b.mtqe_score = 1.0
                    continue
                if b.skip_translate or b.block_type in (
                    BlockType.CODE,
                    BlockType.IMAGE,
                    BlockType.FORMULA,
                ):
                    b.skip_translate = True
                    b.status = BlockStatus.MTQE_PASSED
                    b.target_text = b.source_text
                    b.mtqe_score = 1.0
                    continue
                # Untranslatable residue (watermarks, bib entries, symbol
                # debris) ships verbatim without spending LLM / QE / repair.
                skip_reason = classify_skip(
                    b.source_text,
                    is_heading=b.block_type is BlockType.HEADING,
                    in_bibliography=in_bibliography,
                )
                if skip_reason is not None:
                    logger.debug("Skip-translate %s: %s", b.id, skip_reason)
                    # Keep the policy layer consistent with the skip flag:
                    # a verdict:translate stamp above must not survive a skip.
                    b.policy_translate = False
                    b.policy_reason = f"skip:{skip_reason}"
                    b.skip_translate = True
                    b.status = BlockStatus.MTQE_PASSED
                    b.target_text = b.source_text
                    b.mtqe_score = 1.0

            if selected_pages is not None:
                cleaned_blocks = [
                    b for b in cleaned_blocks if b.bbox is None or b.bbox.page in selected_pages
                ]

            if not cleaned_blocks:
                logger.warning(
                    "Chapter %d of %s produced 0 content blocks after cleaning; "
                    "if the whole book is scanned and OCR is off this job will "
                    "fail at ingest with a loud error instead of exporting an "
                    "empty book.",
                    parsed_count,
                    input_path.name,
                )

            chapter.blocks = cleaned_blocks
            await asyncio.to_thread(ledger.append_chapter, actual_job_id, chapter)
            loaded_count += 1
            if max_chapters is not None and loaded_count >= max_chapters:
                break

    final_stats = await asyncio.to_thread(ledger.get_job_stats, actual_job_id)
    # A job that ends ingest with zero blocks is an empty book however it got
    # there, and must fail here instead of exporting an empty deliverable:
    #   * an adapter that streams chapters whose blocks all cleaned or filtered
    #     away (a fully scanned PDF with OCR off, a --pages window over nothing);
    #   * an adapter that streams *no chapter at all* — MarkdownAdapter only
    #     yields a chapter once it has blocks, so an empty or whitespace-only
    #     .md/.txt reaches here with parsed_count == 0.
    # The former ``parsed_count > 0`` guard let the second family through (its
    # adapter yielded nothing, the guard never fired, and the run finalized
    # "completed" with an empty deliverable). A legitimate resume is not caught
    # by dropping the condition: the fast path skips parsing only when the
    # ledger already holds blocks, so total is > 0 on every resume.
    if int(final_stats.get("total", 0)) == 0:
        raise DocumentParseError(
            f"Document {input_path.name} produced 0 content blocks after cleaning; "
            "cannot translate an empty document."
        )

    # Always stamp a fingerprint at the end of a completed ingest so that a
    # missing one reliably means "ingest never finished" (→ safe to clear).
    # When hashing failed, record the sentinel rather than leaving it blank,
    # so the next resume neither clears a fully translated book nor misreads
    # the absent value as a crash.
    await asyncio.to_thread(
        ledger.set_job_fingerprint,
        actual_job_id,
        current_fingerprint or _FINGERPRINT_UNAVAILABLE,
    )
    if create_event_fn:
        yield await create_event_fn(
            EventType.PREPROCESSING_DONE,
            actual_job_id,
            ledger,
            message="Document chapters partitioned and loaded into ledger",
        )
