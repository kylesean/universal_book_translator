"""Whole-book translation bible terminology extraction stage."""

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Terminology
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import IRBlock
from ubt.core.memory.abbreviation_backfill import backfill_abbreviation_translations
from ubt.core.memory.abbreviation_miner import mine_abbreviations_stream
from ubt.core.memory.bible import (
    BibleEntry,
    BookBible,
    TerminologyConflict,
    clean_bible_entry,
    detect_terminology_conflicts,
    merge_bible_entries,
)
from ubt.core.memory.character_miner import mine_characters_stream
from ubt.core.memory.seed_glossary import load_external_glossary, seed_entries_for_profile
from ubt.core.memory.tm import PROMPT_VERSION
from ubt.core.ports import is_fast_lane_eligible

logger = logging.getLogger(__name__)

# Resume-stable bible cache: mining is deterministic but the LLM
# backfill is not, so re-running extraction on resume would produce different
# glossary translations — invalidating tm_context and every cross-run TM hit.
# The merged bible is persisted to job_meta after extraction and reused on
# resume while this cache key matches. `clear_job_blocks` (--fresh) drops it.
_BIBLE_CACHE_KEY = "bible_cache"


def _glossary_fingerprint(glossary_path: Path | str | None) -> str:
    """Content fingerprint of the external glossary file.

    The cache key must carry the glossary *content*, never just its path:
    keyed on path, editing the CSV in place and re-running the same document
    would reuse the prior renderings. Hashing the file bytes makes any content
    change a MISS (re-mine + re-backfill) while an untouched file still hits.
    """
    if not glossary_path:
        return ""
    path = Path(glossary_path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        # Missing/unreadable file: keep the key deterministic (load_external_glossary
        # logs the real problem and returns no entries).
        return f"unreadable:{path}"


def _split_bible_entries(
    entries: list[BibleEntry],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split bible entries into (glossary, abbreviation-channel) dicts.

    Translated entries feed the enforced term glossary. Untranslated *terms* feed
    the abbreviation channel, whose prompt says "keep the abbreviation
    unchanged". Person and place entries are excluded: a name whose backfill
    failed must not be rendered as an unchanged abbreviation.
    """
    glossary = [e.model_dump() for e in entries if e.translation]
    abbreviations = [
        e.model_dump() for e in entries if not e.translation and e.kind not in ("person", "place")
    ]
    return glossary, abbreviations


def _record_terminology_conflicts(
    ledger: Any,
    job_id: str,
    entries: list[BibleEntry],
) -> list[TerminologyConflict]:
    """Detect, log, and persist terminology conflicts for the report and ``ubt status``.

    Must be called on the **unmerged** entries: :func:`merge_bible_entries` keeps
    one rendering per concept by construction, so scanning its output can never
    find a conflict. An earlier check did exactly that and was unreachable — it
    reported nothing on the document it was written for.

    The unresolved ones are also warned about, because they change what the
    operator can expect: a run whose glossary enforces two renderings of one
    concept will leave blocks below the pass line no matter how well they are
    translated, and without this record that is indistinguishable from a
    translation defect.
    """
    conflicts = detect_terminology_conflicts(entries)
    unresolved = [c for c in conflicts if not c.resolved]
    for conflict in unresolved:
        logger.warning(
            "Terminology conflict: %r is enforced as %r but also appears as %s. "
            "A translation can only satisfy one, so blocks using the other will "
            "read as terminology drift. Keep one entry, or add the rejected "
            "rendering as an admitted variant (inflected_variants) in the glossary.",
            conflict.concept,
            conflict.preferred,
            ", ".join(f"{r!r} (from {s!r})" for r, s in conflict.superseded),
        )
    if conflicts:
        logger.info(
            "Terminology: %d concept(s) carried more than one rendering "
            "(%d auto-resolved by precedence, %d need an operator decision)",
            len(conflicts),
            len(conflicts) - len(unresolved),
            len(unresolved),
        )
    try:
        ledger.set_job_metadata_value(
            job_id, "terminology_conflicts", [c.as_dict() for c in conflicts]
        )
    except Exception as exc:  # noqa: BLE001 - never fail a run over a report field
        logger.debug("Could not persist terminology conflicts for %s: %s", job_id, exc)
    return conflicts


async def run_bible_stage(
    ctx: StageContext,
    terminology: Terminology,
) -> AsyncIterator[TranslationProgressEvent]:
    """Extract, mine, and backfill translation bible for the whole book.

    The extracted terminology is written into the plan-owned ``terminology``
    value (explicit stage execution context) for the stages that follow, and the stage
    yields its own ``BIBLE_EXTRACTED`` event when it has one.

    ``ctx.fast_lane`` (short docs): seeds + chapter titles only —
    skips 0-token mining and the LLM backfill channel. Same stage graph,
    subset policy: every fast-lane entry also exists in the full run.

    ``use_cache``: reuse the bible persisted by a previous run of the same
    job when prompt version, languages, profile inputs, and the glossary
    file's *content* all match (content, not path — an in-place edit
    invalidates the cache). Deterministic seeds and chapter-title translations
    are re-applied on top either way, so user-facing inputs stay live.
    """
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    manifest = ctx.manifest
    profile_name = ctx.profile_name
    target_lang = ctx.target_lang
    source_lang = ctx.source_lang
    fast_lane = ctx.fast_lane
    complete_raw_fn = ctx.raw_completion
    create_event_fn = ctx.create_event
    glossary_path = ctx.config.glossary_path
    use_cache = not ctx.config.fresh
    # An explicit ``exec_mode`` outranks the page-count probe. The route already
    # applied it (``router_mode.decide`` returns ``long`` for
    # ``exec_mode=long``), so re-deriving the fast lane here from
    # ``is_fast_lane_eligible`` silently discarded the caller's decision: a
    # ``--exec-mode long`` run logged "Fast lane enabled" and mined nothing.
    # The probe fills in only the undecided ``auto`` case.
    exec_mode = getattr(ctx.config, "exec_mode", "auto")
    if ctx.source_pdf_path is not None and not fast_lane and exec_mode == "auto":
        # A short born-digital PDF also takes the shortcut on its own merits:
        # same stage graph, subset policy, every fast-lane entry also exists in
        # the full run.
        try:
            fast_lane = await asyncio.to_thread(is_fast_lane_eligible, ctx.input_path)
        except Exception as exc:
            logger.debug("Fast-lane eligibility check skipped for %s: %s", actual_job_id, exc)
    if fast_lane:
        logger.info(
            "Fast lane enabled for job %s (seed-only bible, full gates kept)",
            actual_job_id,
        )
    cache_key = {
        "version": PROMPT_VERSION,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "fast_lane": fast_lane,
        # The profile selects the mining policy (is_fiction -> allow_bare_tokens
        # below), so resuming the same job under a different profile must not
        # reuse terminology mined under the prior one.
        "profile_name": profile_name,
        "glossary_fingerprint": _glossary_fingerprint(glossary_path),
    }
    cached_payload: dict[str, Any] | None = None
    if use_cache:
        raw = await asyncio.to_thread(
            ledger.get_job_metadata_value, actual_job_id, _BIBLE_CACHE_KEY
        )
        if isinstance(raw, dict) and all(raw.get(k) == v for k, v in cache_key.items()):
            cached_payload = raw

    bible_entries: list[BibleEntry] = []
    # 1. User-supplied external glossary has the HIGHEST priority (first-translation-wins)
    if glossary_path:
        user_entries = load_external_glossary(glossary_path)
        logger.info("Loaded %d custom glossary entries from %s", len(user_entries), glossary_path)
        bible_entries.extend(user_entries)

    # 2. Curated domain profile seeds
    bible_entries.extend(
        seed_entries_for_profile(profile_name, source_lang=source_lang, target_lang=target_lang)
    )
    ch_translations = manifest.metadata.get("chapter_translations", {})
    for ch in manifest.chapters:
        if ch.title and ch.title in ch_translations:
            tr = str(ch_translations[ch.title]).strip()
            if tr and tr != ch.title:
                e = clean_bible_entry(source=ch.title, translation=tr, kind="term")
                if e:
                    bible_entries.append(e)

    if cached_payload is not None:
        # Reuse mined+backfilled entries from the previous run of this job;
        # the deterministic entries above were recomputed fresh.
        cached_glossary = [
            dict(e) for e in cached_payload.get("glossary_dicts", []) if isinstance(e, dict)
        ]
        cached_abbreviations = [
            dict(e) for e in cached_payload.get("abbreviation_entries", []) if isinstance(e, dict)
        ]
        cached_entries = [BibleEntry.model_validate(d) for d in cached_glossary]
        cached_entries += [BibleEntry.model_validate(d) for d in cached_abbreviations]
        bible_entries.extend(cached_entries)
        # Record conflicts on the UNMERGED list: the merge is what resolves
        # them, so a scan of its output can never see one.
        await asyncio.to_thread(_record_terminology_conflicts, ledger, actual_job_id, bible_entries)
        merged_entries = merge_bible_entries(bible_entries)
        bible = BookBible(
            doc_id=manifest.doc_id,
            language=target_lang,
            glossary=merged_entries,
        )
        # Return the MERGED view, not the cached payload. The freshly read
        # external glossary (and the fresh seeds) live in bible.glossary;
        # returning the payload would discard them, so the pipeline would
        # enforce the previous run's renderings.
        glossary_dicts, abbreviation_entries = _split_bible_entries(bible.glossary)
        logger.info(
            "Reusing cached translation bible for %s (%d translated, %d pending entries)",
            actual_job_id,
            len(glossary_dicts),
            len(abbreviation_entries),
        )
        event = None
        event = await create_event_fn(
            EventType.BIBLE_EXTRACTED,
            actual_job_id,
            ledger,
            message=(
                f"Translation Bible reused from previous run: {len(merged_entries)} entries "
                f"({len(glossary_dicts)} translated, {len(abbreviation_entries)} pending)"
            ),
        )
        terminology.glossary_dicts = glossary_dicts
        terminology.abbreviation_entries = abbreviation_entries
        if event is not None:
            yield event
        return

    mined_abbreviations: list[dict[str, Any]] = []
    mined_characters: list[dict[str, Any]] = []
    block_texts: list[str] = []
    blocks: list[IRBlock] = []
    if hasattr(ledger, "get_all_blocks"):
        blocks = await asyncio.to_thread(ledger.get_all_blocks, actual_job_id)

    if blocks:
        block_texts = [b.source_text for b in blocks if b.source_text]
    elif hasattr(ledger, "fetch_source_texts"):
        block_texts = await asyncio.to_thread(ledger.fetch_source_texts, actual_job_id)

    # Tier-2: Language-agnostic LLM Document-Skeleton Terminology Extraction.
    #
    # This runs on the fast lane too. Gating it left the short route with **no
    # terminology mechanism at all**: the fast lane produced no mined terms, the
    # skeleton call was skipped, and ``deterministic_glossary`` (which the short
    # route turns on) was then fed an empty glossary, so it enforced nothing.
    # Measured on `vit`: the short route had 0 measurable terms while the long
    # route had 28 at 0.9545 precision. The cost is one call over the document
    # skeleton (a few thousand characters, built once) — not a per-block call —
    # and the lane's actual saving comes from the batch API, which is unaffected.
    skeleton_input = blocks if blocks else block_texts
    if complete_raw_fn is not None and skeleton_input:
        from ubt.core.memory.skeleton_extractor import extract_skeleton_terms_llm

        skeleton_entries = await extract_skeleton_terms_llm(
            skeleton_input,
            complete_raw_fn=complete_raw_fn,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        if skeleton_entries:
            logger.info(
                "Extracted %d domain terminology entries from document skeleton for %s",
                len(skeleton_entries),
                actual_job_id,
            )
            bible_entries.extend(skeleton_entries)
    # 0-token mining: no model call, so the fast lane does not skip it.
    mined_abbreviations = mine_abbreviations_stream(block_texts)
    bible_entries.extend(
        BibleEntry(
            source=m["source"],
            translation="",
            aliases=list(m["aliases"]),
            kind="term",
            frequency=int(m.get("frequency", 0)),
        )
        for m in mined_abbreviations
    )

    # 0-token person-name mining (same: free, so not gated on the fast lane).
    is_fiction = profile_name.lower() in ("fiction", "novel", "literature", "classic")
    mined_characters = mine_characters_stream(
        block_texts, source_lang=source_lang, allow_bare_tokens=is_fiction
    )
    bible_entries.extend(
        BibleEntry(
            source=m["source"],
            translation="",
            aliases=list(m["aliases"]),
            kind="person",
            frequency=int(m.get("frequency", 0)),
        )
        for m in mined_characters
    )

    merged_entries = merge_bible_entries(bible_entries)

    # Translation backfill channel. It runs on the fast lane too, for the same
    # reason the skeleton call does: an untranslated mined entry is a term the
    # glossary cannot enforce, so gating this left the short route with mined
    # entries that no gate could use. The comment that used to sit here ("seeds
    # already carry translations, and untranslated mined entries do not exist
    # there") stopped being true once the 0-token miners were un-gated above.
    backfilled_count = 0
    if any(not e.translation for e in merged_entries):
        try:
            merged_dicts, backfilled_count = await backfill_abbreviation_translations(
                [e.model_dump() for e in merged_entries],
                complete_raw_fn,
                target_lang,
            )
            merged_entries = [BibleEntry.model_validate(d) for d in merged_dicts]
        except Exception as exc:
            logger.warning("Abbreviation translation backfill failed: %s", exc)
            backfilled_count = 0
        # Fold again, now that the mined entries carry renderings.
        #
        # The merge above runs before backfill, when a mined entry still has
        # ``translation=""``, and a concept can only be *contradictory* once both
        # sides have a rendering — so the fold never fired for the exact case it
        # exists for. Measured on arXiv 2606.23344: the LLM skeleton supplied
        # ``state-of-the-art (SOTA) -> 当前最优（SOTA）`` while the miner supplied
        # ``state-of-the-art -> ""``, backfill filled the latter with
        # ``当前最先进的``, and the two survived as separate enforced rules, so
        # b0002 scored 0.25 for correctly writing 当前最优（SOTA）.
        #
        # Detect BEFORE this merge, on the list it is about to resolve: the
        # merged output carries one rendering per concept by construction, so a
        # scan after it finds nothing (that unreachable check shipped once).
        await asyncio.to_thread(
            _record_terminology_conflicts, ledger, actual_job_id, merged_entries
        )
        merged_entries = merge_bible_entries(merged_entries)
    else:
        # No backfill ran, so no second merge will run either: the first merge
        # above already resolved everything, and detection has to happen here.
        await asyncio.to_thread(_record_terminology_conflicts, ledger, actual_job_id, bible_entries)

    bible = BookBible(
        doc_id=manifest.doc_id,
        language=target_lang,
        glossary=merged_entries,
    )
    glossary_dicts, abbreviation_entries = _split_bible_entries(bible.glossary)

    await asyncio.to_thread(
        ledger.set_job_metadata_value,
        actual_job_id,
        _BIBLE_CACHE_KEY,
        {
            **cache_key,
            "glossary_dicts": glossary_dicts,
            "abbreviation_entries": abbreviation_entries,
        },
    )

    event = None
    event = await create_event_fn(
        EventType.BIBLE_EXTRACTED,
        actual_job_id,
        ledger,
        message=(
            f"Translation Bible established with {len(merged_entries)} entries "
            f"(incl. {len(mined_abbreviations)} mined abbreviations, "
            f"{len(mined_characters)} mined characters, {backfilled_count} backfilled)"
        ),
    )

    terminology.glossary_dicts = glossary_dicts
    terminology.abbreviation_entries = abbreviation_entries
    if event is not None:
        yield event
