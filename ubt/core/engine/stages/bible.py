"""Whole-book translation bible terminology extraction stage."""

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.stage_context import StageContext
from ubt.core.memory.abbreviation_backfill import backfill_abbreviation_translations
from ubt.core.memory.abbreviation_miner import mine_abbreviations_stream
from ubt.core.memory.bible import BibleEntry, BookBible, clean_bible_entry, merge_bible_entries
from ubt.core.memory.character_miner import mine_characters_stream
from ubt.core.memory.seed_glossary import load_external_glossary, seed_entries_for_profile
from ubt.core.memory.tm import PROMPT_VERSION
from ubt.core.ports import is_fast_lane_eligible
from ubt.pipeline.facts import Terminology

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
    would reuse the old renderings. Hashing the file bytes makes any content
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
    if ctx.source_pdf_path is not None and not fast_lane:
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
        # reuse terminology mined under the old one.
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
        if create_event_fn:
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
    if not fast_lane:
        block_texts = await asyncio.to_thread(ledger.fetch_source_texts, actual_job_id)
        # Tier-2: Language-agnostic LLM Document-Skeleton Terminology Extraction
        if complete_raw_fn is not None and block_texts:
            from ubt.core.memory.skeleton_extractor import extract_skeleton_terms_llm

            skeleton_entries = await extract_skeleton_terms_llm(
                block_texts,
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

    # 0-token person-name mining
    is_fiction = profile_name.lower() in ("fiction", "novel", "literature", "classic")
    if not fast_lane:
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

    # Translation backfill channel (skipped on the fast lane: seeds already
    # carry translations, and untranslated mined entries do not exist there).
    backfilled_count = 0
    if not fast_lane and any(not e.translation for e in merged_entries):
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
    if create_event_fn:
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
