"""Translation Memory unit tests."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from ubt.core.memory.tm import (
    PROVENANCE_HUMAN_PE,
    PROVENANCE_MACHINE,
    TMPendingEntry,
    TranslationMemory,
    format_few_shot_reference,
    normalize_for_tm,
)


@pytest.fixture
def tm(tmp_path: Path) -> Generator[TranslationMemory, Any, None]:
    memory = TranslationMemory(tmp_path / "tm.sqlite")
    yield memory
    memory.close()


def test_normalize_for_tm_collapses_whitespace() -> None:
    assert normalize_for_tm("  Hello\n   World  ") == "Hello World"


def test_exact_hit_and_miss(tm: TranslationMemory) -> None:
    assert tm.lookup_exact("en", "zh", "Hello world.") is None

    tm.writeback([TMPendingEntry("en", "zh", "Hello world.", "你好，世界。", PROVENANCE_MACHINE)])
    hit = tm.lookup_exact("en", "zh", "Hello\n world.  ")
    assert hit is not None
    assert hit.target_text == "你好，世界。"
    assert hit.similarity == 1.0
    assert hit.provenance == PROVENANCE_MACHINE

    # Language-pair isolation: the same source in another pair must miss.
    assert tm.lookup_exact("en", "ja", "Hello world.") is None


def test_fuzzy_hit_above_threshold(tm: TranslationMemory) -> None:
    base = "The quick brown fox jumps over the lazy dog near the river bank."
    tm.writeback([TMPendingEntry("en", "zh", base, "敏捷的棕色狐狸跳过了懒惰的狗。")])

    near_identical = base.replace("river", "riverbank")
    hit = tm.lookup_fuzzy("en", "zh", near_identical, threshold=0.85)
    assert hit is not None
    assert hit.similarity >= 0.85
    assert hit.target_text == "敏捷的棕色狐狸跳过了懒惰的狗。"


def test_fuzzy_below_threshold_returns_none(tm: TranslationMemory) -> None:
    tm.writeback(
        [TMPendingEntry("en", "zh", "A completely different sentence.", "完全不同的句子。")]
    )
    assert (
        tm.lookup_fuzzy("en", "zh", "Something entirely unlike it in every way.", threshold=0.85)
        is None
    )


def test_writeback_does_not_downgrade_human_pe(tm: TranslationMemory) -> None:
    """Human post-edits are the highest-trust provenance: machines never override."""
    tm.writeback([TMPendingEntry("en", "zh", "Same text.", "人类润色版", PROVENANCE_HUMAN_PE)])
    tm.writeback([TMPendingEntry("en", "zh", "Same text.", "机器版", PROVENANCE_MACHINE)])
    hit = tm.lookup_exact("en", "zh", "Same text.")
    assert hit is not None
    assert hit.provenance == PROVENANCE_HUMAN_PE


def test_machine_writeback_updates_target(tm: TranslationMemory) -> None:
    tm.writeback([TMPendingEntry("en", "zh", "Text.", "v1")])
    tm.writeback([TMPendingEntry("en", "zh", "Text.", "v2")])
    hit = tm.lookup_exact("en", "zh", "Text.")
    assert hit is not None
    assert hit.target_text == "v2"


def test_entry_count_and_pair_isolation(tm: TranslationMemory) -> None:
    assert tm.entry_count() == 0
    tm.writeback(
        [
            TMPendingEntry("en", "zh", "One.", "一"),
            TMPendingEntry("en", "ja", "One.", "一"),
        ]
    )
    assert tm.entry_count() == 2
    assert tm.entry_count("en", "zh") == 1


def test_empty_sources_are_rejected(tm: TranslationMemory) -> None:
    assert tm.writeback([TMPendingEntry("en", "zh", "   ", "x")]) == 0
    assert tm.writeback([TMPendingEntry("en", "zh", "x", "  ")]) == 0
    assert tm.entry_count() == 0


def test_few_shot_reference_formatting(tm: TranslationMemory) -> None:
    tm.writeback([TMPendingEntry("en", "zh", "A base sentence for reference.", "参考译文。")])
    hit = tm.lookup_fuzzy("en", "zh", "A base sentence for reference.", threshold=0.85)
    assert hit is not None
    rendered = format_few_shot_reference(hit, "A base sentence for reference.")
    assert "Reference Translation" in rendered
    assert "A base sentence for reference." in rendered
    assert "参考译文。" in rendered
    assert f"{round(hit.similarity * 100)}%" in rendered
    # F2: the reference is terminology-only evidence, never phrasing to copy.
    assert "reuse" not in rendered.lower()
    assert "phrasing" not in rendered.lower()
    assert "terminology reference ONLY" in rendered
    assert "do NOT copy its wording" in rendered
    assert "do NOT infer the sentence's content" in rendered


# ---------------------------------------------------------------------------
# F2 (review): whole-string similarity admits semantic inversions. Each pair
# below is ADMITTED by fuzz.ratio at the 0.85 threshold (0.881-0.945) and used
# to be injected as a reference the prompt told the model to imitate.
# ---------------------------------------------------------------------------

_INVERTED_PAIRS = (
    ("The reaction is exothermic.", "The reaction is endothermic."),
    ("The rate increases with temperature.", "The rate decreases with temperature."),
    ("This is the case.", "This is not the case."),
    ("He turned left at the corner.", "He turned right at the corner."),
)


@pytest.mark.parametrize(("query", "stored"), _INVERTED_PAIRS)
def test_polarity_inverted_fuzzy_hit_is_not_injected(
    tm: TranslationMemory, query: str, stored: str
) -> None:
    """A >=0.85 fuzzy neighbour that inverts polarity must not become a reference."""
    from rapidfuzz import fuzz

    assert fuzz.ratio(query, stored) / 100.0 >= 0.85  # still a fuzzy CANDIDATE
    tm.writeback([TMPendingEntry("en", "zh", stored, "相反的译文。")])
    hit = tm.lookup_fuzzy("en", "zh", query, threshold=0.85)
    assert hit is not None  # candidacy unchanged — only the reference is dropped
    assert format_few_shot_reference(hit, query) == ""


def test_same_polarity_fuzzy_hit_is_still_injected(tm: TranslationMemory) -> None:
    """The guard must not silence genuinely similar, same-polarity neighbours."""
    stored = "The reaction is exothermic and fast."
    tm.writeback([TMPendingEntry("en", "zh", stored, "反应是放热的且很快。")])
    hit = tm.lookup_fuzzy("en", "zh", "The reaction is exothermic and fast.", threshold=0.85)
    assert hit is not None
    rendered = format_few_shot_reference(hit, "The reaction is exothermic and fast.")
    assert "Reference Translation" in rendered
    assert "反应是放热的且很快。" in rendered


def test_polarity_guard_ignores_substrings_and_agreement() -> None:
    """Word-level cues: "north" is not "no", and two negations agree."""
    from ubt.core.memory.tm import _polarity_diverges

    assert not _polarity_diverges("He went north.", "He went north by train.")
    assert not _polarity_diverges("Nothing was found.", "No result was found.")
    assert _polarity_diverges("Nothing was found.", "A result was found.")
    assert _polarity_diverges("Values above the threshold.", "Values below the threshold.")


def test_abbreviation_entries_change_tm_context() -> None:
    """The abbreviation channel is prompt-visible, so it must hash.

    ``build_chunk_glossary_table`` merges mined abbreviation entries into the
    same prompt table as the term glossary; hashing only the term table let a
    prompt-visible abbreviation change keep stale exact hits alive.
    """
    from ubt.core.memory.abbreviation_miner import format_abbreviations_markdown_table
    from ubt.core.memory.tm import compute_tm_context

    terms = "| MOSFET | 金氧半场效晶体管 |"
    abbr_a = [{"source": "Working Memory", "aliases": ["WM"], "translation": ""}]
    abbr_b = [{"source": "Working Memory", "aliases": ["WM"], "translation": ""}]
    abbr_b[0]["aliases"] = ["WMem"]

    ctx_a = compute_tm_context(
        "v1", "general", terms, "en", "zh", format_abbreviations_markdown_table(abbr_a)
    )
    ctx_b = compute_tm_context(
        "v1", "general", terms, "en", "zh", format_abbreviations_markdown_table(abbr_b)
    )
    assert ctx_a != ctx_b
    # Identical inputs stay identical (stability is what makes reuse possible).
    assert ctx_a == compute_tm_context(
        "v1", "general", terms, "en", "zh", format_abbreviations_markdown_table(abbr_a)
    )
    # Pre-P8 empty-hash semantics are untouched: the 5-arg default stays "".
    assert compute_tm_context("v1", "general", terms, "en", "zh") == compute_tm_context(
        "v1", "general", terms, "en", "zh", ""
    )


def test_abbreviation_change_invalidates_exact_hit(tm: TranslationMemory) -> None:
    """A changed abbreviation table must not serve the stale hit.

    ``format_abbreviations_markdown_table`` renders the (alias, source) pair —
    that table is the whole prompt-visible abbreviation channel.
    """
    from ubt.core.memory.abbreviation_miner import format_abbreviations_markdown_table
    from ubt.core.memory.tm import compute_tm_context

    before = format_abbreviations_markdown_table(
        [{"source": "Working Memory", "aliases": ["WM"], "translation": ""}]
    )
    after = format_abbreviations_markdown_table(
        [{"source": "Working Memory Capacity", "aliases": ["WM"], "translation": ""}]
    )
    assert before != after
    ctx_before = compute_tm_context("v1", "general", "", "en", "zh", before)
    ctx_after = compute_tm_context("v1", "general", "", "en", "zh", after)
    tm.writeback(
        [TMPendingEntry("en", "zh", "WM holds state.", "旧译文。", context_hash=ctx_before)]
    )
    assert tm.lookup_exact("en", "zh", "WM holds state.", ctx_before) is not None
    assert tm.lookup_exact("en", "zh", "WM holds state.", ctx_after) is None


def test_shared_db_reopens_with_entries(tmp_path: Path) -> None:
    """Cross-job reuse: entries persist across TranslationMemory instances."""
    db_path = tmp_path / "tm.sqlite"
    first = TranslationMemory(db_path)
    first.writeback([TMPendingEntry("en", "zh", "Persisted text.", "持久化文本。")])
    first.close()

    second = TranslationMemory(db_path)
    try:
        hit = second.lookup_exact("en", "zh", "Persisted text.")
        assert hit is not None
        assert hit.target_text == "持久化文本。"
    finally:
        second.close()


def _seed_corpus(tm: TranslationMemory) -> list[tuple[str, str]]:
    pairs = [
        ("The quick brown fox jumps over the lazy dog near the river bank.", "狐狸跳河。"),
        ("Neural network weights are updated by gradient descent optimizer.", "梯度下降更新权重。"),
        (
            "王小明先生今天在北京参加了人工智能学术会议。",
            "Mr. Wang attended the AI conference in Beijing today.",
        ),
        (
            "Please confirm the delivery schedule before Friday noon.",
            "请在周五中午前确认交货计划。",
        ),
    ]
    tm.writeback([TMPendingEntry("en", "zh", s, t) for s, t in pairs])
    return pairs


def test_fts_index_created_and_synced(tm: TranslationMemory) -> None:
    import sqlite3

    # NOTE: COUNT(*) on an external-content FTS table reads the content
    # table, so sync is probed via MATCH (which reads the real index).
    def match_rows(term: str) -> list[tuple[int]]:
        with sqlite3.connect(tm.db_path) as conn:
            return conn.execute("SELECT rowid FROM tm_fts WHERE tm_fts MATCH ?", (term,)).fetchall()

    _seed_corpus(tm)
    assert len(match_rows('"quick"')) == 1
    assert len(match_rows('"gradient"')) == 1

    # Upsert of the same source keeps the 1:1 index mapping (update trigger
    # deletes + reinserts instead of duplicating).
    tm.writeback(
        [
            TMPendingEntry(
                "en", "zh", "Please confirm the delivery schedule before Friday noon.", "v2"
            )
        ]
    )
    assert len(match_rows('"schedule"')) == 1


def test_fts_backfills_legacy_db(tmp_path: Path) -> None:
    """Databases written before the FTS index get rebuilt on open."""
    import sqlite3

    db_path = tmp_path / "tm.sqlite"
    legacy = TranslationMemory(db_path)
    _seed_corpus(legacy)
    legacy.close()
    # NOTE: bare DELETE FROM is a no-op on FTS5 tables; delete-all is the
    # documented way to clear the index. COUNT(*) on an external-content
    # table reads the content table, so emptiness is probed via MATCH.
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO tm_fts(tm_fts) VALUES('delete-all')")
        conn.execute("PRAGMA user_version = 0")
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute("SELECT rowid FROM tm_fts WHERE tm_fts MATCH '\"quick\"'").fetchall() == []
        )

    reopened = TranslationMemory(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            assert (
                conn.execute("SELECT rowid FROM tm_fts WHERE tm_fts MATCH '\"quick\"'").fetchall()
                != []
            )
        hit = reopened.lookup_fuzzy(
            "en",
            "zh",
            "The quick brown fox jumps over the lazy dog near the riverbank.",
            threshold=0.85,
        )
        assert hit is not None
    finally:
        reopened.close()


def test_fuzzy_prefilter_matches_full_scan(tmp_path: Path) -> None:
    """Forced-prefilter and full-scan paths agree on hits, misses, and CJK."""
    db_path = tmp_path / "tm.sqlite"
    setup = TranslationMemory(db_path)
    pairs = _seed_corpus(setup)
    setup.close()

    scan = TranslationMemory(db_path, prefilter_min_pool=10**9)
    prefilter = TranslationMemory(db_path, prefilter_min_pool=1, prefilter_limit=200)
    try:
        queries = [
            pairs[0][0].replace("river bank", "riverbank"),
            pairs[1][0].replace("gradient descent", "gradient-descent"),
            pairs[2][0].replace("北京", "上海"),
            "Something entirely unrelated to anything stored here at all.",
            "!!! ??? ...",
        ]
        for q in queries:
            scan_hit = scan.lookup_fuzzy("en", "zh", q, threshold=0.85)
            fts_hit = prefilter.lookup_fuzzy("en", "zh", q, threshold=0.85)
            assert (scan_hit is None) == (fts_hit is None)
            if scan_hit is not None and fts_hit is not None:
                assert scan_hit.source_text == fts_hit.source_text
                assert scan_hit.target_text == fts_hit.target_text
                assert scan_hit.similarity == fts_hit.similarity
    finally:
        scan.close()
        prefilter.close()


def test_fuzzy_prefilter_respects_pair_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "tm.sqlite"
    setup = TranslationMemory(db_path)
    _seed_corpus(setup)
    setup.close()

    prefilter = TranslationMemory(db_path, prefilter_min_pool=1)
    try:
        assert (
            prefilter.lookup_fuzzy(
                "en",
                "ja",
                "The quick brown fox jumps over the lazy dog near the riverbank.",
                threshold=0.85,
            )
            is None
        )
    finally:
        prefilter.close()


def test_exact_hit_is_context_gated(tm: TranslationMemory) -> None:
    """Same source under another prompt/glossary context must not hit."""
    from ubt.core.memory.tm import compute_tm_context

    ctx_a = compute_tm_context("v1", "general", "| Sky | 天空 |", "en", "zh")
    ctx_b = compute_tm_context("v1", "general", "| Sky | 天空穹 |", "en", "zh")
    assert ctx_a != ctx_b
    tm.writeback(
        [TMPendingEntry("en", "zh", "The sky was dark.", "天空是黑的。", context_hash=ctx_a)]
    )
    assert tm.lookup_exact("en", "zh", "The sky was dark.", ctx_a) is not None
    assert tm.lookup_exact("en", "zh", "The sky was dark.", ctx_b) is None
    # Legacy context-'' rows still match context-'' lookups (backward compat).
    tm.writeback([TMPendingEntry("en", "zh", "Legacy entry.", "旧条目。")])
    assert tm.lookup_exact("en", "zh", "Legacy entry.") is not None
    assert tm.lookup_exact("en", "zh", "Legacy entry.", ctx_a) is None


def test_human_pe_falls_back_across_contexts(tm: TranslationMemory) -> None:
    """Human-reviewed translations stay valid across context changes."""
    from ubt.core.memory.tm import PROVENANCE_HUMAN_PE, compute_tm_context

    ctx = compute_tm_context("v1", "general", "", "en", "zh")
    tm.writeback(
        [TMPendingEntry("en", "zh", "Reviewed sentence.", "审校句。", PROVENANCE_HUMAN_PE)]
    )
    hit = tm.lookup_exact("en", "zh", "Reviewed sentence.", ctx)
    assert hit is not None
    assert hit.provenance == PROVENANCE_HUMAN_PE


def test_machine_legacy_does_not_fall_back(tm: TranslationMemory) -> None:
    """Stale machine rows from pre-context days must not skip the LLM."""
    from ubt.core.memory.tm import compute_tm_context

    ctx = compute_tm_context("v1", "general", "", "en", "zh")
    tm.writeback([TMPendingEntry("en", "zh", "Machine legacy.", "机器旧条目。")])
    assert tm.lookup_exact("en", "zh", "Machine legacy.", ctx) is None


def test_domain_none_is_strict_but_domain_is_opt_in(tm: TranslationMemory) -> None:
    """Production passes domain=None, so a glossary change must invalidate.

    Passing profile_name as the domain used to trip the same-domain
    pass and serve stale machine hits across a context change.
    """
    from ubt.core.memory.tm import compute_tm_context

    ctx_a = compute_tm_context("v1", "general", "| Sky | 天空 |", "en", "zh")
    ctx_b = compute_tm_context("v1", "general", "| Sky | 苍穹 |", "en", "zh")
    tm.writeback(
        [
            TMPendingEntry(
                "en", "zh", "Domain row.", "域条目。", context_hash=ctx_a, domain="general"
            )
        ]
    )
    # domain=None (production) is strict: the context change wins.
    assert tm.lookup_exact("en", "zh", "Domain row.", ctx_b, None) is None
    # An explicit same-domain lookup still opts in (documented behaviour).
    assert tm.lookup_exact("en", "zh", "Domain row.", ctx_b, "general") is not None


def test_prompt_version_bump_invalidates(tm: TranslationMemory) -> None:
    """Bumping PROMPT_VERSION retires all machine exact hits."""
    from ubt.core.memory.tm import compute_tm_context

    old = compute_tm_context("v1", "general", "", "en", "zh")
    new = compute_tm_context("v2", "general", "", "en", "zh")
    tm.writeback([TMPendingEntry("en", "zh", "Versioned.", "版本化。", context_hash=old)])
    assert tm.lookup_exact("en", "zh", "Versioned.", old) is not None
    assert tm.lookup_exact("en", "zh", "Versioned.", new) is None


def test_exact_hit_increments_use_count(tm: TranslationMemory) -> None:
    """Successful lookups increment use_count."""
    tm.writeback([TMPendingEntry("en", "zh", "Track usage.", "跟踪使用。")])
    assert tm.get_use_count("en", "zh", "Track usage.") == 0

    hit1 = tm.lookup_exact("en", "zh", "Track usage.")
    assert hit1 is not None
    assert tm.get_use_count("en", "zh", "Track usage.") == 1

    hit2 = tm.lookup_exact("en", "zh", "Track usage.")
    assert hit2 is not None
    assert tm.get_use_count("en", "zh", "Track usage.") == 2


def test_exact_hit_same_domain_cross_context_fallback(tm: TranslationMemory) -> None:
    """Same domain entries can be reused across context hash changes."""
    from ubt.core.memory.tm import compute_tm_context

    ctx_ch1 = compute_tm_context("v1", "semiconductor", "| MOSFET | 金氧半场效晶体管 |", "en", "zh")
    ctx_ch2 = compute_tm_context("v1", "semiconductor", "| FinFET | 鳍式场效晶体管 |", "en", "zh")
    assert ctx_ch1 != ctx_ch2

    tm.writeback(
        [
            TMPendingEntry(
                "en",
                "zh",
                "The gate oxide thickness is critical.",
                "栅极氧化层厚度至关重要。",
                domain="semiconductor",
                context_hash=ctx_ch1,
            )
        ]
    )

    # Without domain, cross-context lookup misses
    assert tm.lookup_exact("en", "zh", "The gate oxide thickness is critical.", ctx_ch2) is None

    # With mismatched domain, cross-context lookup misses
    assert (
        tm.lookup_exact(
            "en", "zh", "The gate oxide thickness is critical.", ctx_ch2, domain="medical"
        )
        is None
    )

    # With matching domain, cross-context lookup succeeds
    hit = tm.lookup_exact(
        "en", "zh", "The gate oxide thickness is critical.", ctx_ch2, domain="semiconductor"
    )
    assert hit is not None
    assert hit.target_text == "栅极氧化层厚度至关重要。"


def test_exact_hit_prefers_exact_context_over_same_domain(tm: TranslationMemory) -> None:
    """Candidate ranking prefers exact context match over domain fallback."""
    from ubt.core.memory.tm import compute_tm_context

    ctx_a = compute_tm_context("v1", "general", "| A | 甲 |", "en", "zh")
    ctx_b = compute_tm_context("v1", "general", "| B | 乙 |", "en", "zh")

    # Storing two versions under different contexts but same domain
    tm.writeback(
        [
            TMPendingEntry(
                "en",
                "zh",
                "Sentence.",
                "上下文A译文",
                domain="general",
                context_hash=ctx_a,
            ),
            TMPendingEntry(
                "en",
                "zh",
                "Sentence.",
                "上下文B译文",
                domain="general",
                context_hash=ctx_b,
            ),
        ]
    )

    hit_a = tm.lookup_exact("en", "zh", "Sentence.", ctx_a, domain="general")
    assert hit_a is not None
    assert hit_a.target_text == "上下文A译文"

    hit_b = tm.lookup_exact("en", "zh", "Sentence.", ctx_b, domain="general")
    assert hit_b is not None
    assert hit_b.target_text == "上下文B译文"


def test_human_pe_with_context_hash_falls_back_across_contexts(tm: TranslationMemory) -> None:
    """Human PE entry with non-empty context_hash falls back to a different context."""
    from ubt.core.memory.tm import PROVENANCE_HUMAN_PE, compute_tm_context

    ctx_orig = compute_tm_context("v1", "general", "| A | 甲 |", "en", "zh")
    ctx_new = compute_tm_context("v1", "general", "| B | 乙 |", "en", "zh")

    tm.writeback(
        [
            TMPendingEntry(
                "en",
                "zh",
                "Vetted text.",
                "专家审定译文。",
                provenance=PROVENANCE_HUMAN_PE,
                context_hash=ctx_orig,
            )
        ]
    )

    hit = tm.lookup_exact("en", "zh", "Vetted text.", ctx_new)
    assert hit is not None
    assert hit.provenance == PROVENANCE_HUMAN_PE
    assert hit.target_text == "专家审定译文。"


def test_data_version_invalidates_cross_process_cache(tmp_path: Path) -> None:
    """External process writes bump PRAGMA data_version and clear cached pools."""
    db_file = tmp_path / "cross_proc_tm.sqlite"
    tm1 = TranslationMemory(db_file)
    tm2 = TranslationMemory(db_file)

    # tm1 writes an entry
    tm1.writeback([TMPendingEntry("en", "zh", "Hello friend.", "你好朋友。")])
    assert tm1.entry_count("en", "zh") == 1

    # tm2 loads pool into its cache
    hit2 = tm2.lookup_fuzzy("en", "zh", "Hello friend!", threshold=0.8)
    assert hit2 is not None
    assert hit2.target_text == "你好朋友。"
    assert ("en", "zh") in tm2._pool_cache

    # tm1 adds another entry via separate connection/instance (simulating worker 2)
    tm1.writeback([TMPendingEntry("en", "zh", "Good morning world.", "早安世界。")])
    assert tm1.entry_count("en", "zh") == 2

    # tm2 should detect data_version change on next lookup and find the new entry
    hit_new = tm2.lookup_fuzzy("en", "zh", "Good morning everyone.", threshold=0.6)
    assert hit_new is not None
    assert hit_new.target_text == "早安世界。"
    assert tm2.entry_count("en", "zh") == 2

    tm1.close()
    tm2.close()


@pytest.mark.fast
def test_tm_writeback_repaired_uses_target_text() -> None:
    from ubt.core.engine.stages.tm_writeback import tm_writeback_text
    from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock

    repaired_block = IRBlock(
        id="ch1#b0001",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="This is an English sentence.",
        draft_text="这是一个失败的残次翻译。",  # defective draft
        target_text="这是一个经过二阶修复的合格翻译。",  # repaired text
        status=BlockStatus.REPAIRED,
    )
    # For REPAIRED blocks, tm_writeback_text MUST return target_text, NOT the defective draft_text
    assert tm_writeback_text(repaired_block) == "这是一个经过二阶修复的合格翻译。"


def test_tm_busy_timeout_default(tm: TranslationMemory) -> None:
    conn = tm._get_conn()
    timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
    assert timeout >= 30000


def test_tm_writeback_retries_on_transient_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    db_file = tmp_path / "lock_retry_tm.sqlite"
    tm_inst = TranslationMemory(db_file)
    try:
        attempts = 0
        real_conn = tm_inst._get_conn()

        class FlakyConn:
            def __init__(self, target: Any) -> None:
                self._target = target

            def __getattr__(self, name: str) -> Any:
                return getattr(self._target, name)

            def execute(self, sql: Any, *args: Any, **kwargs: Any) -> Any:
                nonlocal attempts
                if "INSERT INTO tm_entries" in str(sql) and attempts < 2:
                    attempts += 1
                    raise sqlite3.OperationalError("database is locked")
                return self._target.execute(sql, *args, **kwargs)

            def __enter__(self) -> Any:
                self._target.__enter__()
                return self

            def __exit__(self, *args: Any) -> Any:
                return self._target.__exit__(*args)

        monkeypatch.setattr(tm_inst, "_get_conn", lambda: FlakyConn(real_conn))
        stored = tm_inst.writeback([TMPendingEntry("en", "zh", "Retry test.", "重试测试。")])
        assert stored == 1
        assert attempts == 2
    finally:
        tm_inst.close()


_r0918_TM_SRC = "The kernel caches key-value tensors across decoding steps."

_r0918_TM_NEAR = _r0918_TM_SRC.replace("decoding", "decode")

_r0918_TM_ZH = "内核在解码步之间缓存键值张量。"


def test_tm_refuses_passthrough_rows_on_both_read_paths(tmp_path: Path) -> None:
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    try:
        tm.writeback(
            [
                TMPendingEntry(
                    "en", "zh", _r0918_TM_SRC, _r0918_TM_SRC, provenance=PROVENANCE_MACHINE
                )
            ]
        )
        assert tm.entry_count() == 1  # write behaviour is unchanged by design
        assert tm.lookup_exact("en", "zh", _r0918_TM_SRC) is None
        assert tm.lookup_fuzzy("en", "zh", _r0918_TM_NEAR, threshold=0.7) is None

        # Control: the same two lookups succeed once a real translation lands.
        tm.writeback(
            [
                TMPendingEntry(
                    "en", "zh", _r0918_TM_SRC, _r0918_TM_ZH, provenance=PROVENANCE_HUMAN_PE
                )
            ]
        )
        assert tm.lookup_exact("en", "zh", _r0918_TM_SRC) is not None
        assert tm.lookup_fuzzy("en", "zh", _r0918_TM_NEAR, threshold=0.7) is not None
    finally:
        tm.close()


_r0921_SRC = "The quick brown fox jumps over the lazy dog near the river bank."

_r0921_ZH = "那只敏捷的棕色狐狸跃过河边懒狗。"


def test_tm_pool_revalidates_per_pair_not_whole_cache(tmp_path: Path) -> None:
    """Per-pair generations replaced the whole-cache clear, because of its cost.

    This test pinned the opposite choice (review 2026-09-21 §6.1): ``PRAGMA
    data_version`` moves on any connection's commit — including the ``use_count``
    bump both lookup paths write — and every cached pool was dropped on that
    signal. It was defended as churn because "real pools are small". Measured on
    this checkout a reload costs 0.4 ms at 500 rows, 4 ms at 5k and 17 ms at 19k,
    while reading one pair's generation costs ~1 µs; a shared ``tm.sqlite``
    crosses 5k rows after a couple of books, and every hit re-dirties the signal,
    so each fuzzy lookup was paying a rescan. What freshness still has to
    survive is asserted below: a foreign commit that adds a row to THIS pair
    reloads it, a use_count bump and a write to ANOTHER pair do not.
    """
    db = tmp_path / "tm.sqlite"
    writer = TranslationMemory(db)
    reader = TranslationMemory(db)
    try:
        writer.writeback([TMPendingEntry("en", "zh", _r0921_SRC, _r0921_ZH, "machine")])
        cached = reader._pool("en", "zh")

        assert writer.lookup_exact("en", "zh", _r0921_SRC) is not None
        assert reader._pool("en", "zh") is cached, "a use_count bump changed no source"

        writer.writeback([TMPendingEntry("fr", "zh", _r0921_SRC, _r0921_ZH, "machine")])
        assert reader._pool("en", "zh") is cached, "another pair must not evict this one"

        writer.writeback([TMPendingEntry("en", "zh", "Second sentence.", "第二句。")])
        reloaded = reader._pool("en", "zh")
        assert reloaded is not cached, "a row added to this pair must reach a live reader"
        assert len(reloaded[1]) == 2

        # The pool and the count are filled on different calls, so reading the
        # count must not certify the pool as fresh.
        writer.writeback([TMPendingEntry("en", "zh", "Third sentence.", "第三句。")])
        assert reader.entry_count("en", "zh") == 3
        assert len(reader._pool("en", "zh")[1]) == 3
    finally:
        writer.close()
        reader.close()
