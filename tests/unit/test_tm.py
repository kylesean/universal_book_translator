"""Translation memory: the pure functions, centred on the polarity guard.

A fuzzy TM hit is injected into the draft prompt as a *terminology* reference.
Whole-string similarity cannot see meaning, so a one-morpheme inversion
("exothermic" vs "endothermic", "is" vs "isn't") still clears the fuzzy
threshold -- and a wrong reference is the most direct route for a wrong sentence
into a book. :func:`_polarity_diverges` is the guard that drops such a reference
before it is shown. It is conservative by design: a dropped reference costs one
LLM draft, never a wrong sentence.

The rest of the module's pure surface is pinned here too -- the context
fingerprint an exact hit is valid under, the exact-key normalization, the FTS5
trigram shingling, and the few-shot rendering that consumes the guard.

The SQLite-backed :class:`TranslationMemory` is exercised against a real
``tmp_path`` database: the writeback upsert policy (human review is never
downgraded, the reuse counter is *not* a write counter), the context/domain
read gate, identity-row refusal, the per-pair generation that lets one writer
process invalidate another's cached pool, and the FTS5 prefilter path.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from ubt.core.memory.tm import (
    PROMPT_VERSION,
    PROVENANCE_HUMAN_PE,
    PROVENANCE_MACHINE,
    TMHit,
    TMPendingEntry,
    TranslationMemory,
    _fts_phrase,
    _fts_query_terms,
    _polarity_diverges,
    _polarity_signature,
    _source_hash,
    compute_tm_context,
    format_few_shot_reference,
    normalize_for_tm,
)

pytestmark = pytest.mark.fast

#: Negation words that must set the negation slot (word-level, not substring).
_NEGATION_CUES = (
    "not",
    "no",
    "never",
    "without",
    "none",
    "neither",
    "nor",
    "nothing",
    "nobody",
    "cannot",
)

#: CJK negators that travel as characters, not ``[a-z]+`` tokens.
_CJK_NEGATION_CUES = (
    "\u4e0d",
    "\u6ca1",
    "\u6ca1\u6709",
    "\u522b",
    "\u65e0",
    "\u52ff",
    "\u306a\u3044",
    "\u305a",
    "\u306c",
)

#: Mutually exclusive antonym groups; each contributes one slot.
_ANTONYM_GROUPS = (
    ("increase", "decrease"),
    ("left", "right"),
    ("exothermic", "endothermic"),
    ("more", "less"),
    ("above", "below"),
)

_NEG = "neg"


# --------------------------------------------------------------------------- #
# _polarity_signature: one slot per cue group.
# --------------------------------------------------------------------------- #


def test_signature_has_a_negation_slot_plus_one_per_antonym_group() -> None:
    assert len(_polarity_signature("neutral text")) == 1 + len(_ANTONYM_GROUPS)


def test_neutral_text_carries_no_cues() -> None:
    assert _polarity_signature("the cat sat on the mat") == ("",) * (1 + len(_ANTONYM_GROUPS))


@pytest.mark.parametrize("cue", _NEGATION_CUES)
def test_each_negation_word_sets_the_negation_slot(cue: str) -> None:
    assert _polarity_signature(f"this is {cue} the case")[0] == _NEG


def test_the_n_t_contraction_sets_the_negation_slot() -> None:
    assert _polarity_signature("this isn't the case")[0] == _NEG


def test_a_curly_apostrophe_contraction_also_sets_the_negation_slot() -> None:
    # Real books use U+2019; "n't" in "isn’t" is false unless it is folded first.
    assert _polarity_signature("this isn\u2019t the case")[0] == _NEG


def test_negation_membership_is_word_level_not_substring() -> None:
    assert _polarity_signature("north of the river")[0] == ""


@pytest.mark.parametrize("cue", _CJK_NEGATION_CUES)
def test_cjk_negators_set_the_negation_slot(cue: str) -> None:
    # CJK negation travels as characters; the [a-z]+ cue set never fired for it.
    assert _polarity_signature(f"{cue}\u662f\u771f\u7684")[0] == _NEG


@pytest.mark.parametrize(
    ("word", "slot_index"),
    [("increase", 1), ("left", 2), ("exothermic", 3), ("more", 4), ("above", 5)],
)
def test_each_antonym_group_puts_its_cue_in_its_own_slot(word: str, slot_index: int) -> None:
    signature = _polarity_signature(f"it will {word} sharply")
    assert signature[slot_index] == word


def test_antonym_inflections_canonicalize_to_the_same_slot() -> None:
    # A tense change is not an antonym divergence -- and the slot must hold the
    # canonical cue, not merely be equal to the other inflection's slot.
    assert _polarity_signature("it increases") == _polarity_signature("it increased")
    assert _polarity_signature("it increased")[1] == "increase"


def test_opposite_antonyms_land_in_different_slots() -> None:
    assert _polarity_signature("rates increase") != _polarity_signature("rates decrease")


# --------------------------------------------------------------------------- #
# _polarity_diverges: the guard itself.
# --------------------------------------------------------------------------- #


def test_identical_polarity_does_not_diverge() -> None:
    assert _polarity_diverges("the cat sat", "the cat sat down") is False


def test_an_added_negation_diverges() -> None:
    assert _polarity_diverges("this is not the case", "this is the case") is True


def test_a_removed_negation_diverges() -> None:
    assert _polarity_diverges("this is the case", "this is not the case") is True


def test_an_antonym_swap_diverges() -> None:
    assert _polarity_diverges("the rate will increase", "the rate will decrease") is True


def test_a_cjk_negation_difference_diverges() -> None:
    assert _polarity_diverges("\u4e0d\u662f\u771f\u7684", "\u662f\u771f\u7684") is True


def test_an_inflection_only_change_does_not_diverge() -> None:
    assert _polarity_diverges("it increases", "it increased") is False


# --------------------------------------------------------------------------- #
# compute_tm_context: the identity an exact hit is valid under.
# --------------------------------------------------------------------------- #


def _context(**overrides: str) -> str:
    parts = {
        "prompt_version": PROMPT_VERSION,
        "profile_name": "general",
        "glossary_table": "g1",
        "src_lang": "en",
        "tgt_lang": "zh",
        "abbreviation_table": "",
    }
    parts.update(overrides)
    return compute_tm_context(**parts)


def test_context_is_deterministic_and_32_hex_chars() -> None:
    first = _context()
    assert first == _context()
    assert len(first) == 32
    int(first, 16)


@pytest.mark.parametrize(
    "field",
    [
        "prompt_version",
        "profile_name",
        "glossary_table",
        "src_lang",
        "tgt_lang",
        "abbreviation_table",
    ],
)
def test_every_prompt_visible_part_changes_the_context(field: str) -> None:
    # A prompt-visible change that did not move the fingerprint would let a
    # stale exact hit survive it.
    assert _context(**{field: "changed"}) != _context()


# --------------------------------------------------------------------------- #
# normalize_for_tm / _source_hash: the exact key.
# --------------------------------------------------------------------------- #


def test_normalize_collapses_whitespace_and_strips() -> None:
    assert normalize_for_tm("  a  b\n\tc  ") == "a b c"


def test_normalize_of_blank_text_is_empty() -> None:
    assert normalize_for_tm("   \n ") == ""


def test_source_hash_is_sha256_hex_and_input_sensitive() -> None:
    assert len(_source_hash("x")) == 64
    assert _source_hash("x") == _source_hash("x")
    assert _source_hash("x") != _source_hash("y")


# --------------------------------------------------------------------------- #
# FTS5 trigram shingling.
# --------------------------------------------------------------------------- #


def test_terms_are_three_codepoint_shingles_of_each_word_run() -> None:
    assert _fts_query_terms("hello world") == ["hel", "ell", "llo", "wor", "orl", "rld"]


def test_runs_shorter_than_three_codepoints_yield_no_terms() -> None:
    assert _fts_query_terms("ab cd") == []


def test_shingles_are_deduplicated() -> None:
    assert _fts_query_terms("aaaa") == ["aaa"]


def test_the_term_count_is_capped() -> None:
    rng = random.Random(0)
    long_run = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(300))
    terms = _fts_query_terms(long_run)
    assert len(terms) == 64
    assert all(len(term) == 3 for term in terms)


def test_a_phrase_quotes_and_doubles_embedded_quotes() -> None:
    assert _fts_phrase('a"b') == '"a""b"'


# --------------------------------------------------------------------------- #
# format_few_shot_reference: the guard at the prompt boundary.
# --------------------------------------------------------------------------- #


def _hit(source: str = "The cat sat.", target: str = "Le chat.", similarity: float = 0.9) -> TMHit:
    return TMHit(source, target, similarity, "machine")


def test_a_polarity_divergent_hit_is_not_rendered() -> None:
    assert format_few_shot_reference(_hit(), "The cat did not sit") == ""


def test_a_surviving_hit_states_its_similarity_percentage() -> None:
    assert "90% similar" in format_few_shot_reference(_hit(), "The cat sat")


def test_a_surviving_hit_is_presented_as_terminology_only() -> None:
    reference = format_few_shot_reference(_hit(), "The cat sat")
    assert "terminology reference ONLY" in reference
    assert "do NOT copy" in reference
    assert "Reference source: The cat sat." in reference
    assert "Reference translation: Le chat." in reference


# --------------------------------------------------------------------------- #
# TranslationMemory: the SQLite-backed store.
# --------------------------------------------------------------------------- #


@pytest.fixture()
def tm(tmp_path: Path) -> Iterator[TranslationMemory]:
    memory = TranslationMemory(tmp_path / "tm.sqlite")
    yield memory
    memory.close()


def _store(
    memory: TranslationMemory,
    source: str,
    target: str,
    *,
    src_lang: str = "en",
    tgt_lang: str = "zh",
    provenance: str = PROVENANCE_MACHINE,
    domain: str | None = None,
    context_hash: str = "",
    runs_json: str = "",
) -> int:
    return memory.writeback(
        [
            TMPendingEntry(
                src_lang,
                tgt_lang,
                source,
                target,
                provenance=provenance,
                domain=domain,
                context_hash=context_hash,
                runs_json=runs_json,
            )
        ]
    )


def test_writeback_then_exact_lookup_round_trips(tm: TranslationMemory) -> None:
    assert _store(tm, "The cat sat.", "\u732b\u5750\u7740\u3002") == 1
    hit = tm.lookup_exact("en", "zh", "The cat sat.")
    assert hit == TMHit("The cat sat.", "\u732b\u5750\u7740\u3002", 1.0, PROVENANCE_MACHINE)


def test_exact_lookup_round_trips_the_stored_emphasis_runs(tm: TranslationMemory) -> None:
    runs = '[{"text": "50.0%", "bold": true}]'
    _store(tm, "Cost fell 50.0%.", "\u6210\u672c\u964d\u4f4e 50.0%\u3002", runs_json=runs)
    hit = tm.lookup_exact("en", "zh", "Cost fell 50.0%.")
    assert hit is not None
    assert hit.runs_json == runs


def test_a_rewrite_updates_the_stored_runs(tm: TranslationMemory) -> None:
    _store(tm, "Cost fell 50.0%.", "\u65e7\u8bd1\u6587", runs_json="")
    runs = '[{"text": "50.0%", "bold": true}]'
    _store(tm, "Cost fell 50.0%.", "\u65b0\u8bd1\u6587 50.0%", runs_json=runs)
    hit = tm.lookup_exact("en", "zh", "Cost fell 50.0%.")
    assert hit is not None
    assert hit.target_text == "\u65b0\u8bd1\u6587 50.0%"
    assert hit.runs_json == runs


def test_exact_lookup_is_normalized(tm: TranslationMemory) -> None:
    _store(tm, "The cat sat.", "\u732b\u3002")
    assert tm.lookup_exact("en", "zh", "  The   cat sat.  ") is not None


def test_exact_lookup_is_scoped_to_the_language_pair(tm: TranslationMemory) -> None:
    _store(tm, "The cat sat.", "\u732b\u3002")
    assert tm.lookup_exact("en", "fr", "The cat sat.") is None


def test_exact_lookup_of_blank_source_is_a_miss(tm: TranslationMemory) -> None:
    assert tm.lookup_exact("en", "zh", "   ") is None


def test_writeback_of_no_entries_stores_nothing(tm: TranslationMemory) -> None:
    assert tm.writeback([]) == 0


def test_writeback_skips_entries_with_a_blank_side(tm: TranslationMemory) -> None:
    assert _store(tm, "  ", "x") == 0
    assert _store(tm, "y", "  ") == 0


def test_machine_writeback_never_downgrades_human_pe(tm: TranslationMemory) -> None:
    _store(tm, "Reviewed.", "\u5ba1\u3002", provenance=PROVENANCE_HUMAN_PE)
    _store(tm, "Reviewed.", "\u673a\u3002", provenance=PROVENANCE_MACHINE)
    hit = tm.lookup_exact("en", "zh", "Reviewed.")
    assert hit is not None
    # The latest translation wins, but the reviewed *label* survives: a machine
    # rewrite must not demote a human-reviewed row back to machine provenance.
    assert hit.target_text == "\u673a\u3002"
    assert hit.provenance == PROVENANCE_HUMAN_PE


def test_a_conflicting_writeback_keeps_the_domain_in_sync_with_the_target(
    tm: TranslationMemory,
) -> None:
    _store(tm, "Domain line.", "\u57df\u3002", domain="math")
    _store(tm, "Domain line.", "\u57df2\u3002", domain="bio")
    row = next(entry for entry in tm.scan() if entry.source_text == "Domain line.")
    assert row.target_text == "\u57df2\u3002"
    assert row.domain == "bio"


def test_writeback_does_not_count_as_reuse(tm: TranslationMemory) -> None:
    _store(tm, "Alpha.", "\u7532\u3002")
    _store(tm, "Alpha.", "\u75322\u3002")
    assert tm.get_use_count("en", "zh", "Alpha.") == 0


def test_exact_lookup_increments_the_reuse_counter(tm: TranslationMemory) -> None:
    _store(tm, "Alpha.", "\u7532\u3002")
    tm.lookup_exact("en", "zh", "Alpha.")
    tm.lookup_exact("en", "zh", "Alpha.")
    assert tm.get_use_count("en", "zh", "Alpha.") == 2


def test_get_use_count_of_an_unknown_source_is_zero(tm: TranslationMemory) -> None:
    assert tm.get_use_count("en", "zh", "never stored") == 0


# --- identity rows --------------------------------------------------------- #


def test_a_machine_row_echoing_its_source_is_refused(tm: TranslationMemory) -> None:
    _store(tm, "Introduction", "introduction", provenance=PROVENANCE_MACHINE)
    assert tm.lookup_exact("en", "zh", "Introduction") is None


def test_a_human_row_that_only_recases_is_served(tm: TranslationMemory) -> None:
    _store(tm, "Recase.", "RECASE.", provenance=PROVENANCE_HUMAN_PE)
    assert tm.lookup_exact("en", "zh", "Recase.") is not None


def test_a_human_row_that_verbatim_echoes_is_refused(tm: TranslationMemory) -> None:
    _store(tm, "Verbatim.", "Verbatim.", provenance=PROVENANCE_HUMAN_PE)
    assert tm.lookup_exact("en", "zh", "Verbatim.") is None


def test_row_rejected_is_case_insensitive_for_machine_rows() -> None:
    assert TranslationMemory._row_rejected(PROVENANCE_MACHINE, "Introduction", "introduction")
    assert not TranslationMemory._row_rejected(PROVENANCE_MACHINE, "Introduction", "Intro.")


def test_row_rejected_is_verbatim_only_for_human_rows() -> None:
    assert TranslationMemory._row_rejected(PROVENANCE_HUMAN_PE, "Same.", "Same.")
    assert not TranslationMemory._row_rejected(PROVENANCE_HUMAN_PE, "Recase.", "RECASE.")


# --- context / domain gate ------------------------------------------------- #


def test_exact_hit_requires_a_matching_context(tm: TranslationMemory) -> None:
    _store(tm, "Ctx.", "\u4e0a\u4e0b\u6587\u3002", context_hash="A")
    assert tm.lookup_exact("en", "zh", "Ctx.", context_hash="A") is not None
    assert tm.lookup_exact("en", "zh", "Ctx.", context_hash="B") is None
    assert tm.lookup_exact("en", "zh", "Ctx.") is None


def test_a_same_domain_row_is_exempt_from_the_context_gate(tm: TranslationMemory) -> None:
    _store(tm, "Dom.", "\u57df\u3002", domain="math", context_hash="A")
    assert tm.lookup_exact("en", "zh", "Dom.", context_hash="B", domain="math") is not None
    assert tm.lookup_exact("en", "zh", "Dom.", context_hash="B", domain="bio") is None


def test_a_human_pe_row_is_exempt_from_the_context_gate(tm: TranslationMemory) -> None:
    _store(tm, "Human.", "\u4eba\u3002", provenance=PROVENANCE_HUMAN_PE, context_hash="A")
    assert tm.lookup_exact("en", "zh", "Human.", context_hash="B") is not None


def test_hit_allowed_gate_matrix() -> None:
    # human_pe bypasses everything.
    assert TranslationMemory._hit_allowed(PROVENANCE_HUMAN_PE, "A", None, "B", None)
    # a machine row needs context equality ...
    assert TranslationMemory._hit_allowed(PROVENANCE_MACHINE, "A", None, "A", None)
    assert not TranslationMemory._hit_allowed(PROVENANCE_MACHINE, "A", None, "B", None)
    # ... or a shared domain.
    assert TranslationMemory._hit_allowed(PROVENANCE_MACHINE, "A", "math", "B", "math")
    assert not TranslationMemory._hit_allowed(PROVENANCE_MACHINE, "A", "math", "B", "bio")


def test_human_pe_outranks_an_exact_context_machine_row(tm: TranslationMemory) -> None:
    # Same normalized source under different context hashes -> two rows; the
    # reviewed rendering must win even though the machine row matches exactly.
    _store(tm, "Rank.", "\u673a\u3002", context_hash="A", provenance=PROVENANCE_MACHINE)
    _store(tm, "Rank.", "\u4eba\u3002", context_hash="", provenance=PROVENANCE_HUMAN_PE)
    hit = tm.lookup_exact("en", "zh", "Rank.", context_hash="A")
    assert hit is not None
    assert hit.target_text == "\u4eba\u3002"
    assert hit.provenance == PROVENANCE_HUMAN_PE


# --- fuzzy lookup (scan path) ---------------------------------------------- #


def test_fuzzy_scan_finds_a_near_identical_source(tm: TranslationMemory) -> None:
    _store(tm, "The quick brown fox jumps over the lazy dog.", "\u72d0\u72f8\u3002")
    hit = tm.lookup_fuzzy("en", "zh", "The quick brown fox jumps over the lazy dog!", 0.8)
    assert hit is not None
    assert hit.target_text == "\u72d0\u72f8\u3002"
    assert 0.8 <= hit.similarity < 1.0


def test_fuzzy_respects_the_threshold(tm: TranslationMemory) -> None:
    _store(tm, "The quick brown fox jumps over the lazy dog.", "\u72d0\u72f8\u3002")
    assert tm.lookup_fuzzy("en", "zh", "The quick brown fox jumps over the lazy dog!", 0.99) is None


def test_fuzzy_of_blank_source_is_a_miss(tm: TranslationMemory) -> None:
    assert tm.lookup_fuzzy("en", "zh", "   ") is None


def test_fuzzy_obeys_the_context_gate(tm: TranslationMemory) -> None:
    _store(tm, "Gated fuzzy source sentence here.", "\u95e8\u3002", context_hash="A")
    assert (
        tm.lookup_fuzzy("en", "zh", "Gated fuzzy source sentence here.", 0.9, context_hash="A")
        is not None
    )
    assert (
        tm.lookup_fuzzy("en", "zh", "Gated fuzzy source sentence here.", 0.9, context_hash="B")
        is None
    )


def test_fuzzy_refuses_an_identity_candidate(tm: TranslationMemory) -> None:
    _store(tm, "Echoed sentence that is identical.", "echoed sentence that is identical.")
    assert tm.lookup_fuzzy("en", "zh", "Echoed sentence that is identical.", 0.9) is None


def test_writeback_refreshes_the_cached_pool(tm: TranslationMemory) -> None:
    # Prime the pool, then write; the new row must be findable without a reopen.
    tm.lookup_fuzzy("en", "zh", "nothing close at all here", 0.99)
    _store(tm, "Brand new sentence for the pool.", "\u65b0\u3002")
    hit = tm.lookup_fuzzy("en", "zh", "Brand new sentence for the pool.", 0.95)
    assert hit is not None
    assert hit.target_text == "\u65b0\u3002"


def test_evict_refreshes_the_cached_pool(tm: TranslationMemory) -> None:
    _store(tm, "Evictable sentence for the pool.", "\u5220\u3002")
    tm.lookup_fuzzy("en", "zh", "Evictable sentence for the pool.", 0.99)
    row = next(entry for entry in tm.scan() if entry.source_text.startswith("Evictable"))
    assert tm.evict_ids([row.id]) == 1
    assert tm.lookup_fuzzy("en", "zh", "Evictable sentence for the pool.", 0.95) is None


# --- FTS prefilter path ---------------------------------------------------- #


def test_the_fts_path_owns_a_large_enough_pool(tmp_path: Path) -> None:
    memory = TranslationMemory(tmp_path / "tm.sqlite", prefilter_min_pool=2)
    try:
        _store(memory, "The riverbank erosion study is important.", "\u6cb3\u5cb8\u3002")
        _store(memory, "Another riverbank sentence about erosion.", "\u53e6\u3002")
        assert memory._fts_path_active("en", "zh") is True
        hit = memory.lookup_fuzzy("en", "zh", "The riverbank erosion study is important!", 0.8)
        assert hit is not None
        assert hit.target_text == "\u6cb3\u5cb8\u3002"
    finally:
        memory.close()


def test_the_fts_path_is_a_miss_without_token_overlap(tmp_path: Path) -> None:
    # A >=threshold hit without trigram overlap is near-impossible; the prefilter
    # deliberately does not fall back to a full scan in that case.
    memory = TranslationMemory(tmp_path / "tm.sqlite", prefilter_min_pool=2)
    try:
        _store(memory, "The riverbank erosion study is important.", "\u6cb3\u5cb8\u3002")
        _store(memory, "Another riverbank sentence about erosion.", "\u53e6\u3002")
        assert memory.lookup_fuzzy("en", "zh", "zzz qqq www", 0.5) is None
    finally:
        memory.close()


def test_a_punctuation_only_query_has_no_fts_terms(tmp_path: Path) -> None:
    memory = TranslationMemory(tmp_path / "tm.sqlite", prefilter_min_pool=1)
    try:
        _store(memory, "Some stored sentence for the pool.", "\u53e5\u3002")
        assert memory.lookup_fuzzy("en", "zh", "!!! ...", 0.5) is None
    finally:
        memory.close()


# --- counts, scan, evict --------------------------------------------------- #


def test_entry_count_filters_by_pair(tm: TranslationMemory) -> None:
    _store(tm, "One.", "\u4e00\u3002")
    _store(tm, "Two.", "\u4e8c\u3002")
    _store(tm, "Trois.", "3.", src_lang="en", tgt_lang="fr")
    assert tm.entry_count("en", "zh") == 2
    assert tm.entry_count("en", "fr") == 1
    assert tm.entry_count() == 3
    assert tm.entry_count("en", "de") == 0


def test_entry_count_reflects_a_later_write(tm: TranslationMemory) -> None:
    _store(tm, "One.", "\u4e00\u3002")
    assert tm.entry_count("en", "zh") == 1
    _store(tm, "Two.", "\u4e8c\u3002")
    assert tm.entry_count("en", "zh") == 2
    assert tm.entry_count() == 2


def test_scan_returns_every_row_in_id_order(tm: TranslationMemory) -> None:
    _store(tm, "One.", "\u4e00\u3002")
    _store(tm, "Two.", "\u4e8c\u3002")
    rows = tm.scan()
    assert [entry.id for entry in rows] == sorted(entry.id for entry in rows)
    assert [entry.source_text for entry in rows] == ["One.", "Two."]


def test_evict_removes_rows_and_returns_the_count(tm: TranslationMemory) -> None:
    _store(tm, "One.", "\u4e00\u3002")
    _store(tm, "Two.", "\u4e8c\u3002")
    ids = [entry.id for entry in tm.scan()]
    assert tm.evict_ids(ids) == 2
    assert tm.entry_count() == 0
    assert tm.scan() == []


def test_evict_of_nothing_is_a_noop(tm: TranslationMemory) -> None:
    assert tm.evict_ids([]) == 0
    assert tm.evict_ids([999_999]) == 0


# --- per-pair generation across processes ---------------------------------- #


def test_one_process_sees_another_process_write(tmp_path: Path) -> None:
    db = tmp_path / "tm.sqlite"
    writer = TranslationMemory(db)
    reader = TranslationMemory(db)
    try:
        _store(writer, "First sentence for sharing.", "\u4e00\u3002")
        # Prime the reader's pool before the second write.
        assert reader.lookup_fuzzy("en", "zh", "First sentence for sharing.", 0.99) is not None
        _store(writer, "Second sentence added by the writer.", "\u4e8c\u3002")
        hit = reader.lookup_fuzzy("en", "zh", "Second sentence added by the writer.", 0.95)
        assert hit is not None
        assert hit.target_text == "\u4e8c\u3002"
    finally:
        writer.close()
        reader.close()


def test_one_process_sees_another_process_evict(tmp_path: Path) -> None:
    db = tmp_path / "tm.sqlite"
    writer = TranslationMemory(db)
    reader = TranslationMemory(db)
    try:
        _store(writer, "First sentence for sharing.", "\u4e00\u3002")
        assert reader.lookup_fuzzy("en", "zh", "First sentence for sharing.", 0.99) is not None
        row = next(entry for entry in writer.scan() if entry.source_text.startswith("First"))
        writer.evict_ids([row.id])
        assert reader.lookup_fuzzy("en", "zh", "First sentence for sharing.", 0.99) is None
    finally:
        writer.close()
        reader.close()


def test_data_survives_close_and_reopen(tmp_path: Path) -> None:
    db = tmp_path / "tm.sqlite"
    first = TranslationMemory(db)
    _store(first, "Persisted sentence.", "\u6301\u4e45\u3002")
    first.close()
    second = TranslationMemory(db)
    try:
        assert second.lookup_exact("en", "zh", "Persisted sentence.") is not None
        assert second.entry_count() == 1
    finally:
        second.close()


def test_generation_starts_at_zero_and_increments(tm: TranslationMemory) -> None:
    with tm._lock, tm._get_conn() as conn:
        assert tm._generation(conn, ("en", "zh")) == 0
        tm._bump_generations(conn, {("en", "zh")})
        tm._bump_generations(conn, {("en", "zh")})
        assert tm._generation(conn, ("en", "zh")) == 2
        # A different pair is untouched by the first pair's bumps.
        assert tm._generation(conn, ("en", "fr")) == 0
