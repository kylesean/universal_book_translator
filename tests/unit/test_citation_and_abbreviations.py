"""Unit tests for academic-strategy 0-token components: citation masking and abbreviation mining."""

from ubt.core.cleaners.citation_masker import CitationMasker, count_citations
from ubt.core.memory.abbreviation_miner import (
    format_abbreviations_markdown_table,
    mine_abbreviations,
    mine_abbreviations_stream,
)


class TestCitationMasker:
    def setup_method(self) -> None:
        self.masker = CitationMasker()

    def test_mask_unmask_roundtrip_single_citation(self) -> None:
        src = "Attention mechanisms have become integral [1]. More details in [12]."
        masked, mapping = self.masker.mask(src)
        assert "[1]" not in masked and "[12]" not in masked
        assert "CITE_MASK" in masked
        assert len(mapping) == 2
        assert self.masker.unmask(masked, mapping) == src

    def test_mask_unmask_range_and_multi_citations(self) -> None:
        src = "Prior work [12-14] and combined results [3, 7, 21] confirm this."
        masked, mapping = self.masker.mask(src)
        assert "[12-14]" not in masked
        restored = self.masker.unmask(masked, mapping)
        assert restored == src
        assert count_citations(src) == 2

    def test_unmask_fuzzy_survives_llm_mutation(self) -> None:
        """LLM may mutate token casing/brackets; fuzzy matching must restore."""
        masked, mapping = self.masker.mask("See [12] for details.")
        token = next(iter(mapping))
        mutated = masked.replace(token, token.lower().replace("⟦", "[").replace("⟧", "]"))
        restored = self.masker.unmask(mutated, mapping)
        assert "[12] for details" in restored

    def test_renumbered_token_is_not_silently_swapped(self) -> None:
        """A renumbered citation token must not restore the wrong marker."""
        src = "See [12] and [14]."
        masked, mapping = self.masker.mask(src)
        tokens = list(mapping)
        tampered = masked.replace(tokens[0], tokens[0].replace("0001", "0002"))

        restored = self.masker.unmask(tampered, mapping)
        assert "[12]" not in restored

        report = self.masker.unmask_checked(tampered, mapping)
        assert not report.clean
        assert report.mismatched == [2]
        assert report.missing == [1]

    def test_bare_token_without_checksum_is_not_restored(self) -> None:
        """Regression M8: an echoed checksum-less token stays masked.

        If the model drops the ``-abc`` suffix (or emits ``CITE_MASK_0001`` as
        ordinary text), restoring it to a real citation could inject the wrong
        reference. It must be left in place so ``unmask_checked`` reports it
        unverified and the block is quarantined.
        """
        masked, mapping = self.masker.mask("See [12] for details.")
        token = next(iter(mapping))
        bare = token.replace("⟦", "").replace("⟧", "").split("-")[0]  # drop checksum
        mutated = masked.replace(token, bare)
        assert "CITE_MASK" in bare and "-" not in bare

        restored = self.masker.unmask(mutated, mapping)
        assert "[12]" not in restored  # not silently restored
        assert "CITE_MASK" in restored  # left visible for a human

        report = self.masker.unmask_checked(mutated, mapping)
        assert not report.clean
        assert report.unverified == [1]

    def test_chinese_context_roundtrip(self) -> None:
        src = "如文献 [12] 所述，注意力机制 [3, 7] 已被广泛应用。"
        masked, mapping = self.masker.mask(src)
        assert "[12]" not in masked and "[3, 7]" not in masked
        assert self.masker.unmask(masked, mapping) == src

    def test_plain_brackets_not_masked(self) -> None:
        """Non-citation brackets must stay untouched."""
        src = "The result is [a, b] and the value is [x]."
        masked, mapping = self.masker.mask(src)
        assert mapping == {}
        assert masked == src

    def test_no_citations_noop(self) -> None:
        src = "No citations here at all."
        masked, mapping = self.masker.mask(src)
        assert masked == src and mapping == {}


class TestAbbreviationMiner:
    def test_mines_canonical_academic_patterns(self) -> None:
        text = (
            "Long Short-Term Memory (LSTM) networks are widely used. "
            "We employ scaled dot-product attention (SDPA) throughout."
        )
        entries = mine_abbreviations(text)
        by_acronym = {e["aliases"][0]: e for e in entries}
        assert "LSTM" in by_acronym
        assert by_acronym["LSTM"]["source"] == "Long Short-Term Memory"
        assert "SDPA" in by_acronym
        assert by_acronym["SDPA"]["source"] == "scaled dot-product attention"

    def test_first_attestation_wins_and_dedup(self) -> None:
        text = "Convolutional Neural Network (CNN) layers. Another CNN line: Convolutional Neural Network (CNN) again."
        entries = mine_abbreviations(text)
        assert len(entries) == 1
        assert entries[0]["source"] == "Convolutional Neural Network"

    def test_rejects_common_word_pairs(self) -> None:
        text = "Television (TV) and personal computer (PC) are not terminology."
        assert mine_abbreviations(text) == []

    def test_rejects_bare_acronyms_without_expansion(self) -> None:
        text = "The (GPU) is fast."
        assert mine_abbreviations(text) == []

    def test_rejects_digit_acronyms(self) -> None:
        text = "Generation 5 (G5) models are fast."
        assert mine_abbreviations(text) == []

    def test_long_acronyms_skip_initialism_check(self) -> None:
        """LSTM-style mixed initialisms must pass; strict initialism would reject."""
        text = "Bidirectional Encoder Representations from Transformers (BERT) is pretrained."
        entries = mine_abbreviations(text)
        assert len(entries) == 1
        assert entries[0]["aliases"] == ["BERT"]

    def test_short_acronym_initialism_check(self) -> None:
        text = "The Convolutional Neural Network (CNN) wins."
        assert mine_abbreviations(text), "CNN initials must match expansion"

    def test_digit_acronyms_are_rejected(self) -> None:
        text = " ".join(f"Fake Term Number {i} (FT{i}N)" for i in range(20))
        entries = mine_abbreviations(text)  # digits in acronym → rejected
        assert entries == []

    def test_format_table_renders_pairs(self) -> None:
        entries = [{"source": "Long Short-Term Memory", "aliases": ["LSTM"]}]
        table = format_abbreviations_markdown_table(entries)
        assert "| LSTM | Long Short-Term Memory |" in table
        assert format_abbreviations_markdown_table([]) == ""


class TestAbbreviationStreamMiner:
    """mine_abbreviations_stream must match whole-text mining results."""

    def test_stream_matches_whole_text(self) -> None:
        blocks = [
            "Bidirectional Encoder Representations from Transformers (BERT) changed NLP.",
            "We also compare against Long Short-Term Memory (LSTM) baselines.",
            "Scaled dot-product attention (SDPA) drives the transformer.",
        ]
        joined = "\n".join(blocks)
        assert mine_abbreviations_stream(blocks) == mine_abbreviations(joined)

    def test_stream_finds_pair_across_block_boundary(self) -> None:
        # Expansion at the end of one block, acronym at the start of the next
        blocks = [
            "This chapter introduces Convolutional Neural Networks",
            "(CNN) in detail.",
        ]
        entries = mine_abbreviations_stream(blocks)
        assert any(e["aliases"] == ["CNN"] for e in entries)

    def test_stream_respects_max_entries(self) -> None:
        blocks = [
            f"Fake Term Name {c} (FTN{c}) appears here." for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ]
        entries = mine_abbreviations_stream(blocks, max_entries=5)
        assert len(entries) == 5

    def test_stream_forced_tiny_chunks_match_whole_text(self) -> None:
        blocks = [
            "Recurrent Neural Networks (RNN) were dominant for years.",
            "Then Attention Is All You Need replaced them.",
            "Generative Pre-trained Transformer (GPT) scaled up.",
            "Deep Belief Networks (DBN) are older history.",
        ]
        joined = "\n".join(blocks)
        assert mine_abbreviations_stream(
            blocks, chunk_chars=48, tail_chars=300
        ) == mine_abbreviations(joined)


def test_citation_drop_of_one_duplicate_is_flagged_missing() -> None:
    """§10.4-3: dropping 1 of two identical ``[12]`` must not read as clean.

    The old ``missing`` predicate used a global ``original not in restored``, so
    one surviving copy vouched for both and the silent loss opened the
    fail-closed checker.
    """
    from ubt.core.cleaners.citation_masker import CitationMasker

    masked, mapping = CitationMasker().mask("As shown [12], and again [12].")
    assert len(mapping) == 2
    t1, t2 = list(mapping)
    report = CitationMasker().unmask_checked(f"As shown {t1}, and again.", mapping)
    assert t2 not in f"As shown {t1}, and again."  # the second token is gone
    assert report.missing, "dropped duplicate citation was silently accepted"


def test_code_drop_of_one_duplicate_is_flagged_missing() -> None:
    """Same fail-open, code spans (CODE_REVIEW §10.4-3)."""
    from ubt.core.cleaners.code_masker import CodeMasker

    masker = CodeMasker()
    masked, mapping = masker.mask("run `foo()` then look `foo()` end")
    assert list(mapping.values()) == ["`foo()`", "`foo()`"], mapping
    t1, t2 = list(mapping)
    report = masker.unmask_checked(f"运行 {t1} 然后 结束", mapping)
    assert report.missing, "dropped duplicate code span was silently accepted"


def test_code_echo_detected_via_duplicated_count() -> None:
    """§10.4-4: code masker must flag the model emitting a span twice.

    Echo twin of the drop check: both tokens restored PLUS a leaked literal
    copy puts the restored count above the token count.
    """
    from ubt.core.cleaners.code_masker import CodeMasker

    masker = CodeMasker()
    masked, mapping = masker.mask("run `foo()` then look `foo()` end")
    t1, t2 = list(mapping)
    assert masker.unmask_checked(f"run {t1} look {t2} 又写 `foo()` 结束", mapping).duplicated
    # faithful: both tokens, no extra copy -> clean
    assert masker.unmask_checked(f"run {t1} look {t2} end", mapping).clean


def test_math_echo_detected_via_duplicated_count() -> None:
    """§10.4-4: math masker must flag a doubled formula (echo direction)."""
    from ubt.core.cleaners.math_masker import MathMasker

    masker = MathMasker()
    masked, mapping = masker.mask("Note $x^2$ and also $x^2$.")
    assert len(mapping) == 2
    t1, t2 = list(mapping)
    assert masker.unmask_checked(f"{t1} {t2} 另附 $x^2$", mapping).duplicated
    assert masker.unmask_checked(f"{t1} 并且 {t2}", mapping).clean


def test_citation_masker_namespaced_prefix_fuzzy_unmask() -> None:
    """Namespaced token prefix ⟦UBT:CITE:0001-...⟧ must fuzzy-unmask when mutated."""
    from ubt.core.cleaners.citation_masker import CitationMasker

    masker = CitationMasker(mask_prefix="⟦UBT:CITE:")
    text = "Refer to [42] for details."
    masked, mapping = masker.mask(text)
    assert len(mapping) == 1
    tok = next(iter(mapping))
    mutated = tok.replace("⟦", "[").replace("⟧", "]")
    draft = f"请参考 {mutated} 查看详情。"
    unmasked = masker.unmask(draft, mapping)
    assert "[42]" in unmasked
    assert "[" not in unmasked.replace("[42]", "") and "UBT:CITE" not in unmasked
