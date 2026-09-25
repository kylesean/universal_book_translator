import re

import pytest

from ubt.core.cleaners.math_masker import (
    MathMasker,
    count_math_spans,
    extract_math_spans,
)


def _masker() -> MathMasker:
    return MathMasker()


def test_inline_dollar_masked_and_restored() -> None:
    src = "When $T$ grows, the cache $M_{KV}$ grows too."
    masked, mapping = _masker().mask(src)
    assert len(mapping) == 2
    assert "$T$" not in masked and "$M_{KV}$" not in masked
    import re

    assert re.search(r"⟦MATH_MASK_0001-[0-9a-f]{3}⟧", masked)
    assert re.search(r"⟦MATH_MASK_0002-[0-9a-f]{3}⟧", masked)
    # Simulated translation around the opaque placeholders.
    translated = masked.replace("When", "当").replace("grows", "增长")
    restored = _masker().unmask(translated, mapping)
    assert "$T$" in restored and "$M_{KV}$" in restored
    assert "⟦MATH" not in restored


def test_display_dollar_masked_before_inline() -> None:
    src = "See $$x^2 + y^2$$ for details."
    masked, mapping = _masker().mask(src)
    assert len(mapping) == 1
    assert "$$" not in masked
    assert _masker().unmask(masked, mapping) == src


def test_latex_paren_bracket_groups_masked() -> None:
    src = r"With \(q_t\) and \[K_{1:t}\] the layer attends."
    masked, mapping = _masker().mask(src)
    assert len(mapping) == 2
    assert _masker().unmask(masked, mapping) == src


def test_currency_never_masked() -> None:
    for src in [
        "It costs $5 today.",
        "From $5 to $10 is double.",
        "Price: $3.50 or $1,000.",
    ]:
        masked, mapping = _masker().mask(src)
        assert mapping == {}, src
        assert masked == src


def test_standalone_dollar_unit_marker_never_pairs_into_a_span() -> None:
    """``($)`` is a currency column marker, not a delimiter (arXiv 2609.20519).

    Two of them on one line used to pair into a $...$ span whose content was a
    row of prose, so every table header written ``Eff. ($) ... Token ($)``
    tripped Gate 4's math-span check and burned a repair round on text the
    draft could not fix. The same pairing happened across an amount:
    ``($286.45) ... 11.6% ($)``.
    """
    for src in [
        "Token Eff. ($) ↓             | Avg. Score ↑                 | Token Eff. ($)",
        "traffic by 4.7~9.0% ($286.45) and cost per solved task by 11.6% ($)",
        "Priced at ($) per call.",
    ]:
        masked, mapping = _masker().mask(src)
        assert mapping == {}, src
        assert masked == src


def test_real_inline_math_still_masked_next_to_dollar_markers() -> None:
    """The guard must not swallow genuine spans that sit near a ($) marker."""
    src = "Eff. ($) drops as $K_{t}$ falls below 0.5 ($)"
    masked, mapping = _masker().mask(src)
    assert list(mapping.values()) == ["$K_{t}$"], mapping


def test_plain_prose_untouched() -> None:
    src = "The cache does not store thoughts or symbolic facts."
    masked, mapping = _masker().mask(src)
    assert mapping == {}
    assert masked == src


def test_unmask_fuzzy_mutation() -> None:
    src = "Scale by $d_h$ please."
    masked, mapping = _masker().mask(src)
    assert len(mapping) == 1
    # LLM rewrote the token brackets/casing but kept the checksum — restores.
    token = next(iter(mapping))
    mutated = masked.replace(token, "[math_mask_0001" + token[-5:])
    assert _masker().unmask(mutated, mapping) == src


def test_count_and_extract_helpers() -> None:
    src = "A $x$ plus $y^2$ equals \\(z\\)."
    assert count_math_spans(src) == 3
    # Token-number order (masking runs display → paren → inline), so compare
    # as multisets — preservation checks use sorted comparison too.
    assert sorted(extract_math_spans(src)) == sorted(["$x$", "$y^2$", "\\(z\\)"])
    assert count_math_spans("No math here $5.") == 0


def test_multiline_align_env_masked_as_unit() -> None:
    src = (
        "The update is\n"
        "\\begin{align}\n"
        "K_{1:t} &= [K_{1:t-1}; k_t] \\\\\n"
        "V_{1:t} &= [V_{1:t-1}; v_t].\n"
        "\\end{align}\n"
        "as shown."
    )
    masked, mapping = _masker().mask(src)
    assert len(mapping) == 1
    assert "\\begin{align}" not in masked
    assert masked.count("⟦MATH_MASK_") == 1
    restored = _masker().unmask(masked, mapping)
    assert restored == src


def test_starred_and_nested_envs() -> None:
    src = "$x = \\begin{cases} a & b \\\\ c & d \\end{cases}$ and\n\\begin{align*} y &= 1 \\\\ z &= 2 \\end{align*}"
    masked, mapping = _masker().mask(src)
    # Inner cases masked first, outer $...$ second, align* third.
    assert len(mapping) == 3
    assert "cases" not in masked and "align*" not in masked
    assert _masker().unmask(masked, mapping) == src


def test_non_math_begin_env_untouched() -> None:
    src = "\\begin{center}\nhello\n\\end{center}"
    masked, mapping = _masker().mask(src)
    assert mapping == {}
    assert masked == src


def test_unmask_checked_clean_when_intact() -> None:
    masker = _masker()
    src = "When $T$ grows, $M_{KV}$ grows."
    masked, mapping = masker.mask(src)
    draft = masked.replace("When", "当")
    report = masker.unmask_checked(draft, mapping)
    assert report.clean
    assert report.text == src.replace("When", "当")


def test_unmask_checked_missing_when_swallowed() -> None:
    masker = _masker()
    src = "A $x$ plus $y$."
    masked, mapping = masker.mask(src)
    tokens = list(mapping)
    assert len(tokens) == 2
    # Model echoed the first token but swallowed the second entirely.
    draft = masked.replace(tokens[1], "")
    report = masker.unmask_checked(draft, mapping)
    assert not report.clean
    assert report.missing == [2]
    assert "$x$" in report.text


def test_unmask_checked_mismatched_when_index_rewritten() -> None:
    masker = _masker()
    src = "A $x$ plus $y$."
    masked, mapping = masker.mask(src)
    tokens = list(mapping)
    # Model rewrote index 0001 -> 0002 with a bogus checksum: silent
    # wrong-formula restore must be caught, not shipped.
    draft = masked.replace(tokens[0], "⟦MATH_MASK_0002-000⟧")
    report = masker.unmask_checked(draft, mapping)
    assert not report.clean
    assert 2 in report.mismatched or report.missing == [1]


def test_unmask_checked_checksum_case_insensitive() -> None:
    """C3: an uppercased checksum hex restores clean (code/cite parity)."""
    import re

    masker = _masker()
    src = "Energy $E=mc^2$ here."
    masked, mapping = masker.mask(src)
    draft = re.sub(r"([0-9a-f]{3}⟧)", lambda m: m.group(1).upper(), masked)
    assert draft != masked  # the fixture really has lowercase hex to flip
    report = masker.unmask_checked(draft, mapping)
    assert report.clean
    assert report.mismatched == []
    assert report.text == src


def test_unmask_checked_mutated_residue_flagged() -> None:
    masker = _masker()
    src = "Scale by $d_h$ please."
    masked, mapping = masker.mask(src)
    # Unknown index the mapping never issued stays visible as residue.
    draft = masked + " ⟦MATH_MASK_0009-abc⟧"
    report = masker.unmask_checked(draft, mapping)
    assert not report.clean
    assert report.mutated == ["9-abc"]


def test_unmask_does_not_restore_wrong_formula_for_renumbered_token() -> None:
    """A renumbered math token must never restore another formula (checksum binding).

    User-visible failure prevented: the model rewrote token ``0001`` as
    ``0002``, the fuzzy restore matched by index alone, and the shipped block
    lost ``$x^2$`` while printing ``$y^3$`` twice. The mismatched token must be
    left in place (visible residue -> repair) instead.
    """
    masker = _masker()
    src = "能量 $x^2$ 与 $y^3$ 相关"
    masked, mapping = masker.mask(src)
    tokens = list(mapping)
    assert len(tokens) == 2
    renumbered_token = tokens[0].replace("0001", "0002")
    renumbered = masked.replace(tokens[0], renumbered_token)

    restored = masker.unmask(renumbered, mapping)
    assert restored.count("$y^3$") == 1, restored
    assert "$x^2$" not in restored
    assert renumbered_token in restored, restored

    report = masker.unmask_checked(renumbered, mapping)
    assert not report.clean
    assert report.mismatched == [2]
    assert report.missing == [1]


def test_unmask_checked_tolerates_checksumless_echo() -> None:
    """A model that faithfully echoes a token without its
    optional -ck checksum suffix must be tolerated (unverified), not flagged
    as a wrong-formula mismatch. Before the fix this triggered needless repair
    rounds for correct restorations."""
    masker = _masker()
    src = "A $x$ plus $y$."
    masked, mapping = masker.mask(src)
    tokens = list(mapping)
    assert len(tokens) == 2
    # Drop the optional checksum suffix from both echoed tokens (the masker
    # emits it as optional, so a faithful model may legally omit it).
    draft = masked
    for tok in tokens:
        idx = re.search(r"(\d+)", tok).group(1)  # type: ignore[union-attr]
        draft = draft.replace(tok, f"⟦MATH_MASK_{idx}⟧")
    report = masker.unmask_checked(draft, mapping)
    assert report.clean
    assert report.mismatched == []
    assert report.unverified == [1, 2]


def test_dropping_one_of_two_identical_math_is_flagged_missing() -> None:
    """§10.4-3: a checker that fails closed must not open on duplicate spans.

    Two identical ``$x^2$`` mask to two tokens. If the draft drops one entirely,
    the surviving copy used to vouch for both (global ``original in restored``),
    so the silent loss read as clean. The count-aware check must flag it.
    """
    src = "Note $x^2$ and also $x^2$."
    masked, mapping = _masker().mask(src)
    assert len(mapping) == 2
    t1, t2 = list(mapping)
    draft = masked.replace(t2, "")  # model kept the first token, lost the second
    report = _masker().unmask_checked(draft, mapping)
    assert report.missing, "dropped duplicate formula was silently accepted"
    assert report.text.count("$x^2$") == 1


def test_math_masker_multiline_and_spaced_display_math() -> None:
    masker = _masker()
    # Multiline display math
    multiline_text = (
        "Before math\n$$\n\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}\n$$\nAfter math"
    )
    masked, spans = masker.mask(multiline_text)
    assert len(spans) == 1, f"Expected 1 masked math span, got {len(spans)}"
    assert "\\int" not in masked

    # Spaced display math
    spaced_text = "Here is $$ E = mc^2 $$ in text"
    masked2, spans2 = masker.mask(spaced_text)
    assert len(spans2) == 1, (
        f"Expected 1 masked math span for spaced display math, got {len(spans2)}"
    )
    assert "E = mc^2" not in masked2


@pytest.mark.fast
def test_math_masker_unmasks_nested_tokens_after_fuzzy_match() -> None:
    from ubt.core.cleaners.mask_tokens import token_checksum

    # Outer token has an inner token inside its original
    masker = MathMasker()
    orig2 = "y^2"
    tok2 = f"⟦MATH_MASK_0002-{token_checksum(2, orig2)}⟧"
    orig1 = f"$x = {tok2}$"
    ck1 = token_checksum(1, orig1)
    tok1 = f"⟦MATH_MASK_0001-{ck1}⟧"

    mapping = {tok1: orig1, tok2: orig2}
    # Draft output mutated the outer token brackets: [MATH_MASK_0001-ck1]
    draft = f"公式为 [MATH_MASK_0001-{ck1}] 完成。"
    unmasked = masker.unmask(draft, mapping)
    assert "⟦MATH_MASK_" not in unmasked, f"Inner mask token was leaked: {unmasked}"
    assert "$x = y^2$" in unmasked


@pytest.mark.fast
def test_math_masker_supports_namespaced_prefix_and_avoids_scott_bracket_collision() -> None:
    """Namespaced token prefix ⟦UBT:MATH:0001-...⟧ prevents collision with Scott semantic brackets ⟦e⟧."""
    masker = MathMasker(mask_prefix="⟦UBT:MATH:")
    src = "In semantics, ⟦e1 + e2⟧ evaluates $x + y$ in env $\\rho$."
    masked, mapping = masker.mask(src)
    assert "⟦UBT:MATH:" in masked
    assert "⟦e1 + e2⟧" in masked  # Scott brackets not damaged or stripped
    assert "$x + y$" not in masked
    unmasked = masker.unmask(masked, mapping)
    assert unmasked == src
    assert "⟦e1 + e2⟧" in unmasked


@pytest.mark.fast
def test_math_masker_namespaced_prefix_fuzzy_unmask() -> None:
    """Namespaced token prefix ⟦UBT:MATH:0001-...⟧ must fuzzy-unmask when LLM mutates brackets."""
    masker = MathMasker(mask_prefix="⟦UBT:MATH:")
    src = "Formula $x^2 + y^2 = z^2$ is classic."
    masked, mapping = masker.mask(src)
    assert len(mapping) == 1
    tok = next(iter(mapping.keys()))
    # Simulate LLM bracket mutation: [UBT:MATH:0001-xxx]
    mutated_tok = tok.replace("⟦", "[").replace("⟧", "]")
    draft = f"公式 {mutated_tok} 是经典的。"
    unmasked = masker.unmask(draft, mapping)
    assert "$x^2 + y^2 = z^2$" in unmasked
    assert "[" not in unmasked and "UBT:MATH" not in unmasked
