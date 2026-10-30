"""A leaked masker token must never reach a reader-visible surface.

Measured on a real 26-page BSIM-CMG chapter (``chapter-3.pdf``, local
``gemma4-e4b``): the draft stage masked two spans, the model echoed one token
with a checksum bound to the *other* span's index, the restore pass refused the
substitution -- correctly, because restoring would have printed the wrong
formula mid-sentence -- and left ``⟦SOUP_MASK_0001-f2d⟧`` in the draft. Nothing
downstream removed it, so the token was printed in the delivered PDF's text
layer and in its HTML, EPUB and XLIFF companions.

Two things must both survive the fix, and they pull in opposite directions:

* the **reader** gets prose with a neutral ``⟨missing⟩`` bracket in place of the
  internal token, and keeps the rest of the sentence (the sentence was
  translated; only the protected span was unrecoverable);
* the **operator** keeps the ``sentinel_leaked`` / ``soup_token_corrupt`` flags,
  so the block still lands in the post-edit queue. A diagnostic written against
  the sanitized surface would go silent at exactly the moment it is needed, so
  :func:`delivered_text` exposes ``sanitize_sentinels=False`` for it.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.mask_tokens import token_checksum
from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.render.delivered_text import (
    SENTINEL_PLACEHOLDER,
    delivered_text,
    strip_pipeline_sentinels,
)
from ubt.render.typst.artifact_guard import has_sentinel

pytestmark = pytest.mark.fast


def _block(
    kind: BlockType = BlockType.NARRATIVE,
    *,
    block_id: str = "b1",
    source: str = "",
    target: str = "",
    draft: str = "",
) -> IRBlock:
    """An IRBlock whose ``id``/``source_text`` come from its element, as always."""
    element = make_element(
        id=block_id,
        spine_index=1,
        block_type=kind,
        source_text=source,
    )
    return IRBlock(element=element, target_text=target, draft_text=draft)


def _masked_token(family: str, index: int = 1, original: str = "x") -> str:
    """A token in the exact shape ``TokenFactory.next`` emits."""
    return f"⟦{family}_MASK_{index:04d}-{token_checksum(index, original)}⟧"


class TestStripIsNarrowAndComplete:
    """Every masker family blanks; meaning-carrying sentinels do not."""

    @pytest.mark.parametrize("family", ["SOUP", "MATH", "CITE", "CODE", "EMAIL"])
    def test_every_masker_family_blanks(self, family: str) -> None:
        token = _masked_token(family)
        text = f"它仅依赖于 y 方向，且 {token} = Vtm ln(Nch/ni)。"
        out = strip_pipeline_sentinels(text)
        assert token not in out
        assert SENTINEL_PLACEHOLDER in out
        # The surrounding prose is the reader's, and survives untouched.
        assert out.startswith("它仅依赖于 y 方向，且")
        assert out.endswith("= Vtm ln(Nch/ni)。")

    def test_typst_escaped_variant_blanks(self) -> None:
        r"""The measured leak shape from a *later* block: ``SOUP\_MASK\_0001``.

        Typst escaping turns ``_`` into ``\_``; a pattern keyed on the raw
        underscore would leave the escaped spelling on the page.
        """
        text = r"见 ⟦SOUP\_MASK\_0001-644⟧ 与 ⟦CITE\_MASK\_0002-e09⟧ 之间"
        out = strip_pipeline_sentinels(text)
        assert "MASK" not in out
        assert out.count(SENTINEL_PLACEHOLDER) == 2

    def test_latex_bracket_wrapped_variant_blanks(self) -> None:
        r"""The measured leak shape from 26-page replay block ``b0058``.

        A model rewrote the mask as ``$\\llbracket SOUP\\_MASK\\_0001-2a3
        \\rrbracket$``. The token is intact, but the delimiter is a LaTeX macro,
        so a stripper keyed on the ``⟦`` glyph missed it and Typst rendered the
        macros back into exactly the glyphs the artifact contract flags as a
        ``placeholder_leak``. The surrounding dollars go with the token: the
        replacement is prose, and leaving them would make Typst read the
        placeholder as a math variable.
        """
        text = (
            "双栅鳍式场效应晶体管的移动电子电荷密度随 $V_G$ 的变化"
            r"（在 $\llbracket SOUP\_MASK\_0001-2a3 \rrbracket$ 下），分别由所提出的模型获得。"
        )
        out = strip_pipeline_sentinels(text)

        assert "MASK" not in out
        assert "llbracket" not in out and "rrbracket" not in out
        assert SENTINEL_PLACEHOLDER in out
        # No orphan math delimiters left behind, and the prose survives.
        assert out.count("$") == 2, "the $V_G$ span keeps its own dollars"
        assert out.startswith("双栅鳍式场效应晶体管的移动电子电荷密度随 $V_G$ 的变化（在 ")
        assert out.endswith(" 下），分别由所提出的模型获得。")

    def test_ascii_and_cjk_bracket_forms(self) -> None:
        assert SENTINEL_PLACEHOLDER in strip_pipeline_sentinels("[SOUP_MASK_0001-f2d]")
        assert SENTINEL_PLACEHOLDER in strip_pipeline_sentinels("【SOUP_MASK_0001-f2d】")

    def test_a_token_without_a_checksum_blanks(self) -> None:
        """A model that drops the checksum suffix still leaks the index."""
        assert SENTINEL_PLACEHOLDER in strip_pipeline_sentinels("前 ⟦SOUP_MASK_0004⟧ 后")

    @pytest.mark.parametrize(
        "text",
        [
            "⟦B⟧粗体⟦/B⟧",
            "⟦I⟧斜体⟦/I⟧",
            "⟦SEMANTIC_BREAK⟧",
            "这里 ⟦x⟧ 不是机器",
            "MASK_0001 没有括号包裹",
            "普通句子，没有哨兵",
        ],
    )
    def test_meaning_bearing_and_non_tokens_are_untouched(self, text: str) -> None:
        """Emphasis markers carry content the renderer still needs.

        ``parse_emphasis`` turns ``⟦B⟧`` into a bold run and the bifurcation
        token splits a block; blanking either would destroy translation, not
        hide machinery.
        """
        assert strip_pipeline_sentinels(text) == text

    def test_a_lone_closer_is_not_swallowed_backward(self) -> None:
        """A stray ``⟧`` must not turn unrelated later text into a "token"."""
        text = "句子甲 ⟧ 句子乙 MASK_0001 结尾"
        assert strip_pipeline_sentinels(text) == text

    def test_empty_and_token_free_inputs_pass_through(self) -> None:
        assert strip_pipeline_sentinels("") == ""
        assert strip_pipeline_sentinels("   ") == "   "


class TestDiagnosticStaysAlive:
    """The operator must still be told, even though the reader no longer sees it."""

    def test_delivered_text_sanitizes_by_default(self) -> None:
        token = _masked_token("SOUP")
        block = _block(target=f"其中 εch 是介电常数，且 {token} = Vtm ln(Nch/ni)。")
        out = delivered_text(block)
        assert token not in out
        assert SENTINEL_PLACEHOLDER in out

    def test_unsanitized_surface_keeps_the_token_for_diagnosis(self) -> None:
        """``sanitize_sentinels=False`` is the diagnostic's view, not the page's."""
        token = _masked_token("SOUP")
        block = _block(target=f"且 {token} = 0")
        assert delivered_text(block, sanitize_sentinels=False) == f"且 {token} = 0"

    def test_the_draft_fallback_is_sanitized_too(self) -> None:
        """A failed block delivers its *draft*, which is where leaks were found."""
        token = _masked_token("SOUP")
        block = _block(target="", draft=f"草稿里也带着 {token}")
        assert token not in delivered_text(block)
        assert SENTINEL_PLACEHOLDER in delivered_text(block)

    def test_the_source_fallback_is_sanitized_too(self) -> None:
        token = _masked_token("CITE")
        block = _block(target="", draft="", source=f"source with {token}")
        assert token not in delivered_text(block)

    def test_a_quarantine_marker_still_falls_back_to_the_draft(self) -> None:
        """The wrapper rules and the sentinel rules must not interfere."""
        token = _masked_token("SOUP")
        block = _block(
            target='<mark class="ubt-blocked-human">原始英文</mark>',
            draft=f"真翻译带着 {token}",
        )
        out = delivered_text(block)
        assert "原始英文" not in out
        assert token not in out
        assert SENTINEL_PLACEHOLDER in out


class TestGuardParity:
    """The render-layer guard and this substitution must agree on "is a leak"."""

    @pytest.mark.parametrize("family", ["SOUP", "MATH", "CITE", "CODE", "EMAIL"])
    def test_a_blanked_text_no_longer_trips_the_guard(self, family: str) -> None:
        """Whatever this module blanks, ``artifact_guard`` must have recognised.

        ``has_sentinel`` is what makes the renderer escape a span as prose
        instead of feeding it to the math transpiler (where ``_`` is subscript
        and the compile dies). If ``strip_pipeline_sentinels`` removed a shape the
        guard still flags, that shape would reach the transpiler unescaped.
        """
        token = _masked_token(family)
        assert has_sentinel(token), "guard must recognise the token it protects against"
        assert not has_sentinel(strip_pipeline_sentinels(f"x {token} y")), (
            f"substituted {family} token must not still trip the guard"
        )

    def test_delivered_output_can_never_trip_the_guard_on_a_masker_token(self) -> None:
        """One block per family: the shipped surface carries no masker token."""
        for family in ("SOUP", "MATH", "CITE", "CODE", "EMAIL"):
            block = _block(block_id=family.lower(), target=_masked_token(family))
            assert not has_sentinel(delivered_text(block)), family


class TestMeasuredRegression:
    """The literal block from the real chapter, kept as the end-to-end anchor."""

    # pdf_main#b0014 of chapter-3.pdf, en->zh, draft+repair on local gemma4-e4b.
    MEASURED_TARGET = (
        "其中 ψ(x, y) 是沟道中的静电势，q 是电子电荷的量，ni 是本征载流子浓度，"
        "εch 是沟道（鳍）的介电常数，Vtm 是由 $k_B T/q$ 给出的热电压，"
        "Nch 是沟道掺杂浓度，且 ⟦SOUP_MASK_0001-f2d⟧ = $V_{tm} \\ln(N_{ch}/n_i)$。"
    )

    def test_the_measured_leak_is_not_shipped(self) -> None:
        block = _block(block_id="pdf_main#b0014", target=self.MEASURED_TARGET)
        out = delivered_text(block)
        assert "SOUP_MASK" not in out
        assert "⟦" not in out
        # And the block is not lost: its prose and its math both still ship.
        assert "$k_B T/q$" in out
        assert "$V_{tm} \\ln(N_{ch}/n_i)$" in out
        assert "沟道掺杂浓度" in out
