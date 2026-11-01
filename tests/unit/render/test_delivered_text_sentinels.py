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
    element = make_element(
        id=block_id,
        spine_index=1,
        block_type=kind,
        source_text=source,
    )
    return IRBlock(element=element, target_text=target, draft_text=draft)


def _masked_token(family: str, index: int = 1, original: str = "x") -> str:
    return f"⟦{family}_MASK_{index:04d}-{token_checksum(index, original)}⟧"


class TestStripIsNarrowAndComplete:
    @pytest.mark.parametrize("family", ["SOUP", "MATH", "CITE", "CODE", "EMAIL"])
    def test_every_masker_family_blanks(self, family: str) -> None:
        token = _masked_token(family)
        text = f"它仅依赖于 y 方向，且 {token} = Vtm ln(Nch/ni)。"
        out = strip_pipeline_sentinels(text)
        assert token not in out
        assert SENTINEL_PLACEHOLDER in out
        assert out.startswith("它仅依赖于 y 方向，且")
        assert out.endswith("= Vtm ln(Nch/ni)。")

    def test_typst_escaped_variant_blanks(self) -> None:
        text = r"见 ⟦SOUP\_MASK\_0001-644⟧ 与 ⟦CITE\_MASK\_0002-e09⟧ 之间"
        out = strip_pipeline_sentinels(text)
        assert "MASK" not in out
        assert out.count(SENTINEL_PLACEHOLDER) == 2

    def test_latex_bracket_wrapped_variant_blanks(self) -> None:
        text = (
            "双栅鳍式场效应晶体管的移动电子电荷密度随 $V_G$ 的变化"
            r"（在 $\llbracket SOUP\_MASK\_0001-2a3 \rrbracket$ 下），分别由所提出的模型获得。"
        )
        out = strip_pipeline_sentinels(text)

        assert "MASK" not in out
        assert "llbracket" not in out and "rrbracket" not in out
        assert SENTINEL_PLACEHOLDER in out
        assert out.count("$") == 2, "the $V_G$ span keeps its own dollars"
        assert out.startswith("双栅鳍式场效应晶体管的移动电子电荷密度随 $V_G$ 的变化（在 ")
        assert out.endswith(" 下），分别由所提出的模型获得。")

    def test_ascii_and_cjk_bracket_forms(self) -> None:
        assert SENTINEL_PLACEHOLDER in strip_pipeline_sentinels("[SOUP_MASK_0001-f2d]")
        assert SENTINEL_PLACEHOLDER in strip_pipeline_sentinels("【SOUP_MASK_0001-f2d】")

    def test_a_token_without_a_checksum_blanks(self) -> None:
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
        assert strip_pipeline_sentinels(text) == text

    def test_a_lone_closer_is_not_swallowed_backward(self) -> None:
        text = "句子甲 ⟧ 句子乙 MASK_0001 结尾"
        assert strip_pipeline_sentinels(text) == text

    def test_empty_and_token_free_inputs_pass_through(self) -> None:
        assert strip_pipeline_sentinels("") == ""
        assert strip_pipeline_sentinels("   ") == "   "


class TestDiagnosticStaysAlive:
    def test_delivered_text_sanitizes_by_default(self) -> None:
        token = _masked_token("SOUP")
        block = _block(target=f"其中 εch 是介电常数，且 {token} = Vtm ln(Nch/ni)。")
        out = delivered_text(block)
        assert token not in out
        assert SENTINEL_PLACEHOLDER in out

    def test_unsanitized_surface_keeps_the_token_for_diagnosis(self) -> None:
        token = _masked_token("SOUP")
        block = _block(target=f"且 {token} = 0")
        assert delivered_text(block, sanitize_sentinels=False) == f"且 {token} = 0"

    def test_the_draft_fallback_is_sanitized_too(self) -> None:
        token = _masked_token("SOUP")
        block = _block(target="", draft=f"草稿里也带着 {token}")
        assert token not in delivered_text(block)
        assert SENTINEL_PLACEHOLDER in delivered_text(block)

    def test_the_source_fallback_is_sanitized_too(self) -> None:
        token = _masked_token("CITE")
        block = _block(target="", draft="", source=f"source with {token}")
        assert token not in delivered_text(block)

    def test_a_quarantine_marker_still_falls_back_to_the_draft(self) -> None:
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
    @pytest.mark.parametrize("family", ["SOUP", "MATH", "CITE", "CODE", "EMAIL"])
    def test_a_blanked_text_no_longer_trips_the_guard(self, family: str) -> None:
        token = _masked_token(family)
        assert has_sentinel(token), "guard must recognise the token it protects against"
        assert not has_sentinel(strip_pipeline_sentinels(f"x {token} y")), (
            f"substituted {family} token must not still trip the guard"
        )

    def test_delivered_output_can_never_trip_the_guard_on_a_masker_token(self) -> None:
        for family in ("SOUP", "MATH", "CITE", "CODE", "EMAIL"):
            block = _block(block_id=family.lower(), target=_masked_token(family))
            assert not has_sentinel(delivered_text(block)), family


class TestMeasuredRegression:
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
        assert "$k_B T/q$" in out
        assert "$V_{tm} \\ln(N_{ch}/n_i)$" in out
        assert "沟道掺杂浓度" in out


class TestStrayHtmlTagsNeverReachThePage:
    MEASURED_TARGET = (
        "内核内部的权限分离。RustyHermit-MPK [59]</span>、CubicleOS [56]</span> "
        "和 KDPM [36]</span> 利用 Intel MPK 进行隔离。"
    )

    def test_the_measured_leak_is_deleted(self) -> None:
        block = _block(block_id="pdf_main#b0220", target=self.MEASURED_TARGET)
        out = delivered_text(block)
        assert "</span>" not in out
        assert "<span" not in out
        assert out == (
            "内核内部的权限分离。RustyHermit-MPK [59]、CubicleOS [56] 和 KDPM [36] "
            "利用 Intel MPK 进行隔离。"
        )

    def test_an_open_tag_with_attributes_is_deleted(self) -> None:
        block = _block(target='<span class="cite">引用</span> 之后继续。')
        assert delivered_text(block) == "引用 之后继续。"

    def test_math_like_comparisons_survive(self) -> None:
        block = _block(target="当 x<a>y 成立时继续。")
        assert delivered_text(block) == "当 x<a>y 成立时继续。"

    def test_diagnostics_still_see_the_raw_surface(self) -> None:
        block = _block(target="RustyHermit-MPK [59]</span> 利用 Intel MPK。")
        assert "</span>" in delivered_text(block, sanitize_sentinels=False)


class TestDegradedListMarkersAreTypedAsBullets:
    def test_a_degraded_marker_is_normalized(self) -> None:
        block = _block(
            kind=BlockType.LIST_ITEM,
            target="·Redis。我们使用官方的 Redis 基准测试工具 [53]。",
        )
        assert delivered_text(block) == "• Redis。我们使用官方的 Redis 基准测试工具 [53]。"

    def test_an_already_clean_marker_keeps_its_space(self) -> None:
        block = _block(kind=BlockType.LIST_ITEM, target="• 已有空格的条目。")
        assert delivered_text(block) == "• 已有空格的条目。"

    def test_prose_middle_dot_survives(self) -> None:
        block = _block(kind=BlockType.NARRATIVE, target="条目一 · 条目二 · 条目三。")
        assert delivered_text(block) == "条目一 · 条目二 · 条目三。"
