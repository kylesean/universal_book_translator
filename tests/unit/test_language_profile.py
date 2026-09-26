"""Multilingual defense-net tests: language profiles, structural numerals, script boundaries."""

import pytest

from ubt.core.language_profile import (
    PROFILES,
    ZH,
    get_pair_policy,
    get_profile,
    is_supported_lang,
    normalize_lang_code,
)
from ubt.core.memory.bible import clean_bible_entry
from ubt.core.memory.cjk_matcher import count_term_in_text
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.validators.consistency import (
    NumericConsistencyValidator,
    normalize_structural_numerals,
)


def test_zh_profile_reproduces_legacy_behaviour() -> None:
    """Default FastPassFilter must stay exactly en→zh: approve zh, reject non-CJK targets."""
    fp = FastPassFilter()
    assert fp.profile == ZH

    zh_decision = fp.evaluate(
        "Psychological research shows that sleep deprivation significantly impairs cognition.",
        "心理学研究表明，睡眠不足会显著损害认知能力。",
    )
    assert zh_decision.passed is True
    assert zh_decision.target_ratio > 0.8

    # The historical en→zh hard-coded gate rejects a perfect French translation.
    fr_decision = fp.evaluate(
        "Psychological research shows that sleep deprivation significantly impairs cognition.",
        "Les recherches en psychologie montrent que le manque de sommeil altère les capacités cognitives.",
    )
    assert fr_decision.passed is False
    assert "Insufficient zh script density" in fr_decision.reason


def test_french_profile_approves_french_translation() -> None:
    """fr profile must approve the exact translation the legacy gate rejected."""
    fp = FastPassFilter(get_profile("fr"))
    decision = fp.evaluate(
        "Psychological research shows that sleep deprivation significantly impairs cognition.",
        "Les recherches en psychologie montrent que le manque de sommeil altère les capacités cognitives.",
    )
    assert decision.passed is True


def test_japanese_profile_approves_japanese_translation() -> None:
    """ja profile gates on kana density instead of CJK ideographs."""
    fp = FastPassFilter(get_profile("ja"))
    decision = fp.evaluate(
        "Psychological research shows that sleep deprivation significantly impairs cognition.",
        "心理学の研究により、睡眠不足が認知能力を著しく損なうことが示されている。",
    )
    assert decision.passed is True


def test_unknown_profile_raises_instead_of_silent_fallback() -> None:
    with pytest.raises(ValueError, match="Unknown language profile"):
        get_profile("klingon")


def test_structural_chinese_numerals_pass_numeric_gate() -> None:
    """'Chapter 7' → '第七章' is register choice, not numeric loss: must not fire."""
    assert normalize_structural_numerals("第七章") == "第7章"
    assert normalize_structural_numerals("第十二章") == "第12章"
    assert normalize_structural_numerals("第一百二十三章") == "第123章"

    validator = NumericConsistencyValidator()
    src = "See Chapter 7 for the method, and Chapter 12 for the appendix."
    tgt = "方法请参见第七章，附录请参见第十二章。"
    assert validator.validate(src, tgt).is_valid is True
    assert FastPassFilter().evaluate(src, tgt).passed is True


def test_numeric_gate_still_blocks_real_losses() -> None:
    """Normalization must not weaken the gate: genuinely lost digits still fail."""
    validator = NumericConsistencyValidator()
    src = "See Chapter 7 and page 42 for details."
    tgt = "详情请参见第七章。"
    res = validator.validate(src, tgt)
    assert res.is_valid is False
    assert "42" in (res.message or "")


def test_fullwidth_digits_are_normalized() -> None:
    validator = NumericConsistencyValidator()
    src = "In 1984 the population reached 2,500,000."
    tgt = "１９８４年，人口达到２５０万。"  # 2500000 written with full-width digits
    assert validator.validate(src, tgt).is_valid is True


def test_cyrillic_and_accented_terms_use_word_boundaries() -> None:
    """Non-ASCII alphabetic terms get Unicode-aware boundary matching, not substring."""
    assert count_term_in_text("дом", "старый дом, домовладелец, дом.") == 2
    assert count_term_in_text("café", "Le café est proche du cafetier.") == 1
    # Legacy behaviour (bare substring) would have returned 3 and 2 respectively.


def test_bible_guardrail_handles_cjk_sources() -> None:
    """CJK source terms are length-capped in characters; alphabetic in words."""
    assert clean_bible_entry("这是一个非常非常长的中文术语词组", "long") is None
    assert clean_bible_entry("雪球", "Snowball") is not None
    # Japanese quote style is stripped alongside simplified book quotes
    entry = clean_bible_entry("Snowball", "「雪球」")
    assert entry is not None
    assert entry.translation == "雪球"


def test_all_profiles_are_consistent() -> None:
    """Profiles resolve case-insensitively and every entry is self-consistent."""
    for code in PROFILES:
        assert get_profile(code.upper()) == get_profile(code)
        assert get_profile(code).min_length_ratio < get_profile(code).max_length_ratio


def test_resolve_font_config_profiles() -> None:
    from ubt.core.language_profile import resolve_font_config

    zh = resolve_font_config("zh-CN")
    assert "Noto Sans CJK SC" in zh.typst_fonts
    assert zh.figure_prefix == "图"

    tw = resolve_font_config("zh-TW")
    assert "Noto Sans CJK TC" in tw.typst_fonts
    assert tw.figure_prefix == "圖"

    ja = resolve_font_config("ja")
    assert "Noto Sans CJK JP" in ja.typst_fonts
    assert ja.figure_prefix == "図"

    ko = resolve_font_config("ko")
    assert "Noto Sans CJK KR" in ko.typst_fonts
    assert ko.figure_prefix == "그림"

    fr = resolve_font_config("fr")
    assert "Liberation Serif" in fr.typst_fonts
    assert fr.figure_prefix == "Figure"

    de = resolve_font_config("de")
    assert fr.figure_prefix != de.figure_prefix
    assert de.figure_prefix == "Abb."


def test_typst_reconstructor_target_lang_typography(noto_cjk_installed: None) -> None:
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
    from ubt.core.ir.models import BlockType, IRBlock

    block = IRBlock(
        id="b1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="FIG. 1.2 shows the schematic.",
        target_text="FIG. 1.2 shows the schematic.",
    )

    # Test Japanese target language
    recon_ja = TypstReconstructor()
    src_ja = recon_ja.generate_typst_source([block], target_lang="ja")
    assert "Noto Sans CJK JP" in src_ja
    assert "図 1.2" in src_ja

    # Test French target language
    recon_fr = TypstReconstructor()
    src_fr = recon_fr.generate_typst_source([block], target_lang="fr")
    assert "Liberation Serif" in src_fr
    assert "Figure 1.2" in src_fr

    # Test Chinese target language
    recon_zh = TypstReconstructor()
    src_zh = recon_zh.generate_typst_source([block], target_lang="zh")
    assert "Noto Sans CJK SC" in src_zh
    assert "图 1.2" in src_zh


def test_math_text_system_prompt_parameterization() -> None:
    from ubt.core.cleaners.math_text import get_math_text_system_prompt

    prompt_zh = get_math_text_system_prompt("en", "zh")
    assert "English to Chinese" in prompt_zh

    prompt_fr = get_math_text_system_prompt("en", "fr")
    assert "English to French" in prompt_fr

    prompt_de_ja = get_math_text_system_prompt("de", "ja")
    assert "German to Japanese" in prompt_de_ja


def test_region_tag_resolves_to_base_profile() -> None:
    """Region/script subtags must resolve to their base profile instead of crashing."""
    assert get_profile("zh-CN") == ZH
    assert get_profile("zh-TW") == ZH
    assert get_profile("zh-Hans") == ZH
    assert get_profile("en-US") == get_profile("en")
    assert get_profile("en_GB") == get_profile("en")
    assert normalize_lang_code("zh-Hans") == "zh"
    assert normalize_lang_code("zh-TW") == "zh"


def test_unsupported_base_language_still_raises() -> None:
    """Normalization must not turn an unsupported language into a silent fallback."""
    with pytest.raises(ValueError, match="Unknown language profile"):
        get_profile("pt-BR")
    with pytest.raises(ValueError, match="Unknown language profile"):
        get_profile("klingon")


def test_is_supported_lang_reports_entry_point_truth() -> None:
    assert is_supported_lang("zh-CN") is True
    assert is_supported_lang("zh-TW") is True
    assert is_supported_lang("en-US") is True
    assert is_supported_lang("zh") is True
    assert is_supported_lang("pt-BR") is False
    assert is_supported_lang("it") is False
    assert is_supported_lang("klingon") is False


def test_pair_policy_uses_calibrated_band_for_region_tags() -> None:
    """en-US -> zh-CN must hit the calibrated (0.2, 1.5) band, not the generic fallback."""
    policy = get_pair_policy("en-US", "zh-CN")
    assert policy.target_code == "zh"
    assert policy.source_code == "en"
    assert (policy.min_length_ratio, policy.max_length_ratio) == (0.2, 1.5)


def test_fast_pass_filter_accepts_region_tag_target() -> None:
    from ubt.core.qe.fast_pass import FastPassFilter

    fp = FastPassFilter(source_lang="en", target_lang="zh-CN")
    decision = fp.evaluate(
        "Psychological research shows that sleep deprivation impairs cognition.",
        "心理学研究表明，睡眠不足会损害认知能力。",
    )
    assert decision.passed is True
