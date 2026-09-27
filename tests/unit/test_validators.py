"""Unit tests for 0-Token validators: HTML delta, numeric, and glossary consistency."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.stage_ctx_factory import build_stage_ctx, drain, inert_event
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.ir.models import BlockStatus, BookManifest, ChapterIR, ChapterMeta, FlowID, IRBlock
from ubt.core.ir.run_metadata import RunMetadata
from ubt.core.language_profile import FR, ZH
from ubt.core.qe.comet_runner import (
    GLOSSARY_VIOLATION_MARKER,
    QE_SCORE_GLOSSARY_VIOLATION,
    HeuristicQERunner,
)
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.validators.consistency import (
    GlossaryConsistencyValidator,
    NumericConsistencyValidator,
    _cn_numeral_value,
    canonicalize_numeric_token,
    normalize_for_numeric_matching,
)
from ubt.core.validators.html_delta import HTMLDeltaValidator

pytestmark = pytest.mark.fast


def test_html_delta_flawless_case() -> None:
    """Validate that well-formed tags and images pass seamlessly."""
    validator = HTMLDeltaValidator()
    src = 'Here is an illustration: <img src="figures/brain.png" alt="Brain Diagram"/> and markdown ![Chart](data/chart.svg).'
    tgt = '这是一张插图：<img src="figures/brain.png" alt="大脑结构图"/> 以及 Markdown 图片 ![图表](data/chart.svg)。'

    res = validator.validate(src, tgt)
    assert res.is_valid is True
    assert res.error_code is None


def test_html_delta_catches_unescaped_quote_corruption() -> None:
    """Validate detection of LLM unescaped quotes breaking alt attributes."""
    validator = HTMLDeltaValidator()
    src = '<img src="cat.jpg" alt="A cute cat"/>'
    # Broken: unescaped quote in alt="A cute "kitty""
    tgt = '<img src="cat.jpg" alt="一只可爱的"小猫""/>'

    res = validator.validate(src, tgt)
    assert res.is_valid is False
    assert res.error_code == "HTML_DELTA_MISMATCH"
    assert res.suggested_action == "RETRY"
    assert "unescaped quote" in (res.message or "")


def test_html_delta_catches_missing_or_extra_images() -> None:
    """Validate detection of dropped or fabricated image URLs."""
    validator = HTMLDeltaValidator()
    src = '<img src="fig1.png"/> <img src="fig2.png"/>'
    tgt = '<img src="fig1.png"/>'  # fig2 missing

    res = validator.validate(src, tgt)
    assert res.is_valid is False
    assert "Missing HTML <img src>" in (res.message or "")


def test_numeric_consistency_validator() -> None:
    """Validate that numbers, percentages, and historical dates are strictly preserved."""
    validator = NumericConsistencyValidator()
    src = "In 1984, the population increased by 15.6% to 2,500,000 citizens."

    # Preserved numbers (commas stripped naturally)
    tgt_good = "在 1984 年，人口增长了 15.6%，达到了 2500000 名公民。"
    assert validator.validate(src, tgt_good).is_valid is True

    # Missing percentage and date
    tgt_bad = "人口增长了很多，达到了两百五十万公民。"
    res = validator.validate(src, tgt_bad)
    assert res.is_valid is False
    assert res.error_code == "NUMERIC_INCONSISTENCY"
    assert "1984" in (res.message or "")
    assert "15.6" in (res.message or "")


def test_glossary_consistency_validator() -> None:
    """Validate that defined terms are faithfully reflected in target translations."""
    glossary = [
        {"source": "Cognitive Dissonance", "translation": "认知失调"},
        {"source": "Working Memory", "translation": "工作记忆"},
    ]
    validator = GlossaryConsistencyValidator(glossary)

    src = "Festinger introduced Cognitive Dissonance theory while studying Working Memory."
    tgt_good = "费斯汀格在研究工作记忆时提出了认知失调理论。"
    assert validator.validate(src, tgt_good).is_valid is True

    tgt_drift = "费斯汀格在研究短期记忆时提出了认知冲突理论。"
    res = validator.validate(src, tgt_drift)
    assert res.is_valid is False
    assert res.error_code == "GLOSSARY_DRIFT"
    assert "Cognitive Dissonance -> 认知失调" in (res.message or "")
    assert "Working Memory -> 工作记忆" in (res.message or "")


def test_glossary_expected_side_is_case_and_boundary_tolerant() -> None:
    """The target side must match like the source side.

    The expected-rendering check used a bare case-sensitive substring test, so a
    Latin term with an ordinary surrounding space (or different case) was flagged
    as drift and sent to needless repair. Both sides now go through the single
    detection primitive (``ubt.core.qe.term_drift``), which folds case on both
    sides and uses the enforcer's boundary-aware matcher.
    """
    glossary = [{"source": "FinFET", "translation": "FinFET器件"}]
    validator = GlossaryConsistencyValidator(glossary)
    # Space inserted between the Latin token and the CJK noun: still correct.
    assert validator.validate("A FinFET scales down.", "一种 FinFET器件 被缩小。").is_valid
    # Casing of the Latin token differs: still the decided rendering.
    assert validator.validate("A FinFET scales down.", "一种 finfet器件被缩小。").is_valid
    # A genuinely absent rendering is still caught.
    assert not validator.validate("A FinFET scales down.", "一种晶体管被缩小。").is_valid


def test_numeric_sub_superscript_normalization() -> None:
    """Spaced-subscript extraction ('β 2') vs normalized translation ('β²'/'β₀').

    Real ch3 narrative-formula failures: the model correctly renders the
    subscript while the source keeps the PDF extraction space.
    """
    validator = NumericConsistencyValidator()
    assert (
        validator.validate(
            "solving Eq. (3.11) for the β 2 term inside the square root",
            "通过对式(3.11)中根号内的β²项求解",
        ).is_valid
        is True
    )
    assert (
        validator.validate(
            "where β 0 is obtained from Eq. (A.7) using F th",
            "其中 β₀ 由式(A.7)求得，所用的 F_th",
        ).is_valid
        is True
    )
    # A genuinely dropped number still fails.
    res = validator.validate("the β 2 term inside the square root", "根号内的β项")
    assert res.is_valid is False


def test_numeric_consistency_boundary_check() -> None:
    """Verify that numbers with matching prefixes/suffixes like 10 in 100 or 5 in 50 do not false-pass."""
    validator = NumericConsistencyValidator()
    # 10 is missing because target has 100 instead of 10
    res = validator.validate("Only 10 participants responded.", "只有100名参与者做出了回应。")
    assert res.is_valid is False
    assert res.error_code == "NUMERIC_INCONSISTENCY"
    assert "10" in (res.message or "")

    # Target has 10
    res_good = validator.validate("Only 10 participants responded.", "只有10名参与者做出了回应。")
    assert res_good.is_valid is True

    # 5 is missing because target has 50
    res2 = validator.validate("Group of 5 students.", "50名学生的群体。")
    assert res2.is_valid is False
    assert "5" in (res2.message or "")


def test_numeric_trailing_decimal_zero_equivalence() -> None:
    """'1.5' and '1.50' are the same magnitude and must not read as a lost number."""
    validator = NumericConsistencyValidator()
    assert validator.validate("the value is 1.5 units.", "值是 1.50 单位。").is_valid
    assert validator.validate("the value is 1.50 units.", "值是 1.5 单位。").is_valid
    # A genuinely different magnitude is still caught.
    assert not validator.validate("the value is 1.5 units.", "值是 1.6 单位。").is_valid


def test_non_numeral_cjk_characters_do_not_mask_lost_numbers() -> None:
    """A numeral character embedded in a CJK word ('统一') is not a number."""
    validator = NumericConsistencyValidator()
    res = validator.validate("See Section 1 for details.", "本章统一描述。")
    assert not res.is_valid
    assert res.error_code == "NUMERIC_INCONSISTENCY"

    # A genuine standalone Chinese numeral still matches (H11 behavior kept).
    assert validator.validate("Section 5 details.", "五项细节。").is_valid


def test_cjk_numeral_with_measure_word_is_not_a_false_positive() -> None:
    """A number rendered as a CJK numeral + unit must not be read as missing.

    Regression (inding A4): the context regex only normalised a numeral
    run bounded by non-CJK text, so a numeral flanked by CJK on both sides was
    never converted. The digit check then reported the source value as missing,
    and because a numeric flag is a *structural* defect no QE score could
    release it: a correct translation was escalated to repair and could end up
    BLOCKED_HUMAN with a source-only placeholder in place of good output.
    """
    validator = NumericConsistencyValidator()
    assert validator.validate(
        "The contract value was 200 yuan in total.", "合同总金额为两百元。"
    ).is_valid
    assert validator.validate("There were 20 people.", "共二十人。").is_valid
    assert validator.validate("It is 30 meters long.", "长达三十米。").is_valid
    assert validator.validate("It lasted 30 seconds.", "持续三十秒。").is_valid
    assert validator.validate("There are five items.", "有五项。").is_valid
    # Digit forms keep working alongside the numeral forms.
    assert validator.validate("There were 20 people.", "共有20人。").is_valid
    # A genuinely wrong value is still caught in the numeral form.
    assert not validator.validate("There were 20 people.", "共三十人。").is_valid


def test_cjk_numeral_normalization_never_fabricates_a_digit() -> None:
    """A numeral character inside a word must not become a number.

    Regression (inding A4, the reverse direction): the old regex
    rewrote a word-final numeral character at a string boundary, so '统一'
    became '统1'. That *fabricated* a "1" in the match-only view, which could
    mask a source '1' that the translation really had dropped — the precise
    failure the previous comment claimed to prevent.
    """
    for word in ("统一", "万一", "十分重要", "一度认为", "进一步", "一面说", "一部分"):
        assert normalize_for_numeric_matching(word, "zh") == word

    validator = NumericConsistencyValidator()
    # Source '1' really is absent from the translation -> must still be caught.
    assert not validator.validate("See Section 1 for details.", "统一描述。").is_valid


def test_cjk_numerals_normalize_only_in_genuine_numeral_contexts() -> None:
    """Normalisation is a match-only view: units and magnitude suffixes included.

    Known, deliberate limitation (see the comment on ``_CN_NUMERAL_CONTEXT_RE``):
    a chengyu shaped like <numeral><measure-word> is still normalised
    ('三番五次' -> '三番5次'), and a lone single-character numeral closed only by
    punctuation ('数量为三。') is not. Tightening either direction would trade the
    common false negative (a correct translation flagged as a missing number and
    quarantined) for the rare false positive, so the trade-off is intentional and
    only a lexicon could improve it.
    """
    cases = {
        "为两百元。": "为200元。",
        "共二十人。": "共20人。",
        "长达三十米。": "长达30米。",
        "五项": "5项",
        "第七章": "第7章",
        "见第七章": "见第7章",
        "人数为二十。": "人数为20。",
        "共三十余次": "共30余次",
        "二十多个": "20多个",
        "三人同行。": "3人同行。",
        # 万/亿 are magnitude suffixes: the digit check then expands '20万'.
        "二十万元": "200000元",
    }
    for source, expected in cases.items():
        assert normalize_for_numeric_matching(source, "zh") == expected, source


def test_cjk_numeral_shorthand_and_zero_placeholder() -> None:
    """Colloquial shorthand fills the next-lower unit; a 零 placeholder disables it.

    Regression: the parser summed a trailing bare digit as a units value after a
    unit was consumed, so '一百二' (120) read as 102 and '两千五' (2500) as 2005.
    A correct shorthand translation therefore mismatched its source digit and was
    escalated as a structural numeric defect. The 零 placeholder ('一千零五' =
    1005) and positional concatenation ('一九八四' = 1984) must keep working.
    """
    assert _cn_numeral_value("一百二") == 120
    assert _cn_numeral_value("两千五") == 2500
    assert _cn_numeral_value("一千五") == 1500
    # Zero placeholder marks an order gap: the trailing digit is a units value.
    assert _cn_numeral_value("一千零五") == 1005
    assert _cn_numeral_value("二百零三") == 203
    # Complete and positional forms are unchanged.
    assert _cn_numeral_value("一百二十三") == 123
    assert _cn_numeral_value("一百二十") == 120
    assert _cn_numeral_value("一九八四") == 1984
    assert _cn_numeral_value("二零二零") == 2020

    validator = NumericConsistencyValidator()
    assert validator.validate("The value is 120.", "数值为一百二。").is_valid
    assert validator.validate("The value is 2500.", "数值为两千五。").is_valid
    assert validator.validate("The value is 1005.", "数值为一千零五。").is_valid
    # A genuinely wrong shorthand value is still caught.
    assert not validator.validate("The value is 120.", "数值为两百五。").is_valid


def test_fullwidth_source_digits_are_not_reported_lost() -> None:
    """'１９８４' in the source must match '1984' in the translation.

    User-visible failure prevented: ``str.isdigit()`` counts full-width digits
    as digits, so ``_NUM.findall`` collected '１９８４' verbatim into
    ``src_nums``, while the target-side match view normalizes full-width
    digits to ASCII ('１９８４' -> '1984'). The check therefore compared an
    un-normalizable token against normalized text, never matched, and
    ``NUMERIC_INCONSISTENCY`` quarantined a correct translation as
    ``BLOCKED_HUMAN`` — a book whose chapters carried full-width digits could
    never pass the gate.
    """
    validator = NumericConsistencyValidator()
    src = "１９８４年に小説が出版された。"
    tgt = "小说于 1984 年出版。"
    assert validator.validate(src, tgt).is_valid

    # A genuinely missing number is still caught after the fold.
    assert not validator.validate(src, "小说于 1990 年出版。").is_valid


def test_fullwidth_source_matches_fullwidth_target_and_superscripts_stay_numeric_neutral() -> None:
    """Both sides fold to the same ASCII domain; superscripts never merge into tokens."""
    validator = NumericConsistencyValidator()
    # Both sides full-width: without the fold the sides never shared a token.
    assert validator.validate("価格は１２３円です。", "价格是123日元。").is_valid
    assert validator.validate("価格は123円です。", "价格是１２３日元。").is_valid
    # Superscript source digits are not numeric tokens: folding them before
    # tokenization would manufacture '102' out of '10²' (and the gate would
    # then quarantine every 'x²' in the book as a lost number). The literal
    # '10' next to the unit is still tracked and matched.
    assert validator.validate("The area is 10² cm².", "面积为 10 平方厘米。").is_valid
    # And the fold does not turn ordinary numbers into data loss.
    assert validator.validate("In 1984 the count was 12.", "1984 年，计数为 12。").is_valid


def test_three_decimal_fraction_is_not_mistaken_for_thousands() -> None:
    """'1.500 g' (three decimals) must also license its decimal reading '1.5'.

    User-visible failure prevented: canonicalize strips a dot followed by
    exactly three digits as a German-style thousands separator
    ('1.500' -> '1500'), so a translation rendering the same quantity as
    '1.5 g' shared no token with the source and NUMERIC_INCONSISTENCY
    quarantined it — same BLOCKED_HUMAN family as the full-width defect.
    """
    validator = NumericConsistencyValidator()
    assert validator.validate("The sample weighed 1.500 g.", "样品质量为 1.5 克。").is_valid
    # The thousand reading still matches a literal integer rendering.
    assert validator.validate("The sample weighed 1.500 g.", "样品质量为 1500 毫克。").is_valid
    # A genuinely different value matches neither reading.
    assert not validator.validate("The sample weighed 1.500 g.", "样品质量为 2.5 克。").is_valid


def test_two_decimal_and_german_thousands_semantics_unchanged() -> None:
    """The ambiguity license must not weaken the unambiguous cases."""
    validator = NumericConsistencyValidator()
    # Two-decimal fractions were never ambiguous and stay exact.
    assert validator.validate("Price: 1.50 euros.", "价格：1.5 欧元。").is_valid
    assert not validator.validate("Price: 1.50 euros.", "价格：2.5 欧元。").is_valid
    # A true thousands dot still canonicalizes to the integer.
    assert validator.validate("The town has 1.500 residents.", "该镇有 1500 名居民。").is_valid
    assert not validator.validate("The town has 1.500 residents.", "该镇有 2500 名居民。").is_valid


def test_century_decade_idiom_is_not_a_lost_number() -> None:
    """'the 1980s' rendered as '20世纪80年代' is correct Chinese, not a defect.

    Found by the real-model baseline (`tests/integration/test_local_model_baseline.py`):
    the local model translated "reported in the 1980s" as
    '短沟效应最早由人们在 20 世纪 80 年代报告。' and the numeric gate reported
    'Missing numeric tokens: [1980]'. A numeric flag is STRUCTURAL, so no QE score
    could release it: a correct, idiomatic translation was escalated to repair and
    could be quarantined as BLOCKED_HUMAN. Same failure family as the CJK-numeral
    case above.
    """
    validator = NumericConsistencyValidator()
    assert validator.validate(
        "Short-channel effects were first reported in the 1980s.",
        "短沟效应最早由人们在 20 世纪 80 年代报告。",
    ).is_valid
    # The digit form needs no rewriting, and both spellings must agree.
    assert validator.validate("It happened in the 1990s.", "这发生在1990年代。").is_valid
    assert validator.validate("It happened in the 1990s.", "这发生在20世纪90年代。").is_valid
    assert normalize_for_numeric_matching("20世纪80年代", "zh") == "1980年代"
    # A genuinely wrong decade is still caught.
    assert not validator.validate("It happened in the 1990s.", "这发生在20世纪80年代。").is_valid
    # Fully-Chinese numerals: '二十世纪八十年代' is the same idiom, and the
    # context normaliser folds only one side, so the century pattern must run
    # before it (or neither spelling matches and a correct translation is lost).
    assert normalize_for_numeric_matching("二十世纪八十年代", "zh") == "1980年代"
    assert validator.validate("It happened in the 1980s.", "这发生在二十世纪八十年代。").is_valid


def test_scientific_notation_expands_to_its_value() -> None:
    """'1e5' denotes 100000: writing the expanded value is not a dropped number.

    ``_NUM`` tokenized '1e5' as '1' and '5', so a target correctly rendering
    100000 shared no token and was quarantined as a lost number.
    """
    validator = NumericConsistencyValidator()
    assert normalize_for_numeric_matching("1e5", "en") == "100000"
    assert validator.validate("The speed is 1e5 m/s.", "速度是 100000 米/秒。").is_valid
    # A genuinely different value is still caught.
    assert not validator.validate("The speed is 1e5 m/s.", "速度是 1000 米/秒。").is_valid


def test_numeric_consistency_validator_with_profiles() -> None:
    """Fix 8: Verify NumericConsistencyValidator integrates cleanly with LanguageProfile."""
    val_zh = NumericConsistencyValidator(profile=ZH)
    val_fr = NumericConsistencyValidator(profile=FR)

    # Chinese structural numerals should pass under ZH profile
    res_zh = val_zh.validate("Chapter 7 begins.", "第七章开始。")
    assert res_zh.is_valid

    # Plain digits
    res_fr = val_fr.validate("In 1984 there were 42 cats.", "En 1984 il y avait 42 chats.")
    assert res_fr.is_valid

    # Missing number detection
    res_fr_fail = val_fr.validate("In 1984 there were 42 cats.", "En 1984 il y avait des chats.")
    assert not res_fr_fail.is_valid
    assert "42" in str(res_fr_fail.details.get("lost_numbers", []))


def test_numeric_consistency_range_support() -> None:
    """Verify that NumericConsistencyValidator accepts ranges like 1984-1985 translated to 1984年至1985年."""
    validator = NumericConsistencyValidator()

    # 1. Standard range translated to natural Chinese
    src = "The research was conducted during 1984-1985."
    tgt_good = "该研究于1984年至1985年期间开展。"
    res = validator.validate(src, tgt_good)
    assert res.is_valid is True, f"Failed: {res.message}"

    # 2. Dash variation (en-dash / em-dash)
    src_dash = "Data from 2010–2020 was analyzed."
    tgt_dash = "分析了2010年至2020年的数据。"
    assert validator.validate(src_dash, tgt_dash).is_valid is True

    # 3. Genuinely missing one of the numbers in the range
    tgt_bad = "该研究于1984年开展。"  # 1985 missing
    res_bad = validator.validate(src, tgt_bad)
    assert res_bad.is_valid is False
    assert "1984-1985" in (res_bad.message or "")


def test_html_delta_math_inequalities_immunity() -> None:
    """Verify that HTMLDeltaValidator does not crash or fail on mathematical inequalities like a < b and c > d."""
    validator = HTMLDeltaValidator()

    src = "For any valid index, if a < b and c > d, the condition holds true."
    tgt = "对于任何有效索引，若 a < b 且 c > d，则该条件成立。"

    res = validator.validate(src, tgt)
    assert res.is_valid is True, f"Failed: {res.message}"

    # Broken HTML img tag should still be caught
    src_img = '<img src="fig.png" alt="diagram"/>'
    tgt_img = '<img src="fig.png" alt="图示"bad_attr""/>'
    res_img = validator.validate(src_img, tgt_img)
    assert res_img.is_valid is False
    assert res_img.error_code == "HTML_DELTA_MISMATCH"


def test_numeric_consistency_validator_idiom_exemptions() -> None:
    """Verify that numbers embedded in standard idioms/rhetorical phrases are exempted."""
    validator = NumericConsistencyValidator()

    # 'at sixes and sevens' -> '乱七八糟' (no literal 6 or 7 in target)
    res1 = validator.validate(
        original="Everything was at sixes and sevens after the announcement.",
        translated="消息公布后，现场一片混乱，毫无头绪。",
    )
    assert res1.is_valid, f"Expected valid, got: {res1.message}"

    # 'Catch-22' -> '无法摆脱的困境' (no literal 22 in target)
    res2 = validator.validate(
        original="It is a classic Catch-22 situation for the researchers.",
        translated="这对研究人员而言是一个典型的进退维谷的困局。",
    )
    assert res2.is_valid, f"Expected valid, got: {res2.message}"

    # Actual factual numbers like 1984 or 42 must still be strictly required
    res3 = validator.validate(
        original="The laboratory was established in 1984 with 42 employees.",
        translated="该实验室成立，拥有员工。",
    )
    assert not res3.is_valid
    message = res3.message or ""
    assert "1984" in message or "42" in message


def test_numeric_consistency_with_digit_idioms() -> None:
    """Verify that genuine digit-containing idioms like 24/7 and 9-to-5 are exempted."""
    validator = NumericConsistencyValidator()

    # 24/7 -> 全天候 (digits 24 and 7 omitted)
    res1 = validator.validate(
        original="The server operates 24/7 without interruption.",
        translated="该服务器全天候不间断运行。",
    )
    assert res1.is_valid, f"Expected valid for 24/7, got: {res1.message}"

    # 9-to-5 -> 朝九晚五 (or 日常工作)
    res2 = validator.validate(
        original="He was tired of the typical 9-to-5 routine.",
        translated="他厌倦了千篇一律的日常上班生活。",
    )
    assert res2.is_valid, f"Expected valid for 9-to-5, got: {res2.message}"

    # top 10 -> 名列前茅
    res3 = validator.validate(
        original="The university ranks in the top 10 globally.",
        translated="这所大学在全世界名列前茅。",
    )
    assert res3.is_valid, f"Expected valid for top 10, got: {res3.message}"


def test_html_delta_formatting_drift_is_advisory_not_failure() -> None:
    """Emphasis-tag drift and a dropped anchor warn; they never gate RETRY."""
    validator = HTMLDeltaValidator()
    src = 'Read the <b>landmark</b> ruling, details <a href="https://ex.com/r">here</a>.'
    tgt = "阅读这项裁决，<a>此处</a>见详情。"  # <b> removed, href lost

    res = validator.validate(src, tgt)
    assert res.is_valid is True  # advisory tier must not block
    joined = " | ".join(res.details.get("formatting_warnings", []))
    assert "formatting tag drift: <b> x1 in source, x0 in target" in joined
    assert "anchor href dropped: https://ex.com/r" in joined


def test_html_delta_balanced_formatting_has_no_warnings() -> None:
    validator = HTMLDeltaValidator()
    src = "The <em>only</em> <strong>true</strong> path."
    tgt = "唯一<em>且</em><strong>真正</strong>的道路。"
    res = validator.validate(src, tgt)
    assert res.is_valid is True
    assert "formatting_warnings" not in res.details


def test_html_delta_structural_error_keeps_formatting_warnings() -> None:
    """A hard image failure stays RETRY and still carries the advisory notes."""
    validator = HTMLDeltaValidator()
    src = '<img src="fig.png"/> <b>bold</b> <i>ital</i>'
    tgt = "<b>粗</b>"  # image dropped, <i> lost too

    res = validator.validate(src, tgt)
    assert res.is_valid is False
    assert res.suggested_action == "RETRY"
    assert "formatting_warnings" in res.details


def test_html_delta_malformed_source_translates_faithfully() -> None:
    """Heritage dirty EPUBs carry unescaped quotes already; a target that
    preserves the same malformed shape (different recovered attribute names,
    since names derive from the translated content) must pass. The old
    name-keyed diff could never cancel and RETRIED every such block to
    exhaustion."""
    validator = HTMLDeltaValidator()
    src = '<p>Text <img src="p.png" alt="say "hi" now"> tail</p>'
    tgt = '<p>正文 <img src="p.png" alt="说“你好”现在"> 尾部</p>'
    res = validator.validate(src, tgt)
    assert res.is_valid, f"Failed: {res.message}"

    # A genuinely NEW malformed tag (target has more than the source) fails.
    tgt2 = '<p>正文 <img src="p.png" alt="图"broken""> 尾部</p>'
    res2 = validator.validate(src + '<img src="q.png" alt="ok"/>', tgt + tgt2)
    assert res2.is_valid is False
    assert res2.suggested_action == "RETRY"


def test_numeric_gate_accepts_si_symbol_abbreviations() -> None:
    """A correct "2.5 MPa" was reported as having LOST 2500000.

    Only spelled-out scale words were known, so the right scientific
    abbreviation sent the block to repair — which cannot invent a better unit —
    and it can end up BLOCKED_HUMAN with the source placeholder shipped.
    A unit word that merely starts with a prefix letter ("3 mol") must not be
    read as milli, or a real 1000x error would pass the gate.
    """
    validator = NumericConsistencyValidator()
    for src, tgt in (
        ("压力为 2500000 Pa。", "压力为 2.5 MPa。"),
        ("频率 5000000 Hz", "频率 5 MHz"),
        ("质量 0.25 g", "质量 250 mg"),
        ("电流 0.01 A", "电流 10 mA"),
    ):
        assert validator.validate(src, tgt).is_valid, (src, tgt)

    assert not validator.validate("数量为 3 mol", "数量为 3000 mol").is_valid
    from ubt.core.validators.consistency import _scale_map

    assert _scale_map("3 mol") == {}


@pytest.mark.fast
def test_consistency_validator_leading_zero_date_matching() -> None:
    validator = NumericConsistencyValidator()

    # Source has 2023-05-09, target translates to 2023年5月9日
    src = "Published on 2023-05-09 in Journal."
    tgt = "发表于 2023年5月9日 的期刊上。"
    res = validator.validate(src, tgt)
    assert res.is_valid is True, f"Failed date matching with leading zero: {res.details}"


@pytest.mark.fast
def test_unicode_math_and_fraktur_not_flagged_as_hallucinated_latex() -> None:
    from ubt.core.validators.math_guard import novel_unsupported_latex_commands

    # \vdash from Unicode ⊢ (U+22A2)
    assert (
        novel_unsupported_latex_commands(
            "a typing judgment Γ ⊢ 𝑡 : 𝑇 states that",
            "类型判断 $\\Gamma \\vdash t : T$ 表明",
        )
        == []
    )
    # \emptyset from Unicode ⌀ (U+2300)
    assert (
        novel_unsupported_latex_commands(
            "where 𝑝 𝑚 ∩ (𝑑 𝑛 ∪ 𝑝 𝑛 ) ≠ ⌀ , Ψ 𝑡 = id Γ ;",
            "其中 $p_m \\cap (d_n \\cup p_n) \\neq \\emptyset$，$\\Psi_t = id_{\\Gamma}$；",
        )
        == []
    )
    # \mathfrak from Unicode Fraktur 𝔈 (U+1D508)
    assert (
        novel_unsupported_latex_commands(
            "set remains an effect function ( 𝔈 ∗ Σ iso )",
            "set 仍为效应函数（$\\mathfrak{E} * \\Sigma_{iso}$）",
        )
        == []
    )


@pytest.mark.fast
def test_typstify_math_handles_unicode_alphanumerics_and_font_commands() -> None:
    from ubt.adapters.pdf.overlay_text import typstify_math

    for expr in (
        r"D(X) = \mathbb{N} \to X",
        r"\mathfrak{E} * \Sigma_{iso}",
        "𝔈Γ",
        "Σ → 𝖬𝖺𝗒𝖻𝖾",
        "𝒜𝑘",
        "𝜎𝛾",
        r"\\varphi",
        "𝑝 : 𝔓 Γ ≔𝖲𝖾𝗍(𝐾)",
        "σγ ⊧ d",
    ):
        res = typstify_math(expr)
        assert res is not None, f"typstify_math returned None for {expr!r}"


@pytest.mark.fast
def test_numeric_validator_allows_glued_page_ranges_and_footnote_superscripts() -> None:
    validator = NumericConsistencyValidator()

    # Glued PDF page range 'pp. 4046' repaired by translator to '第 40–46 页'
    res_range = validator.validate(
        "D. Birsan, 'On Plug-ins and Extensible Architectures,' ACM Queue , vol. 3, no. 2, pp. 4046, 2005",
        "D. Birsan，“论插件与可扩展架构”，ACM Queue，第 3 卷，第 2 期，第 40–46 页，2005",
    )
    assert res_range.is_valid is True, res_range.message

    # Flattened inline footnote superscript marker 'plugins 5 ,' omitted in translation
    res_fn = validator.validate(
        "Koishi is an open-source chatbot application framework built on Cordis 4 . Over four years of development, it has accumulated over 4000 community-contributed plugins 5 , ranging from instant-messaging adapters.",
        "Koishi 是构建于 Cordis 之上的开源聊天机器人应用框架。经过四年多的发展，已积累 4000 余个社区贡献的插件，涵盖即时通讯适配器。",
    )
    assert res_fn.is_valid is True, res_fn.message


@pytest.mark.fast
def test_unicode_script_and_fraktur_preserve_math_font_style() -> None:
    from ubt.adapters.pdf.overlay_text import render_overlay_line, typstify_math

    assert typstify_math("𝒞") == "cal(C)"
    assert typstify_math("𝒞\ufe00") == "cal(C)"
    assert typstify_math("𝔈") == "frak(E)"
    # Bare 𝒞 in prose should also be wrapped as $cal(C)$ so Typst renders
    # Computer Modern Calligraphic instead of Chancery script (mathscr)
    rendered = render_overlay_line("范畴 𝒞 上的单子 $(T, \\eta, \\mu)$", math_probe=lambda b: True)
    assert "$cal(C)$" in rendered
    assert "𝒞" not in rendered


def test_numeric_idiom_exemption_is_occurrence_scoped() -> None:
    from ubt.core.validators.consistency import NumericConsistencyValidator

    v = NumericConsistencyValidator()
    src = "The model ranks in the top 10 globally. Chapter 10 explains the method."
    tgt = "该模型在全球排名前十。"
    assert v.validate(src, tgt).is_valid is False
    # The idiom on its own still passes (idiomatic rendering without the digit).
    assert v.validate("It is in the top 10.", "它位列前十。").is_valid is True


def test_numeric_gate_rejects_changed_decimal() -> None:
    from ubt.core.validators.consistency import NumericConsistencyValidator

    v = NumericConsistencyValidator()
    assert v.validate("The value is 3 units.", "The value is 3.5 units.").is_valid is False
    assert v.validate("There are 5 items.", "There are 3.5 items.").is_valid is False
    # A sentence-final period is not a decimal: "3." is still the number 3.
    assert v.validate("The value is 3.", "The value is 3.").is_valid is True


@pytest.mark.fast
def test_superscript_power_is_not_a_lost_number() -> None:
    from ubt.core.validators.consistency import NumericConsistencyValidator

    result = NumericConsistencyValidator().validate("The area is 10^2 m.", "面积为 10² 米。")
    assert result.is_valid, result.message


@pytest.mark.fast
def test_canonicalize_numeric_token_preserves_zero_leading_three_digit_decimals() -> None:
    """0.125 and 0.500 are decimals, never thousands-separated integers."""
    assert canonicalize_numeric_token("0.125") == "0.125"
    assert canonicalize_numeric_token("0.500") == "0.5"
    assert canonicalize_numeric_token("1.500") == "1500"
    assert canonicalize_numeric_token("1,500") == "1500"


_r0917_BAD_TARGET = "当沟道长度缩短至 20 nm 时，短沟道效应会加剧。"

_r0917_GLOSSARY: list[dict[str, Any]] = [
    {"source": "subthreshold swing", "translation": "亚阈值摆幅", "aliases": []}
]

_r0917_GOOD_TARGET = "当沟道长度缩短至 20 nm 时，亚阈值摆幅会退化。"

_r0917_SRC = "The subthreshold swing degrades as the channel length shrinks to 20 nm."


def _r0917_manifest(**metadata: Any) -> BookManifest:
    """Run decisions go on the typed contract; source keys stay in the dict."""
    run_keys = {k: v for k, v in metadata.items() if k in RunMetadata.model_fields}
    artifact = {k: v for k, v in metadata.items() if k not in RunMetadata.model_fields}
    return BookManifest(
        doc_id="d1",
        title="Regression",
        source_path="/tmp/regression.md",
        chapters=[ChapterMeta(chapter_id="c1", title="Chapter One", spine_index=1)],
        metadata=artifact,
        run=RunMetadata(**run_keys),
    )


def test_quality_gate_flags_glossary_violation(tmp_path: Path) -> None:
    """A fluent target that alters an enforced term cannot auto-pass.

    Ensures terminology violations are scored and routed to repair rather than auto-passing.
    """
    ledger = SQLiteJobLedger(tmp_path / "glossary_gate.sqlite")
    manifest = _r0917_manifest()
    ledger.init_job_from_manifest("job_gloss", manifest)
    blocks = [
        IRBlock(
            id="b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            source_text=_r0917_SRC,
            target_text=_r0917_GOOD_TARGET,
            status=BlockStatus.DRAFTED,
        ),
        IRBlock(
            id="b2",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            source_text=_r0917_SRC,
            target_text=_r0917_BAD_TARGET,
            status=BlockStatus.DRAFTED,
        ),
    ]
    ledger.append_chapter(
        "job_gloss",
        ChapterIR(doc_id="d1", chapter_id="c1", title="Chapter One", spine_index=1, blocks=blocks),
    )

    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_gloss",
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
        qe_runner=HeuristicQERunner(),
        create_event=inert_event,
        glossary_dicts=_r0917_GLOSSARY,
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))

    by_id = {b.id: b for b in ledger.get_all_blocks("job_gloss")}
    assert by_id["b1"].status is BlockStatus.MTQE_PASSED
    assert by_id["b2"].status is BlockStatus.REPAIR_PENDING
    assert any(GLOSSARY_VIOLATION_MARKER in f for f in by_id["b2"].error_flags)
    assert by_id["b2"].mtqe_score == QE_SCORE_GLOSSARY_VIOLATION
    assert (by_id["b2"].mtqe_score or 0.0) < 0.75  # below the auto-pass band
    # The correct rendering is untouched by the new signal.
    assert by_id["b1"].mtqe_score is None


def test_structural_only_verdict_still_passes_without_glossary(tmp_path: Path) -> None:
    """With no glossary threaded in, clean blocks auto-pass without flags."""
    ledger = SQLiteJobLedger(tmp_path / "no_glossary.sqlite")
    ledger.init_job_from_manifest("job_plain", _r0917_manifest())
    blocks = [
        IRBlock(
            id="b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            source_text=_r0917_SRC,
            target_text=_r0917_BAD_TARGET,
            status=BlockStatus.DRAFTED,
        )
    ]
    ledger.append_chapter(
        "job_plain",
        ChapterIR(doc_id="d1", chapter_id="c1", title="Chapter One", spine_index=1, blocks=blocks),
    )
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_plain",
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
        qe_runner=HeuristicQERunner(),
        create_event=inert_event,
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))
    by_id = {b.id: b for b in ledger.get_all_blocks("job_plain")}
    assert by_id["b1"].status is BlockStatus.MTQE_PASSED
    assert by_id["b1"].error_flags == []


def test_scale_rewriting_is_accepted_and_real_omissions_are_not() -> None:
    from ubt.core.validators.consistency import NumericConsistencyValidator

    validator = NumericConsistencyValidator()
    # (source, target, must_pass). Measured against the pre-fix code: both
    # scale cases reported a missing number and were caught below.
    cases = [
        (
            "该产品配备250万像素摄像头，售价1,200元。",
            "The device features a 2.5-megapixel camera priced at 1,200 yuan.",
            True,
        ),
        ("The device has a 2.5 million pixel camera.", "该产品配备250万像素摄像头。", True),
        ("a 150 million person country", "一个1.5亿人口的国家", True),
        ("a 30 millisecond delay", "30ms 延迟", True),
        # A genuinely dropped number stays a defect.
        ("a 2.5-megapixel camera and 7 sensors", "a camera with 7 sensors", False),
        ("配备250万像素摄像头", "配备摄像头", False),
        ("in 1984 the ratio was 15.6", "八十年代的比率", False),
    ]
    for source, target, must_pass in cases:
        result = validator.validate(source, target)
        assert result.is_valid is must_pass, f"{source!r} -> {target!r}: {result.message}"


def test_dropped_magnitude_is_rejected() -> None:
    """A quantity scaled in the source must keep its magnitude in the target."""
    v = NumericConsistencyValidator()
    # Bare digits surviving must NOT satisfy a 万-scaled source quantity.
    assert not v.validate("250万", "250").is_valid
    assert not v.validate("2.5 million", "2.5").is_valid
    # The value restated (same magnitude, any spelling) passes.
    assert v.validate("250万", "2500000").is_valid
    assert v.validate("250万", "2.5 million").is_valid


def test_bare_number_alongside_a_same_key_scaled_number_is_still_required() -> None:
    """A digit run that occurs both bare and with a magnitude word keeps both
    obligations. The scale map is keyed by the canonical digits, so a bare
    ``250`` and a scaled ``250万`` collapsed to one entry and the bare
    occurrence was checked only against the magnitude -- a dropped *or changed*
    bare number then shipped as valid. The magnitude must still be restated even
    when the bare digits survive.
    """
    v = NumericConsistencyValidator()
    src = "共 250 项，另一处 250万。"
    # Both obligations met: the bare digit is kept and the magnitude restated.
    assert v.validate(src, "共 250 items, i.e. 2.5 million.").is_valid
    # Bare number dropped while the magnitude survives.
    assert not v.validate(src, "共 2.5 million.").is_valid
    # Bare number altered while the magnitude survives.
    assert not v.validate(src, "共 251 items, i.e. 2.5 million.").is_valid
    # Magnitude dropped while the bare number survives (pre-existing guard).
    assert not v.validate(src, "共 250 items.").is_valid


def test_unit_prefix_is_not_a_magnitude() -> None:
    """'千/百' inside a measure unit ('千克', '千米') is not a x1000/x100 scale."""
    v = NumericConsistencyValidator()
    assert not v.validate("10千克", "10000").is_valid
    assert not v.validate("5千米", "5000").is_valid
    assert v.validate("10千克", "10 千克").is_valid


def test_negative_sign_is_preserved() -> None:
    v = NumericConsistencyValidator()
    assert not v.validate("Temperature is -5 C", "温度为 5 C").is_valid
    assert v.validate("Temperature is -5 C", "温度为 -5 摄氏度").is_valid
    assert v.validate("Temperature is -5 C", "温度为 −5 摄氏度").is_valid


def test_scientific_notation_expansion_does_not_desync_exemptions() -> None:
    """Exemption spans must be computed on the same expanded view as the tokens."""
    v = NumericConsistencyValidator()
    assert v.validate("Foo 1e5 plugins 5 , bar", "Foo 100000 个插件 ，bar").is_valid
