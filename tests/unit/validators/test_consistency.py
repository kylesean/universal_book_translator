"""Contract tests for the 0-token numeric/glossary consistency validators.

These pin the *matching-only* normalisation view (stored text is never touched)
and the false-positive guards that keep a correct translation from being
quarantined as a dropped number: CJK numerals, locale separators, scientific
notation, sub/superscripts, magnitude scale words and page-range glue.
"""

from __future__ import annotations

import pytest

from ubt.core.language_profile import ZH
from ubt.core.validators.base import ValidationResult
from ubt.core.validators.consistency import (
    GlossaryConsistencyValidator,
    NumericConsistencyValidator,
    _cn_compound_runs,
    _cn_numeral_value,
    _cn_or_ascii_value,
    _en_number_phrase_value,
    _expand_scientific_not,
    _glued_page_range_is_preserved,
    _has_negative_token,
    _has_numeric_token,
    _strip_trailing_decimal_zeros,
    canonicalize_numeric_token,
    denoted_numeric_values,
    magnitude_rewritten,
    normalize_for_numeric_matching,
    scale_equivalent_values,
)

pytestmark = pytest.mark.fast


def _check(original: str, translated: str, profile: object = None) -> ValidationResult:
    return NumericConsistencyValidator(profile).validate(original, translated)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Chinese numerals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一百二十三", 123),  # complete form
        ("一九八四", 1984),  # positional concatenation
        ("一百二", 120),  # colloquial shorthand, ×10
        ("两千五", 2500),  # colloquial shorthand, ×100
        ("一千零五", 1005),  # 零 gap disables the shorthand
        ("十", 10),  # bare 十 is 10, not 0
        ("十五", 15),
        ("二十五", 25),
        ("两百", 200),  # 两 == 二
        ("零", 0),
    ],
)
def test_cn_numeral_value(text: str, expected: int) -> None:
    assert _cn_numeral_value(text) == expected


def test_cn_or_ascii_value_dispatches_on_script() -> None:
    assert _cn_or_ascii_value("123") == 123
    assert _cn_or_ascii_value("一百") == 100


# --------------------------------------------------------------------------- #
# Scientific notation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1e5", "100000"),
        ("2E2", "200"),
        ("1.5e3", "1500.0"),
        ("1e-3", "0.001"),
        ("1e-2", "0.01"),
        ("1e+3", "1000"),
        ("a 1e5 b", "a 100000 b"),
        ("(1e5)", "(100000)"),
        ("x1e5", "x1e5"),  # word char on the left: not a number
        ("1e5y", "1e5y"),  # word char on the right: not a number
        ("1e2000", "1e2000"),  # exponent bound: left untouched
        ("1e-1001", "1e-1001"),  # negative bound too
    ],
)
def test_expand_scientific_not(text: str, expected: str) -> None:
    assert _expand_scientific_not(text) == expected


# --------------------------------------------------------------------------- #
# Decimal canonicalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.50", "1.5"),
        ("1.500", "1.5"),
        ("1.00", "1"),
        ("1.0", "1"),
        ("1.0500", "1.05"),
        ("10", "10"),
        ("0.125", "0.125"),
    ],
)
def test_strip_trailing_decimal_zeros(text: str, expected: str) -> None:
    assert _strip_trailing_decimal_zeros(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,500", "1500"),  # EN thousands comma
        ("1,234,567", "1234567"),
        ("1.500", "1500"),  # DE thousands dot
        ("1.000", "1000"),
        ("15,6", "15.6"),  # decimal comma
        ("1.50", "1.5"),
        ("0.125", "0.125"),  # leading 0 is never a separator
        ("1984", "1984"),
    ],
)
def test_canonicalize_numeric_token(text: str, expected: str) -> None:
    assert canonicalize_numeric_token(text) == expected


# --------------------------------------------------------------------------- #
# normalize_for_numeric_matching
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("１９８４", "1984"),  # full-width digits
        ("١٩٨٤", "1984"),  # Arabic-Indic digits
        ("第七章", "第7章"),  # structural marker
        ("250万", "2500000"),  # magnitude scaling
        ("1.5亿", "150000000"),
        ("1亿2000万", "120000000"),  # adjacent magnitudes sum, not concatenate
        ("二十世纪八十年代", "1980年代"),  # century/decade idiom
        ("20世纪80年代", "1980年代"),
        ("20世纪八十年代", "1980年代"),
        ("统一", "统一"),  # numeral char inside a word is not a number
        ("十分", "十分"),
        ("一部分", "一部分"),
        ("五项", "5项"),  # followed by a measure word
        ("三百米", "300米"),
        ("β²", "β^2"),  # superscript gains a separator
        ("H₂O", "H_2O"),  # subscript gains a separator
        ("1e5", "100000"),
        ("3,000", "3000"),
        ("1.50", "1.5"),
        ("三番五次", "三番5次"),  # documented residual (chengyu)
        ("数量为三。", "数量为三。"),  # documented residual (lone numeral)
    ],
)
def test_normalize_for_numeric_matching(text: str, expected: str) -> None:
    assert normalize_for_numeric_matching(text) == expected


def test_lang_gate_only_skips_chinese_magnitudes_without_digit_chars() -> None:
    # 万/亿 carry no digit char, so a non-zh target leaves them untouched...
    assert normalize_for_numeric_matching("250万", lang="en") == "250万"
    # ...but a genuine Chinese numeral char still triggers the fold.
    assert normalize_for_numeric_matching("一九八四", lang="en") == "1984"
    assert normalize_for_numeric_matching("统一", lang="en") == "统一"


# --------------------------------------------------------------------------- #
# Scale equivalence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2.5 million", {"2.5": {"2500000"}}),
        ("250万", {"250": {"2500000"}}),
        ("12.5 百万", {"12.5": {"12500000"}}),
        ("2.5 MPa", {"2.5": {"2500000"}}),  # SI symbol, case-sensitive
        ("50 kHz", {"50": {"50000"}}),
        ("3 nm", {"3": {"0.000000003"}}),
        ("10千克", {}),  # 千 as a unit prefix, not a magnitude
        ("10千米", {}),
        ("3 mol", {}),  # bare unit word, not an SI symbol
    ],
)
def test_scale_equivalent_values(text: str, expected: dict[str, set[str]]) -> None:
    assert scale_equivalent_values(text) == expected


def test_denoted_values_combine_bare_and_scaled_readings() -> None:
    assert denoted_numeric_values("12.5 million") == {"12.5", "12500000"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12.5 million", "12500000"),
        ("1.5 million", "1500000"),
        ("250万", "2500000"),
        ("12.5 百万", "12500000"),
        ("20 nm", "20 nm"),  # SI prefix keeps the number literal
        ("10千克", "10千克"),  # unit prefix guard
        ("3 mol", "3 mol"),
    ],
)
def test_magnitude_rewritten(text: str, expected: str) -> None:
    assert magnitude_rewritten(text) == expected


# --------------------------------------------------------------------------- #
# Compound magnitudes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3亿5000万", [(["3", "5000"], "350000000")]),
        ("1万2千", [(["1", "2"], "12000")]),
        ("plain text", []),
    ],
)
def test_cn_compound_runs(text: str, expected: list[tuple[list[str], str]]) -> None:
    assert _cn_compound_runs(text) == expected


# --------------------------------------------------------------------------- #
# Token presence helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("target", "num", "expected"),
    [
        ("value 3 here", "3", True),
        ("3.", "3", True),  # sentence period is not a decimal
        ("3.5", "3", False),  # fragment of a larger number
        ("0.5", "5", False),
        ("12345", "1234", False),  # prefix fragment
        ("12345", "234", False),  # interior fragment
        ("5", "05", True),  # two-digit day/month leading zero
        ("007", "7", False),  # identifier change is not a numeric match
        ("1500", "1500", True),
    ],
)
def test_has_numeric_token(target: str, num: str, expected: bool) -> None:
    assert _has_numeric_token(target, num) is expected


@pytest.mark.parametrize(
    ("target", "num", "expected"),
    [
        ("-5", "5", True),
        ("−5", "5", True),  # unicode minus
        ("负5", "5", True),
        ("5", "5", False),  # no sign
        ("a5", "5", False),  # not preceded by a sign
    ],
)
def test_has_negative_token(target: str, num: str, expected: bool) -> None:
    assert _has_negative_token(target, num) is expected


# --------------------------------------------------------------------------- #
# Glued page ranges
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("original", "translated", "num", "expected"),
    [
        ("pp. 4046", "40-46", "4046", True),
        ("pp. 4046", "40–46", "4046", True),  # en dash
        ("pp. 4046", "40/46", "4046", True),  # slash
        ("pp. 4046", "4046", "4046", False),  # no restored range
        ("pp. 4046", "40 to 46", "4046", False),  # not a range delimiter
        ("pp. 4046", "4046", "4047", False),  # digits must match
        ("pp. 1111", "11-11", "1111", False),  # first == last is not a range
        ("see 4046", "40-46", "4046", False),  # missing the pp. cue
    ],
)
def test_glued_page_range_is_preserved(
    original: str, translated: str, num: str, expected: bool
) -> None:
    assert _glued_page_range_is_preserved(original, translated, num) is expected


# --------------------------------------------------------------------------- #
# NumericConsistencyValidator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("original", "translated", "valid", "lost"),
    [
        ("The year 1984 was cold.", "那一年很冷。", False, ["1984"]),
        ("The year 1984 was cold.", "1984 年很冷。", True, None),
        ("It costs 250万 yuan.", "花费 2500000 元。", True, None),
        ("It costs 250万 yuan.", "花费 250 元。", False, ["250"]),
        ("1.5 million units sold.", "售出 150万 台。", True, None),
        ("1.5 million units sold.", "售出 150 台。", False, ["1.5"]),
        ("See pp. 4046 for details.", "见 40-46 页。", True, None),
        ("See pp. 4046 for details.", "见 4046 页。", True, None),
        ("Temperature -5 degrees.", "温度 5 度。", False, ["5"]),
        ("Temperature -5 degrees.", "温度 -5 度。", True, None),
        ("Values 1,500 and 15,6.", "数值 1500 与 15.6。", True, None),
        ("It costs 1.500 g.", "质量为 1.5 g。", True, None),  # ambiguous reading
        ("It costs 1.500 g.", "质量为 1500 g。", True, None),
        ("It costs 1.500 g.", "质量为 2 g。", False, ["1500"]),
        ("Read pages 1984-1985 now.", "现在读 1984 到 1985 年。", True, None),
        ("Read pages 1984-1985 now.", "现在读 1984 年。", False, ["1984-1985"]),
        ("ratio 10/20 here", "比率 10 比 20", True, None),
        ("ratio 10/20 here", "比率 10", False, ["10/20"]),
        ("value 3亿5000万 here", "值是 350000000", True, None),
        ("value 3亿5000万 here", "值是 3", False, ["3", "5000"]),
    ],
)
def test_numeric_validator(
    original: str, translated: str, valid: bool, lost: list[str] | None
) -> None:
    result = _check(original, translated)
    assert result.is_valid is valid
    if valid:
        assert result.error_code is None
    else:
        assert result.error_code == "NUMERIC_INCONSISTENCY"
        assert result.details["lost_numbers"] == lost
        assert result.suggested_action == "RETRY"


def test_no_source_numbers_is_always_success() -> None:
    assert _check("No numbers here.", "这里没有数字。").is_valid


def test_idiom_exemption_is_occurrence_scoped() -> None:
    # 'Top 10' alone exempts the 10...
    assert _check("Top 10 books.", "十大书籍。").is_valid
    # ...but a second, standalone '10' in the same source is still required.
    result = _check("Top 10 books and Chapter 10 covers it.", "十大书籍涵盖它。")
    assert not result.is_valid
    assert result.details["lost_numbers"] == ["10"]


def test_flattened_footnote_is_exempt() -> None:
    # "plugins 5 ," is PDF-flattened superscript metadata, not a numeric fact.
    assert _check("the plugins 5 , which are useful", "插件很有用。").is_valid


def test_the_zh_profile_is_the_no_argument_default() -> None:
    assert _check("Chapter 3.", "第三章。").is_valid
    assert NumericConsistencyValidator(ZH).validate("Chapter 3.", "第三章。").is_valid


# --------------------------------------------------------------------------- #
# GlossaryConsistencyValidator
# --------------------------------------------------------------------------- #


_GLOSSARY: list[dict[str, object]] = [
    {"source": "FinFET", "translation": "鳍式场效应晶体管"},
    {"source": "GPU", "translation": "图形处理器"},
]


def test_glossary_consistency_passes_on_a_rendered_term() -> None:
    validator = GlossaryConsistencyValidator(_GLOSSARY)
    assert validator.validate("The FinFET is fast.", "鳍式场效应晶体管很快。").is_valid


def test_glossary_consistency_reports_a_drifted_term() -> None:
    validator = GlossaryConsistencyValidator(_GLOSSARY)
    result = validator.validate("The FinFET is fast.", "它很快。")
    assert not result.is_valid
    assert result.error_code == "GLOSSARY_DRIFT"
    assert result.details["missing_terms"] == [{"source": "FinFET", "expected": "鳍式场效应晶体管"}]


def test_glossary_consistency_accepts_none_sides() -> None:
    validator = GlossaryConsistencyValidator(_GLOSSARY)
    assert validator.validate(None, None).is_valid
    assert validator.validate("", "").is_valid


def test_glossary_is_sorted_by_source_length_descending() -> None:
    validator = GlossaryConsistencyValidator(
        [
            {"source": "A", "translation": "甲"},
            {"source": "AlphaBeta", "translation": "甲乙"},
        ]
    )
    assert [entry["source"] for entry in validator.glossary] == ["AlphaBeta", "A"]


def test_percentage_points_pp_not_treated_as_pico_scale() -> None:
    validator = NumericConsistencyValidator()
    # "14.5 pp" is percentage points, not 14.5 pico-p.
    src = "DS-Vision shows a similar 14.5 pp gap."
    tgt = "DS-Vision 表现出类似的 14.5 个百分点差距。"
    res = validator.validate(src, tgt)
    assert res.is_valid, res.message

    src2 = "12.3, 14.8, 17.4 and 11.6 pp from shortest to longest"
    tgt2 = "从最短到最长分别为 12.3、14.8、17.4 和 11.6 个百分点"
    res2 = validator.validate(src2, tgt2)
    assert res2.is_valid, res2.message


def test_pdf_spaced_decimals_collapse() -> None:
    validator = NumericConsistencyValidator()
    src = "in 19 . 6% of facts versus 1 . 1% before, with p = 0 . 007"
    tgt = "在 19.6% 的事实中，相比之前的 1.1%，p = 0.007"
    res = validator.validate(src, tgt)
    assert res.is_valid, res.message

    src_table = "+ 24 . 7 pp and + 11 . 8 pp"
    tgt_table = "+24.7 个百分点和 +11.8 个百分点"
    res_table = validator.validate(src_table, tgt_table)
    assert res_table.is_valid, res_table.message


def test_hyphenated_words_not_treated_as_negative_tokens() -> None:
    validator = NumericConsistencyValidator()
    src = "trained on epoch-5 checkpoints and tier-1 models"
    tgt = "在第 5 轮检查点和第 1 梯队模型上训练"
    res = validator.validate(src, tgt)
    assert res.is_valid, res.message


def test_dash_variant_ranges_canonicalize_to_hyphen() -> None:
    # PDF extraction emits en/em dashes for ranges; a hyphen in the target is
    # the same numeric fact.
    assert canonicalize_numeric_token("10\u201320") == "10-20"
    assert canonicalize_numeric_token("10\u201420") == "10-20"
    validator = NumericConsistencyValidator()
    res = validator.validate(
        "Latency drops by 10\u201320 ms versus 5\u201310\u00d7 before.",
        "延迟降低 10-20 毫秒，此前为 5-10 倍。",
    )
    assert res.is_valid, res.message


def test_bare_count_multiplier_scale_equivalence() -> None:
    validator = NumericConsistencyValidator()
    src = "measured over more than 1.5 million containers and 390K microVMs"
    tgt = "基于超过 150 万个容器和 39 万个微虚拟机测得"
    res = validator.validate(src, tgt)
    assert res.is_valid, res.message


def test_compound_satisfaction_does_not_waive_bare_occurrences() -> None:
    # Restating the compound total ("3亿5000万" -> "3.5 亿") satisfies only the
    # compound's constituents; a bare "3" elsewhere in the source still needs
    # its own occurrence, exactly like the scaled-number path.
    validator = NumericConsistencyValidator()
    res = validator.validate(
        "The cluster spans 3亿5000万 requests in total, stored on 3 disks.",
        "集群总共处理 3.5 亿个请求。",
    )
    assert not res.is_valid, res.message
    assert "3" in res.details["lost_numbers"]


def test_wan_yi_scale_folding_is_exact() -> None:
    # 9007199254740993亿 exceeds float's 2^53 exact-integer range; the fold
    # must use Decimal so the normalized value keeps every digit (same
    # arithmetic as _cn_compound_runs).
    assert "900719925474099300000000" in normalize_for_numeric_matching("9007199254740993亿")


def test_a_leading_list_marker_is_not_a_lost_number() -> None:
    # "(1) Rollout..." translated without the marker is correct: the marker is
    # editorial structure (the renderer restores it), not a numeric fact.
    res = _check(
        "(1) Rollout and evaluation jobs create sandboxes in a bursty manner. "
        "A single job may request up to 32K sandbox instances.",
        "部署与评估任务以突发式方式创建沙箱。单个任务可能请求多达 32K 个沙箱实例。",
    )
    assert res.is_valid


def test_a_real_number_after_a_list_marker_is_still_required() -> None:
    # Exempting the marker must not exempt a genuine number elsewhere.
    res = _check("(1) The limit is 42 units.", "（1）上限为个单元。")
    assert not res.is_valid
    assert "42" in res.details["lost_numbers"]


# --------------------------------------------------------------------------- #
# Invented (unauthorized) target numbers — the reverse of the loss check.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("original", "translated", "invented"),
    [
        # A source with no digits cannot license any target digit.
        ("No figures here.", "其实有 42 个。", ["42"]),
        ("The dose was small.", "剂量为 500mg。", ["500"]),
        # A digit the source never stated, alongside one it did.
        ("The dose is 50mg.", "剂量为 500mg。", None),  # 50 lost, caught as lost not invented
    ],
)
def test_invented_number_is_rejected(
    original: str, translated: str, invented: list[str] | None
) -> None:
    res = _check(original, translated)
    assert not res.is_valid
    if invented is not None:
        assert res.details["invented_numbers"] == invented


def test_invented_number_absent_from_source_is_the_new_gate() -> None:
    # The pure case the old validator could not see: source has no number at all.
    res = _check("The results were conclusive.", "结果有 95% 的确信度。")
    assert not res.is_valid
    assert res.details["invented_numbers"] == ["95"]


@pytest.mark.parametrize(
    ("original", "translated"),
    [
        # A spelled-out source quantity may legitimately become a digit.
        ("Chapter Seven covers this.", "第7章涵盖了这一点。"),
        ("It sold two million copies.", "售出200万册。"),
        ("It sold two million copies.", "售出2000000册。"),
        ("one hundred fifty people came", "来了150人"),
        ("a dozen eggs", "12个鸡蛋"),
        # A date written out is not three invented numbers.
        ("The meeting is on 2020-01-01.", "会议在2020年1月1日。"),
        # A glued page range restored as a range states values the source token
        # does not, and is already covered by the loss check's own acceptance.
        ("See pp. 4046 for details.", "见 40-46 页。"),
    ],
)
def test_a_faithful_expansion_is_not_an_invented_number(original: str, translated: str) -> None:
    res = _check(original, translated)
    assert res.is_valid, res.message


def test_en_number_phrase_value_reads_magnitudes() -> None:
    assert 2_000_000 in _en_number_phrase_value("two million")
    assert 150 in _en_number_phrase_value("one hundred and fifty")
    # A phrase the parser cannot value yields nothing — the gate stays
    # conservative rather than guessing.
    assert _en_number_phrase_value("the quick brown fox") == set()


# --------------------------------------------------------------------------- #
# Numeric advisories (order inversion / elided restatement) — never blocking.
# --------------------------------------------------------------------------- #


def test_a_numeric_order_inversion_is_advisory_not_a_failure() -> None:
    # A clinical dose swap is the motivating case, but an order change is also
    # how a legitimate translation restates a comparison, so the gate must not
    # block: it reports instead.
    res = _check(
        "Patient took 50mg of Drug A and 5mg of Drug B.",
        "患者服用了 5mg 的 Drug A 和 50mg 的 Drug B.",
    )
    assert res.is_valid
    assert any("numeric_order_differs" in a for a in res.details["numeric_advisories"])


def test_a_legitimate_reorder_is_only_advised() -> None:
    res = _check("A 50% increase over 30 days.", "30天内增长50%。")
    assert res.is_valid
    assert any("numeric_order_differs" in a for a in res.details["numeric_advisories"])


def test_a_repeated_source_number_elided_in_target_is_advised() -> None:
    res = _check(
        "In 2019 sales rose. In 2019 profits fell.",
        "2019年销售额上升，当年利润下降。",
    )
    assert res.is_valid
    assert any("numeric_repeat_shortfall" in a for a in res.details["numeric_advisories"])


def test_a_clean_translation_carries_no_advisory() -> None:
    res = _check("The year 1984 was cold.", "1984 年很冷。")
    assert res.is_valid
    assert res.details == {}
