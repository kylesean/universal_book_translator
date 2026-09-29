import pytest

from ubt.core.cleaners.cjk_spacing import (
    apply_pangu_spacing,
    normalize_cjk_punctuation,
    normalize_publishing_cjk,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("在GCA条件下", "在 GCA 条件下"),
        ("FinFET器件", "FinFET 器件"),
        ("在FinFET和GAA架构中", "在 FinFET 和 GAA 架构中"),
        ("第3章", "第 3 章"),
        ("3.1节", "3.1 节"),
        ("在2026年出版", "在 2026 年出版"),
        ("性能提升了15%的水平", "性能提升了 15% 的水平"),
    ],
)
def test_apply_pangu_spacing_cjk_latin_and_digits(raw: str, expected: str) -> None:
    """Space insertion between CJK and Latin characters or digits."""
    assert apply_pangu_spacing(raw) == expected


def test_apply_pangu_spacing_math_and_masks() -> None:
    """Test space insertion between CJK and LaTeX math formulas or mask placeholders."""
    # LaTeX inline math
    assert apply_pangu_spacing("根据公式$V_{gs}$计算") == "根据公式 $V_{gs}$ 计算"
    assert apply_pangu_spacing("由$E=mc^2$可得") == "由 $E=mc^2$ 可得"

    # Display math
    assert (
        apply_pangu_spacing("方程如下：$$\\nabla \\cdot D = \\rho$$由此可见")
        == "方程如下：$$\\nabla \\cdot D = \\rho$$ 由此可见"
    )

    # Mask placeholders — real tokens from the maskers, not hand-written
    # lookalikes. mask() binds a lowercase-hex checksum with a hyphen
    # (⟦MATH_MASK_0001-d42⟧); fake tokens like ⟦MATH_INLINE_0⟧ once masked
    # a regex gap where pangu never spaced genuine tokens.
    from ubt.core.cleaners.mask_tokens import token_checksum as _token_checksum
    from ubt.core.cleaners.math_masker import MathMasker

    _, math_mapping = MathMasker().mask("其中$E=mc^2$为常数")
    (math_tok,) = math_mapping.keys()
    assert apply_pangu_spacing(f"其中{math_tok}为常数") == f"其中 {math_tok} 为常数"
    soup_tok = f"⟦SOUP_MASK_0001-{_token_checksum(1, 'ΔE')}⟧"
    assert apply_pangu_spacing(f"由{soup_tok}推导") == f"由 {soup_tok} 推导"

    # Complex formula with commands and internal CJK text must not be mangled
    formula = "函数 $f(x)$ 满足 $\\frac{1}{2}$ 和 $x_{\\text{输入}}$ 的要求"
    res = apply_pangu_spacing(formula)
    assert "\\fra c" not in res
    assert "$\\frac{1}{2}$" in res
    assert "\\text{ 输入}" not in res


def test_apply_pangu_spacing_cjk_punctuation_guards() -> None:
    """Test that full-width CJK punctuation marks do not get unwanted spaces."""
    # Full-width parenthesis / brackets
    assert apply_pangu_spacing("（GCA）和“FinFET”是关键概念") == "（GCA）和“FinFET”是关键概念"
    assert apply_pangu_spacing("双栅（DG）结构") == "双栅（DG）结构"

    # Full-width commas and periods
    assert apply_pangu_spacing("FinFET，其沟道长为5nm。") == "FinFET，其沟道长为 5nm。"
    assert apply_pangu_spacing("代入$x=0$，得到$y=1$。") == "代入 $x=0$，得到 $y=1$。"


def test_apply_pangu_spacing_idempotent_and_non_cjk() -> None:
    """Test that existing spacing is preserved idempotently and non-CJK targets are ignored."""
    # Already spaced
    assert apply_pangu_spacing("在 GCA 条件下") == "在 GCA 条件下"
    assert apply_pangu_spacing("FinFET 和 GAA") == "FinFET 和 GAA"

    # Non-zh target
    assert apply_pangu_spacing("Hello world 123", target_lang="en") == "Hello world 123"
    assert apply_pangu_spacing("在GCA条件下", target_lang="en") == "在GCA条件下"

    # Empty
    assert apply_pangu_spacing("") == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "因此，将从式(3.1)出发建立紧凑模型。",
            "因此，将从式 (3.1) 出发建立紧凑模型。",
        ),
        (
            "式(3.7)和(3.8)可以合并为一个方程：",
            "式 (3.7) 和 (3.8) 可以合并为一个方程：",
        ),
        ("由方程(3.1)的数值解获得。", "由方程 (3.1) 的数值解获得。"),
        ("参见图 3.1(a)所示。", "参见图 3.1 (a) 所示。"),
        ("由式（3.1）推导得出", "由式（3.1）推导得出"),
    ],
)
def test_apply_pangu_spacing_bracketed_references(raw: str, expected: str) -> None:
    """Space insertion around half-width bracketed equation/figure references (W3C CLReq)."""
    assert apply_pangu_spacing(raw) == expected


def test_normalize_publishing_cjk_full_pipeline() -> None:
    """Test the integrated publishing pipeline: CJK spacing + Pangu + punctuation."""
    raw = (
        "在GCA条件下，FinFET器件--正如第3章所述，公式$V_{gs}$计算结果...表明改进， 而不是彻底变革。"
    )
    expected = "在 GCA 条件下，FinFET 器件——正如第 3 章所述，公式 $V_{gs}$ 计算结果……表明改进，而不是彻底变革。"
    res = normalize_publishing_cjk(raw, target_lang="zh")
    assert res == expected


def test_publishing_cjk_does_not_corrupt_markdown_tables_or_ranges() -> None:
    """Dashes/ellipses only convert adjacent to CJK, never in tables/ranges/URLs."""
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert normalize_publishing_cjk(table) == table

    assert "10--20" in normalize_publishing_cjk("范围 10--20 之间")
    assert "1.0...2" in normalize_publishing_cjk("值为 1.0...2 的范围")
    assert "a--b" in normalize_publishing_cjk("参见 https://example.com/a--b 页面")


def test_publishing_cjk_never_rewrites_math_or_mask_spans() -> None:
    """Punctuation rewrites must not touch $...$ or ⟦...⟧ spans."""
    assert normalize_cjk_punctuation("公式 $a--b$ 定义") == "公式 $a--b$ 定义"
    assert normalize_cjk_punctuation("公式 $x...y$ 定义") == "公式 $x...y$ 定义"
    masked = "结果 ⟦MATH_MASK_0001-a3f⟧ 与中文"
    assert normalize_cjk_punctuation(masked) == masked


def test_cjk_publishing_punctuation_normalization() -> None:
    """Verify that CJK punctuation normalizes dashes to double em-dashes and ellipses to 6 dots."""
    # Double dashes to Chinese double em-dash
    res1 = normalize_cjk_punctuation("这是一个重要发现--甚至颠覆了传统认知。")
    assert res1 == "这是一个重要发现——甚至颠覆了传统认知。"

    # 3-dot ellipsis to 6-dot ellipsis
    res2 = normalize_cjk_punctuation("未完待续...")
    assert res2 == "未完待续……"

    # Combined with CJK spacing
    res3 = normalize_publishing_cjk("改进，  而不是--彻底变革...")
    assert res3 == "改进，而不是——彻底变革……"


def test_cjk_punctuation_isolated_em_dash_and_ellipses() -> None:
    """Verify that single isolated em-dash (—) and 3-dot ellipses convert correctly in CJK context."""
    # Single isolated em-dash
    res1 = normalize_cjk_punctuation("突破—正是我们期待的。")
    assert res1 == "突破——正是我们期待的。"

    # Double em-dash already correct is preserved
    res2 = normalize_cjk_punctuation("突破——正是我们期待的。")
    assert res2 == "突破——正是我们期待的。"

    # Single isolated ellipsis
    res3 = normalize_cjk_punctuation("探索…永无止境。")
    assert res3 == "探索……永无止境。"

    # Double ellipsis already correct is preserved
    res4 = normalize_cjk_punctuation("探索……永无止境。")
    assert res4 == "探索……永无止境。"

    # Western text guarded
    res5 = normalize_cjk_punctuation("Method -- fast, result... okay.", target_lang="en")
    assert res5 == "Method -- fast, result... okay."


@pytest.mark.fast
def test_cjk_spacing_protects_code_spans() -> None:
    # Inline code with function call inside CJK text
    sample = "在函数`foo(1)`中我们观察到`x--`计数递减。"
    result = normalize_publishing_cjk(sample, target_lang="zh")
    assert "`foo(1)`" in result
    assert "`x--`" in result
    assert "foo (1)" not in result
    assert "x——" not in result


def test_cjk_space_removal_only_fires_for_spaceless_targets() -> None:
    from ubt.core.cleaners.cjk_spacing import normalize_publishing_cjk

    korean = "이것은 테스트 입니다."
    assert normalize_publishing_cjk(korean, target_lang="ko") == korean
    assert normalize_publishing_cjk(korean, target_lang="en") == korean
    # Chinese still loses the stray space between Han characters.
    assert normalize_publishing_cjk("中 文 书", target_lang="zh") == "中文书"
    # Trailing whitespace before a newline is noise in any language.
    assert normalize_publishing_cjk("il y a  \nrien", target_lang="fr") == "il y a\nrien"


def test_table_separator_gate_is_linear_on_adversarial_line() -> None:
    """A long near-miss separator row must not backtrack exponentially.

    The old pattern partitioned the dash run in exponentially many ways; a
    ~26-dash line took seconds and LLM output is untrusted.
    """
    from time import perf_counter

    from ubt.core.cleaners.cjk_spacing import _TABLE_SEP_RE, normalize_publishing_cjk

    line = "-" * 4000 + "x"
    start = perf_counter()
    assert _TABLE_SEP_RE.fullmatch(line) is None
    assert perf_counter() - start < 2.0

    # Real separator rows still match and stay byte-identical.
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert _TABLE_SEP_RE.fullmatch("|---|---|") is not None
    assert normalize_publishing_cjk(table) == table


def test_pangu_spacing_does_not_break_bare_function_calls() -> None:
    """``sin(x)`` is a call, not a bracketed reference; leave it alone."""
    assert apply_pangu_spacing("当 sin(x) 趋近时") == "当 sin(x) 趋近时"
    assert apply_pangu_spacing("函数 f(x) 连续") == "函数 f(x) 连续"
    # ...but a numeric bracketed reference still gets its space.
    assert apply_pangu_spacing("参见图 3.1(a)所示。") == "参见图 3.1 (a) 所示。"


def test_cjk_spacing_does_not_treat_currency_as_protected_math() -> None:
    """Currency runs like $5到$10 must not be treated as protected math spans."""
    res = apply_pangu_spacing("价格为$5到$10。")
    assert "$5到$" not in res
    assert "价格为 $5" not in res
    assert "5 到" in res
