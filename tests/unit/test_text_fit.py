"""Width model and the greedy capacity kernel (line breaking is Typst's)."""

from ubt.adapters.pdf.text_fit import EmWidth, FlowFitter


def _fitter(**kwargs: object) -> FlowFitter:
    return FlowFitter(measure=EmWidth(), **kwargs)  # type: ignore[arg-type]


def test_em_width_ordering() -> None:
    m = EmWidth()
    cjk = m("汉字", 10.0)
    ascii_text = m("ab", 10.0)
    assert cjk == 20.0
    assert ascii_text == 11.0
    assert m("a b", 10.0) < m("a好b", 10.0)


def test_fit_chars_ascii_breaks_on_space() -> None:
    f = _fitter()
    n = f.fit_chars("aaa bbb ccc", width_pt=EmWidth()("aaa bbb ", 10.0), size_pt=10.0)
    assert n == 8  # "aaa bbb " — never mid-word


def test_flow_paragraph_fits_and_overflows() -> None:
    f = _fitter()
    boxes = [(0.0, 0.0, 100.0, 10.0), (0.0, 10.0, 100.0, 20.0)]
    # Full-length alignment: unused tail boxes keep "" (paint-only covers).
    assert f.flow_paragraph("汉字测试", boxes, 10.0) == ["汉字测试", ""]
    # 25 CJK chars = 250pt > 200pt capacity -> None.
    assert f.flow_paragraph("汉字测试内容太多放不下呀呀呀呀呀呀呀呀呀呀", boxes, 10.0) is None


def test_squeezed_width_zero_for_pure_ascii() -> None:
    f = _fitter()
    m = EmWidth()
    assert f.squeezed_width("abc xyz", 10.0) == m("abc xyz", 10.0)


def test_squeezed_width_discounts_punct_density() -> None:
    f = _fitter()
    m = EmWidth()
    raw = m("汉字，。测试", 10.0)
    assert f.squeezed_width("汉字，。测试", 10.0) < raw
    assert f.squeezed_width("汉字，。测试", 10.0) >= raw * (1.0 - 0.06)


def test_fit_chars_kinsoku_pushes_opener_down() -> None:
    f = _fitter()
    # Width fits "汉字「" raw (10+10+10=30) — the opener must not dangle.
    assert f.fit_chars("汉字「测试", width_pt=30.0, size_pt=10.0) == 2


def test_fit_chars_kinsoku_closer_paths() -> None:
    f = _fitter()
    # "汉字，" raw 26.2pt: fits, closer rides along via the plain flow.
    assert f.fit_chars("汉字，测试", width_pt=26.5, size_pt=10.0) == 3
    # Too narrow even squeezed (26.0 > 25.9): closer stays down.
    assert f.fit_chars("汉字，测试", width_pt=25.9, size_pt=10.0) == 2


def _balanced(line: str) -> bool:
    return line.count("$") % 2 == 0


def test_flow_never_splits_math_span() -> None:
    """Span-atomic flow (chapter-1 $T_{\text{si}}$ class): a cut inside
    $...$ pulls back to the span open instead of shipping unbalanced $
    per overlay line (which degrades to literal backslash text)."""
    f = _fitter()
    text = "薄体结构（$T_{\\text{si}} \\ll L_g$）可显著抑制短沟道效应。"
    # Narrow first box: greedy cut would land mid-span (unbalanced $ on
    # two lines); atomic flow pulls back to the span open instead.
    boxes = [(0.0, 0.0, 100.0, 10.0), (0.0, 10.0, 150.0, 20.0), (0.0, 20.0, 150.0, 30.0)]
    lines = f.flow_paragraph(text, boxes, 10.0)
    assert lines == [
        "薄体结构（",
        "$T_{\\text{si}} \\ll L_g$）",
        "可显著抑制短沟道效应。",
    ]
    assert all(_balanced(ln) for ln in lines)


def test_flow_fails_closed_when_span_exceeds_box() -> None:
    """A span wider than every box cannot flow atomically -> overflow
    (clean source-visible English), never a split literal."""
    f = _fitter()
    boxes = [(0.0, 0.0, 30.0, 10.0)]
    assert f.flow_paragraph("看$T_{\\text{si}} \\ll L_g$好", boxes, 10.0) is None


def test_flow_without_spans_unchanged() -> None:
    f = _fitter()
    boxes = [(0.0, 0.0, 100.0, 10.0), (0.0, 10.0, 100.0, 20.0)]
    assert f.flow_paragraph("汉字测试内容", boxes, 10.0) == ["汉字测试内容", ""]


def test_flow_kinsoku_line_start_closer_prevented() -> None:
    """Kinsoku: a line must never start with a closer punctuation mark like '，'."""
    f = _fitter()
    # 40pt box fits '开发完成' (4 chars x 10pt) but not comma.
    # Hanging allows the comma to stay on line 1, never dangling onto line 2.
    text = "开发完成，并在系统测试。"
    boxes = [(0.0, 0.0, 46.0, 10.0), (0.0, 10.0, 100.0, 20.0)]
    lines = f.flow_paragraph(text, boxes, 10.0)
    assert lines is not None
    assert lines[0] == "开发完成，"
    assert lines[1] == "并在系统测试。"
    assert not lines[1].startswith("，")

    # Tight box where comma cannot hang: pulls back '成' so line 2 starts with '成'
    boxes_tight = [(0.0, 0.0, 35.0, 10.0), (0.0, 10.0, 100.0, 20.0)]
    lines_tight = f.flow_paragraph(text, boxes_tight, 10.0)
    assert lines_tight is not None
    assert lines_tight[0] == "开发完"
    assert lines_tight[1] == "成，并在系统测试。"
    assert not lines_tight[1].startswith("，")


def test_flow_kinsoku_consecutive_closers_no_infinite_loop() -> None:
    """Kinsoku: consecutive closer punctuation (e.g. '），' or '）。') must not oscillate."""
    f = _fitter()
    text = "测试模型（黑色符号），针对不同情况。"
    # Box width cuts between '）' and '，'
    boxes = [(0.0, 0.0, 95.0, 10.0), (0.0, 10.0, 200.0, 20.0)]
    lines = f.flow_paragraph(text, boxes, 10.0)
    assert lines is not None
    assert not lines[1].startswith("，")
    assert not lines[1].startswith("）")
