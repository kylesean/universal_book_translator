from ubt.core.cleaners.soup_math import (
    SoupMathMasker,
    find_soup_spans,
    is_soup_line,
    strip_fuzzy,
)


def _covered(text: str) -> list[str]:
    return sorted(text[s:e] for s, e in find_soup_spans(text))


def test_inline_greek_glue_masked() -> None:
    assert _covered("器件ψ_pert较大而β趋于0") == ["ψ_pert"]


def test_greek_paren_masked() -> None:
    assert _covered("通过cos(β)求解") == ["cos(β)"]


def test_greek_digit_masked() -> None:
    spans = _covered("得到β2项")
    assert spans == ["β2"]


def test_snake_identifier_masked() -> None:
    assert _covered("由F_th,SI得到") == ["F_th,SI"]


def test_plain_prose_untouched() -> None:
    assert find_soup_spans("其中Ag0、r、F和Fth,SI定义为") == []
    assert find_soup_spans("file_name where Q 0") == []
    assert find_soup_spans("(A.4)") == []


def test_display_lines_detected() -> None:
    assert is_soup_line("βSI = e−ψpert/(2Vtm) Ag0r F − Fth,SI Ag0 + 1 − 1.")
    assert is_soup_line("F = VG − VFB − VCH − ψpert 2Vtm + ln TFIN 2 qn2i 2EsiNchVtm")


def test_prose_lines_not_detected() -> None:
    assert not is_soup_line("其中Ag0、r、F和Fth,SI定义为")
    assert not is_soup_line("where ψ(x, y) is the electrostatic potential in the channel")
    assert not is_soup_line("式 (A.1) 仅在 (F - Fth,SI)/Ag0 > -1 时成立。")
    assert not is_soup_line("Vch为沟道的准费米势，Vtm为热电压，由kBT/q给出")
    assert not is_soup_line("(A.4)")


def test_strip_fuzzy_spacing_variants() -> None:
    assert strip_fuzzy("求解 where Q 0 完毕", "whereQ0") == "求解  完毕"
    assert strip_fuzzy("abc", "zzz") is None
    assert strip_fuzzy("abc", "") is None


def test_masker_roundtrip_clean() -> None:
    masker = SoupMathMasker()
    src = "由cos(β)与ψ_pert求得"
    masked, mapping = masker.mask(src)
    assert len(mapping) == 2
    assert "cos(β)" not in masked and "ψ_pert" not in masked
    report = masker.unmask_checked(masked, mapping)
    assert report.clean
    assert report.text == src


def test_masker_corrupt_token_flagged() -> None:
    masker = SoupMathMasker()
    src = "由cos(β)求得"
    masked, mapping = masker.mask(src)
    assert len(mapping) == 1
    tampered = masked.replace("0001", "0007")
    report = masker.unmask_checked(tampered, mapping)
    assert not report.clean
    assert report.missing == [1] or report.mismatched == [1] or report.mutated


def test_param_unit_never_splits_english_word() -> None:
    """C1: "5 square meters" must not match "Area = 5 s"."""
    from ubt.core.cleaners.soup_math import SoupMathMasker

    assert _covered("The area = 5 square meters") == ["area = 5"]
    masker = SoupMathMasker()
    masked, _ = masker.mask("The area = 5 square meters")
    assert "square meters" in masked
    # True single-letter units still match.
    assert _covered("x = 5 s") == ["x = 5 s"]
    assert _covered("x = 5 cm.") == ["x = 5 cm"]


def test_existing_mask_tokens_are_never_rewrapped() -> None:
    """C2: soup must not nest tokens inside already-masked text."""
    from ubt.core.cleaners.math_masker import MathMasker

    math_masker = SoupMathMasker()
    _, math_mapping = MathMasker().mask("Energy $E=mc^2$ here")
    (tok,) = math_mapping.keys()
    src = f"M3 with {tok} inline"
    masked, mapping = math_masker.mask(src)
    assert "⟦⟦" not in masked
    assert all("⟦⟦" not in t for t in mapping)
