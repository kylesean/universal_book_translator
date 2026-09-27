"""Regression tests for Typst image-ref sanitization and asset staging."""

import time
from pathlib import Path

import pytest

from ubt.adapters.pdf import typst_fragments as tf
from ubt.adapters.pdf.typst_fragments import _prose_to_typst


def test_sanitize_image_ref_rejects_traversal() -> None:
    assert tf._sanitize_image_ref("../../etc/passwd") == ""
    assert tf._sanitize_image_ref("assets/../secret.png") == ""
    assert tf._sanitize_image_ref("assets/fig.png") == "assets/fig.png"
    # Absolute pipeline asset paths stay valid inputs for staging.
    assert tf._sanitize_image_ref("/tmp/ubt_assets/x.png") == "/tmp/ubt_assets/x.png"
    assert tf._sanitize_image_ref('a"b\\c') == "abc"


def _isolate_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    import tempfile

    allowed_tmp = tmp_path / "system_tmp"
    allowed_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(allowed_tmp))
    return allowed_tmp


def test_stage_image_assets_rejects_absolute_outside_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_temp(monkeypatch, tmp_path)
    out_root = tmp_path / "out"
    out_root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")

    source = f'#image("{secret.resolve()}", width: 70%)'
    result = tf._stage_image_assets(source, out_root / "book.typ")

    assert result == source
    assert not any(p.is_file() for p in (out_root / "_assets").iterdir())


def test_stage_image_assets_rejects_relative_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_temp(monkeypatch, tmp_path)
    out_root = tmp_path / "out"
    out_root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")

    source = '#image("../secret.txt", width: 70%)'
    result = tf._stage_image_assets(source, out_root / "book.typ")

    assert result == source
    assert not any(p.is_file() for p in (out_root / "_assets").iterdir())


def test_stage_image_assets_copies_allowlisted_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed_tmp = _isolate_temp(monkeypatch, tmp_path)
    asset = allowed_tmp / "fig1.png"
    asset.write_bytes(b"pngbytes")
    out_root = tmp_path / "out"
    out_root.mkdir()

    source = f'#image("{asset}", width: 70%)'
    result = tf._stage_image_assets(source, out_root / "book.typ")

    staged = list((out_root / "_assets").glob("fig1_*.png"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == b"pngbytes"
    assert f'#image("/_assets/{staged[0].name}"' in result


def test_stage_image_assets_stages_math_cache_svg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MathJax backend writes #image refs into MATH_CACHE_DIR (XDG cache,
    outside the compile root and the temp dir); staging must accept it or every
    engine formula fails to compile with file-not-found."""
    from ubt.adapters.pdf import math_renderer

    _isolate_temp(monkeypatch, tmp_path)
    cache = tmp_path / "xdg_cache" / "ubt" / "math_svg"
    cache.mkdir(parents=True)
    monkeypatch.setattr(math_renderer, "MATH_CACHE_DIR", cache)
    svg = cache / "eq-abc123.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    out_root = tmp_path / "out"
    out_root.mkdir()

    source = f'#image("{svg}", width: 42.0pt)'
    result = tf._stage_image_assets(source, out_root / "book.typ")

    staged = list((out_root / "_assets").glob("eq-abc123_*.svg"))
    assert len(staged) == 1
    assert f'#image("/_assets/{staged[0].name}"' in result


def test_stage_image_assets_resolves_a_symlinked_math_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache root reached through a symlink must still match the allowlist.

    ``_within`` compares canonical paths; leaving MATH_CACHE_DIR unresolved made
    a symlinked ($HOME on NFS, /tmp -> /private/tmp) or relative root fail
    ``relative_to``, so every engine formula's #image was rejected as "outside
    allowed roots" and the compile failed.
    """
    from ubt.adapters.pdf import math_renderer

    _isolate_temp(monkeypatch, tmp_path)
    real = tmp_path / "real_cache" / "math_svg"
    real.mkdir(parents=True)
    link = tmp_path / "linked_cache"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(math_renderer, "MATH_CACHE_DIR", link)
    svg = real / "eq-linked.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    out_root = tmp_path / "out"
    out_root.mkdir()

    source = f'#image("{svg}", width: 42.0pt)'
    result = tf._stage_image_assets(source, out_root / "book.typ")

    staged = list((out_root / "_assets").glob("eq-linked_*.svg"))
    assert len(staged) == 1
    assert f'#image("/_assets/{staged[0].name}"' in result


def test_stage_image_assets_stages_docling_cache_relative_path_in_subfolder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docling extracts figures to DOCLING_CACHE_DIR with relative paths (.ubt/docling_cache/...).

    When output is in a subfolder (e.g. out_dir/book.typ), staging must resolve the asset relative to CWD
    and allow it under DOCLING_CACHE_DIR, instead of failing on out_dir/.ubt/... or rejecting it as outside allowed roots.
    """
    from ubt.adapters.pdf import docling_parser

    _isolate_temp(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".ubt" / "docling_cache"
    asset_dir = cache / "assets" / "fake_sha"
    asset_dir.mkdir(parents=True)
    monkeypatch.setattr(docling_parser, "DOCLING_CACHE_DIR", cache)

    pic = asset_dir / "pic_p1_1.png"
    pic.write_bytes(b"picbytes")
    rel_ref = ".ubt/docling_cache/assets/fake_sha/pic_p1_1.png"

    out_root = tmp_path / "output_subfolder"
    out_root.mkdir()

    source = f'#image("{rel_ref}", width: 65%)'
    result = tf._stage_image_assets(source, out_root / "book.typ")

    staged = list((out_root / "_assets").glob("pic_p1_1_*.png"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == b"picbytes"
    assert f'#image("/_assets/{staged[0].name}"' in result


def test_stage_image_assets_idempotent_on_already_staged_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_temp(monkeypatch, tmp_path)
    out_root = tmp_path / "out"
    staged_dir = out_root / "_assets"
    staged_dir.mkdir(parents=True)
    (staged_dir / "fig_12345678.png").write_bytes(b"bytes")

    source = '#image("/_assets/fig_12345678.png", width: 70%)'
    result = tf._stage_image_assets(source, out_root / "book.typ")
    assert result == source


def test_prose_to_typst_decouples_inline_hash_call_before_paren() -> None:
    text = "并令 $\\psi$(x = 0, y) = 0"

    def mock_inline(latex: str) -> str | None:
        if "psi" in latex:
            return '#box(baseline: 0.232em)[#image("/_assets/psi.svg", height: 0.899em)]'
        return None

    res = tf._prose_to_typst(text, inline_math=mock_inline)
    assert (
        '#box(baseline: 0.232em)[#image("/_assets/psi.svg", height: 0.899em)]\\(x = 0, y) = 0'
        in res
    )


def test_decouple_inline_box_calls() -> None:
    source = (
        '为了求解 #box(baseline: 0.232em)[#image("/_assets/psi.svg", height: 0.899em)](x = 0, y) = 0\n'
        '#image("diag.png")(ref)\n'
        "#box[test](arg)\n"
        "#box[already]\\(escaped)\n"
    )
    decoupled = tf._decouple_inline_box_calls(source)
    assert (
        '#box(baseline: 0.232em)[#image("/_assets/psi.svg", height: 0.899em)]\\(x = 0, y) = 0'
        in decoupled
    )
    assert '#image("diag.png")\\(ref)' in decoupled
    assert "#box[test]\\(arg)" in decoupled
    assert "#box[already]\\(escaped)" in decoupled
    assert "\\\\(" not in decoupled  # idempotent


def test_prose_to_typst_neutralizes_code_in_inline_math() -> None:
    """A book's own math span can carry ``$#read("x")$``.

    In Typst math mode a bare ``#`` switches into code execution, so the prose
    path (narrative, headings, captions, footnotes, table cells) must run the
    same ``_sanitize_math_content`` guard the display path already applies.
    """
    out = tf._prose_to_typst('x $#read("/etc/passwd")$ y')
    assert "#read" not in out, f"code injection survived: {out!r}"
    assert "\\$" not in out, f"delimiters were escaped instead of sanitized: {out!r}"


def test_prose_to_typst_keeps_pandoc_layout_calls() -> None:
    """Sanitizing inline math must not strip pandoc's side-effect-free calls.

    ``\\boxed`` becomes ``#box(...)`` in Typst; blanket-stripping every ``#``
    would demote the call to literal glyphs (the chapter-3 Eq. 3.9 regression).
    """
    out = tf._prose_to_typst(r"a $\boxed{b}$ c")
    assert "#box" in out, f"pandoc layout call was stripped: {out!r}"


def test_prose_polish_does_not_rewrite_generic_assignments() -> None:
    """Parameter-assignment polish fired on any
    "Word = number unit" prose, rewriting general books into math mode.
    It now requires a physics-quantity variable AND a known SI unit."""
    out = tf._prose_to_typst("Price = 30 dollars, and team_size = 5 people.")
    assert "$" not in out
    assert "Price = 30 dollars" in out
    hit = tf._prose_to_typst("TFIN = 20 nm keeps working")
    assert '$T_"FIN" = 20 "nm"$' in hit


@pytest.mark.fast
def test_prose_to_typst_handles_spaced_inline_latex_and_single_digit_math() -> None:
    """Spaced inline LaTeX like '$ \\Gamma $' or '$ \\Gamma \\to \\Gamma $' and '$0$' must
    not leak raw escaped dollar signs ('\\$') into Typst output."""
    text = "其中 $ \\Gamma $ 为上下文类型，栖居于 $ \\Gamma \\to \\Gamma $ 中：$0$ 表示未使用，$1$ 表示线性使用。"
    out = _prose_to_typst(text)
    assert "\\$" not in out, f"Leaked escaped dollar sign in Typst output: {out!r}"
    assert "Gamma" in out


def test_equation_number_is_stripped_only_after_an_explicit_separator() -> None:
    """A parenthesized tail is stripped only when a separator precedes it.

    Commit 32111ed narrowed the tail rule: a space-only tail may be content
    (``f(x) = (1 - x)``), so only ``\\quad`` / ``\\qquad`` / ``\\\\`` followed by
    a bare number is an equation number. A bare ``(3.2)`` with only a space
    before it is deliberately left alone.
    """
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    # Explicit separator -> the trailing equation number is removed.
    assert _clean_ocr_formula("E = mc^2 \\quad (3.2)") == "E = mc^2"
    assert _clean_ocr_formula("E = mc^2 \\\\ (3.2) \\\\") == "E = mc^2"
    # No separator -> the parenthesized tail is content and survives.
    assert _clean_ocr_formula("f(x) = (1 - x)") == "f(x) = (1 - x)"
    assert _clean_ocr_formula("f(x) = 1 (1)  (2)  (3)") == "f(x) = 1 (1)  (2)  (3)"
    assert _clean_ocr_formula("E = mc^2") == "E = mc^2"


def test_trailing_equation_number_run_is_matched_in_linear_time() -> None:
    """The nested quantifiers used to backtrack exponentially on a near-miss."""
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    adversarial = "x = y " + "(1)  " * 16 + "X"
    start = time.perf_counter()
    _clean_ocr_formula(adversarial)
    elapsed = time.perf_counter() - start
    # Measured 12.1 s pre-fix; the linear peel is sub-millisecond.
    assert elapsed < 2.0, f"equation-number strip took {elapsed:.3f}s"
