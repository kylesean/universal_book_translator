from __future__ import annotations

from pathlib import Path

import pytest

from ubt.render.typst.font_routing import (
    FontFace,
    route_stack,
    script_class,
    typst_font_list,
)
from ubt.render.typst.fonts import (
    BUNDLED_FONTS,
    FontDigestMismatch,
    bundled_font_dir,
    describe,
    digest,
)

pytestmark = pytest.mark.fast


def _make_set(tmp_path: Path, *, skip: str | None = None) -> Path:
    for face in BUNDLED_FONTS:
        if face.filename == skip:
            continue
        (tmp_path / face.filename).write_bytes(b"fake-font-" + face.filename.encode())
    return tmp_path


def test_a_complete_set_is_reported_as_bundled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_set(tmp_path)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    assert bundled_font_dir() == tmp_path
    assert describe()["bundled"] is True


def test_a_partial_set_reads_as_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _make_set(tmp_path, skip=BUNDLED_FONTS[0].filename)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    assert bundled_font_dir() is None
    report = describe()
    assert report["bundled"] is False
    missing = report["missing"]
    assert isinstance(missing, list)
    assert BUNDLED_FONTS[0].filename in missing


def test_a_zero_byte_font_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_set(tmp_path, skip=BUNDLED_FONTS[0].filename)
    (tmp_path / BUNDLED_FONTS[0].filename).write_bytes(b"")
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    assert bundled_font_dir() is None


def test_a_missing_directory_is_absent_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path / "nope"))
    assert bundled_font_dir() is None
    assert describe()["bundled"] is False


def test_digest_verification_catches_a_substituted_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_set(tmp_path)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    face = BUNDLED_FONTS[0]
    (tmp_path / face.filename).write_bytes(b"tampered")

    with pytest.raises(FontDigestMismatch) as excinfo:
        bundled_font_dir(verify_digests=True)
    assert excinfo.value.filename == face.filename
    assert "fetch_fonts" in str(excinfo.value)


def test_every_declared_face_is_redistributable() -> None:
    assert BUNDLED_FONTS, "the bundled set must not be empty"
    for face in BUNDLED_FONTS:
        assert face.license == "OFL-1.1", f"{face.filename} is not redistributable"
        assert face.sha256, f"{face.filename} has no pinned digest"


def test_digest_is_stable_for_the_same_bytes(tmp_path: Path) -> None:
    target = tmp_path / "f.bin"
    target.write_bytes(b"abc")
    assert digest(target) == digest(target)
    assert len(digest(target)) == 64


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("zh", "cjk"),
        ("zh-TW", "cjk"),
        ("zh_Hans", "cjk"),
        ("ja", "cjk"),
        ("ko", "cjk"),
        ("ar", "rtl"),
        ("he", "rtl"),
        ("fa-IR", "rtl"),
        ("en", "latin"),
        ("fr", "latin"),
        ("ru", "latin"),
    ],
)
def test_script_class_picks_the_routing(lang: str, expected: str) -> None:
    assert script_class(lang) == expected


def test_a_cjk_target_scopes_the_latin_face() -> None:
    routed = route_stack(("Noto Serif", "Noto Serif CJK SC"), "zh")
    assert routed[0] == FontFace(family="Noto Serif", covers="latin-in-cjk")
    assert routed[1] == FontFace(family="Noto Serif CJK SC")


def test_a_latin_target_gains_no_covers() -> None:
    routed = route_stack(("Noto Serif", "DejaVu Serif"), "en")
    assert all(face.covers is None for face in routed)


def test_an_rtl_target_scopes_the_script_face_by_range() -> None:
    routed = route_stack(("Noto Serif", "Noto Naskh Arabic"), "ar")
    latin, arabic = routed
    assert latin.covers is None, "embedded Latin must stay Latin"
    assert arabic.covers is not None and arabic.covers.startswith("regex(")


def test_the_latin_in_cjk_keyword_is_quoted() -> None:
    emitted = FontFace(family="Noto Serif", covers="latin-in-cjk").to_typst()
    assert emitted == '(name: "Noto Serif", covers: "latin-in-cjk")'


def test_a_regex_covers_is_emitted_as_an_expression() -> None:
    emitted = FontFace(family="Noto Naskh Arabic", covers='regex("[\\u0600-\\u06FF]")').to_typst()
    assert emitted == '(name: "Noto Naskh Arabic", covers: regex("[\\u0600-\\u06FF]"))'
    assert 'covers: "regex' not in emitted, "a regex must not be quoted into a string"


def test_the_emitted_list_keeps_the_resolved_order() -> None:
    listed = typst_font_list(("Noto Serif CJK SC", "Noto Sans CJK SC", "Noto Serif"), "zh")
    assert listed.index("Noto Serif CJK SC") < listed.index("Noto Sans CJK SC")
    assert listed.endswith('(name: "Noto Serif", covers: "latin-in-cjk")')


def test_a_plain_target_emits_a_bare_tuple_body() -> None:
    assert typst_font_list(("Noto Serif", "DejaVu Serif"), "fr") == '"Noto Serif", "DejaVu Serif"'


def test_isolation_is_decided_per_target_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ubt.render.typst.fonts import can_isolate

    _make_set(tmp_path)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))

    for lang in (
        "zh",
        "zh-CN",
        "zh-TW",
        "ja",
        "ko",
        "ar",
        "he",
        "en",
        "fr",
        "ru",
        "uk",
        "bg",
        "el",
        "th",
        "hi",
        "mr",
        "ne",
    ):
        assert can_isolate(lang), f"{lang} should be reproducible from the set alone"
    for lang in ("bn", "pa", "gu", "my", "lo"):
        assert not can_isolate(lang), f"{lang} must keep system fonts"


def test_no_set_means_no_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ubt.render.typst.fonts import can_isolate

    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path / "absent"))
    assert not can_isolate("zh")


def test_a_partial_set_never_isolates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ubt.render.typst.fonts import can_isolate

    _make_set(tmp_path, skip=BUNDLED_FONTS[0].filename)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    assert not can_isolate("zh")


def test_compile_flags_follow_the_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ubt.adapters.pdf.typst_compile import _font_args

    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path / "absent"))
    assert _font_args("zh") == []

    _make_set(tmp_path)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    zh_args = _font_args("zh")
    assert "--font-path" in zh_args
    assert "--ignore-system-fonts" in zh_args

    ar_args = _font_args("ar")
    assert "--ignore-system-fonts" in ar_args

    bn_args = _font_args("bn")
    assert "--font-path" in bn_args
    assert "--ignore-system-fonts" not in bn_args


def test_isolation_never_claims_a_script_the_set_lacks() -> None:
    from ubt.render.typst.fonts import _LANG_SCRIPT_TOKEN, bundled_script_coverage

    coverage = bundled_script_coverage()
    for lang, token in _LANG_SCRIPT_TOKEN.items():
        assert token in coverage, f"{lang} claims script {token!r}, which no bundled face declares"


def test_a_script_alias_is_never_used_for_a_different_script() -> None:
    from ubt.core.language_profile import _SCRIPT_OF_LANGUAGE, resolve_font_config

    not_devanagari = {"bn", "pa", "gu", "my"}
    not_thai = {"lo"}
    for lang in not_devanagari:
        face = resolve_font_config(lang).typst_fonts[0]
        assert "Devanagari" not in face, f"{lang} is not a Devanagari language but asks for {face}"
    for lang in not_thai:
        face = resolve_font_config(lang).typst_fonts[0]
        assert "Thai" not in face, f"{lang} is Lao, not Thai, but asks for {face}"

    for lang in not_devanagari | not_thai:
        assert lang in _SCRIPT_OF_LANGUAGE, f"{lang} has no script mapping at all"


def test_the_sc_subset_keeps_kana_and_hangul_out() -> None:
    from scripts.fetch_fonts import CJK_SUBSET_UNICODES as spec

    assert "U+3040-30FF" not in spec, "kana in the SC cut re-creates the CID-0 defect"
    assert "U+AC00-D7A3" not in spec
    assert "U+1100-11FF" not in spec
    assert "U+4E00-9FFF" in spec
    assert "U+3000-303F" in spec
    assert "U+FF00-FFEF" in spec


def test_the_regional_cuts_do_keep_kana_and_hangul() -> None:
    from scripts.fetch_fonts import KANA_HANGUL_UNICODES as extra

    assert "U+3040-30FF" in extra
    assert "U+AC00-D7A3" in extra
    assert "U+1100-11FF" in extra


def test_each_cjk_region_has_its_own_face() -> None:
    cjk = [f for f in BUNDLED_FONTS if "CJK" in f.filename and "Sans" not in f.filename]
    families = {f.family for f in cjk}
    assert {"Noto Serif CJK SC", "Noto Serif CJK JP", "Noto Serif CJK KR"} <= families
    digests = [f.sha256 for f in cjk]
    assert len(set(digests)) == len(digests), "regional faces must be distinct files"


def test_the_regional_faces_own_their_script_tokens() -> None:
    from ubt.render.typst.fonts import BUNDLED_FONTS

    owners: dict[str, list[str]] = {}
    for face in BUNDLED_FONTS:
        for script in face.scripts:
            owners.setdefault(script, []).append(face.family)
    for script in ("jpan", "kore", "arabic", "hebrew"):
        assert len(owners.get(script, [])) == 1, f"{script} has {owners.get(script)}"


def test_the_bundled_cjk_families_have_bold_faces() -> None:
    bold_families = {face.family for face in BUNDLED_FONTS if "-Bold-" in face.filename}
    assert "Noto Serif CJK SC" in bold_families
    assert "Noto Sans CJK SC" in bold_families


def test_every_declared_face_has_a_fetch_recipe() -> None:
    from scripts.fetch_fonts import RECIPE_PREFIXES

    for face in BUNDLED_FONTS:
        assert any(face.filename.startswith(prefix) for prefix in RECIPE_PREFIXES), (
            f"{face.filename} matches no download recipe prefix; add a branch "
            "to scripts/fetch_fonts.py before shipping the manifest entry"
        )
