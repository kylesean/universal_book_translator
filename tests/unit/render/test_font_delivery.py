"""The renderer's font set must be owned, and mixed-script punctuation routed.

Two layers, each with a failure mode that only a test can catch:

* **Bundled set** — the delivery used to depend on whatever the host installed,
  so one book rendered differently on a workstation and in an Alpine container.
  ``fonts.bundled_font_dir`` answers "is the hermetic set present, complete and
  unmodified", and a *partial* set must read as absent: rendering a Chinese
  book with only the Latin faces installed is the silent degradation the probe
  exists to prevent.
* **``covers`` routing** — a bare priority list lets a Latin face claim shared
  punctuation (curly quotes, dashes, ellipsis) in a Chinese book. The typographic
  convention hands those to the CJK face, and Typst 0.13+ spells that as
  ``covers: "latin-in-cjk"``. The keyword must be **quoted**: unquoted is a
  variable reference and fails to compile with ``unknown variable:
  latin-in-cjk`` — verified against Typst 0.15.1.
"""

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


# --------------------------------------------------------------------------- #
# Bundled font set
# --------------------------------------------------------------------------- #


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
    """Latin-only must never satisfy a Chinese book's font requirement.

    A caller that gets a directory back is entitled to assume every declared
    face exists; returning the path for a partial set breaks that contract and
    ships tofu.
    """
    _make_set(tmp_path, skip=BUNDLED_FONTS[0].filename)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    assert bundled_font_dir() is None
    report = describe()
    assert report["bundled"] is False
    missing = report["missing"]
    assert isinstance(missing, list)
    assert BUNDLED_FONTS[0].filename in missing


def test_a_zero_byte_font_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted download leaves a 0-byte file; it must not be trusted."""
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
    """A pinned digest is what makes "the same book renders the same" verifiable."""
    _make_set(tmp_path)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    face = BUNDLED_FONTS[0]
    (tmp_path / face.filename).write_bytes(b"tampered")

    with pytest.raises(FontDigestMismatch) as excinfo:
        bundled_font_dir(verify_digests=True)
    assert excinfo.value.filename == face.filename
    assert "fetch_fonts" in str(excinfo.value)


def test_every_declared_face_is_redistributable() -> None:
    """A delivery engine must not depend on a font it may not ship.

    System-vendor faces (YaHei, PingFang, MS Gothic) are covered by the probe
    as *optional* substitutions; the bundled set is the part we redistribute,
    so every entry must carry an OFL-style license.
    """
    assert BUNDLED_FONTS, "the bundled set must not be empty"
    for face in BUNDLED_FONTS:
        assert face.license == "OFL-1.1", f"{face.filename} is not redistributable"
        assert face.sha256, f"{face.filename} has no pinned digest"


def test_digest_is_stable_for_the_same_bytes(tmp_path: Path) -> None:
    target = tmp_path / "f.bin"
    target.write_bytes(b"abc")
    assert digest(target) == digest(target)
    assert len(digest(target)) == 64


# --------------------------------------------------------------------------- #
# covers routing
# --------------------------------------------------------------------------- #


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
    """Shared punctuation must belong to the CJK face, not the Latin one."""
    routed = route_stack(("Noto Serif", "Noto Serif CJK SC"), "zh")
    assert routed[0] == FontFace(family="Noto Serif", covers="latin-in-cjk")
    # The CJK face stays plain so it owns everything the Latin scope excludes.
    assert routed[1] == FontFace(family="Noto Serif CJK SC")


def test_a_latin_target_gains_no_covers() -> None:
    """One face serves the page; a ``covers`` entry would only add a no-op."""
    routed = route_stack(("Noto Serif", "DejaVu Serif"), "en")
    assert all(face.covers is None for face in routed)


def test_an_rtl_target_scopes_the_script_face_by_range() -> None:
    routed = route_stack(("Noto Serif", "Noto Naskh Arabic"), "ar")
    latin, arabic = routed
    assert latin.covers is None, "embedded Latin must stay Latin"
    assert arabic.covers is not None and arabic.covers.startswith("regex(")


def test_the_latin_in_cjk_keyword_is_quoted() -> None:
    """Unquoted, Typst reads it as a variable and the compile fails.

    Measured on Typst 0.15.1: ``covers: latin-in-cjk`` →
    ``error: unknown variable: latin-in-cjk``.
    """
    emitted = FontFace(family="Noto Serif", covers="latin-in-cjk").to_typst()
    assert emitted == '(name: "Noto Serif", covers: "latin-in-cjk")'


def test_a_regex_covers_is_emitted_as_an_expression() -> None:
    emitted = FontFace(family="Noto Naskh Arabic", covers='regex("[\\u0600-\\u06FF]")').to_typst()
    assert emitted == '(name: "Noto Naskh Arabic", covers: regex("[\\u0600-\\u06FF]"))'
    assert 'covers: "regex' not in emitted, "a regex must not be quoted into a string"


def test_the_emitted_list_keeps_the_resolved_order() -> None:
    """Routing must not reorder: the probe's preference still decides the face."""
    listed = typst_font_list(("Noto Serif CJK SC", "Noto Sans CJK SC", "Noto Serif"), "zh")
    assert listed.index("Noto Serif CJK SC") < listed.index("Noto Sans CJK SC")
    assert listed.endswith('(name: "Noto Serif", covers: "latin-in-cjk")')


def test_a_plain_target_emits_a_bare_tuple_body() -> None:
    assert typst_font_list(("Noto Serif", "DejaVu Serif"), "fr") == '"Noto Serif", "DejaVu Serif"'


# --------------------------------------------------------------------------- #
# Per-target isolation: reproducible where the set covers the script
# --------------------------------------------------------------------------- #


def test_isolation_is_decided_per_target_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sealing system fonts is right only where the set carries the glyphs.

    The bundled set measures 0% kana and 0% hangul coverage. Applying
    ``--ignore-system-fonts`` to a Japanese book would typeset every page as
    tofu — worse than a good page that is merely host-dependent.
    """
    from ubt.render.typst.fonts import can_isolate

    _make_set(tmp_path)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))

    # The pan-CJK faces carry every CJK script (kana 96.9%, hangul 100%),
    # Arabic/Hebrew ship their own small files, and the Latin face measures
    # 100% of assigned Cyrillic (through ext-B) and Greek — so all of these
    # render reproducibly from the set alone.
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
    # Scripts with no bundled face. These are *different* Unicode blocks from
    # the ones above, not variants: a Devanagari face renders no Bengali, and
    # Thai is not Lao. They keep system fonts, which is the middle state — a
    # useful --font-path, but not a sealed render.
    for lang in ("bn", "pa", "gu", "my", "lo"):
        assert not can_isolate(lang), f"{lang} must keep system fonts"


def test_no_set_means_no_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ubt.render.typst.fonts import can_isolate

    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path / "absent"))
    assert not can_isolate("zh")


def test_a_partial_set_never_isolates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolation with an incomplete set would seal off the only working fonts."""
    from ubt.render.typst.fonts import can_isolate

    _make_set(tmp_path, skip=BUNDLED_FONTS[0].filename)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    assert not can_isolate("zh")


def test_compile_flags_follow_the_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The three states: no args, path only, path + isolation."""
    from ubt.adapters.pdf.typst_compile import _font_args

    # No set installed.
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path / "absent"))
    assert _font_args("zh") == []

    # Set present, target covered -> sealed.
    _make_set(tmp_path)
    monkeypatch.setenv("UBT_FONT_DIR", str(tmp_path))
    zh_args = _font_args("zh")
    assert "--font-path" in zh_args
    assert "--ignore-system-fonts" in zh_args

    # Set present, target covered by a script-specific file.
    ar_args = _font_args("ar")
    assert "--ignore-system-fonts" in ar_args

    # Set present, target not covered -> path only, system fonts stay.
    # (Bengali/Lao/Myanmar have no bundled face; sealing them off would ship
    # tofu, so the intermediate state is the correct one.)
    bn_args = _font_args("bn")
    assert "--font-path" in bn_args
    assert "--ignore-system-fonts" not in bn_args


def test_isolation_never_claims_a_script_the_set_lacks() -> None:
    """Every isolatable language's script must be declared by a real face.

    ``can_isolate`` seals system fonts, so a wrong entry here is not a cosmetic
    labelling bug — it is the difference between a typeset book and a PDF whose
    every page is tofu. Bengali, Gurmukhi, Gujarati and Myanmar were previously
    mapped to ``devanagari``, and Lao to ``thai``, which meant a Bengali book
    asked for "Noto Serif Devanagari": a face that carries none of its
    characters.

    The check is structural: the token a language maps to must be one that some
    face in the set actually declares.
    """
    from ubt.render.typst.fonts import _LANG_SCRIPT_TOKEN, bundled_script_coverage

    coverage = bundled_script_coverage()
    for lang, token in _LANG_SCRIPT_TOKEN.items():
        assert token in coverage, f"{lang} claims script {token!r}, which no bundled face declares"


def test_a_script_alias_is_never_used_for_a_different_script() -> None:
    """Each language maps to its own script, not to a neighbouring one.

    The script fallback table decides which face a target asks for. Aliasing a
    language to another script's fallback names a face that cannot render it,
    which is strictly worse than the Latin default: DejaVu at least draws the
    digits and punctuation.
    """
    from ubt.core.language_profile import _SCRIPT_OF_LANGUAGE, resolve_font_config

    # Languages the engine accepts whose script is neither Devanagari nor Thai.
    not_devanagari = {"bn", "pa", "gu", "my"}
    not_thai = {"lo"}
    for lang in not_devanagari:
        face = resolve_font_config(lang).typst_fonts[0]
        assert "Devanagari" not in face, f"{lang} is not a Devanagari language but asks for {face}"
    for lang in not_thai:
        face = resolve_font_config(lang).typst_fonts[0]
        assert "Thai" not in face, f"{lang} is Lao, not Thai, but asks for {face}"

    # And the map must actually route them somewhere, not silently drop them.
    for lang in not_devanagari | not_thai:
        assert lang in _SCRIPT_OF_LANGUAGE, f"{lang} has no script mapping at all"


def test_the_sc_subset_keeps_kana_and_hangul_out() -> None:
    """The Chinese cut must not carry kana *and* hangul together.

    Measured with Typst 0.15.1 under ``--ignore-system-fonts``: a subset holding
    ideographs plus both kana and hangul (33,624 glyphs) typesets ``：``, ``；``
    and ``？`` as CID 0. The page still looks correct — the glyphs are drawn —
    but the PDF's CIDSet declares only CID 0, so those characters are absent
    from the text layer and copy/search/artifact checks all fail on text a
    reader can plainly see. Neither block alone triggers it (ideographs+hangul,
    32,597 glyphs, is clean), and the un-subsetted 65,535-glyph face is clean,
    so the defect is in the subsetting path rather than the font.

    Since JP and KR have their own faces, the SC cut has nothing to gain from
    either block; excluding both removes the trigger and 2.7 MB. Chinese
    coverage is unaffected: ideographs 100%, CJK punctuation 100%, fullwidth
    forms 93.8%, ASCII 100% (measured on the fetched file).
    """
    from scripts.fetch_fonts import CJK_SUBSET_UNICODES as spec

    # Kana and hangul belong to the regional cuts only.
    assert "U+3040-30FF" not in spec, "kana in the SC cut re-creates the CID-0 defect"
    assert "U+AC00-D7A3" not in spec
    assert "U+1100-11FF" not in spec
    # Chinese coverage stays complete.
    assert "U+4E00-9FFF" in spec
    assert "U+3000-303F" in spec
    assert "U+FF00-FFEF" in spec


def test_the_regional_cuts_do_keep_kana_and_hangul() -> None:
    """JP and KR own those scripts, so their spec must include them.

    The complement of the test above: removing kana/hangul from the SC cut is
    only safe because a Japanese book resolves to the JP face, which still
    carries kana. A future edit that "simplifies" both specs to one would leave
    Japanese rendering as tofu under isolation.
    """
    from scripts.fetch_fonts import KANA_HANGUL_UNICODES as extra

    assert "U+3040-30FF" in extra
    assert "U+AC00-D7A3" in extra
    assert "U+1100-11FF" in extra


def test_each_cjk_region_has_its_own_face() -> None:
    """JP/KR must not be served by the SC face.

    Shared ideographs are drawn differently per region (直, 骨, 角 …), so an SC
    face in a Japanese book is legible but reads as Chinese typesetting. Each
    regional face is a separate file with its own digest — a shared digest means
    the subset step extracted the same face three times, which is exactly what
    happened before the branch order and weight selection were pinned.
    """
    cjk = [f for f in BUNDLED_FONTS if "CJK" in f.filename and "Sans" not in f.filename]
    families = {f.family for f in cjk}
    assert {"Noto Serif CJK SC", "Noto Serif CJK JP", "Noto Serif CJK KR"} <= families
    digests = [f.sha256 for f in cjk]
    assert len(set(digests)) == len(digests), "regional faces must be distinct files"


def test_the_regional_faces_own_their_script_tokens() -> None:
    """A script token claimed by two faces would make the isolation decision ambiguous."""
    from ubt.render.typst.fonts import BUNDLED_FONTS

    owners: dict[str, list[str]] = {}
    for face in BUNDLED_FONTS:
        for script in face.scripts:
            owners.setdefault(script, []).append(face.family)
    for script in ("jpan", "kore", "arabic", "hebrew"):
        assert len(owners.get(script, [])) == 1, f"{script} has {owners.get(script)}"
