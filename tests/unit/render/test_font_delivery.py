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
    # and Arabic/Hebrew ship their own small files, so all of these render
    # reproducibly from the set alone.
    for lang in ("zh", "zh-CN", "zh-TW", "ja", "ko", "ar", "he", "en", "fr"):
        assert can_isolate(lang), f"{lang} should be reproducible from the set alone"
    # Scripts the set still does not carry: Cyrillic text beyond Latin-1,
    # Thai and the Indic scripts need faces this set does not bundle.
    for lang in ("ru", "th", "hi"):
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
    # (Cyrillic/Thai/Indic have no bundled face; sealing them off would ship
    # tofu, so the intermediate state is the correct one.)
    th_args = _font_args("th")
    assert "--font-path" in th_args
    assert "--ignore-system-fonts" not in th_args


def test_the_cjk_subset_spec_includes_kana_and_hangul() -> None:
    """The subset spec is what decides whether Japanese/Korean ship as tofu.

    The first version cut only the ideograph block, so the set rendered Japanese
    and Korean as blank squares even though the upstream pan-CJK face carries
    both scripts. The ranges are data, so a future edit could silently drop them
    again; this pins the intent, not the encoding.
    """
    from scripts.fetch_fonts import CJK_SUBSET_UNICODES as spec

    # Hiragana/Katakana
    assert "U+3040-30FF" in spec
    # Hangul syllables and Jamo
    assert "U+AC00-D7A3" in spec
    assert "U+1100-11FF" in spec
    # The ideograph block and CJK punctuation stay.
    assert "U+4E00-9FFF" in spec
    assert "U+3000-303F" in spec


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
