"""Tests for the renderer-facing font probe (D10).

The behavior that matters: a family the compiler cannot resolve is dropped from
the emitted stack, and a machine with no CJK font at all is reported instead of
quietly rendering tofu. An *unavailable* probe must change nothing — a failed
probe is not evidence that a font is missing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf import font_probe
from ubt.adapters.pdf.font_probe import (
    FontStack,
    available_font_families,
    is_cjk_capable,
    needs_cjk,
    resolve_font_stack,
)
from ubt.adapters.pdf.typst_fragments import sanitize_font_family
from ubt.core.language_profile import resolve_font_config

ZH_STACK = [
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Noto Sans CJK SC",
    "Liberation Serif",
]


@pytest.fixture(autouse=True)
def _clear_probe_caches() -> Any:
    """Keep the process-level probe caches from leaking between tests."""
    font_probe._typst_families.cache_clear()
    font_probe._fontconfig_families.cache_clear()
    yield
    font_probe._typst_families.cache_clear()
    font_probe._fontconfig_families.cache_clear()


def _fake_proc(stdout: str, returncode: int = 0) -> Any:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


# -- stack resolution -----------------------------------------------------


def test_drops_phantom_family_and_keeps_order() -> None:
    avail = frozenset({"Noto Serif CJK SC", "Noto Sans CJK SC", "Liberation Serif"})
    got = resolve_font_stack(ZH_STACK, target_lang="zh", available=avail)
    assert got.families == ("Noto Serif CJK SC", "Noto Sans CJK SC", "Liberation Serif")
    assert got.unavailable == ("Source Han Serif SC",)
    assert got.probed is True
    assert got.cjk_available is True


def test_matching_ignores_spelling_variants() -> None:
    avail = frozenset({"Noto-Serif-CJK-SC", "noto sans cjk sc"})
    got = resolve_font_stack(ZH_STACK, target_lang="zh", available=avail)
    assert "Noto Serif CJK SC" in got.families
    assert "Source Han Serif SC" in got.unavailable


def test_nothing_installed_reports_every_name_missing() -> None:
    """Empty renderer view: emit as requested, but claim no CJK coverage."""
    got = resolve_font_stack(ZH_STACK, target_lang="zh", available=frozenset())
    assert got.families == tuple(ZH_STACK)
    assert got.unavailable == tuple(ZH_STACK)
    assert got.probed is True
    assert got.cjk_available is False


def test_latin_only_machine_flags_cjk_missing() -> None:
    avail = frozenset({"Liberation Serif", "DejaVu Sans"})
    got = resolve_font_stack(ZH_STACK, target_lang="zh", available=avail)
    assert got.families == ("Liberation Serif",)
    assert got.cjk_available is False


def test_latin_target_is_not_flagged() -> None:
    avail = frozenset({"Liberation Serif"})
    got = resolve_font_stack(
        ["Liberation Serif", "DejaVu Serif"], target_lang="en", available=avail
    )
    assert got.families == ("Liberation Serif",)
    assert got.cjk_available is True


def test_unknown_cjk_family_is_substituted_in() -> None:
    """A machine holding CJK under a name the profile lacks must still render."""
    avail = frozenset({"WenQuanYi Zen Hei", "Liberation Serif"})
    got = resolve_font_stack(ZH_STACK, target_lang="zh", available=avail)
    assert "WenQuanYi Zen Hei" in got.families
    assert got.substituted == ("WenQuanYi Zen Hei",)
    assert got.cjk_available is True


def test_substitution_is_deterministic() -> None:
    avail = frozenset({"WenQuanYi Zen Hei", "AR PL UMing CN"})
    first = resolve_font_stack(ZH_STACK, target_lang="zh", available=avail)
    second = resolve_font_stack(ZH_STACK, target_lang="zh", available=avail)
    assert first.substituted == second.substituted


def test_no_substitution_for_latin_target() -> None:
    avail = frozenset({"Noto Serif CJK SC"})
    got = resolve_font_stack(["Unknown Latin"], target_lang="en", available=avail)
    assert got.substituted == ()


def test_unprobeable_environment_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    got = resolve_font_stack(ZH_STACK, target_lang="zh", available=None)
    assert got.families == tuple(ZH_STACK)
    assert got.unavailable == ()
    assert got.probed is False


def test_empty_configured_stack() -> None:
    got = resolve_font_stack([], target_lang="zh", available=frozenset({"Foo"}))
    assert got.families == ()
    assert got.primary == ""


def test_blank_names_are_discarded() -> None:
    got = resolve_font_stack(
        ["  ", "Noto Serif CJK SC", ""],
        target_lang="zh",
        available=frozenset({"Noto Serif CJK SC"}),
    )
    assert got.families == ("Noto Serif CJK SC",)


def test_override_family_stays_first_when_installed() -> None:
    """Callers prepend an explicit font_family override before resolving."""
    avail = frozenset({"My Serif", "Noto Serif CJK SC"})
    got = resolve_font_stack(["My Serif", *ZH_STACK], target_lang="zh", available=avail)
    assert got.families[0] == "My Serif"
    assert got.unavailable == ("Source Han Serif SC", "Noto Sans CJK SC", "Liberation Serif")


def test_protected_override_survives_even_when_not_installed() -> None:
    """A user's explicit choice is never pruned; it is only reported missing.

    Silently swapping the typeface someone asked for by name is worse than
    letting Typst apply its own fallback.
    """
    avail = frozenset({"Noto Serif CJK SC"})
    got = resolve_font_stack(
        ["My Body Font", *ZH_STACK],
        target_lang="zh",
        available=avail,
        protected=("My Body Font",),
    )
    assert got.families[0] == "My Body Font"
    assert "My Body Font" in got.unavailable  # still reported as not installed
    assert "Source Han Serif SC" not in got.families  # profile names do get pruned


def test_unprotected_profile_name_is_pruned() -> None:
    avail = frozenset({"Noto Serif CJK SC"})
    got = resolve_font_stack(
        ["My Body Font", *ZH_STACK], target_lang="zh", available=avail, protected=()
    )
    assert "My Body Font" not in got.families


def test_as_typst_tuple() -> None:
    assert FontStack(families=("A", "B C")).as_typst_tuple() == '("A", "B C")'
    assert FontStack(families=()).as_typst_tuple() == "()"


def test_primary_is_first() -> None:
    assert FontStack(families=("A", "B")).primary == "A"


# -- language and family classification -----------------------------------


@pytest.mark.parametrize(
    "lang,expected",
    [
        ("zh", True),
        ("zh-CN", True),
        ("zh_hans", True),
        ("zh-Hant-TW", True),
        ("ja", True),
        ("ko", True),
        ("en", False),
        ("fr", False),
        ("de", False),
        ("", False),
        (None, False),
    ],
)
def test_needs_cjk(lang: str | None, expected: bool) -> None:
    assert needs_cjk(lang) is expected


@pytest.mark.parametrize(
    "family,expected",
    [
        ("Noto Serif CJK SC", True),
        ("Noto Sans CJK JP", True),
        ("Source Han Serif TC", True),
        ("WenQuanYi Micro Hei", True),
        ("SimHei", True),
        ("MSYGOTH", True),
        ("MS PGothic", True),
        ("Hiragino Sans", True),
        ("Noto Sans", False),  # Latin release; only the "-CJK" one carries the scripts
        ("Liberation Serif", False),
        ("DejaVu Sans", False),
        ("Times New Roman", False),
    ],
)
def test_is_cjk_capable(family: str, expected: bool) -> None:
    assert is_cjk_capable(family) is expected


# -- subprocess probes ----------------------------------------------------


def test_typst_probe_parses_family_list(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        return _fake_proc("Noto Serif CJK SC\nDejaVu Sans\n\n")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/typst")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    got = available_font_families("typst")
    assert got == frozenset({"Noto Serif CJK SC", "DejaVu Sans"})
    assert calls[0] == ["/usr/bin/typst", "fonts"]


def test_fontconfig_parses_comma_separated_families(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fontconfig line lists style variants after the file path, split on commas."""

    def _fake_run(argv: list[str], **kwargs: Any) -> Any:
        assert argv[0] == "/usr/bin/fc-list"
        return _fake_proc(
            "/f/a.ttf: Noto Serif CJK SC,Noto Serif CJK SC Bold\n/f/b.ttf: Liberation Serif\n"
        )

    def _which(name: str) -> str | None:
        return None if name == "typst" else "/usr/bin/fc-list"

    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    got = available_font_families("typst")
    assert got is not None
    assert {"Noto Serif CJK SC", "Noto Serif CJK SC Bold", "Liberation Serif"} <= got


def test_both_probes_absent_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert available_font_families("typst") is None


def test_probe_failure_is_not_evidence_of_missing_font(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing `typst fonts` must not prune the stack down to nothing."""
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/typst")
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _fake_proc("", returncode=1))
    got = resolve_font_stack(ZH_STACK, target_lang="zh")
    assert got.probed is False
    assert got.families == tuple(ZH_STACK)


def test_oserror_in_probe_degrades_to_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(argv: list[str], **kwargs: Any) -> Any:
        raise OSError("exec format error")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/typst")
    monkeypatch.setattr(subprocess, "run", _raise)
    assert available_font_families("typst") is None


def test_binary_path_off_path_still_works(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An explicit path outside PATH counts when the file exists."""
    seen: list[str] = []
    off_path = tmp_path / "typst"
    off_path.write_text("#!/bin/sh\n", encoding="utf-8")

    def _fake_run(argv: list[str], **kwargs: Any) -> Any:
        seen.append(argv[0])
        return _fake_proc("Foo Sans\n")

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert available_font_families(str(off_path)) == frozenset({"Foo Sans"})
    assert seen == [str(off_path)]


# -- substitution correctness (the tofu bug the Windows runner surfaced) ----

#: What a stock Windows install reports, plus the Latin faces it shares with
#: every Office install. "Franklin Gothic" and "Century Gothic" are the traps:
#: they are Latin-only, and both sort before "Microsoft YaHei".
STOCK_WINDOWS = frozenset(
    {
        "Segoe UI",
        "Arial",
        "Times New Roman",
        "Calibri",
        "Consolas",
        "Franklin Gothic",
        "Century Gothic",
        "Microsoft YaHei",
        "SimSun",
        "SimHei",
        "MS Gothic",
        "MS PGothic",
        "MS Mincho",
        "Malgun Gothic",
    }
)


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        pytest.param("Franklin Gothic", False, id="latin-gothic"),
        pytest.param("Century Gothic", False, id="latin-gothic-2"),
        pytest.param("Cooper Gothic", False, id="latin-gothic-3"),
        pytest.param("MS PGothic", True, id="japanese-pgothic"),
        pytest.param("Yu Gothic", True, id="japanese-yugothic"),
        pytest.param("UI Gothic", True, id="japanese-uigothic"),
        pytest.param("MS PMincho", True, id="japanese-pmincho"),
        pytest.param("Yu Mincho", True, id="japanese-yumincho"),
    ],
)
def test_gothic_and_mincho_are_style_words_before_they_are_scripts(
    family: str, expected: bool
) -> None:
    """ "Gothic"/"mincho" were bare stems, so Latin faces claimed CJK coverage.

    The stem matched "Franklin Gothic" — a Windows system font with no CJK glyph —
    and it won the alphabetical substitution race on every Chinese book on that
    machine: the emitted stack was Latin-only, ``cjk_available`` read True, and the
    warning that exists to catch exactly this stayed silent.
    """
    assert is_cjk_capable(family) is expected


@pytest.mark.parametrize("lang", ["zh", "ja", "ko"])
def test_stock_windows_gets_a_face_that_can_actually_render_the_script(lang: str) -> None:
    """The regression the 2026-09-20 Windows CI run exposed.

    Before, all three languages resolved to ``("Franklin Gothic",)`` — readable in
    the log, blank on the page.
    """
    got = resolve_font_stack(
        tuple(resolve_font_config(lang).typst_fonts), target_lang=lang, available=STOCK_WINDOWS
    )
    assert got.families, lang
    assert is_cjk_capable(got.primary), f"{lang} substituted a non-CJK face: {got.primary}"
    assert got.primary not in {"Franklin Gothic", "Century Gothic"}, lang
    assert got.cjk_available is True, lang


def test_substitution_does_not_cross_scripts() -> None:
    """A Korean face in a Chinese book is the same tofu in a different font."""
    avail = frozenset({"Malgun Gothic", "SimSun", "Segoe UI"})
    assert resolve_font_stack(ZH_STACK, target_lang="zh", available=avail).substituted == (
        "SimSun",
    )
    assert resolve_font_stack(ZH_STACK, target_lang="ko", available=avail).substituted == (
        "Malgun Gothic",
    )


def test_region_tagged_family_wins_for_that_language() -> None:
    """…CJK SC beats …CJK KR for Chinese, and the reverse for Korean.

    The requested stack names nothing installed, so the *choice* between two
    installed CJK families is what is under test here. The region token is a suffix
    test, not a substring: "SC" appears inside plenty of unrelated Latin names.
    """
    avail = frozenset({"Noto Sans CJK KR", "Noto Sans CJK SC"})
    missing = ["Unobtanium Serif"]
    assert resolve_font_stack(missing, target_lang="zh", available=avail).substituted == (
        "Noto Sans CJK SC",
    )
    assert resolve_font_stack(missing, target_lang="ko", available=avail).substituted == (
        "Noto Sans CJK KR",
    )


def test_installed_requested_family_is_not_substituted_away() -> None:
    """When the profile's own name resolves, nothing is appended — even if a
    nearer-region family is also installed."""
    avail = frozenset({"Noto Sans CJK KR", "Noto Sans CJK SC"})
    got = resolve_font_stack(ZH_STACK, target_lang="zh", available=avail)
    assert got.substituted == ()
    assert got.families == ("Noto Sans CJK SC",)


def test_font_family_reaches_both_render_engines() -> None:
    """Before: the pipeline assigned ``adapter.font_family`` while the render
    strategy and the emitter each kept their own construction-time ``None``."""
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    adapter = DoclingPDFAdapter()
    assert adapter.font_family is None
    adapter.font_family = "Noto Serif Test"
    assert adapter._renderer.font_family == "Noto Serif Test"
    assert adapter.reconstructor.font_family == "Noto Serif Test"

    # A family name is interpolated into ``#set text(font: "...")``.
    assert sanitize_font_family('x") #import "evil') is None
    adapter.font_family = 'x") #import "evil'
    assert adapter.font_family is None
    assert adapter._renderer.font_family is None
