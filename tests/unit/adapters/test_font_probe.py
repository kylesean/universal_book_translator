from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ubt.adapters.pdf.font_probe import (
    FontStack,
    is_cjk_capable,
    is_rtl_capable,
    needs_cjk,
    needs_rtl,
    resolve_font_stack,
    script_need,
)

if TYPE_CHECKING:
    from ubt.core.engine.stage_context import StageContext

pytestmark = pytest.mark.fast

_LINUX_CJK = frozenset({"Noto Serif CJK SC", "Noto Sans CJK SC", "Liberation Serif"})
_MACOS_CJK = frozenset({"PingFang SC", "Songti SC", "Hiragino Sans GB", "Helvetica"})
_WINDOWS_CJK = frozenset({"Microsoft YaHei", "SimSun", "SimHei", "Arial"})
_JA = frozenset({"Noto Serif CJK JP", "Noto Sans CJK JP", "Liberation Serif"})
_KO = frozenset({"Noto Serif CJK KR", "Noto Sans CJK KR", "Liberation Serif"})
_AR = frozenset({"Noto Naskh Arabic", "Amiri", "DejaVu Sans"})


@pytest.mark.parametrize(
    "lang",
    ["zh", "zh-CN", "zh-Hans", "zh_TW", "ja", "ja-JP", "ko", "ko-KR"],
)
def test_cjk_targets_are_recognised(lang: str) -> None:
    assert needs_cjk(lang)


@pytest.mark.parametrize("lang", ["en", "fr", "de", "es", "ru", "tr", "vi", "th"])
def test_latin_and_other_scripts_are_not_mistaken_for_cjk(lang: str) -> None:
    assert not needs_cjk(lang)


@pytest.mark.parametrize("lang", ["ar", "he", "fa", "ur", "ar-EG"])
def test_rtl_targets_are_recognised(lang: str) -> None:
    assert needs_rtl(lang)


def test_script_need_reports_one_answer_per_target() -> None:
    assert script_need("zh-TW") == "cjk"
    assert script_need("ar-SA") == "rtl"
    assert script_need("en") == ""
    assert script_need(None) == ""
    assert script_need("zh-Arab") == "cjk"


@pytest.mark.parametrize(
    "family",
    [
        "Noto Serif CJK SC",
        "Source Han Serif SC",
        "PingFang SC",
        "Songti SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimSun",
        "Malgun Gothic",
        "MS Gothic",
        "WenQuanYi Zen Hei",
    ],
)
def test_cjk_capable_families_are_recognised_on_every_platform(family: str) -> None:
    assert is_cjk_capable(family)


@pytest.mark.parametrize(
    "family",
    ["Liberation Serif", "DejaVu Sans", "Arial", "Helvetica", "Franklin Gothic", "Century Gothic"],
)
def test_latin_families_are_not_mistaken_for_cjk(family: str) -> None:
    assert not is_cjk_capable(family)


@pytest.mark.parametrize(
    "family", ["Noto Naskh Arabic", "Noto Sans Hebrew", "Amiri", "Scheherazade New"]
)
def test_rtl_capable_families_are_recognised(family: str) -> None:
    assert is_rtl_capable(family)


def test_installed_families_are_kept_in_profile_order() -> None:
    stack = resolve_font_stack(
        ["Noto Serif CJK SC", "Source Han Serif SC", "Noto Sans CJK SC", "Liberation Serif"],
        target_lang="zh",
        available=_LINUX_CJK,
    )
    assert stack.families == ("Noto Serif CJK SC", "Noto Sans CJK SC", "Liberation Serif")
    assert stack.unavailable == ("Source Han Serif SC",)
    assert stack.probed is True
    assert stack.script_available is True


def test_the_absent_spare_never_reaches_the_emitted_preamble() -> None:
    stack = resolve_font_stack(
        ["Noto Serif CJK SC", "Source Han Serif SC", "Noto Sans CJK SC"],
        target_lang="zh",
        available=_LINUX_CJK,
    )
    assert "Source Han Serif SC" not in stack.families
    assert "Source Han Serif SC" in stack.unavailable


def test_a_non_latin_target_gets_a_same_script_substitute() -> None:
    stack = resolve_font_stack(
        ["Source Han Serif SC"],
        target_lang="zh",
        available=_MACOS_CJK,
    )
    assert stack.substituted, "a same-script family must be added"
    assert stack.script_available is True
    assert is_cjk_capable(stack.substituted[0])


def test_a_foreign_script_face_never_wins_the_substitution_race() -> None:
    available = frozenset({"Malgun Gothic", "Noto Serif CJK KR", "Noto Sans CJK JP"})
    stack = resolve_font_stack(["Source Han Serif SC"], target_lang="zh", available=available)
    assert stack.substituted
    chosen = stack.substituted[0]
    assert "Malgun" not in chosen
    assert "KR" not in chosen
    assert "CJK" in chosen


def test_a_japanese_target_prefers_its_own_region_over_pan_cjk() -> None:
    available = frozenset({"Noto Serif CJK JP", "Noto Serif CJK SC", "Noto Sans CJK KR"})
    stack = resolve_font_stack(["Source Han Serif JP"], target_lang="ja", available=available)
    assert stack.substituted == ("Noto Serif CJK JP",)


def test_an_arabic_target_gets_a_shaping_capable_face() -> None:
    stack = resolve_font_stack(["Noto Naskh Arabic"], target_lang="ar", available=_AR)
    assert stack.script_available is True
    assert stack.families == ("Noto Naskh Arabic",)
    assert stack.substituted == ()


def test_a_rtl_target_with_no_arabic_face_reports_no_script_coverage() -> None:
    stack = resolve_font_stack(
        ["Noto Naskh Arabic"], target_lang="ar", available=frozenset({"Liberation Serif"})
    )
    assert stack.script_available is False


def test_no_probe_answer_passes_the_stack_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = ["Noto Serif CJK SC", "Source Han Serif SC"]
    monkeypatch.setattr("ubt.adapters.pdf.font_probe._typst_families", lambda _b, _lang=None: None)
    monkeypatch.setattr("ubt.adapters.pdf.font_probe._fontconfig_families", lambda: None)

    stack = resolve_font_stack(requested, target_lang="zh")
    assert stack.probed is False
    assert stack.families == tuple(requested)
    assert stack.script_available is True


def test_a_user_override_survives_even_when_it_is_not_installed() -> None:
    stack = resolve_font_stack(
        ["My Private Face", "Noto Serif CJK SC"],
        target_lang="zh",
        available=_LINUX_CJK,
        protected=("My Private Face",),
    )
    assert stack.families[0] == "My Private Face"
    assert "My Private Face" in stack.unavailable
    assert stack.script_available is True


def test_an_empty_request_never_produces_a_bogus_stack() -> None:
    stack = resolve_font_stack([], target_lang="zh", available=_LINUX_CJK)
    assert stack.families == ()
    assert stack.script_available is False


def test_a_latin_target_needs_no_substitution() -> None:
    stack = resolve_font_stack(
        ["Liberation Serif"], target_lang="fr", available=frozenset({"Liberation Serif"})
    )
    assert stack.substituted == ()
    assert stack.script_available is True


def test_fontstack_primary_is_the_first_emitted_face() -> None:
    stack = FontStack(families=("A", "B"))
    assert stack.primary == "A"
    assert FontStack(families=()).primary == ""


def test_every_script_exclusive_marker_is_also_script_capable() -> None:
    import ubt.adapters.pdf.font_probe as probe

    for target, markers in probe._SCRIPT_EXCLUSIVE.items():
        need = "cjk" if target in {"zh", "ja", "ko"} else "rtl"
        for marker in markers:
            assert probe._is_script_capable(f"{marker} test family", need), (
                f"{marker!r} ranks as a {target} face but is not {need}-capable"
            )


def test_resolve_for_render_prunes_through_the_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.adapters.pdf.font_probe import resolve_for_render

    monkeypatch.setattr(
        "ubt.adapters.pdf.font_probe.available_font_families",
        lambda _binary="typst", _lang=None: _LINUX_CJK,
    )
    resolved = resolve_for_render("zh")
    assert resolved.families == ("Noto Serif CJK SC", "Noto Sans CJK SC", "Liberation Serif")
    assert "Source Han Serif SC" in resolved.unavailable


def test_resolve_for_render_keeps_a_user_override_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.adapters.pdf.font_probe import resolve_for_render

    monkeypatch.setattr(
        "ubt.adapters.pdf.font_probe.available_font_families",
        lambda _binary="typst", _lang=None: _LINUX_CJK,
    )
    resolved = resolve_for_render("zh", font_override="My Private Face")
    assert resolved.families[0] == "My Private Face"
    assert "My Private Face" in resolved.unavailable


def test_the_font_verdict_lands_in_the_quality_report() -> None:
    from ubt.core.engine.reporter import QualityReport

    assert "font_delivery" in QualityReport.model_fields
    field = QualityReport.model_fields["font_delivery"]
    assert field.default is None, "unmeasured must be None, never a guessed verdict"


def test_font_delivery_helper_reports_no_verdict_when_unprobed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.adapters.pdf.font_probe import FontStack

    monkeypatch.setattr(
        "ubt.adapters.pdf.font_probe.resolve_for_render",
        lambda *_a, **_k: FontStack(families=("X",), probed=False, script_available=True),
    )
    from ubt.core.engine.stages.export import _font_delivery_for

    class _Manifest:
        target_lang = "zh"

    class _Ctx:
        manifest = _Manifest()
        config = type("C", (), {"font_family": None})()
        job_id = "t"

    assert _font_delivery_for(cast("StageContext", _Ctx())) is None


def test_the_probe_passes_the_bundled_font_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import ubt.adapters.pdf.font_probe as probe

    probe._typst_families.cache_clear()
    captured: dict[str, list[str]] = {}

    class _Result:
        returncode = 0
        stdout = "Noto Serif CJK SC\nNoto Serif\n"
        stderr = ""

    def _fake_run(cmd: list[str], **_kwargs: object) -> _Result:
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(probe, "_bundled_font_path", lambda: "/repo/fonts")

    families = probe.available_font_families()

    assert captured["cmd"][:2] == ["typst", "fonts"] or "fonts" in captured["cmd"]
    assert "--font-path" in captured["cmd"], "the probe must mirror the compile's flag"
    assert "/repo/fonts" in captured["cmd"]
    assert "Noto Serif CJK SC" in (families or ())
    probe._typst_families.cache_clear()


def test_no_bundled_set_means_no_extra_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    import ubt.adapters.pdf.font_probe as probe

    probe._typst_families.cache_clear()
    captured: dict[str, list[str]] = {}

    class _Result:
        returncode = 0
        stdout = "Liberation Serif\n"
        stderr = ""

    def _fake_run(cmd: list[str], **_kwargs: object) -> _Result:
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(probe, "_bundled_font_path", lambda: None)

    probe.available_font_families()
    assert "--font-path" not in captured["cmd"]
    probe._typst_families.cache_clear()


def test_a_changed_font_path_is_not_served_from_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ubt.adapters.pdf.font_probe as probe

    probe._typst_families.cache_clear()
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = "Something\n"
        stderr = ""

    def _fake_run(cmd: list[str], **_kwargs: object) -> _Result:
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr(probe, "_bundled_font_path", lambda: None)
    probe.available_font_families()
    first = len(calls)

    monkeypatch.setattr(probe, "_bundled_font_path", lambda: "/newly/installed/fonts")
    probe.available_font_families()
    assert len(calls) > first, "the new font path must trigger a fresh probe"
    probe._typst_families.cache_clear()
