"""The font stack must be resolved against the machine, not just the profile.

``language_profile`` names a per-language stack, but the names are static and
the machine is not: a family that is not installed produces ``unknown font
family`` per compile, and a stack with no face for the target script produces
tofu without failing the build.

``resolve_font_stack`` is the guard for both, and it sits on the delivery path
(one wrong answer ships blank squares), so each branch is pinned here:

* probe reports a family -> it is kept, in the profile's order,
* probe reports it missing -> dropped and reported,
* non-Latin target with nothing usable -> a same-script substitute is chosen,
  and a *foreign*-script face never wins that race,
* no probe answered -> the stack passes through untouched (an unavailable
  probe is not evidence a font is missing),
* a user's explicit ``font_family`` survives even when unresolvable.
"""

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


# --------------------------------------------------------------------------- #
# Script classification
# --------------------------------------------------------------------------- #


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
    # A CJK tag with an RTL region subtag is still CJK: the base subtag decides.
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
    """The stacks ship per-OS names, so the classifier must cover each vendor.

    macOS and Windows names are the ones a name-stem check tends to miss:
    ``PingFang``/``Songti``/``Hiragino`` carry no literal "cjk", and a false
    negative here reports "no CJK font installed" on a Mac that has three.
    """
    assert is_cjk_capable(family)


@pytest.mark.parametrize(
    "family",
    ["Liberation Serif", "DejaVu Sans", "Arial", "Helvetica", "Franklin Gothic", "Century Gothic"],
)
def test_latin_families_are_not_mistaken_for_cjk(family: str) -> None:
    """ "Gothic" is a style word Latin faces carry; matching it would ship tofu.

    A false "available" suppresses the tofu warning, which is strictly worse
    than an extra warning about a font that is only missing.
    """
    assert not is_cjk_capable(family)


@pytest.mark.parametrize(
    "family", ["Noto Naskh Arabic", "Noto Sans Hebrew", "Amiri", "Scheherazade New"]
)
def test_rtl_capable_families_are_recognised(family: str) -> None:
    assert is_rtl_capable(family)


# --------------------------------------------------------------------------- #
# resolve_font_stack
# --------------------------------------------------------------------------- #


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
    """This is the exact shape of the warning observed on arXiv 2608.25512.

    The profile lists Source Han Serif SC as the second face; the machine has
    only the Noto release. Emitting both makes Typst warn on every compile —
    pruning turns that into one debug-level fact instead.
    """
    stack = resolve_font_stack(
        ["Noto Serif CJK SC", "Source Han Serif SC", "Noto Sans CJK SC"],
        target_lang="zh",
        available=_LINUX_CJK,
    )
    assert "Source Han Serif SC" not in stack.families
    assert "Source Han Serif SC" in stack.unavailable


def test_a_non_latin_target_gets_a_same_script_substitute() -> None:
    """The profile's faces are all missing, but the machine has a usable one."""
    stack = resolve_font_stack(
        ["Source Han Serif SC"],
        target_lang="zh",
        available=_MACOS_CJK,
    )
    assert stack.substituted, "a same-script family must be added"
    assert stack.script_available is True
    # The substitute has to carry the script, not merely exist.
    assert is_cjk_capable(stack.substituted[0])


def test_a_foreign_script_face_never_wins_the_substitution_race() -> None:
    """A Korean face in a Chinese book is the same blank squares by another name.

    Pan-CJK families serve every CJK target; a region-tagged or
    language-specific face serves only its own.
    """
    available = frozenset({"Malgun Gothic", "Noto Serif CJK KR", "Noto Sans CJK JP"})
    stack = resolve_font_stack(["Source Han Serif SC"], target_lang="zh", available=available)
    assert stack.substituted
    chosen = stack.substituted[0]
    assert "Malgun" not in chosen
    assert "KR" not in chosen
    # The pan-CJK JP release is the only face left that covers simplified
    # Chinese, so it wins over the Korean-only family.
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
    """No silent tofu: the caller must be able to say the page will not shape."""
    stack = resolve_font_stack(
        ["Noto Naskh Arabic"], target_lang="ar", available=frozenset({"Liberation Serif"})
    )
    assert stack.script_available is False


def test_no_probe_answer_passes_the_stack_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable probe is not evidence that a font is missing."""
    requested = ["Noto Serif CJK SC", "Source Han Serif SC"]
    # Both probes silent: no typst binary and no fontconfig.
    monkeypatch.setattr("ubt.adapters.pdf.font_probe._typst_families", lambda _b: None)
    monkeypatch.setattr("ubt.adapters.pdf.font_probe._fontconfig_families", lambda: None)

    stack = resolve_font_stack(requested, target_lang="zh")
    assert stack.probed is False
    assert stack.families == tuple(requested)
    # No verdict is offered about script coverage without a probe.
    assert stack.script_available is True


def test_a_user_override_survives_even_when_it_is_not_installed() -> None:
    """Silently replacing an explicit choice is worse than Typst's fallback."""
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


# --------------------------------------------------------------------------- #
# Marker-table consistency and the shared resolver
# --------------------------------------------------------------------------- #


def test_every_script_exclusive_marker_is_also_script_capable() -> None:
    """Two tables describe the same fonts; a drift between them ships tofu.

    ``_SCRIPT_EXCLUSIVE`` decides the substitution ranking and
    ``_CJK_NAME_MARKERS`` decides "can this face carry the script". A name in
    one but not the other means a face that ranks as usable is then judged
    incapable (or the reverse), so the substitute race picks wrongly.
    """
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
    """The shared entry point: profile stack + probe + protected override."""
    from ubt.adapters.pdf.font_probe import resolve_for_render

    monkeypatch.setattr(
        "ubt.adapters.pdf.font_probe.available_font_families",
        lambda _binary="typst": _LINUX_CJK,
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
        lambda _binary="typst": _LINUX_CJK,
    )
    resolved = resolve_for_render("zh", font_override="My Private Face")
    assert resolved.families[0] == "My Private Face"
    assert "My Private Face" in resolved.unavailable


# --------------------------------------------------------------------------- #
# The verdict reaches the delivery report
# --------------------------------------------------------------------------- #


def test_the_font_verdict_lands_in_the_quality_report() -> None:
    """A tofu deliverable must be visible in the report, not only in stderr.

    ``script_available=False`` is a delivery-level fact: the PDF carries blank
    squares. Logging it means only whoever watched the console learns it; the
    report is what the console page and the KPI artifact read.
    """
    from ubt.core.engine.reporter import QualityReport

    # The field exists on the model and survives serialization untouched.
    assert "font_delivery" in QualityReport.model_fields
    field = QualityReport.model_fields["font_delivery"]
    assert field.default is None, "unmeasured must be None, never a guessed verdict"


def test_font_delivery_helper_reports_no_verdict_when_unprobed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable probe must not become ``script_available: True`` in a report."""
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
