#!/usr/bin/env python
"""§12 Q4 acceptance: right-to-left targets work end to end.

There is no Arabic/Hebrew corpus document, so this exercises every gate the RTL
path crosses with real Arabic/Hebrew text:

- admission: the language profiles exist, so all four entry points accept ``ar``
  and ``he`` (and region tags), and ``get_pair_policy`` resolves instead of
  raising;
- identity: the Arabic/Hebrew script ratios and the QE gates (near-echo
  exemption, sentence terminators, Arabic-Indic digit folding, artifact-parity
  script ranges) behave on real RTL text;
- fonts: an RTL target substitutes a same-script face rather than shipping tofu,
  and reports script coverage;
- output: the live Typst preamble emits ``dir: rtl`` (and ``lang``), and the
  HTML/EPUB views emit the target language and direction.

Exit 0 iff every check passes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ubt.core.language_profile import (
    get_pair_policy,
    get_profile,
    is_supported_lang,
    resolve_font_config,
    supported_lang_codes,
)


def _check_admission(problems: list[str]) -> None:
    from ubt.core.language_profile import is_rtl_lang

    for code in ("ar", "he", "ar-EG", "he-IL"):
        if not is_supported_lang(code):
            problems.append(f"{code!r} is not admitted")
    for code in ("ar", "he"):
        profile = get_profile(code)
        if not profile.name:
            problems.append(f"profile {code!r} has no name")
        if not is_rtl_lang(code):
            problems.append(f"is_rtl_lang({code!r}) is False")
    if "ar" not in supported_lang_codes() or "he" not in supported_lang_codes():
        problems.append("supported_lang_codes() does not list ar/he")


def _check_pair_policy(problems: list[str]) -> None:
    for target in ("ar", "he"):
        try:
            policy = get_pair_policy("en", target)
        except Exception as exc:  # noqa: BLE001 - surface the exact failure
            problems.append(f"get_pair_policy(en,{target}) raised {exc!r}")
            continue
        if policy.target_code != target:
            problems.append(f"pair policy target {policy.target_code!r} != {target!r}")
        if policy.min_target_ratio <= 0:
            problems.append(f"{target}: cross-script identity gate disabled")


def _check_identity(problems: list[str]) -> None:
    from ubt.adapters.pdf.artifact_parity import _script_ranges, target_script_ratio
    from ubt.core.language_profile import arabic_script_ratio, hebrew_script_ratio
    from ubt.core.qe.fast_pass import is_near_verbatim_echo
    from ubt.core.qe.term_shape import count_sentences
    from ubt.core.validators.consistency import normalize_for_numeric_matching

    arabic = "تتيح معمارية المحوّل التدريب المتوازي."
    hebrew = "הארכיטקטורה מאפשרת אימון מקבילי."
    if arabic_script_ratio(arabic) < 0.5:
        problems.append(f"arabic_script_ratio too low: {arabic_script_ratio(arabic)}")
    if hebrew_script_ratio(hebrew) < 0.5:
        problems.append(f"hebrew_script_ratio too low: {hebrew_script_ratio(hebrew)}")

    # A correct Arabic target carrying many Latin proper nouns is not an echo.
    source = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa"
    target = "ألفا بيتا غاما Alpha Beta Gamma Delta Epsilon Zeta Eta Theta"
    if is_near_verbatim_echo(source, target, target_is_cjk=False, target_is_rtl=True):
        problems.append("RTL near-echo exemption did not fire")
    if not is_near_verbatim_echo(source, target, target_is_cjk=False, target_is_rtl=False):
        problems.append("Latin retention no longer detects the echo (regression)")

    # Arabic sentence terminators split sentences.
    if count_sentences("جملة واحدة؟ ثم جملة ثانية۔") != 2:
        problems.append("Arabic sentence terminators are not counted")

    # Arabic-Indic digits fold to ASCII for numeric matching.
    for raw in ("١٩٨٤", "۱۹۸۴"):
        if normalize_for_numeric_matching(raw, "ar") != "1984":
            problems.append(f"Arabic-Indic digits not folded: {raw!r}")

    # Artifact-parity covers the RTL script it is asked to check.
    if not _script_ranges("ar") or not _script_ranges("he"):
        problems.append("artifact-parity has no RTL script ranges")
    if target_script_ratio(arabic, "ar") <= 0.0:
        problems.append("artifact-parity target_script_ratio(ar) is zero")


def _check_fonts(problems: list[str]) -> None:
    from ubt.adapters.pdf.font_probe import (
        is_rtl_capable,
        needs_rtl,
        resolve_font_stack,
        script_need,
    )

    if not needs_rtl("ar") or needs_rtl("en"):
        problems.append("needs_rtl is wrong")
    if script_need("ar") != "rtl" or script_need("zh") != "cjk" or script_need("en") != "":
        problems.append("script_need is wrong")
    if not is_rtl_capable("Noto Naskh Arabic") or is_rtl_capable("Liberation Serif"):
        problems.append("is_rtl_capable misclassifies")

    configured = resolve_font_config("ar").typst_fonts
    if not configured:
        problems.append("resolve_font_config(ar) returned no stack")

    # A machine with no Arabic family substitutes one rather than shipping tofu.
    available = frozenset({"Liberation Serif", "Noto Naskh Arabic", "Noto Sans CJK SC"})
    resolved = resolve_font_stack(list(configured), target_lang="ar", available=available)
    if "Noto Naskh Arabic" not in resolved.families:
        problems.append("no Arabic substitute was added")
    if not resolved.script_available:
        problems.append("script_available is False despite an Arabic face")

    # A machine with no Arabic family at all reports no script coverage.
    bare = resolve_font_stack(
        list(configured), target_lang="ar", available=frozenset({"Liberation Serif"})
    )
    if bare.script_available:
        problems.append("script_available is True with no RTL font installed")


def _check_live_typst(problems: list[str]) -> None:
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
    from ubt.core.ir.models import BlockType, IRBlock, make_element

    def preamble(target: str) -> str:
        reconstructor = TypstReconstructor(target_lang=target)
        block = IRBlock(
            element=make_element(
                id="b1",
                spine_index=1,
                block_type=BlockType.NARRATIVE,
                source_text="Hello world",
            ),
            target_text="مرحبا بالعالم" if target == "ar" else "你好世界",
        )
        return reconstructor.generate_typst_source([block], target_lang=target)

    arabic = preamble("ar")
    if "dir: rtl" not in arabic:
        problems.append("live Typst preamble has no dir: rtl for ar")
    if 'lang: "ar"' not in arabic:
        problems.append("live Typst preamble has no lang for ar")
    for ltr in ("en", "zh"):
        if "dir: rtl" in preamble(ltr):
            problems.append(f"live Typst preamble emitted dir: rtl for {ltr}")


def _check_views(problems: list[str]) -> None:
    import zipfile

    from ubt.core.qe.fast_pass import FastPassFilter
    from ubt.layout.theme import Direction, resolve_theme
    from ubt.model.ast import Document, Paragraph, Region, RegionKind
    from ubt.model.span import CanonicalSource, Span
    from ubt.pipeline.steps import realize
    from ubt.render.epub_view import compose_epub
    from ubt.render.html_view import compose_html, document_head
    from ubt.render.typst_backend import TypstBackend
    from ubt.verify.verifier import build_verifiers

    if resolve_theme("en", "ar").target_direction is not Direction.RTL:
        problems.append("resolve_theme(en, ar) is not RTL")

    text = "Hello world"
    document = Document(
        source=CanonicalSource(doc_id="rtl", text=text),
        regions=(
            Region(
                id="r0",
                kind=RegionKind.BODY,
                elements=(
                    Paragraph(
                        id="p1", spine_index=0, span=Span(page=0, chars=(0, len(text))), text=text
                    ),
                ),
            ),
        ),
    )
    target = "مرحبا بالعالم"
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="ar"))
    backend = TypstBackend({"p1": target}, theme=resolve_theme("en", "ar"))
    attestations = [realize(el, backend, verifiers, document.source) for el in document.elements]

    if "rtl" not in document_head("ar", "rtl"):
        problems.append("document_head carries no rtl direction")

    with tempfile.TemporaryDirectory(prefix="ubt-rtl-") as tmp:
        html_path = Path(tmp) / "out.html"
        compose_html(document, attestations, {"p1": target}, html_path, lang="ar", direction="rtl")
        html = html_path.read_text(encoding="utf-8")
        if 'lang="ar"' not in html or 'dir="rtl"' not in html:
            problems.append("HTML view has no lang/dir for ar")
        if target not in html:
            problems.append("HTML view lost the Arabic target")

        epub_path = Path(tmp) / "out.epub"
        compose_epub(document, attestations, {"p1": target}, epub_path, lang="ar", direction="rtl")
        with zipfile.ZipFile(epub_path) as archive:
            opf = archive.read("OEBPS/content.opf").decode("utf-8")
            text_xhtml = archive.read("OEBPS/text.xhtml").decode("utf-8")
        if 'page-progression-direction="rtl"' not in opf:
            problems.append("EPUB spine has no rtl page progression")
        if 'dir="rtl"' not in text_xhtml:
            problems.append("EPUB content has no dir for ar")


def main() -> int:
    problems: list[str] = []
    _check_admission(problems)
    _check_pair_policy(problems)
    _check_identity(problems)
    _check_fonts(problems)
    _check_live_typst(problems)
    _check_views(problems)

    print("\nRTL acceptance (ar / he)")
    print(f"  supported base languages: {', '.join(supported_lang_codes())}")
    print(f"  problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"    {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
