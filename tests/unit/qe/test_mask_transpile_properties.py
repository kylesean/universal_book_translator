from __future__ import annotations

import random
import re

import pytest

from ubt.render.typst.math_transpiler import typstify_math
from ubt.segment.placeholders import default_placeholder_engine

pytestmark = pytest.mark.fast

SEEDS = (20261013, 7, 424242)


_SPAN_TEMPLATES = [
    "contact me at {email} today",
    "see {email} or mail {email2}",
    "the code is {code} here",
    "inline `fenced` and {code}",
    "value ${math}$ in text",
    "$${display}$$ equation",
    "\\[{display}\\] shown",
    "cite {cite} and {cite2} together",
    "{soup} as narrative",
    "mixed {email} and ${math}$ plus {cite}",
    "CJK 邮箱 {email} 与公式 ${math}$ 引用 {cite} 混排",
    "brackets [({math})] and $x_i$ end",
    "{code} with ${math}$ adjacent",
    "no placeholders at all, plain prose sentence",
]


def _random_span(rng: random.Random, kind: str) -> str:
    letters = "abcdefgh"
    if kind == "email":
        return f"{rng.choice(letters)}{rng.randrange(99)}@example.org"
    if kind == "code":
        return f"x_{rng.randrange(9)} = {rng.randrange(99)}"
    if kind == "math":
        return rng.choice(["x_i^2 + y", "\\frac{a}{b}", "E = mc^2", "\\alpha + \\beta"])
    if kind == "display":
        return rng.choice(["\\int_0^1 f(x) dx", "\\sum_i a_i = b", "\\frac{1}{2}"])
    if kind == "cite":
        return f"[{rng.randrange(1, 60)}]"
    return rng.choice(["φ + γ ≤ Σ × 2", "λ_k ≈ 3.14", "μ ≥ 0.05 且 σ² ≤ 1"])


def _generate(rng: random.Random) -> str:
    template = rng.choice(_SPAN_TEMPLATES)
    fills: dict[str, str] = {}
    if "{email2}" in template:
        fills["email2"] = _random_span(rng, "email")
    if "{cite2}" in template:
        fills["cite2"] = _random_span(rng, "cite")
    for kind in ("email", "code", "math", "display", "cite", "soup"):
        placeholder = f"{{{kind}}}"
        if placeholder in template:
            fills[kind] = _random_span(rng, kind)
    return template.format(**fills)


@pytest.mark.parametrize("seed", SEEDS)
def test_mask_round_trip_is_lossless(seed: int) -> None:
    rng = random.Random(seed)
    engine = default_placeholder_engine()
    for _ in range(300):
        text = _generate(rng)
        masked = engine.mask(text)
        outcome = engine.restore(masked.text, masked)
        assert outcome.text == text, (
            f"seed={seed} round-trip changed the text\n"
            f"input:   {text!r}\nrestored:{outcome.text!r}\nmasked:  {masked.text!r}"
        )
        assert outcome.clean, (
            f"seed={seed} restore reported corruption on an untouched echo: "
            f"{[name for name, report in outcome.reports if not report.clean]}"
        )


def test_a_mutated_token_fails_clean_instead_of_restoring_wrong_span() -> None:
    engine = default_placeholder_engine()
    text = "first $x_i^2$ then $\\frac{a}{b}$ end"
    masked = engine.mask(text)
    assert len(masked.math_map) == 2, "fixture must produce two math tokens"
    (token_1, original_1), (token_2, original_2) = masked.math_map.items()
    assert original_1 != original_2, "spans must differ or the swap is invisible"
    index_1 = re.search(r"(\d+)", token_1)
    index_2 = re.search(r"(\d+)", token_2)
    assert index_1 and index_2, "token shape does not carry an index"
    forged = token_1.replace(index_1.group(1), index_2.group(1), 1)
    echoed = masked.text.replace(token_1, forged)
    outcome = engine.restore(echoed, masked)
    assert not outcome.clean, (
        f"a rewritten index must fail the integrity check: "
        f"reports={[name for name, report in outcome.reports if not report.clean]}"
    )
    assert outcome.text.count(original_2) == 1, (
        f"span 2's content was restored into the forged slot: {outcome.text!r}"
    )


_MATH_ATOMS = [
    "x",
    "x_i",
    "\\alpha",
    "\\frac{a}{b}",
    "\\frac{a + c}{b - d}",
    "\\sqrt{n}",
    "\\sqrt[3]{x}",
    "\\binom{n}{k}",
    "a^2 + b^2",
    "e^{i\\pi}",
    "\\int_0^1",
    "\\sum_{i=1}^{n}",
    "\\left( x \\right)",
    "\\mathrm{d}x",
    "\\text{loss}",
    "\\llbracket B \\rrbracket",
]


def _random_math(rng: random.Random, depth: int = 0) -> str:
    parts: list[str] = []
    for _ in range(rng.randrange(1, 4)):
        parts.append(rng.choice(_MATH_ATOMS))
        parts.append(rng.choice([" + ", " ", " = ", ", ", " - ", "\\;"]))
    body = "".join(parts).strip()
    if depth < 2 and rng.random() < 0.3:
        body = f"\\frac{{{body}}}{{{_random_math(rng, depth + 1)}}}"
    return body


@pytest.mark.parametrize("seed", SEEDS)
def test_transpiled_math_never_emits_unbalanced_or_untranslated_output(seed: int) -> None:
    rng = random.Random(seed)
    engine = default_placeholder_engine()
    converted_count = 0
    for _ in range(300):
        body = _random_math(rng)
        out = typstify_math(body)
        if out is None:
            continue
        converted_count += 1
        assert out.count("$") % 2 == 0, f"seed={seed} unbalanced $ in {body!r} -> {out!r}"
        for untranslated in ("\\frac", "\\binom", "\\sqrt["):
            assert untranslated not in out, (
                f"seed={seed} {untranslated!r} leaked into output: {body!r} -> {out!r}"
            )
        if "\\frac" in body:
            assert "frac(" in out, f"seed={seed} \\frac not converted: {body!r} -> {out!r}"
        if "\\binom" in body:
            assert "binom(" in out, f"seed={seed} \\binom not converted: {body!r} -> {out!r}"
        if "\\sqrt{" in body:
            assert "sqrt(" in out, f"seed={seed} \\sqrt{{}} not converted: {body!r} -> {out!r}"
        masked = engine.mask(f"v ${body}$ w")
        if masked.math_map:
            (token, original) = next(iter(masked.math_map.items()))
            again = typstify_math(token.strip("$"))
            if again is not None:
                assert token[1] in again or "⟦" in again, (
                    f"seed={seed} token mangled: {token!r} -> {again!r}"
                )
    assert converted_count > 100, "generator stopped producing convertible bodies"
