"""Prompt hygiene (module B, item 3): positive academic guidance, no negative rules.

The spec removed the "pink elephant" negative hardcode (anti-passive-voice and
OCR-denoising) from the rich/macro prompts — naming a forbidden construction
tends to *elicit* it — and replaced it with positive register guidance. It also
removed the "placeholders keep their original relative order" constraint, which
contradicted the reorder-tolerant unmask fix. HYBRID is the default strategy.
"""

from __future__ import annotations

from typing import Any

import pytest

from ubt.core.router.capabilities import PromptStrategy
from ubt.core.router.prompts import (
    build_hybrid_draft_prompt,
    build_macro_chunk_draft_prompt,
    build_rich_draft_prompt,
)
from ubt.core.router.registry import get_default_registry

pytestmark = pytest.mark.fast

#: Fragments the spec removed from the prompts.
_REMOVED = (
    "relative order",
    "Restorative Denoising",
    "DO NOT transcribe or echo nonsensical OCR",
    "avoid mechanical passive constructions",
)


def _system(prompt: tuple[str, str]) -> str:
    return prompt[0]


@pytest.mark.parametrize(
    "builder",
    [build_rich_draft_prompt, build_macro_chunk_draft_prompt, build_hybrid_draft_prompt],
)
def test_prompts_carry_no_negative_rules(builder: Any) -> None:
    # The macro builder takes a list of (id, text) blocks; the others a string.
    if builder is build_macro_chunk_draft_prompt:
        system, _user = builder([("b1", "Hello world.")], target_lang="zh")
    else:
        system, _user = builder("Hello world.", target_lang="zh")
    for fragment in _REMOVED:
        assert fragment not in system, f"negative rule survived: {fragment!r}"


def test_rich_prompt_keeps_positive_register_guidance() -> None:
    system, _ = build_rich_draft_prompt("Measurements were taken.", target_lang="zh")
    assert "fluent" in system.lower()
    assert "Natural Register" in system


def test_hybrid_is_the_default_strategy() -> None:
    registry = get_default_registry()
    for model in ("gpt-4o", "claude-3-7-sonnet", "deepseek-v3"):
        profile = registry.resolve(model)
        if profile is not None:
            assert profile.prompt_strategy is PromptStrategy.HYBRID
