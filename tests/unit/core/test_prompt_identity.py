from __future__ import annotations

import hashlib

import pytest

from ubt.core.memory.tm import PROMPT_VERSION, compute_tm_context
from ubt.core.router import prompts

pytestmark = pytest.mark.fast


def test_the_version_is_derived_from_the_layout_literals() -> None:
    rebuilt = hashlib.sha256("\x00".join(prompts._PROMPT_LAYOUT_PARTS).encode("utf-8")).hexdigest()[
        :12
    ]
    assert f"v1-{rebuilt}" == PROMPT_VERSION


def test_changing_a_layout_literal_changes_the_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    edited = ("a layout literal that never existed before", *prompts._PROMPT_LAYOUT_PARTS[1:])
    digest = hashlib.sha256("\x00".join(edited).encode("utf-8")).hexdigest()[:12]
    assert f"v1-{digest}" != PROMPT_VERSION


def test_the_version_is_stable_across_imports() -> None:
    import importlib

    reloaded = importlib.reload(prompts)
    try:
        assert reloaded.PROMPT_VERSION == PROMPT_VERSION
        from ubt.core.memory.tm import PROMPT_VERSION as from_tm

        assert from_tm == PROMPT_VERSION
    finally:
        importlib.reload(prompts)


def test_the_version_reaches_the_tm_context() -> None:
    base = compute_tm_context(
        PROMPT_VERSION, "general", "glossary", "en", "zh", abbreviation_table="", model="m"
    )
    different = compute_tm_context(
        "v1-deadbeef0000", "general", "glossary", "en", "zh", abbreviation_table="", model="m"
    )
    assert base != different


def test_a_layout_literal_edit_would_change_stored_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = prompts._DRAFT_SOURCE_MARKER
    monkeypatch.setattr(prompts, "_DRAFT_SOURCE_MARKER", original + " (edited)\n")
    parts = (prompts._DRAFT_SOURCE_MARKER, *prompts._PROMPT_LAYOUT_PARTS[1:])
    edited_version = f"v1-{hashlib.sha256(chr(0).join(parts).encode('utf-8')).hexdigest()[:12]}"
    assert edited_version != PROMPT_VERSION
    before = compute_tm_context(
        PROMPT_VERSION, "general", "g", "en", "zh", abbreviation_table="", model=""
    )
    after = compute_tm_context(
        edited_version, "general", "g", "en", "zh", abbreviation_table="", model=""
    )
    assert before != after
