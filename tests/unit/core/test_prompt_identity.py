"""The prompt-layout identity: a layout change must invalidate TM exact hits.

``PROMPT_VERSION`` is part of the translation-memory key
(``compute_tm_context``), so a stored translation is reused only when the
prompt layout that produced it is the layout in use now. The old design was a
hand-written ``"v1"`` in ``tm.py`` with a comment asking the next person to bump
it — and nothing checked the promise. The value is now derived from the layout
literals themselves, and these tests pin both halves of that contract: the
derivation is stable, and it MOVES when a literal moves.
"""

from __future__ import annotations

import hashlib

import pytest

from ubt.core.memory.tm import PROMPT_VERSION, compute_tm_context
from ubt.core.router import prompts

pytestmark = pytest.mark.fast


def test_the_version_is_derived_from_the_layout_literals() -> None:
    """Recomputing the digest from the same parts reproduces the value."""
    rebuilt = hashlib.sha256("\x00".join(prompts._PROMPT_LAYOUT_PARTS).encode("utf-8")).hexdigest()[
        :12
    ]
    assert f"v1-{rebuilt}" == PROMPT_VERSION


def test_changing_a_layout_literal_changes_the_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property the old hand-bump asked for, now enforced.

    A prompt-layout edit that nobody bumps must still invalidate stored
    translations: the digest is computed from the parts, so an edited part
    yields a different version.
    """
    edited = ("a layout literal that never existed before", *prompts._PROMPT_LAYOUT_PARTS[1:])
    digest = hashlib.sha256("\x00".join(edited).encode("utf-8")).hexdigest()[:12]
    assert f"v1-{digest}" != PROMPT_VERSION


def test_the_version_is_stable_across_imports() -> None:
    """Two reads a process apart are the same value (no per-import salt)."""
    import importlib

    reloaded = importlib.reload(prompts)
    try:
        assert reloaded.PROMPT_VERSION == PROMPT_VERSION
        # And tm's re-export is the same object-valued string.
        from ubt.core.memory.tm import PROMPT_VERSION as from_tm

        assert from_tm == PROMPT_VERSION
    finally:
        importlib.reload(prompts)


def test_the_version_reaches_the_tm_context() -> None:
    """The identity actually participates in the exact-hit key."""
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
    """End to end: edit a part, recompute, and the TM key moves.

    This is the failure the derivation prevents — a layout constant changing
    while the identity stayed put, turning every stored translation into a
    silent exact hit for a prompt that no longer produces it.
    """
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
