"""Contract tests for the LLM document-skeleton terminology extractor (Tier-2 Bible)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest

from ubt.core.ir.models import BlockType, FlowID, IRBlock, make_element
from ubt.core.memory.skeleton_extractor import (
    _count_term_occurrences,
    _document_text,
    _parse_terms_json,
    build_document_skeleton,
    extract_skeleton_terms_llm,
)

pytestmark = pytest.mark.fast

_CompleteFn = Callable[[str, str], Awaitable[str]]


def _block(
    block_id: str,
    source: str,
    *,
    block_type: BlockType = BlockType.NARRATIVE,
    skip_translate: bool = False,
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=block_type,
        flow_id=FlowID.MAIN_STORY,
        source_text=source,
        skip_translate=skip_translate,
    )
    return IRBlock(element=element)


# --------------------------------------------------------------------------- #
# _document_text
# --------------------------------------------------------------------------- #


def test_document_text_joins_strings() -> None:
    assert _document_text(["a", "b"]) == "a\nb"


def test_document_text_reads_block_sources() -> None:
    assert _document_text([_block("x", "S1"), _block("y", "S2")]) == "S1\nS2"


# --------------------------------------------------------------------------- #
# _count_term_occurrences
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("term", "text", "expected"),
    [
        ("coeffect", "coeffect coeffects and coeffect", 3),  # inflections count
        ("agent harness", "agent harnesses here", 1),  # multi-word, last inflected
        ("cat", "category cat", 1),  # prefix of a longer word does not match
        ("", "anything", 0),
        ("\u795e\u7ecf\u7f51\u7edc", "\u795e\u7ecf\u7f51\u7edc\u5c42", 1),  # CJK substring
    ],
)
def test_count_term_occurrences(term: str, text: str, expected: int) -> None:
    assert _count_term_occurrences(term, text) == expected


# --------------------------------------------------------------------------- #
# build_document_skeleton
# --------------------------------------------------------------------------- #


def test_skeleton_of_no_blocks_is_empty() -> None:
    assert build_document_skeleton([]) == ""


def test_skeleton_from_plain_strings() -> None:
    assert build_document_skeleton(["Heading", "Body text"]) == "Heading\nBody text"


def test_skeleton_from_strings_keeps_a_partial_tail_when_it_fits() -> None:
    skeleton = build_document_skeleton(["x" * 300], max_chars=200)
    assert skeleton == "x" * 200


def test_skeleton_from_strings_drops_a_too_small_remainder() -> None:
    assert build_document_skeleton(["x" * 200], max_chars=50) == ""


def test_skeleton_from_blocks_keeps_headings_and_intro_prose() -> None:
    blocks = [
        _block("h1", "Chapter One", block_type=BlockType.HEADING),
        _block("p1", "Intro prose here"),
        _block("f1", "skipme", skip_translate=True),
        _block("c1", "code", block_type=BlockType.CODE),
    ]
    assert build_document_skeleton(blocks) == "[H] Chapter One\nIntro prose here"


# --------------------------------------------------------------------------- #
# _parse_terms_json
# --------------------------------------------------------------------------- #


def test_parse_terms_from_a_json_fence() -> None:
    raw = '```json\n{"terms":[{"source":"a"}]}\n```'
    assert _parse_terms_json(raw) == [{"source": "a"}]


def test_parse_terms_filters_non_mapping_entries() -> None:
    assert _parse_terms_json('{"terms":[{"source":"a"},1]}') == [{"source": "a"}]


def test_parse_terms_from_surrounding_prose() -> None:
    assert _parse_terms_json('prefix {"terms":[{"source":"a"}]} suffix') == [{"source": "a"}]


@pytest.mark.parametrize("raw", ["not json", "", '{"domain":"x"}'])
def test_parse_terms_returns_empty_on_unusable_input(raw: str) -> None:
    assert _parse_terms_json(raw) == []


# --------------------------------------------------------------------------- #
# extract_skeleton_terms_llm
# --------------------------------------------------------------------------- #


def _stub(response: str) -> _CompleteFn:
    async def complete(_system: str, _user: str) -> str:
        return response

    return complete


def _blocks() -> list[IRBlock]:
    return [
        _block("h", "Chapter One", block_type=BlockType.HEADING),
        _block("p", "The coeffect calculus and coeffects matter."),
    ]


def test_extraction_without_a_completer_is_empty() -> None:
    result = asyncio.run(
        extract_skeleton_terms_llm(
            _blocks(), complete_raw_fn=None, source_lang="en", target_lang="zh"
        )
    )
    assert result == []


def test_extraction_of_a_too_small_skeleton_is_empty() -> None:
    # The skeleton is below the 30-char floor, so no LLM call is made — even
    # though the stub would return a term that is present in the document.
    response = '{"terms":[{"source":"x","translation":"y"}]}'
    result = asyncio.run(
        extract_skeleton_terms_llm(
            ["x"], complete_raw_fn=_stub(response), source_lang="en", target_lang="zh"
        )
    )
    assert result == []


def test_extraction_anchors_terms_to_the_document_and_ranks_by_occurrence() -> None:
    response = json.dumps(
        {
            "domain": "PL",
            "terms": [
                {"source": "coeffect", "translation": "\u4f59\u6548\u5e94", "kind": "term"},
                {"source": "absent-term", "translation": "x", "kind": "term"},
                {"source": "", "translation": "y"},
                {"source": "coeffects", "translation": "\u4f59\u6548\u5e94\u4eec", "kind": "weird"},
            ],
        }
    )
    entries = asyncio.run(
        extract_skeleton_terms_llm(
            _blocks(), complete_raw_fn=_stub(response), source_lang="en", target_lang="zh"
        )
    )
    by_source = {e.source: e for e in entries}
    assert "absent-term" not in by_source  # not present in the document
    assert by_source["coeffect"].translation == "\u4f59\u6548\u5e94"
    assert by_source["coeffect"].frequency == 2  # matches 'coeffect' + 'coeffects'
    assert by_source["coeffects"].frequency == 1
    # An unknown kind collapses to the default "term".
    assert by_source["coeffects"].kind == "term"


def test_extraction_degrades_on_a_provider_error() -> None:
    async def boom(_system: str, _user: str) -> str:
        raise RuntimeError("boom")

    assert (
        asyncio.run(
            extract_skeleton_terms_llm(
                _blocks(), complete_raw_fn=boom, source_lang="en", target_lang="zh"
            )
        )
        == []
    )


@pytest.mark.asyncio
async def test_run_bible_stage_feeds_ir_blocks_to_skeleton_extractor() -> None:
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Any, cast

    from ubt.core.engine.facts import Terminology
    from ubt.core.engine.stages.bible import run_bible_stage

    received_input: str | None = None

    async def mock_complete(_system: str, user: str) -> str:
        nonlocal received_input
        received_input = user
        return json.dumps({"domain": "AI", "terms": []})

    blocks = [
        _block("h1", "Quantum Computing", block_type=BlockType.HEADING),
        _block("p1", "Ignored intro prose", block_type=BlockType.NARRATIVE, skip_translate=True),
        _block("p2", "Valid intro prose", block_type=BlockType.NARRATIVE),
    ]

    class FakeLedger:
        def get_job_metadata_value(self, *a: Any) -> None:
            return None

        def set_job_metadata_value(self, *a: Any) -> None:
            pass

        def get_all_blocks(self, job_id: str) -> list[IRBlock]:
            return blocks

        def fetch_source_texts(self, job_id: str) -> list[str]:
            return [b.source_text for b in blocks if b.source_text]

    async def mock_create_event(*a: Any, **k: Any) -> None:
        return None

    ctx = SimpleNamespace(
        job_id="test-job",
        ledger=FakeLedger(),
        manifest=SimpleNamespace(doc_id="d1", metadata={}, chapters=[]),
        profile_name="general",
        target_lang="zh",
        source_lang="en",
        fast_lane=False,
        raw_completion=mock_complete,
        create_event=mock_create_event,
        config=SimpleNamespace(glossary_path=None, fresh=True),
        source_pdf_path=None,
        input_path=Path("doc.pdf"),
    )

    term = Terminology()
    async for _ in run_bible_stage(cast("Any", ctx), term):
        pass

    assert received_input is not None
    # Because IRBlocks were fed, the heading "[H] Quantum Computing" is present,
    # and skip_translate block "Ignored intro prose" was excluded!
    assert "[H] Quantum Computing" in received_input
    assert "Ignored intro prose" not in received_input
    assert "Valid intro prose" in received_input
