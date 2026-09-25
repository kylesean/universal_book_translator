"""Contract tests for the offline baseline double (``tests/mock_providers.py``).

These bind the double to the *product's own* gate predicates: if
``FastPassFilter`` tightens what it accepts in a table, or the mock starts
answering a table block with prose, the failure lands here in milliseconds
instead of surfacing as an unexplained ``pass_rate`` drop inside a multi-minute
end-to-end baseline. Nothing here touches the filesystem or the network.
"""

import re

import pytest

from tests.mock_providers import TokenEchoMockProvider, render_grid
from ubt.core.qe.fast_pass import FastPassFilter, grid_columns, markdown_grid_shape

#: The shape both corpus tables actually have: 4 columns, a ``:---`` alignment
#: row, identifiers and decimals in the cells.
TABLE_4X6 = """| Subsystem | Primary Modality | Anatomical Correlates | Typical Assessment Paradigm |
| :--- | :--- | :--- | :--- |
| Central Executive | Modality-independent | Dorsolateral Prefrontal Cortex (DLPFC) | N-back Task, Stroop Task |
| Phonological Loop | Acoustic / Verbal | Left Inferior Parietal & Broca's Area | Digit Span, Nonword Repetition |
| Visuospatial Sketchpad | Visual / Spatial | Right Occipito-Parietal Network | Corsi Block-Tapping Task |
| Episodic Buffer | Multimodal Integration | Anterior Cingulate & Hippocampus | Prose Recall, Complex Span |
"""

TABLE_3X3 = """| Tier | COMET-22 | Latency (s/page) |
| :--- | :--- | :--- |
| Zero-Shot Vanilla | 0.741 | 1.82 |
"""

PROSE = "Working memory is a limited-capacity system that maintains a few items."

_SINGLE_ROW = "| Only | One | Row |"

#: FastPass's own hallucination-loop predicate: a 4-20 char unit four times in
#: a row. Rendered cells must never give it a match to hold on to.
_REPETITION_PATTERN = re.compile(r"(.{4,20}?)\1{3,}")


def _draft_prompt(source: str) -> str:
    """A prompt shaped like the draft stage's, enough for ``_extract_source``."""
    return f"### Glossary\n| Term | Translation |\n| --- | --- |\n\n{source}\n\n### Instructions"


def _source_scoped_prompt(source: str) -> str:
    return f"### Source Paragraph to Translate\n{source}\n\n### Additional context\nirrelevant"


@pytest.mark.parametrize("table", [TABLE_4X6, TABLE_3X3])
def test_render_grid_preserves_shape(table: str) -> None:
    assert markdown_grid_shape(render_grid(table)) == markdown_grid_shape(table)


@pytest.mark.parametrize("table", [TABLE_4X6, TABLE_3X3])
async def test_rendered_table_clears_the_structural_gate(table: str) -> None:
    """The whole point of the contract: the gate accepts the double's tables."""
    target = render_grid(table)
    decision = FastPassFilter().validate_structural_invariants(table, target)
    assert decision.passed, decision.reason


def test_prose_is_not_rendered_as_a_grid() -> None:
    assert render_grid(PROSE) == ""


def test_single_row_table_is_ignored_by_the_gate() -> None:
    """One pipe row is not a grid, so the gate cannot fail it; the double still
    renders it column-for-column (the two-row floor is the gate's, not the
    renderer's)."""
    assert markdown_grid_shape(_SINGLE_ROW) == []
    rendered = render_grid(_SINGLE_ROW)
    assert grid_columns(rendered) == grid_columns(_SINGLE_ROW)
    assert not any(ch.isdigit() for ch in rendered)


@pytest.mark.parametrize("table", [TABLE_4X6, TABLE_3X3])
def test_cell_text_is_unique(table: str) -> None:
    """Identical cells repeated across a grid read as a loop to the gate."""
    rendered = render_grid(table)
    cells = [cell for line in rendered.splitlines() for cell in line.strip("|").split("|")]
    assert len(cells) == len(set(cells))
    assert _REPETITION_PATTERN.search(rendered) is None


def test_render_grid_is_deterministic() -> None:
    assert render_grid(TABLE_4X6) == render_grid(TABLE_4X6)


@pytest.mark.parametrize("table", [TABLE_4X6, TABLE_3X3])
async def test_provider_keeps_the_grid(table: str) -> None:
    provider = TokenEchoMockProvider(default_response="离线基线通用译文。")
    target = await provider.generate(_source_scoped_prompt(table))
    assert markdown_grid_shape(target) == markdown_grid_shape(table)


async def test_provider_does_not_echo_the_prompt_scaffolding_table() -> None:
    """The glossary table lives in the prompt, outside the source section; a
    prose block must not come back holding it."""
    provider = TokenEchoMockProvider(default_response="离线基线通用译文。")
    target = await provider.generate(_draft_prompt(PROSE))
    assert markdown_grid_shape(target) == []


async def test_provider_renders_a_real_draft_prompt_for_a_table_block() -> None:
    provider = TokenEchoMockProvider(default_response="离线基线通用译文。")
    target = await provider.generate(_source_scoped_prompt(TABLE_4X6))
    decision = FastPassFilter().validate_structural_invariants(TABLE_4X6, target)
    assert decision.passed, decision.reason


async def test_provider_echoes_cell_payloads() -> None:
    """Figures and identifiers inside cells survive into the rendered grid, so
    the numeric and omission gates still find them."""
    provider = TokenEchoMockProvider(default_response="离线基线通用译文。")
    latency = await provider.generate(_source_scoped_prompt(TABLE_3X3))
    for token in ("COMET-22", "0.741", "1.82"):
        assert token in latency
    anatomy = await provider.generate(_source_scoped_prompt(TABLE_4X6))
    assert "DLPFC" in anatomy
