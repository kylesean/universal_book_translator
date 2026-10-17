"""Real blocks the quality gate wrongly rejected, replayed against the live gate.

Every case here is verbatim text from an arXiv 2606.23344 run — not a
hand-written fixture. Each one was delivered as ``NEEDS_HUMAN`` with a flag that
says the translation is wrong (a terminology drift, a dropped number, an
untranslated passage) when the translation is in fact correct. The test asserts
the gate now passes them, and that the *same gate* still fails its near misses.

Why a fixture instead of the assertion only: reproducing these took four rounds
of a ~20-minute translation, because each gate that let a block through exposed
the next gate hidden behind it. The fixture freezes the text so a future change
to any of those gates is checked in milliseconds.

The near-miss assertions living beside each case are the point: "this block
passes" is only meaningful if the same rule still rejects the defect it exists
to catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ubt.core.qe.fast_pass import FastPassFilter, is_verbatim_echo
from ubt.core.qe.term_drift import detect_term_drift, normalize_glossary
from ubt.core.validators.consistency import (
    GlossaryConsistencyValidator,
    NumericConsistencyValidator,
)

pytestmark = pytest.mark.fast

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "measured_gate_false_positives.json"


def _cases() -> dict[str, dict[str, Any]]:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return {case["block_id"]: case for case in payload["cases"]}


CASES = _cases()


@pytest.mark.parametrize("block_id", sorted(CASES))
def test_a_correct_translation_is_not_flagged_by_any_gate(block_id: str) -> None:
    """The block must clear every gate that once rejected it.

    ``test_defects_rt_doclayout`` proves each *rule* in isolation with a
    minimal pair. This proves the rule against the text it was derived from —
    which is where the earlier attempts went wrong (a fix that looked right on
    a constructed sentence still rejected the real one).
    """
    case = CASES[block_id]
    src, tgt = case["source_text"], case["target_text"]
    block_type = case["block_type"]

    # Untranslated-echo gate (headings of pure identifiers).
    assert not is_verbatim_echo(src, tgt), f"{block_id}: reported as an untranslated echo"

    # Numeric fidelity gate.
    numeric = NumericConsistencyValidator().validate(src, tgt)
    assert numeric.is_valid, f"{block_id}: {numeric.message}"

    # Target-language density gate (full evaluate path, as the pipeline runs it).
    decision = FastPassFilter().evaluate(src, tgt, block_type=block_type)
    assert decision.passed, f"{block_id}: {decision.reason}"


#: The five terminology flags the run raised, and what the ledger text shows.
#: Only these three are false positives: the translation carries the canonical
#: term (or an accepted synonym, which the validator does not see). The other two
#: are NOT — see ``test_the_terminology_flags_that_were_right_still_fail``.
_TERMINOLOGY_FALSE_POSITIVES = {
    "pdf_main#b0029": {"source": "Detection", "translation": "检测"},
    "pdf_main#b0049": {"source": "Mask", "translation": "掩码"},
    "pdf_main#b0099": {"source": "Mask", "translation": "掩码"},
}


@pytest.mark.parametrize("block_id", sorted(_TERMINOLOGY_FALSE_POSITIVES))
def test_the_terminology_gate_passes_the_blocks_it_wrongly_rejected(block_id: str) -> None:
    """Terminology needs the run's glossary, which the ledger does not keep.

    Each entry is the term the false-positive message named, so the check is the
    one that actually fired rather than a guess.
    """
    case = CASES[block_id]
    result = GlossaryConsistencyValidator([_TERMINOLOGY_FALSE_POSITIVES[block_id]]).validate(
        case["source_text"], case["target_text"]
    )
    assert result.is_valid, f"{block_id}: {result.message}"


def test_the_terminology_flags_that_were_right_still_fail() -> None:
    """Two of the five were real: the gate must keep rejecting them.

    Recording this is the point. b0010 and b0015 were counted as fixed when their
    *violation* scan stopped reporting them, but the validator that gates delivery
    uses the drift scan, which still fails them — and reading the text shows why
    the gate was right both times:

    * b0010 ends ``…/PP-DocLayoutV3 获取。`` — the URL did not stop at its end and
      swallowed 公开, so "publicly available" became just "obtain";
    * b0015 renders "content interference" as ``相邻元素的干扰`` ("interference
      from adjacent elements"), dropping 内容.

    Both are translation defects, not measurement artefacts. The gate catching
    them is the behaviour a delivery needs; softening the drift scan to make them
    pass would hide two real errors on a shipped page.
    """
    for block_id, entry in (
        ("pdf_main#b0010", {"source": "Publicly Available", "translation": "公开可用"}),
        ("pdf_main#b0015", {"source": "Content Interference", "translation": "内容干扰"}),
    ):
        case = CASES[block_id]
        result = GlossaryConsistencyValidator([entry]).validate(
            case["source_text"], case["target_text"]
        )
        assert not result.is_valid, f"{block_id}: the defect was expected to be caught"
        # ...and the text is broken in the way the message says.
        assert entry["translation"] not in case["target_text"]


# --- the near misses: the same rules must still catch the real defects --------- #


def test_a_term_only_kept_as_a_label_does_not_excuse_prose() -> None:
    """The b0099 exemption must not leak into running text."""
    glossary = normalize_glossary([{"source": "Mask", "translation": "掩码"}])
    (finding,) = detect_term_drift(
        "The mask head predicts masks for every query.",
        "头部为每个查询预测结果。",
        glossary,
    )
    assert finding.drifted is True


def test_a_count_suffix_still_fails_when_the_count_is_dropped() -> None:
    """The 38k reading must not authorize losing the quantity."""
    assert (
        not NumericConsistencyValidator()
        .validate("we scaled our corpus to over 38k samples.", "我们扩大了训练语料库。")
        .is_valid
    )


def test_a_scientific_exponent_does_not_excuse_a_dropped_minus() -> None:
    """The 10-4 reading must not launder a genuine sign loss."""
    assert (
        not NumericConsistencyValidator()
        .validate("the temperature is -4 degrees", "温度是 4 度")
        .is_valid
    )


def test_an_identifier_heading_still_fails_when_it_has_prose() -> None:
    """The echo exemption must not cover a heading that DOES contain prose."""
    assert is_verbatim_echo("3. Proposed Approach", "3. Proposed Approach")


def test_an_untranslated_table_still_fails_the_density_gate() -> None:
    """The density fix for Real5-OmniDocBench must not release an untouched table.

    This is the assertion that caught the first (wrong) attempt: stripping
    source terms before digits erased the residue that proves a table was never
    translated.
    """
    src = "| Model | GLUE | SQuAD |\n|---|---|---|\n| T5-Base | 84.3 | 85.5 |"
    decision = FastPassFilter().evaluate(src, src, block_type="table")
    assert not decision.passed
    assert "script density" in decision.reason


# --- the real defect the URL bug hid ------------------------------------------ #


def test_a_url_terminates_at_a_cjk_glyph() -> None:
    """A word glued to a link is not part of the link.

    Measured on arXiv 2606.23344 (b0010): the translation wrote
    "…/PP-DocLayoutV3获取。" and every URL pattern in the codebase used "not
    whitespace", so 获取 became part of the URL. That single character had three
    downstream effects: the Pangu spacer was blocked from separating it, the
    glossary scan could not see that 公开 had been dropped, and the density
    residue lost a word the translator wrote.
    """
    from ubt.core.cleaners.cjk_spacing import normalize_publishing_cjk
    from ubt.core.qe.fast_pass import _URL_RE
    from ubt.core.validators.glossary_enforcer import extract_protected_spans

    text = "模型权重可在 https://huggingface.co/PaddlePaddle/PP-DocLayoutV3获取。"
    url = "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3"

    assert _URL_RE.findall(text) == [url]
    assert [text[s:e] for s, e in extract_protected_spans(text)] == [url]
    # ...and the spacer can now do its job.
    assert normalize_publishing_cjk(text) == f"模型权重可在 {url} 获取。"


def test_a_url_path_is_still_read_whole() -> None:
    """The near miss: only CJK ends a URL, nothing else does."""
    from ubt.core.qe.fast_pass import _URL_RE

    for url in (
        "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3",
        "https://arxiv.org/abs/2606.23344v1",
        "https://example.org/a_b-c~d/e?x=1&y=2#frag",
        "https://example.org/path_with_underscores/file.tar.gz",
    ):
        assert _URL_RE.findall(f"see {url} for details") == [url]


def test_the_url_swallowed_word_is_now_caught_as_a_lost_term() -> None:
    """With the boundary fixed, b0010's defect is visible instead of hidden.

    The block was delivered as correct while its text read "…/PP-DocLayoutV3
    获取。" — "publicly available" had become "obtain". Before the boundary fix
    the term scan could not see the missing 公开 at all.
    """
    from ubt.core.validators.consistency import GlossaryConsistencyValidator

    case = CASES["pdf_main#b0010"]
    assert "公开" not in case["target_text"]
    result = GlossaryConsistencyValidator(
        [{"source": "Publicly Available", "translation": "公开可用"}]
    ).validate(case["source_text"], case["target_text"])
    assert not result.is_valid
