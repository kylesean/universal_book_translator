"""The document index behind the omission gate's table-label discriminator.

Pinned here:

* :func:`document_vocabulary` / :func:`identifier_terms_in` answer exactly what
  the historical per-call whole-document scan answered;
* the index is built once per run, not once per stage or per block — the defect
  this replaced re-scanned the whole book for every tracked term of every block
  (143 ms/block on a 3.9M-character book, 5581 s across a 20k-block run);
* ``OmissionGate.evaluate(document_text=...)`` actually honours its argument,
  which the documented "an explicit argument wins" promise did not;
* ``SourceDocumentIndex`` reads the book once and shares one vocabulary.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.source_document import SourceDocumentIndex
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    IRBlock,
    make_element,
)
from ubt.core.qe.omission import (
    _ALL_CAPS_TOKEN_RE,
    _LATIN_RUN_RE,
    DocumentVocabulary,
    OmissionGate,
    _is_identifier_shaped,
    document_vocabulary,
    identifier_terms,
    identifier_terms_in,
)

pytestmark = pytest.mark.fast


def _legacy_identifier_terms(source_text: str, document_text: str) -> set[str]:
    """The pre-index implementation, verbatim, as the equivalence oracle.

    Kept as the reference the new set-membership path must match: it is the
    code that produced every historical verdict (a whole-document ``re.sub``
    plus one boundary-aware ``re.search`` per candidate term).
    """
    runs = [
        match.group(0)
        for match in _LATIN_RUN_RE.finditer(source_text.strip())
        if len(match.group(0)) >= 2
    ]
    _freq = Counter(runs)
    tracked = {t for t in set(runs) if _is_identifier_shaped(t)}
    lowered = _ALL_CAPS_TOKEN_RE.sub(" ", document_text)
    return {
        t
        for t in tracked
        if not (
            t.isupper()
            and len(t) >= 3
            and t.isalpha()
            and re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", lowered, re.IGNORECASE)
        )
    }


# --------------------------------------------------------------------------- #
# Equivalence with the algorithm this replaced
# --------------------------------------------------------------------------- #

_BOOK = (
    "The mask is applied uniformly. Same as before. We use a random round.\n"
    "We use MHA and LSTM and GQA layers here.\n"
    "| MASK | SAME | RND |\n"
    "| MHA | LSTM | GQA |\n"
)

_CASES = (
    ("| MASK | SAME | RND |", _BOOK),
    ("| MHA | LSTM | GQA |", _BOOK),
    ("| MASK | SAME | MHA |", _BOOK),
    ("The MHA block.", _BOOK),
    ("The FinFET device whereQ0", _BOOK),
    ("| MASK | SAME | RND |", ""),
    ("| MASK | SAME | MHA |", ""),
)


@pytest.mark.parametrize(("source", "document"), _CASES)
def test_the_indexed_vocabulary_agrees_with_a_whole_document_scan(
    source: str, document: str
) -> None:
    expected = (
        _legacy_identifier_terms(source, document) if document else set(identifier_terms(source))
    )
    actual = identifier_terms(source, document)
    assert actual == expected


def test_the_verdict_is_identical_against_a_prepared_vocabulary() -> None:
    vocabulary = document_vocabulary(_BOOK)
    for source, _document in _CASES:
        assert identifier_terms_in(source, vocabulary) == identifier_terms(source, _BOOK)


def test_an_acronym_written_in_capitals_in_prose_is_not_evidence() -> None:
    """Stripping every all-caps run before indexing is what keeps this honest."""
    vocabulary = document_vocabulary("The MHA layer and the LSTM block differ.")
    assert identifier_terms_in("| MHA | LSTM |", vocabulary) == {"MHA", "LSTM"}


# --------------------------------------------------------------------------- #
# evaluate(document_text=...) must honour its argument
# --------------------------------------------------------------------------- #


def test_an_explicit_document_argument_wins_over_the_bound_context() -> None:
    """The docstring promised this and the implementation read ``self`` instead.

    A caller that passes ``document_text=`` gets the verdict for *that* text.
    Reading the bound context here made the public parameter inert: the two
    calls below were indistinguishable before this was fixed.
    """
    source = "Table 2: MASK | SAME | RND strategies. The model uses MHA layers."
    translation = "表2：掩码策略。模型使用 MHA 层。"
    book = "We apply a mask. The same token. Random choice. An MHA layer exists."

    bound = OmissionGate(target_lang="zh")
    bound.set_document_context(book)
    with_argument = bound.evaluate(source, translation, document_text=book)
    without_argument = bound.evaluate(source, translation)
    assert (with_argument.passed, with_argument.reason) == (
        without_argument.passed,
        without_argument.reason,
    )

    # ... and an explicit argument that differs from the bound context changes
    # the verdict (this is the half that was inert).
    unrelated = OmissionGate(target_lang="zh")
    unrelated.set_document_context("Nothing relevant here at all.")
    overridden = unrelated.evaluate(source, translation, document_text=book)
    assert overridden.reason == without_argument.reason


def test_a_prepared_vocabulary_may_be_passed_instead_of_raw_text() -> None:
    vocabulary = document_vocabulary(_BOOK)
    gate = OmissionGate(target_lang="zh")
    from_vocab = gate.evaluate(
        "| MASK | SAME | RND |", "| 掩码 | 相同 | 随机 |", document_text=vocabulary
    )
    from_text = gate.evaluate(
        "| MASK | SAME | RND |", "| 掩码 | 相同 | 随机 |", document_text=_BOOK
    )
    assert from_vocab.reason == from_text.reason
    assert from_vocab.metrics == from_text.metrics


def test_rebinding_the_same_document_does_not_reindex_the_book() -> None:
    """The chapter-streaming draft stage binds once per chapter.

    Re-indexing a 3.9M-character book per chapter cost 130 ms x chapters; the
    guard is that the second bind is a no-op, observable as the same object.
    """
    gate = OmissionGate(target_lang="zh")
    gate.set_document_context(_BOOK)
    first = gate._document_vocabulary
    gate.set_document_context(_BOOK)
    assert gate._document_vocabulary is first


# --------------------------------------------------------------------------- #
# The per-block cost this exists to remove
# --------------------------------------------------------------------------- #


def test_the_per_block_cost_no_longer_scales_with_the_book() -> None:
    """A 4000-block book must not make each evaluation scan it.

    Before the index, one ``evaluate`` did a whole-document ``re.sub`` plus a
    boundary search per candidate term, so the cost grew with the *book* while
    the question stayed per *block*. The ratio, not an absolute time, is pinned:
    a same-machine regression to rescanning the book per block blows it up by
    three orders of magnitude, while normal timing noise does not.
    """
    small_book = "\n".join(f"Paragraph {i} mentions MASK and MHA alike." for i in range(50))
    large_book = "\n".join(f"Paragraph {i} mentions MASK and MHA alike." for i in range(4_000))
    block = "Paragraph 7 mentions MASK and MHA alike."

    def measure(document: str, repeats: int = 20) -> float:
        vocabulary = document_vocabulary(document)
        gate = OmissionGate(target_lang="zh")
        gate.bind_document_vocabulary(vocabulary)
        gate.evaluate(block, "第 7 段同时提到掩码和 MHA。")  # warm
        started = time.perf_counter()
        for _ in range(repeats):
            gate.evaluate(block, "第 7 段同时提到掩码和 MHA。")
        return (time.perf_counter() - started) / repeats

    small = measure(small_book)
    large = measure(large_book)
    # An 80x larger book must not make the per-block question materially slower.
    assert large < max(small * 4.0, 2e-3), (
        f"per-block cost scales with the book ({small * 1e3:.3f} ms vs "
        f"{large * 1e3:.3f} ms): the whole-document scan is back"
    )


# --------------------------------------------------------------------------- #
# SourceDocumentIndex: read once, index once
# --------------------------------------------------------------------------- #


def _block(index: int) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=f"ch001#b{index:04d}",
            spine_index=index,
            block_type=BlockType.NARRATIVE,
            source_text=f"Paragraph {index} mentions MASK and MHA alike.",
        ),
        status=BlockStatus.MTQE_PASSED,
        target_text=f"第 {index} 段。",
    )


def test_the_source_index_reads_the_book_exactly_once(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    try:
        ledger.init_job_from_manifest(
            "job", BookManifest(doc_id="doc", title="t", source_path="s.pdf")
        )
        ledger.append_chapter(
            "job",
            ChapterIR(
                doc_id="doc",
                chapter_id="ch001",
                title="C1",
                spine_index=0,
                blocks=[_block(i) for i in range(5)],
            ),
        )
        index = SourceDocumentIndex()
        assert index.bound is False
        index.load(ledger, "job")
        first_text = index.text()
        first_vocab = index.vocabulary()
        assert index.bound is True
        assert "Paragraph 0" in first_text

        # Every later chapter's load is a no-op on the same objects.
        for _ in range(10):
            index.load(ledger, "job")
        assert index.text() is first_text
        assert index.vocabulary() is first_vocab
        # The vocabulary holds ordinary-case words only: every all-caps run is
        # stripped before indexing, or an acronym in prose would excuse itself.
        assert "mentions" in first_vocab.words
        assert "mha" not in first_vocab.words
    finally:
        ledger.close()


def test_the_index_survives_a_concurrent_load(tmp_path: Path) -> None:
    """Two chapter workers may race the first load; exactly one must read."""
    import threading

    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    try:
        ledger.init_job_from_manifest(
            "job", BookManifest(doc_id="doc", title="t", source_path="s.pdf")
        )
        ledger.append_chapter(
            "job",
            ChapterIR(
                doc_id="doc",
                chapter_id="ch001",
                title="C1",
                spine_index=0,
                blocks=[_block(i) for i in range(3)],
            ),
        )
        index = SourceDocumentIndex()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                index.load(ledger, "job")
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert index.bound is True
        assert index.text().count("Paragraph") == 3
    finally:
        ledger.close()


def test_an_unbound_index_answers_empty_instead_of_none() -> None:
    index = SourceDocumentIndex()
    assert index.text() == ""
    assert isinstance(index.vocabulary(), DocumentVocabulary)
    assert not index.vocabulary()
