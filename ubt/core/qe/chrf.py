"""chrF: character n-gram F-score (Popović 2015).

The masker-collapse acceptance is a *lower bound*: the refactored segment/translate path
must not degrade the translation relative to the pre-refactor masker path. chrF
compares the two outputs character-n-gram by character-n-gram, so a faithful
collapse scores ~1.0 and any drift in the mask/restore contract shows up as a
lower score.

This is deliberately a small, self-contained implementation (the pipeline
already carries a chrF-style residue metric in :mod:`ubt.core.qe.omission`); it
is not a general MT-evaluation framework.
"""

from __future__ import annotations

from collections import Counter


def _char_ngrams(text: str, n: int) -> Counter[str]:
    """Character n-grams over whitespace-normalized, word-boundary-padded text."""
    normalized = " ".join(text.split())
    padded = f" {normalized} "
    if n <= 0:
        n = 1
    if len(padded) < n:
        return Counter({padded: 1}) if padded.strip() else Counter()
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


def chrf(hypothesis: str, reference: str, *, n: int = 6, beta: float = 2.0) -> float:
    """The chrF score of ``hypothesis`` against ``reference`` in ``[0, 1]``.

    ``n`` is the character n-gram order (sacrebleu's default is 6) and ``beta``
    weights recall over precision (default 2, as in sacrebleu). Two empty strings
    are a perfect match (1.0); one empty and one non-empty is 0.0.
    """
    hyp = _char_ngrams(hypothesis, n)
    ref = _char_ngrams(reference, n)
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    overlap = sum((hyp & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp.values())
    recall = overlap / sum(ref.values())
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


__all__ = ["chrf"]
