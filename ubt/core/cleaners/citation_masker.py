"""Read-only citation masking for bracketed numeric references.

Academic text carries inline citation markers ([12], [12-14], [3, 7, 21],
``cf. [5]``) that LLM translation routinely corrupts: numbers get re-numbered,
ranges are translated as prose ("12 到 14"), or markers are dropped entirely.
Since citations must survive translation verbatim, they are masked behind
deterministic tokens before drafting and unmasked afterwards — the same
placeholder-protection contract as :class:`~ubt.core.cleaners.code_masker.CodeMasker`.

Spoken-form citations like "(Smith et al., 2021)" are intentionally NOT masked
in this iteration: they are free text whose translation is expected to adapt
("et al." conventions differ per language). Bracketed numeric markers are the
hard invariant.
"""

import functools
import re
from collections import Counter

from ubt.core.cleaners.mask_tokens import UnmaskReport
from ubt.core.cleaners.mask_tokens import find_reordered as _find_reordered
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position
from ubt.core.cleaners.mask_tokens import token_checksum as _token_checksum

_MASK_PREFIX = "⟦CITE_MASK_"

# Bracketed numeric citation: [12], [12-14], [1, 2, 3], [5-7, 9]
_CITATION_PATTERN = re.compile(r"\[\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]")


@functools.lru_cache(maxsize=16)
def _get_citation_fuzzy_pattern(mask_prefix: str) -> re.Pattern[str]:
    # Core name survives casing/bracket mutations: ⟦CITE_MASK_0001-abc⟧,
    # [cite_mask_0001], 【CITE_MASK_0001】, bare CITE_MASK_0001.
    clean_prefix = re.sub(r"^[⟦\[【(\s]+|[⟧\]】)\s]+$", "", mask_prefix).rstrip(":_")
    prefix_pattern = re.sub(r"[:_]+", r"[:_]?", re.escape(clean_prefix))
    if "CITE" in clean_prefix.upper() and clean_prefix.upper() != "CITE_MASK":
        prefix_pattern = rf"(?:{prefix_pattern}|CITE[:_]?MASK|CITE)"
    return re.compile(
        r"[⟦\[【(]?\s*" + prefix_pattern + r"[:_]?(\d{1,6})(?:-([0-9a-z]{3}))?\s*[⟧\]】)]?",
        re.IGNORECASE,
    )


@functools.lru_cache(maxsize=16)
def _get_citation_scan_pattern(mask_prefix: str) -> re.Pattern[str]:
    clean_prefix = re.sub(r"^[⟦\[【(\s]+|[⟧\]】)\s]+$", "", mask_prefix).rstrip(":_")
    prefix_pattern = re.sub(r"[:_]+", r"[:_]?", re.escape(clean_prefix))
    if "CITE" in clean_prefix.upper() and clean_prefix.upper() != "CITE_MASK":
        prefix_pattern = rf"(?:{prefix_pattern}|CITE[:_]?MASK|CITE)"
    return re.compile(prefix_pattern + r"[:_]?(\d{1,6})(?:-([0-9a-z]{3}))?", re.IGNORECASE)


class CitationMasker:
    """Masks bracketed numeric citations before translation and unmasks afterwards."""

    def __init__(self, mask_prefix: str = _MASK_PREFIX) -> None:
        self.mask_prefix = mask_prefix

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace citations with protective tokens; returns (masked_text, mapping).

        Tokens carry a checksum suffix (``⟦CITE_MASK_0001-a3f⟧``) binding the
        index to the masked citation; see
        :func:`~ubt.core.cleaners.math_masker._token_checksum`.
        """
        mapping: dict[str, str] = {}
        counter = 1

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            original = match.group(0)
            token = f"{self.mask_prefix}{counter:04d}-{_token_checksum(counter, original)}⟧"
            mapping[token] = original
            counter += 1
            return token

        masked = _CITATION_PATTERN.sub(_replace, text)
        return masked, _order_by_position(masked, mapping)

    def _index_table(self, mapping: dict[str, str]) -> dict[int, tuple[str, str, str]]:
        """index -> (token, original, expected checksum)."""
        table: dict[int, tuple[str, str, str]] = {}
        for token, original in mapping.items():
            m = re.search(r"(\d+)", token)
            if m:
                idx = int(m.group(1))
                table[idx] = (token, original, _token_checksum(idx, original))
        return table

    def _scan_refs(self, text: str) -> list[tuple[int, str | None]]:
        """All token-shaped references in text: (index, checksum-or-None)."""
        pattern = _get_citation_scan_pattern(self.mask_prefix)
        return [(int(idx), ck) for idx, ck in pattern.findall(text)]

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore citation tokens; fuzzy-matches LLM mutations like CodeMasker.

        Only a token carrying a *matching* checksum is restored. A present-but-
        wrong checksum (renumbered token) and a checksum-less bare token are both
        left in place: a leftover mask is reported unverified and quarantined,
        whereas silently restoring a citation the model merely echoed back would
        corrupt the book (M8).
        """
        result = text
        for token, original in mapping.items():
            result = result.replace(token, original)

        table = self._index_table(mapping)
        if not table:
            return result

        fuzzy = _get_citation_fuzzy_pattern(self.mask_prefix)

        def _fuzzy_repl(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            checksum = match.group(2)
            entry = table.get(idx)
            if entry is None:
                return match.group(0)
            if not checksum or checksum.lower() != entry[2]:
                return match.group(0)
            return entry[1]

        return fuzzy.sub(_fuzzy_repl, result)

    def _duplicated_indices(
        self, table: dict[int, tuple[str, str, str]], restored: str
    ) -> list[int]:
        """Indices whose protected citation leaked as literal text in the draft.

        Masking replaces *every* bracketed numeric citation in the block, so the
        model can only emit the original form ``[12]`` by echoing it from its
        own memory (or from unmasked neighbor context in the prompt). After the
        tokens restore, a faithful draft contains each citation exactly as many
        times as the mapping has tokens for it. A draft that kept the citation
        as free text *in addition* to its token inflates that count, and the
        published text would repeat the reference (defect 10.4-3: every
        existing bucket stayed empty and the echo passed as clean). A citation-
        shaped string the mapping never issued is a *foreign* citation — added
        content, which the QE added-content gate owns, not this masker.
        """
        expected = Counter(original for _, original, _ in table.values())
        leaked = {
            idx
            for idx, (_, original, _) in table.items()
            if restored.count(original) > expected[original]
        }
        return sorted(leaked)

    def _missing_indices(
        self, table: dict[int, tuple[str, str, str]], seen: dict[int, str | None], restored: str
    ) -> list[int]:
        """Indices whose citation did not survive into the restored text.

        Multiple tokens can mask the same original — two ``[12]`` in one block.
        A test satisfied whenever *any* copy of the original is present lets a
        draft that dropped 1 of 2 identical citations read as clean, and the
        fail-closed checker opens on symmetric input. Attribute the shortfall
        to the tokens the model never emitted, letting a genuine free-text copy
        cover one that it did drop. Bare/mismatched tokens keep
        going to ``unverified``/``mismatched`` and are not double-flagged here.
        """
        by_original: dict[str, list[int]] = {}
        for idx, (_, original, _) in table.items():
            by_original.setdefault(original, []).append(idx)
        missing: list[int] = []
        for original, idxs in by_original.items():
            valid_emitted = [
                idx for idx in idxs if seen.get(idx) and str(seen[idx]).lower() == table[idx][2]
            ]
            omitted = [idx for idx in idxs if idx not in seen]
            surplus = restored.count(original) - len(valid_emitted)
            covered = min(len(omitted), max(0, surplus))
            missing.extend(omitted[covered:])
        return sorted(set(missing))

    def unmask_checked(self, text: str, mapping: dict[str, str]) -> UnmaskReport:
        """Restore plus integrity verification (see :class:`UnmaskReport`)."""
        table = self._index_table(mapping)
        seen: dict[int, str | None] = {}
        for idx, checksum in self._scan_refs(text):
            seen.setdefault(idx, checksum)

        restored = self.unmask(text, mapping)

        missing = self._missing_indices(table, seen, restored)
        unverified = sorted(idx for idx, ck in seen.items() if idx in table and not ck)
        mismatched = sorted(
            idx for idx, ck in seen.items() if idx in table and ck and ck.lower() != table[idx][2]
        )
        # A swap of two intact spans leaves every other bucket empty; only the
        # order check can see a protected citation move within the block.
        reordered = _find_reordered(
            list(table), [idx for idx, _ in self._scan_refs(text) if idx in table]
        )
        # A token echoed *plus* its citation emitted as free text leaves the
        # restore character-perfect and every bucket above empty — only the
        # occurrence count sees the duplicated reference (defect 10.4-3).
        duplicated = self._duplicated_indices(table, restored)
        residue = [f"{idx}-{ck}" if ck else str(idx) for idx, ck in self._scan_refs(restored)]
        return UnmaskReport(
            text=restored,
            missing=sorted(missing),
            mismatched=mismatched,
            mutated=residue,
            unverified=unverified,
            reordered=reordered,
            duplicated=duplicated,
        )


def count_citations(text: str) -> int:
    """Number of bracketed numeric citations in text (0-token diagnostics)."""
    return len(_CITATION_PATTERN.findall(text))
