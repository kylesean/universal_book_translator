"""Code block and inline identifier token masking for syntax protection."""

import functools
import re
from collections import Counter

from ubt.core.cleaners.mask_tokens import UnmaskReport
from ubt.core.cleaners.mask_tokens import find_reordered as _find_reordered
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position
from ubt.core.cleaners.mask_tokens import token_checksum as _token_checksum

_FENCED_CODE_PATTERN = re.compile(r"```[\w]*\n[\s\S]*?\n```|```[\s\S]*?```")
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")


@functools.lru_cache(maxsize=16)
def _get_code_fuzzy_pattern(mask_prefix: str) -> re.Pattern[str]:
    """Token shape surviving bracket/casing/spacing mutations and an optional checksum."""
    clean_prefix = re.sub(r"^[⟦\[【(\s]+|[⟧\]】)\s]+$", "", mask_prefix).rstrip(":_")
    prefix_pattern = re.sub(r"[:_]+", r"[:_]?", re.escape(clean_prefix))
    if "CODE" in clean_prefix.upper() and clean_prefix.upper() != "CODE_MASK":
        prefix_pattern = rf"(?:{prefix_pattern}|CODE[:_]?MASK|CODE)"
    return re.compile(
        r"[⟦\[【(]?\s*" + prefix_pattern + r"[:_]?(\d{1,6})(?:-([0-9a-z]{3}))?\s*[⟧\]】)]?",
        re.IGNORECASE,
    )


@functools.lru_cache(maxsize=16)
def _get_code_scan_pattern(mask_prefix: str) -> re.Pattern[str]:
    clean_prefix = re.sub(r"^[⟦\[【(\s]+|[⟧\]】)\s]+$", "", mask_prefix).rstrip(":_")
    prefix_pattern = re.sub(r"[:_]+", r"[:_]?", re.escape(clean_prefix))
    if "CODE" in clean_prefix.upper() and clean_prefix.upper() != "CODE_MASK":
        prefix_pattern = rf"(?:{prefix_pattern}|CODE[:_]?MASK|CODE)"
    return re.compile(prefix_pattern + r"[:_]?(\d{1,6})(?:-([0-9a-z]{3}))?", re.IGNORECASE)


class CodeMasker:
    """Masks code blocks and inline code spans before translation, and unmasks them afterward."""

    def __init__(self, mask_prefix: str = "⟦CODE_MASK_") -> None:
        self.mask_prefix = mask_prefix

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace code blocks and inline code with deterministic protective tokens.

        Tokens carry a checksum suffix (``⟦CODE_MASK_0001-a3f⟧``) binding the
        index to the masked original; see
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

        # 1. Mask fenced multi-line code blocks first
        masked_text = _FENCED_CODE_PATTERN.sub(_replace, text)
        # 2. Mask remaining inline code spans
        masked_text = _INLINE_CODE_PATTERN.sub(_replace, masked_text)
        return masked_text, _order_by_position(masked_text, mapping)

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
        pattern = _get_code_scan_pattern(self.mask_prefix)
        return [(int(idx), ck) for idx, ck in pattern.findall(text)]

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore masked code tokens back into their original code text with fuzzy matching.

        A checksum that is present but does not match the expected one is left
        in place rather than restoring the wrong span (renumbered tokens).
        Checksum-less echoes stay backward compatible and restore by index.
        """
        result = text
        for token, original in mapping.items():
            result = result.replace(token, original)

        table = self._index_table(mapping)
        if not table:
            return result

        fuzzy = _get_code_fuzzy_pattern(self.mask_prefix)

        def _fuzzy_repl(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            checksum = match.group(2)
            entry = table.get(idx)
            if entry is None:
                return match.group(0)
            if checksum and checksum.lower() != entry[2]:
                return match.group(0)
            return entry[1]

        return fuzzy.sub(_fuzzy_repl, result)

    def _missing_indices(
        self, table: dict[int, tuple[str, str, str]], seen: dict[int, str | None], restored: str
    ) -> list[int]:
        """Indices whose code span did not survive into the restored text.

        Count-aware per-original check; two identical spans masked by different
        tokens must not let one surviving copy vouch for the other. See
        :meth:`~ubt.core.cleaners.citation_masker.CitationMasker._missing_indices`
        for the rationale.
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

    def _duplicated_indices(
        self, table: dict[int, tuple[str, str, str]], restored: str
    ) -> list[int]:
        """Indices whose protected code span leaked as literal text in the draft.

        The mirror of the drop check: masking replaced *every* span, so a
        faithful draft holds each original exactly as many times as there are
        tokens for it. A restored count above that means the model echoed the
        token *and* emitted the span as free text, publishing a duplicated
        reference (the echo twin of :meth:`_missing_indices`).
        """
        expected = Counter(original for _, original, _ in table.values())
        leaked = {
            idx
            for idx, (_, original, _) in table.items()
            if restored.count(original) > expected[original]
        }
        return sorted(leaked)

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
        # order check can see a protected code span move within the block.
        reordered = _find_reordered(
            list(table), [idx for idx, _ in self._scan_refs(text) if idx in table]
        )
        residue = [f"{idx}-{ck}" if ck else str(idx) for idx, ck in self._scan_refs(restored)]
        duplicated = self._duplicated_indices(table, restored)
        return UnmaskReport(
            text=restored,
            missing=sorted(missing),
            mismatched=mismatched,
            mutated=residue,
            unverified=unverified,
            reordered=reordered,
            duplicated=duplicated,
        )
