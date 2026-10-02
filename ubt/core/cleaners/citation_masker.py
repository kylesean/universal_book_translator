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

import re

from ubt.core.cleaners.mask_tokens import RestoreStyle, UnmaskReport, restore_masked, token_patterns
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position
from ubt.core.cleaners.mask_tokens import token_checksum as _token_checksum

_MASK_PREFIX = "⟦CITE_MASK_"

# Bracketed numeric citation: [12], [12-14], [1, 2, 3], [5-7, 9]
_CITATION_PATTERN = re.compile(r"\[\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]")


class CitationMasker:
    """Masks bracketed numeric citations before translation and unmasks afterwards."""

    def __init__(self, mask_prefix: str = _MASK_PREFIX) -> None:
        self.mask_prefix = mask_prefix

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace citations with protective tokens; returns (masked_text, mapping).

        Tokens carry a checksum suffix (``⟦CITE_MASK_0001-a3f⟧``) binding the
        index to the masked citation; see
        :func:`~ubt.core.cleaners.mask_tokens.token_checksum`.
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

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore citations; only checksum-verified tokens (see restore_masked)."""
        return restore_masked(text, mapping, self._restore_style()).text

    def unmask_checked(self, text: str, mapping: dict[str, str]) -> UnmaskReport:
        """Restore plus integrity verification; see :class:`UnmaskReport`."""
        return restore_masked(text, mapping, self._restore_style())

    def _restore_style(self) -> RestoreStyle:
        fuzzy, scan = token_patterns(self.mask_prefix, "CITE")
        return RestoreStyle(
            scan_pattern=scan,
            fuzzy_pattern=fuzzy,
            checksumless_restores=False,
            nested=False,
        )
