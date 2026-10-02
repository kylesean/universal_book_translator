"""Code block and inline identifier token masking for syntax protection."""

import re

from ubt.core.cleaners.mask_tokens import RestoreStyle, UnmaskReport, restore_masked, token_patterns
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position
from ubt.core.cleaners.mask_tokens import token_checksum as _token_checksum

_FENCED_CODE_PATTERN = re.compile(r"```[\w]*\n[\s\S]*?\n```|```[\s\S]*?```")
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")


class CodeMasker:
    """Masks code blocks and inline code spans before translation, and unmasks them afterward."""

    def __init__(self, mask_prefix: str = "⟦CODE_MASK_") -> None:
        self.mask_prefix = mask_prefix

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace code blocks and inline code with deterministic protective tokens.

        Tokens carry a checksum suffix (``⟦CODE_MASK_0001-a3f⟧``) binding the
        index to the masked original; see
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

        # 1. Mask fenced multi-line code blocks first
        masked_text = _FENCED_CODE_PATTERN.sub(_replace, text)
        # 2. Mask remaining inline code spans
        masked_text = _INLINE_CODE_PATTERN.sub(_replace, masked_text)
        return masked_text, _order_by_position(masked_text, mapping)

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore code tokens (mutations included); see :func:`restore_masked`."""
        return restore_masked(text, mapping, self._restore_style()).text

    def unmask_checked(self, text: str, mapping: dict[str, str]) -> UnmaskReport:
        """Restore plus integrity verification; see :class:`UnmaskReport`."""
        return restore_masked(text, mapping, self._restore_style())

    def _restore_style(self) -> RestoreStyle:
        fuzzy, scan = token_patterns(self.mask_prefix, "CODE")
        return RestoreStyle(
            scan_pattern=scan,
            fuzzy_pattern=fuzzy,
            checksumless_restores=True,
            nested=False,
        )
