"""Code block and inline identifier token masking for syntax protection.

A fenced block or ``inline code`` span is syntax, not prose: a model that
"localizes" an identifier or reflows a line inside a block has changed the
artifact, so the span is frozen for the duration of drafting.
"""

import re

from ubt.core.cleaners.mask_tokens import (
    BaseMasker,
    RestoreStyle,
    TokenFactory,
    token_patterns,
)
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position

_FENCED_CODE_PATTERN = re.compile(r"```[\w]*\n[\s\S]*?\n```|```[\s\S]*?```")
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")


class CodeMasker(BaseMasker):
    """Masks code blocks and inline code spans before translation, and unmasks them afterward."""

    default_prefix = "⟦CODE_MASK_"

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace code blocks and inline code with deterministic protective tokens."""
        factory = TokenFactory(self.mask_prefix)

        # 1. Mask fenced multi-line code blocks first
        masked_text = _FENCED_CODE_PATTERN.sub(lambda m: factory.next(m.group(0)), text)
        # 2. Mask remaining inline code spans
        masked_text = _INLINE_CODE_PATTERN.sub(lambda m: factory.next(m.group(0)), masked_text)
        return masked_text, _order_by_position(masked_text, factory.mapping)

    def _restore_style(self) -> RestoreStyle:
        fuzzy, scan = token_patterns(self.mask_prefix, "CODE")
        return RestoreStyle(
            scan_pattern=scan,
            fuzzy_pattern=fuzzy,
            checksumless_restores=True,
            nested=False,
        )
