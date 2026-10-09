"""Read-only citation masking for inline reference markers.

Academic text carries inline citation markers ([12], [12-14], [3, 7, 21],
``cf. [5]``) and author-year references ((Guo et al., 2025; Jimenez et al.,
2024), (DeepSeek-AI, 2026)) that LLM translation routinely corrupts: numbers get
re-numbered, ranges are translated as prose ("12 到 14"), "et al." is localized
into the middle of a reference, or markers are dropped entirely. Citations are
identifiers — they must stay retrievable and match the reference list — so they
are masked behind deterministic tokens before drafting and unmasked afterwards,
the same placeholder-protection contract as
:class:`~ubt.core.cleaners.code_masker.CodeMasker`.

Both forms are protected: bracketed numeric markers and parenthesised
author-year references. Only a *reference-shaped* parenthetical is masked (an
author token immediately followed by ``(et al.)? , YYYY``), so an ordinary aside
like ``(DSH)``, ``(RL)`` or ``(see note)`` stays prose and is still translated.
"""

import re

from ubt.core.cleaners.mask_tokens import (
    BaseMasker,
    RestoreStyle,
    TokenFactory,
    token_patterns,
)
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position

_MASK_PREFIX = "⟦CITE_MASK_"

# Bracketed numeric citation: [12], [12-14], [1, 2, 3], [5-7, 9]. A citation
# stands as its own token, so the leading lookbehind and trailing lookahead keep
# out the look-alikes that are not citations: a code/array subscript glued to a
# lowercase identifier (``arr[0]``, ``matrix[12]``) and Markdown link text
# (``[1](url)``). The glue blocklist is deliberately lowercase-ASCII only: the
# previous ``\w`` blocklist also un-invited real citations glued to CJK text
# (``结果表明[12]``) or to a previous citation (``[3][12]``). Known residual:
# a citation glued to a Latin name ("Smith[12]", no space) still reads as a
# subscript — the spaced style ("Smith [12]") is the dominant Latin form and
# is masked normally.
_CITATION_PATTERN = re.compile(
    r"(?<![a-z0-9_)])"
    r"\[\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]"
    r"(?!\()"
)
# Parenthesised author-year reference: (Guo et al., 2025), (DeepSeek-AI, 2026),
# and the semicolon-joined multi form (Xie et al., 2024; Zhou et al., 2024). An
# author token is a capitalised word (optionally ``et al.``) immediately followed
# by ``, YYYY``; the immediate-comma requirement is what keeps ``(Section 2,
# 2024)`` and ``(DSH)`` out.
_PAREN_CITATION_PATTERN = re.compile(
    r"\("
    r"[A-Z][\w.'’\-]*(?:\s+et\s+al\.?)?\s*,\s*(?:19|20)\d{2}[a-z]?"
    r"(?:\s*;\s*[A-Z][\w.'’\-]*(?:\s+et\s+al\.?)?\s*,\s*(?:19|20)\d{2}[a-z]?)*"
    r"\s*\)"
)


class CitationMasker(BaseMasker):
    """Masks inline citations before translation and unmasks afterwards."""

    default_prefix = _MASK_PREFIX

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace citations with protective tokens; returns (masked_text, mapping).

        Tokens carry a checksum suffix (``⟦CITE_MASK_0001-a3f⟧``) binding the
        index to the masked citation; see
        :func:`~ubt.core.cleaners.mask_tokens.token_checksum`.
        """
        factory = TokenFactory(self.mask_prefix)

        masked = _CITATION_PATTERN.sub(lambda m: factory.next(m.group(0)), text)
        masked = _PAREN_CITATION_PATTERN.sub(lambda m: factory.next(m.group(0)), masked)
        return masked, _order_by_position(masked, factory.mapping)

    def _restore_style(self) -> RestoreStyle:
        fuzzy, scan = token_patterns(self.mask_prefix, "CITE")
        return RestoreStyle(
            scan_pattern=scan,
            fuzzy_pattern=fuzzy,
            checksumless_restores=False,
            nested=False,
        )
