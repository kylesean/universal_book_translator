"""Email address and URL masking for verbatim protection.

Academic and production PDFs carry contact addresses (``research@deepseek.com``)
and links (``https://…``) that an LLM routinely mangles: ``research@`` gets
translated as a word, a URL's path is rewritten, or the address is split across
a line and half of it is dropped. They are identifiers, not prose, so they are
masked behind deterministic tokens before drafting and unmasked afterwards —
the same placeholder-protection contract as
:class:`~ubt.core.cleaners.code_masker.CodeMasker`.

URLs are masked before emails so a ``mailto:user@host`` or ``https://user@host``
does not leave its ``user@host`` half to be caught as a bare email. Both are the
most literal spans, so this family runs first in the engine's mask order.
"""

import re

from ubt.core.cleaners.mask_tokens import (
    BaseMasker,
    RestoreStyle,
    TokenFactory,
    token_patterns,
)
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position

_MASK_PREFIX = "⟦EMAIL_MASK_"

#: A bare address. The local part and domain are conservative (no spaces, no
#: angle brackets) so prose like ``user @ host`` is never welded together.
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
#: A scheme-prefixed locator. Closing brackets and quotes are excluded so the
#: match stops at the end of the URL rather than swallowing a following clause.
_URL_PATTERN = re.compile(r"(?:https?://|ftp://|mailto:)[^\s<>()\[\]{}\"']+")
#: Punctuation that commonly trails a URL in prose and belongs to the sentence.
_URL_TRAILING = ".,;:!?。，；：！？"


class EmailMasker(BaseMasker):
    """Masks email addresses and URLs before translation, unmasks afterwards."""

    default_prefix = _MASK_PREFIX

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace addresses/URLs with protective tokens; returns (masked, mapping).

        Tokens carry a checksum suffix (``⟦EMAIL_MASK_0001-a3f⟧``) binding the
        index to the masked original; see
        :func:`~ubt.core.cleaners.mask_tokens.token_checksum`.
        """
        factory = TokenFactory(self.mask_prefix)

        def _replace_url(match: re.Match[str]) -> str:
            original = match.group(0)
            # A trailing period/comma is the sentence's, not the URL's; keep it
            # outside the token so it is not frozen into the restored address.
            core = original.rstrip(_URL_TRAILING)
            if not core:
                return original
            return factory.next(core) + original[len(core) :]

        def _replace_email(match: re.Match[str]) -> str:
            return factory.next(match.group(0))

        masked = _URL_PATTERN.sub(_replace_url, text)
        masked = _EMAIL_PATTERN.sub(_replace_email, masked)
        return masked, _order_by_position(masked, factory.mapping)

    def _restore_style(self) -> RestoreStyle:
        fuzzy, scan = token_patterns(self.mask_prefix, "EMAIL")
        return RestoreStyle(
            scan_pattern=scan,
            fuzzy_pattern=fuzzy,
            checksumless_restores=False,
            nested=False,
        )
