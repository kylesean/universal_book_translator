"""Reject password-protected PDFs before an engine-specific parser sees them.

Docling reports an encrypted PDF as ``ConversionError: ... could not load
document`` and pdf_oxide adds a burst of ``Decryption failed ... Not
authenticated`` noise. Neither names the cause, so a user cannot tell an
encrypted file from a corrupt one. This probe runs at manifest extraction --
before engine dispatch -- so both PDF engines fail with the same actionable
message.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ubt.core.exceptions import DocumentParseError

logger = logging.getLogger(__name__)


def reject_encrypted_pdf(path: Path) -> None:
    """Raise ``DocumentParseError`` when ``path`` needs a password to open.

    Only a password refusal is intercepted. Corrupt or otherwise unreadable
    files are left to the parser, which reports them with engine-specific
    context; an encrypted file that opens with an empty user password is
    readable and must proceed.
    """
    import pikepdf

    try:
        with pikepdf.open(str(path)):
            pass
    except pikepdf.PasswordError as exc:
        raise DocumentParseError(
            f"PDF is password-protected or encrypted; decrypt it before translating: {path}"
        ) from exc
    except Exception as exc:  # a probe must not change how corrupt files are reported
        logger.debug("PDF encryption probe could not open %s: %s", path, exc)
