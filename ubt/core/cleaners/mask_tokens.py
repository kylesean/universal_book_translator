"""Shared placeholder-token infrastructure for the masking cleaners.

The math / code / citation maskers all protect verbatim spans behind an opaque,
checksum-tagged token before drafting and verify the echo on restore. The
integrity machinery — the short checksum, the positional ordering, the
inversion detector, and the :class:`UnmaskReport` verdict — is mask-agnostic, so
it lives here rather than being reached into from ``math_masker``'s private
namespace by its siblings.
"""

import hashlib
from dataclasses import dataclass, field


def token_checksum(index: int, original: str) -> str:
    """Short integrity tag binding a token index to its masked original.

    Clean-room of the typed-placeholder-checksum idea: the model only ever
    sees the opaque token, but on restore we can verify the echoed index
    still belongs to the same span. A model that rewrites ``0001`` as
    ``0007`` (silent wrong-span restore) fails this check: the token is left
    in place, the restore is reported as not clean, and the draft stage
    checkpoints the block as REPAIR_PENDING instead of a finished draft — so a
    rewritten index can neither restore the wrong span nor ship.
    """
    digest = hashlib.blake2s(f"{index}\0{original}".encode(), digest_size=2).hexdigest()
    return digest[:3]


def position_of(token: str, text: str) -> int:
    """Index of ``token`` in ``text``, or one past the end when it is hidden."""
    at = text.find(token)
    return at if at >= 0 else len(text)


def order_by_position(masked_text: str, mapping: dict[str, str]) -> dict[str, str]:
    """Reorder a token mapping to the order the tokens appear in ``masked_text``.

    Indices are assigned per masking pass (display math before inline, fenced
    code before inline code), so index order is not textual order. Restore-time
    verification compares the draft's token order against the order the model
    actually saw, so the mapping must carry that order. Tokens hidden inside
    another token's original (a nested math environment) are not in the masked
    text; they are placed last and keep their relative order.
    """
    ordered = sorted(mapping, key=lambda token: position_of(token, masked_text))
    return {token: mapping[token] for token in ordered}


def find_reordered(expected: list[int], observed: list[int]) -> list[int]:
    """Spans whose draft position violates the expected (masked-source) order.

    ``expected`` is the order masked indices appear in the text handed to the
    model; ``observed`` is their order in the draft, restricted to indices the
    mapping issued. A checksum-verified swap restores both spans intact, so
    every other bucket stays empty — only this order check can see that the
    spans moved. Returns the indices participating in an inversion (both sides
    of a swap), or ``[]`` when the observed order is non-decreasing.

    An index absent from ``expected`` (a mapping that never issued it) is
    skipped rather than raising: callers pre-filter today, but a future
    masker that forgets degrades to a partial order check instead of
    crashing the restore path.
    """
    rank = {idx: pos for pos, idx in enumerate(expected)}
    involved: set[int] = set()
    high_pos = -1
    high_idx: int | None = None
    for idx in observed:
        pos = rank.get(idx)
        if pos is None:
            continue
        if pos < high_pos:
            involved.add(idx)
            if high_idx is not None:
                involved.add(high_idx)
        else:
            high_pos, high_idx = pos, idx
    return sorted(involved)


@dataclass
class UnmaskReport:
    """Result of a checksum-verified restore."""

    text: str
    missing: list[int] = field(default_factory=list)
    mismatched: list[int] = field(default_factory=list)
    mutated: list[str] = field(default_factory=list)
    unverified: list[int] = field(default_factory=list)
    reordered: list[int] = field(default_factory=list)
    duplicated: list[int] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when every masked span restored exactly once, intact, in place.

        ``unverified`` tokens (echoed without the checksum suffix the masker
        always emits) are *reported* but do not by themselves fail this
        property; whether they also fail closed is masker-specific. Most
        maskers restore a checksum-less token verbatim and report it only here;
        :class:`CitationMasker` refuses to restore it and additionally records
        it in ``mutated``, so a citation's ``clean`` is False in that case.
        Callers must not assume ``unverified`` implies a pass.

        "Reordered" tokens (checksums intact, but the spans appear in a
        different order than in the masked source) do fail closed: the text is
        not corrupt character-for-character, yet a protected span moved, which
        only a re-draft can undo.

        "Duplicated" tokens (the draft also emitted the protected span's
        original text alongside its intact token, so restoration printed it
        twice) fail closed for the same reason: the token restore is
        character-perfect, yet the published text would repeat a protected
        citation/formula/code span. Only a re-draft can remove the stray copy.
        """
        return (
            not self.missing
            and not self.mismatched
            and not self.mutated
            and not self.reordered
            and not self.duplicated
        )
