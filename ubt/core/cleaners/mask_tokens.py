"""Shared placeholder-token infrastructure for the masking cleaners.

The math / code / citation maskers all protect verbatim spans behind an opaque,
checksum-tagged token before drafting and verify the echo on restore. The
integrity machinery — the short checksum, the positional ordering, the
inversion detector, and the :class:`UnmaskReport` verdict — is mask-agnostic, so
it lives here rather than being reached into from ``math_masker``'s private
namespace by its siblings.
"""

import hashlib
import re
from collections import Counter
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


@dataclass(frozen=True, slots=True)
class RestoreStyle:
    """How one placeholder family's tokens are shaped and restored.

    The math / code / citation / soup maskers differ only in these four knobs;
    everything else about restoring a masked block is identical. Collapsing the
    four near-identical ``unmask_checked`` implementations into
    :func:`restore_masked` is what lets "protect a span" have one owner.
    """

    scan_pattern: re.Pattern[str]
    fuzzy_pattern: re.Pattern[str]
    #: Restore a checksum-less echo by index (math/code/soup) or refuse it
    #: (citation: an echoed marker we cannot verify must not be re-published).
    checksumless_restores: bool = True
    #: Re-run restore to a fixed point so a token revealed by restoring an
    #: enclosing token (a math environment nested inside inline math) expands.
    nested: bool = False


def _index_table(mapping: dict[str, str]) -> dict[int, tuple[str, str, str]]:
    """index -> (token, original, expected checksum)."""
    table: dict[int, tuple[str, str, str]] = {}
    for token, original in mapping.items():
        match = re.search(r"(\d+)", token)
        if match:
            idx = int(match.group(1))
            table[idx] = (token, original, token_checksum(idx, original))
    return table


def _restore_text(
    text: str, mapping: dict[str, str], table: dict[int, tuple[str, str, str]], style: RestoreStyle
) -> str:
    """Replace tokens verbatim, then fuzzy-match the mutations the model made."""

    def _fuzzy_repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        checksum = match.group(2)
        entry = table.get(idx)
        if entry is None:
            return match.group(0)
        if checksum:
            if checksum.lower() != entry[2]:
                return match.group(0)  # present-but-wrong: never restore the wrong span
        elif not style.checksumless_restores:
            return match.group(0)
        return entry[1]

    result = text
    # A no-nesting family needs one pass; a nested one (a token whose original
    # contains another token) needs a fixed point, bounded so a pathological
    # mapping cannot spin.
    passes = min(len(mapping) + 2, 6) if style.nested else 1
    for _ in range(passes):
        changed = False
        for token, original in mapping.items():
            if token in result:
                replaced = result.replace(token, original)
                if replaced != result:
                    changed = True
                    result = replaced
        subbed = style.fuzzy_pattern.sub(_fuzzy_repl, result)
        if subbed != result:
            changed = True
            result = subbed
        if not changed:
            break
    return result


def _missing_indices(
    table: dict[int, tuple[str, str, str]], seen: dict[int, str | None], restored: str
) -> list[int]:
    """Indices whose span did not survive into the restored text.

    Count-aware per-original: two identical spans masked by different tokens
    must not let one surviving copy vouch for the other.
    """
    by_original: dict[str, list[int]] = {}
    for idx, (_, original, _) in table.items():
        by_original.setdefault(original, []).append(idx)
    missing: list[int] = []
    for original, idxs in by_original.items():
        valid_emitted = [
            idx for idx in idxs if seen.get(idx) and str(seen[idx]).lower() == table[idx][2].lower()
        ]
        omitted = [idx for idx in idxs if idx not in seen]
        surplus = restored.count(original) - len(valid_emitted)
        covered = min(len(omitted), max(0, surplus))
        missing.extend(omitted[covered:])
    return sorted(set(missing))


def _duplicated_indices(table: dict[int, tuple[str, str, str]], restored: str) -> list[int]:
    """Indices whose protected span leaked as free text in the draft.

    A faithful draft holds each original exactly as many times as there are
    tokens for it; a higher restored count is the model emitting the span
    twice (echo twin of :func:`_missing_indices`).
    """
    expected: Counter[str] = Counter(original for _, original, _ in table.values())
    leaked = {
        idx
        for idx, (_, original, _) in table.items()
        if restored.count(original) > expected[original]
    }
    return sorted(leaked)


def restore_masked(text: str, mapping: dict[str, str], style: RestoreStyle) -> UnmaskReport:
    """Restore a masked block and verify it — the one restore implementation.

    Replaces the four near-identical ``unmask_checked`` methods the math / code /
    citation / soup maskers each carried. Every integrity bucket is computed here
    once; a family only supplies its :class:`RestoreStyle`.
    """
    table = _index_table(mapping)
    draft_refs = [(int(idx), checksum) for idx, checksum in style.scan_pattern.findall(text)]
    seen: dict[int, str | None] = {}
    for idx, checksum in draft_refs:
        seen.setdefault(idx, checksum)

    restored = _restore_text(text, mapping, table, style)

    missing = _missing_indices(table, seen, restored)
    # A token echoed without its checksum is "unverified", not "mismatched":
    # the suffix is optional, so a faithful model may drop it. Only a checksum
    # that is present but wrong (a rewritten index) is a wrong-span risk.
    unverified = sorted(idx for idx, ck in seen.items() if idx in table and not ck)
    mismatched = sorted(
        idx
        for idx, ck in seen.items()
        if idx in table and ck and ck.lower() != table[idx][2].lower()
    )
    # A swap of two intact tokens leaves every other bucket empty; only the
    # order check can see a protected span move within the block.
    reordered = find_reordered(list(table), [idx for idx, _ in draft_refs if idx in table])
    # Post-restore residue: anything still token-shaped is unaccounted for.
    residue = [
        f"{int(idx)}-{ck}" if ck else str(int(idx))
        for idx, ck in style.scan_pattern.findall(restored)
    ]
    duplicated = _duplicated_indices(table, restored)
    return UnmaskReport(
        text=restored,
        missing=sorted(missing),
        mismatched=mismatched,
        mutated=residue,
        unverified=unverified,
        reordered=reordered,
        duplicated=duplicated,
    )


__all__ = [
    "RestoreStyle",
    "UnmaskReport",
    "find_reordered",
    "order_by_position",
    "position_of",
    "restore_masked",
    "token_checksum",
]
