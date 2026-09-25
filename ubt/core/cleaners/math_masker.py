"""Inline-math masking for LLM translation protection (BabelDOC placeholder paradigm).

Display math normally arrives as whole ``FORMULA`` blocks (``skip_translate``)
and never reaches the LLM. But *inline* math inside narrative prose —
``$K_{t}$``, ``\\(x^2\\)``, ``$$…$$`` pasted into Markdown/EPUB/HTML sources —
would otherwise be translated as prose ("K 的 t" etc.) or corrupted by
glossary substitution. Like citations and code, inline math is therefore
masked behind deterministic tokens before drafting and unmasked afterwards —
the same placeholder-protection contract as
:class:`~ubt.core.cleaners.code_masker.CodeMasker`.

Deliberately there is NO "formula hint" instruction to the LLM (BabelDOC's
``--add-formula-placehold-hint`` is off by default upstream for the same
reason: hints degrade quality). The token simply replaces the math, so the
model translates around an opaque placeholder.

Currency guard: ``$5``, ``$100`` and ``$3.50`` are money, not math, and are
never masked. A ``$…$`` span is math only when its content is not a bare
number.
"""

import functools
import re
from collections import Counter

from ubt.core.cleaners.mask_tokens import UnmaskReport
from ubt.core.cleaners.mask_tokens import find_reordered as _find_reordered
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position
from ubt.core.cleaners.mask_tokens import token_checksum as _token_checksum

_MASK_PREFIX = "⟦MATH_MASK_"


@functools.lru_cache(maxsize=16)
def _get_math_fuzzy_pattern(mask_prefix: str) -> re.Pattern[str]:
    """Token shape surviving bracket/casing/spacing mutations and an optional checksum."""
    clean_prefix = re.sub(r"^[⟦\[【(\s]+|[⟧\]】)\s]+$", "", mask_prefix).rstrip(":_")
    prefix_pattern = re.sub(r"[:_]+", r"[:_]?", re.escape(clean_prefix))
    if "MATH" in clean_prefix.upper() and clean_prefix.upper() != "MATH_MASK":
        prefix_pattern = rf"(?:{prefix_pattern}|MATH[:_]?MASK|MATH)"
    return re.compile(
        r"[⟦\[【(]?\s*" + prefix_pattern + r"[:_]?(\d{1,6})(?:-([0-9a-z]{3}))?\s*[⟧\]】)]?",
        re.IGNORECASE,
    )


@functools.lru_cache(maxsize=16)
def _get_math_scan_pattern(mask_prefix: str) -> re.Pattern[str]:
    clean_prefix = re.sub(r"^[⟦\[【(\s]+|[⟧\]】)\s]+$", "", mask_prefix).rstrip(":_")
    prefix_pattern = re.sub(r"[:_]+", r"[:_]?", re.escape(clean_prefix))
    if "MATH" in clean_prefix.upper() and clean_prefix.upper() != "MATH_MASK":
        prefix_pattern = rf"(?:{prefix_pattern}|MATH[:_]?MASK|MATH)"
    return re.compile(
        prefix_pattern + r"[:_]?(\d{1,6})(?:-([0-9a-z]{3}))?",
        re.IGNORECASE,
    )


_DISPLAY_DOLLAR_PATTERN = re.compile(r"\$\$\s*(.+?)\s*\$\$", re.DOTALL)
# Multi-line LaTeX math environments (B-track): align/gather/equation blocks
# span newlines so none of the single-line patterns below can see them.
# Runs FIRST — a nested \begin{cases} inside $...$ is masked as one unit and
# the surviving outer delimiters are picked up by the inline pass after.
_MATH_ENV_NAMES = (
    "align",
    "aligned",
    "gather",
    "gathered",
    "multline",
    "flalign",
    "eqnarray",
    "equation",
    "split",
    "cases",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "array",
)
_MATH_ENV_PATTERN = re.compile(
    r"\\begin\{(" + "|".join(_MATH_ENV_NAMES) + r")\*?\}"
    r".*?"
    r"\\end\{\1\*?\}",
    re.DOTALL,
)
# LaTeX \(...\) and \[...\] groups (single-line; display blocks are FORMULA).
_PAREN_MATH_PATTERN = re.compile(r"\\\((.+?)\\\)")
_BRACKET_MATH_PATTERN = re.compile(r"\\\[(.+?)\\\]")
# Inline $...$: neither delimiter may touch whitespace (excludes "$5 and $10"
# chains where the inner edge hits a space).
_INLINE_DOLLAR_PATTERN = re.compile(r"\$(?!\s)([^$\n]+?)(?<!\s)\$")

# Bare numbers with optional thousands/decimal separators: money, not math.
_CURRENCY_PATTERN = re.compile(r"^\d[\d,.]*$")
# Currency chains/ranges ($10-$20, $5–$10, $10-20$)
_CURRENCY_RANGE_PATTERN = re.compile(r"^\d[\d,.]*\s*[-–—]\s*(?:\$?\d[\d,.]*)?$")

# Signs the interior is real math: a \command, sub/superscript, or brace.
_LATEX_SIGNAL_PATTERN = re.compile(r"\\[A-Za-z]+|[_^{}]")


def _is_math_content(content: str) -> bool:
    """True when a $...$ interior is math rather than currency/prose."""
    stripped = content.strip()
    if not stripped:
        return False
    if _CURRENCY_PATTERN.match(stripped) or _CURRENCY_RANGE_PATTERN.match(stripped):
        return False
    # A lone ``$`` is a currency-unit marker, not a delimiter. Two of them on
    # one line -- a table header's ``Eff. ($) ... Token ($)`` -- pair up into a
    # span whose content is a row of prose, so Gate 4 failed the block for a
    # "math span mismatch" the draft can never fix. The tell is a span glued to
    # brackets with no math inside it; a stray-dollar-shattered formula
    # (``$\psi = V \ln($``) ends with ``(`` too, but carries \commands.
    bracket_glued = stripped.startswith(")") or stripped.endswith("(")
    return not bracket_glued or bool(_LATEX_SIGNAL_PATTERN.search(stripped))


class MathMasker:
    """Masks inline math spans before translation and unmasks afterwards."""

    def __init__(self, mask_prefix: str = _MASK_PREFIX) -> None:
        self.mask_prefix = mask_prefix

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace math spans with protective tokens; returns (masked, mapping).

        Tokens carry a checksum suffix (``⟦MATH_MASK_0001-a3f⟧``) binding the
        index to the masked original; see :func:`_token_checksum`.
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

        def _replace_guarded(match: re.Match[str]) -> str:
            if not _is_math_content(match.group(1)):
                return match.group(0)
            return _replace(match)

        masked = _MATH_ENV_PATTERN.sub(_replace, text)
        masked = _DISPLAY_DOLLAR_PATTERN.sub(_replace, masked)
        masked = _PAREN_MATH_PATTERN.sub(_replace, masked)
        masked = _BRACKET_MATH_PATTERN.sub(_replace, masked)
        masked = _INLINE_DOLLAR_PATTERN.sub(_replace_guarded, masked)
        return masked, _order_by_position(masked, mapping)

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore math tokens; fuzzy-matches LLM mutations like CodeMasker.

        A checksum that is present but does not match the expected one is left
        in place rather than restoring the wrong formula (renumbered tokens).
        Checksum-less echoes stay backward compatible and restore by index.
        """
        table = self._index_table(mapping)
        if not table:
            return text

        fuzzy = _get_math_fuzzy_pattern(self.mask_prefix)

        def _fuzzy_repl(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            checksum = match.group(2)
            entry = table.get(idx)
            if entry is None:
                return match.group(0)
            if checksum and checksum.lower() != entry[2]:
                return match.group(0)
            return entry[1]

        result = text
        # Fixed-point loop: an env token nested inside an inline-math token
        # (e.g. $x = \begin{cases}...\end{cases}$) reintroduces the inner
        # token when the outer one is restored, so a single pass can leave
        # tokens behind. Integrating verbatim replace and fuzzy match ensures
        # newly revealed inner tokens from fuzzy-restored outer tokens are expanded.
        for _ in range(min(len(mapping) + 2, 6)):
            changed = False
            for token, original in mapping.items():
                if token in result:
                    replaced = result.replace(token, original)
                    if replaced != result:
                        changed = True
                        result = replaced
            subbed = fuzzy.sub(_fuzzy_repl, result)
            if subbed != result:
                changed = True
                result = subbed
            if not changed:
                break

        return result

    def _index_table(self, mapping: dict[str, str]) -> dict[int, tuple[str, str, str]]:
        """index -> (token, original, expected checksum)."""
        table: dict[int, tuple[str, str, str]] = {}
        for token, original in mapping.items():
            m = re.search(r"(\d+)", token)
            if m:
                idx = int(m.group(1))
                table[idx] = (token, original, _token_checksum(idx, original))
        return table

    def _scan_token_refs(self, text: str) -> list[tuple[int, str | None]]:
        """All token-shaped references in text: (index, checksum-or-None)."""
        pattern = _get_math_scan_pattern(self.mask_prefix)
        return [(int(idx), ck) for idx, ck in pattern.findall(text)]

    def _missing_indices(
        self, table: dict[int, tuple[str, str, str]], seen: dict[int, str | None], restored: str
    ) -> list[int]:
        """Indices whose formula did not survive into the restored text.

        Count-aware per-original check so one restored copy cannot vouch for a
        dropped identical one; see
        :meth:`~ubt.core.cleaners.citation_masker.CitationMasker._missing_indices`
        for the rationale. A no-suffix (bare) token has an
        empty-string checksum and is left to ``unverified``, not counted here.
        """
        by_original: dict[str, list[int]] = {}
        for idx, (_, original, _) in table.items():
            by_original.setdefault(original, []).append(idx)
        missing: list[int] = []
        for original, idxs in by_original.items():
            valid_emitted = [
                idx
                for idx in idxs
                if seen.get(idx) and str(seen[idx]).lower() == table[idx][2].lower()
            ]
            omitted = [idx for idx in idxs if idx not in seen]
            surplus = restored.count(original) - len(valid_emitted)
            covered = min(len(omitted), max(0, surplus))
            missing.extend(omitted[covered:])
        return sorted(set(missing))

    def _duplicated_indices(
        self, table: dict[int, tuple[str, str, str]], restored: str
    ) -> list[int]:
        """Indices whose formula leaked as literal text in the draft.

        Echo twin of :meth:`_missing_indices`: a faithful draft holds each
        original exactly as many times as there are tokens for it, so a
        restored count above that is the model emitting the formula twice.
        """
        expected = Counter(original for _, original, _ in table.values())
        leaked = {
            idx
            for idx, (_, original, _) in table.items()
            if restored.count(original) > expected[original]
        }
        return sorted(leaked)

    def unmask_checked(self, text: str, mapping: dict[str, str]) -> UnmaskReport:
        """Restore plus integrity verification.

        Buckets:
        - ``missing``: expected indices the draft never echoed (swallowed);
        - ``mismatched``: echoed index whose checksum is present but wrong
          (wrong-formula risk — the model rewrote the token);
        - ``mutated``: token-shaped residue with an unknown index;
        - ``reordered``: intact spans the draft emitted in a different order
          than the masked source (position moved, content unchanged).
        Callers fail closed (repair retry) unless :attr:`UnmaskReport.clean`;
        the draft stage checkpoints a non-clean restore as REPAIR_PENDING, so
        a corrupt or reordered draft never reaches the auto-pass path.
        """
        table = self._index_table(mapping)
        draft_refs = self._scan_token_refs(text)
        seen: dict[int, str | None] = {}
        for idx, ck in draft_refs:
            seen.setdefault(idx, ck)

        restored = self.unmask(text, mapping)

        missing = self._missing_indices(table, seen, restored)
        # Tokens echoed without a -ck suffix are *tolerated*: the masker emits
        # the checksum as optional, so a faithful model may drop it. Such a
        # token we cannot verify is "unverified", NOT "mismatched" — flagging it
        # as mismatched would trigger needless repair rounds for correct
        # Restorations. Only a checksum that is *present but
        # wrong* (index rewritten) is a real wrong-formula risk.
        # NOTE: ``_scan_token_refs`` yields an empty-string checksum (not None)
        # for the no-suffix case, so we test truthiness, not ``is None``.
        unverified = sorted(idx for idx, ck in seen.items() if idx in table and not ck)
        mismatched = sorted(
            idx
            for idx, ck in seen.items()
            if idx in table and ck and ck.lower() != table[idx][2].lower()
        )
        # Token order in the draft must match the order of the masked source
        # (``table`` preserves the mapping's textual order). A swap of two
        # intact tokens leaves every other bucket empty, so this is the only
        # check that can see a protected span move within the block.
        reordered = _find_reordered(list(table), [idx for idx, _ in draft_refs if idx in table])
        # Post-restore residue: anything still token-shaped is unaccounted for
        # (unknown index, or a valid duplicate the model emitted twice).
        residue = [f"{idx}-{ck}" if ck else str(idx) for idx, ck in self._scan_token_refs(restored)]
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


def count_math_spans(text: str) -> int:
    """Number of inline-math spans in text (0-token diagnostics)."""
    probe = MathMasker()
    _, mapping = probe.mask(text)
    return len(mapping)


def extract_math_spans(text: str) -> list[str]:
    """Inline-math span contents in order (for preservation comparison)."""
    _, mapping = MathMasker().mask(text)
    ordered = sorted(
        mapping.items(),
        key=lambda kv: int(re.search(r"(\d+)", kv[0]).group(1)),  # type: ignore[union-attr]
    )
    return [original for _, original in ordered]
