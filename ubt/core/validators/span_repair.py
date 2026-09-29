"""Span-level error masking and targeted in-place repair (GEMBA-MQM Infilling).

Implements fine-grained error span annotation and deterministic in-place splicing
to eliminate over-editing and preserve fluent, correct text around defects.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ubt.core.qe.term_drift import detect_target_term_violations
from ubt.core.validators.consistency import canonicalize_numeric_token
from ubt.core.validators.glossary_enforcer import is_cjk_char


def _is_cjk_expansion_blocked(
    text: str, start: int, end: int, surface: str, rendering: str
) -> bool:
    """CJK compound guard — the enforcer's own rule, mirrored for repair.

    An EXPANSION rule (canonical rendering longer than, and containing, the
    surface it replaces — e.g. alias ``网络`` -> ``神经网络``) must not be
    flagged when the occurrence sits inside a CJK run: the exporter skips it
    there, so repair would otherwise manufacture the very damage the exporter
    refuses to make. Latin word boundaries and protected-span shielding are
    *not* re-checked here — they come from
    ``detect_target_term_violations`` (the enforcer's own
    ``find_term_occurrences``), so they exist in exactly one place.
    """
    return (
        len(rendering) > len(surface)
        and surface in rendering
        and (
            (start > 0 and is_cjk_char(text[start - 1]))
            or (end < len(text) and is_cjk_char(text[end]))
        )
    )


@dataclass(frozen=True, slots=True)
class MQMErrorSpan:
    """Represents a localized translation error span."""

    id: str
    error_type: str
    reason: str
    expected: str
    start_pos: int
    end_pos: int
    erroneous_text: str
    severity: str = "minor"  # MQM triage tier: "critical" | "major" | "minor"


# MQM severity ladder. Numeric distortion is Critical: a
# wrong figure silently ships a factual error. Terminology violations are
# Major (consistency damage, recoverable). Anything unrecognized is Minor.
SEVERITY_ORDER: dict[str, int] = {"minor": 0, "major": 1, "critical": 2}

SEVERITY_BY_ERROR_TYPE: dict[str, str] = {
    "numeric": "critical",
    "terminology": "major",
    "terminology_leak": "major",
}


def severity_for_error_type(error_type: str) -> str:
    """Map a MQM error type to its triage severity tier."""
    if error_type in SEVERITY_BY_ERROR_TYPE:
        return SEVERITY_BY_ERROR_TYPE[error_type]
    lowered = error_type.lower()
    if "numeric" in lowered:
        return "critical"
    if "termin" in lowered:
        return "major"
    return "minor"


def max_severity(severities: Iterable[str]) -> str:
    """Return the highest severity tier among the given values (defaults to minor)."""
    best = "minor"
    for s in severities:
        if SEVERITY_ORDER.get(s, 0) > SEVERITY_ORDER[best]:
            best = s
    return best


def span_to_dict(span: MQMErrorSpan) -> dict[str, Any]:
    """Serialize a span for ledger persistence and PE-queue interchange."""
    return {
        "id": span.id,
        "error_type": span.error_type,
        "reason": span.reason,
        "expected": span.expected,
        "start_pos": span.start_pos,
        "end_pos": span.end_pos,
        "erroneous_text": span.erroneous_text,
        "severity": span.severity,
    }


class MQMSpanAnnotator:
    """Locates and annotates fine-grained error spans in translation drafts."""

    def __init__(self) -> None:
        pass

    def annotate_draft(
        self,
        source_text: str,
        draft_text: str,
        error_flags: list[str],
        glossary_entries: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[MQMErrorSpan]]:
        """Identify localized error spans in draft_text and wrap them with XML annotations.

        Returns:
            (annotated_draft, list_of_error_spans)
        """
        if not draft_text:
            return draft_text, []

        spans: list[MQMErrorSpan] = []
        span_id_counter = 1

        # 1. Terminology mismatches & aliases: detected with boundary-aware and
        #    protected-span-aware matching matching the exporter's expectations.
        #    Ordered by glossary sequence with stable span identifiers.
        if glossary_entries:
            for violation in detect_target_term_violations(draft_text, glossary_entries):
                is_alias = violation.kind == "alias"
                reason = (
                    f"Use standard terminology '{violation.expected}' instead of '{violation.surface}'"
                    if is_alias
                    else (
                        f"Untranslated source term '{violation.surface}' leaked; "
                        f"replace with '{violation.expected}'"
                    )
                )
                error_type = "terminology" if is_alias else "terminology_leak"
                for start, end in violation.hits:
                    if is_alias and _is_cjk_expansion_blocked(
                        draft_text, start, end, violation.surface, violation.expected
                    ):
                        continue
                    # Avoid overlapping spans (full containment counts: a
                    # longer alias matched later must not nest inside an
                    # earlier span, or the splice builder emits
                    # duplicated/nested content).
                    if any(s.start_pos < end and start < s.end_pos for s in spans):
                        continue
                    spans.append(
                        MQMErrorSpan(
                            id=str(span_id_counter),
                            error_type=error_type,
                            severity=severity_for_error_type(error_type),
                            reason=reason,
                            expected=violation.expected,
                            start_pos=start,
                            end_pos=end,
                            erroneous_text=violation.surface,
                        )
                    )
                    span_id_counter += 1

        # 2. Number discrepancies (if flagged in error_flags or numeric distortion present)
        has_numeric_flag = any("numeric" in f.lower() for f in error_flags)
        if has_numeric_flag:
            num_pattern = re.compile(r"(?<![0-9a-zA-Z])\d+(?:[\.,]\d+)*(?![0-9a-zA-Z])")
            src_nums = num_pattern.findall(source_text)
            draft_nums = num_pattern.findall(draft_text)
            # Compare by canonical value, not raw string: '1,234' vs '1234' and
            # '1.500' vs '1.5' are one figure under a different thousands/decimal
            # convention. Treating them as a critical numeric mismatch here while
            # consistency.py already considers them equal lets the same pair both
            # pass QE and spawn a pointless repair round.
            src_canon = {canonicalize_numeric_token(s) for s in src_nums}
            draft_canon = {canonicalize_numeric_token(d) for d in draft_nums}
            missing_src = [s for s in src_nums if canonicalize_numeric_token(s) not in draft_canon]
            explained_missing: set[str] = set()
            for d_num in draft_nums:
                if canonicalize_numeric_token(d_num) not in src_canon and src_nums:
                    # Pick candidate from missing source numbers, or explain if ambiguous
                    if len(missing_src) == 1:
                        expected_num = missing_src[0]
                        reason_msg = f"Mismatched number '{d_num}', expected '{expected_num}'"
                        explained_missing.add(missing_src[0])
                    elif len(src_nums) == 1:
                        expected_num = src_nums[0]
                        reason_msg = f"Mismatched number '{d_num}', expected '{expected_num}'"
                        explained_missing.add(src_nums[0])
                    else:
                        expected_num = missing_src[0] if missing_src else ""
                        reason_msg = (
                            f"Extraneous or altered number '{d_num}' not found in source numbers "
                            f"({', '.join(src_nums)})"
                        )
                    d_patt = re.compile(rf"(?<![0-9a-zA-Z]){re.escape(d_num)}(?![0-9a-zA-Z])")
                    for m in d_patt.finditer(draft_text):
                        start, end = m.start(), m.end()
                        if not any(s.start_pos < end and start < s.end_pos for s in spans):
                            spans.append(
                                MQMErrorSpan(
                                    id=str(span_id_counter),
                                    error_type="numeric",
                                    severity=severity_for_error_type("numeric"),
                                    reason=reason_msg,
                                    expected=expected_num,
                                    start_pos=start,
                                    end_pos=end,
                                    erroneous_text=d_num,
                                )
                            )
                            span_id_counter += 1
                            break

            # Missing-source numbers (deleted from the translation) must also
            # Produce spans: a dropped figure is a factual error
            # and must escalate to Critical rather than be silently auto-passed.
            # Numbers already explained as the expected value of a mismatch
            # span above are skipped to avoid double-reporting the same fact.
            # There is no location inside the draft to mark, so the span is
            # anchored at the end of the text.
            for s_num in missing_src:
                if s_num in explained_missing:
                    continue
                spans.append(
                    MQMErrorSpan(
                        id=str(span_id_counter),
                        error_type="numeric",
                        severity=severity_for_error_type("numeric"),
                        reason=f"Source number '{s_num}' missing from translation",
                        expected=s_num,
                        start_pos=len(draft_text),
                        end_pos=len(draft_text),
                        erroneous_text=s_num,
                    )
                )
                span_id_counter += 1

        if not spans:
            return draft_text, []

        # Sort spans ascending by start_pos to build annotated string
        spans.sort(key=lambda s: s.start_pos)

        # Build annotated draft
        pieces: list[str] = []
        curr_idx = 0
        for span in spans:
            if span.start_pos > curr_idx:
                pieces.append(draft_text[curr_idx : span.start_pos])
            # Attributes/content come from source text and defect labels, which
            # may contain `"`/`&`/`<`; unescaped they yield malformed annotation
            # XML and feed the repair model a broken prompt (html_delta escapes
            # the same data).
            pieces.append(
                f'<error_span id="{span.id}" type="{html.escape(str(span.error_type))}" '
                f'reason="{html.escape(str(span.reason), quote=True)}" '
                f'expected="{html.escape(str(span.expected), quote=True)}">'
                f"{html.escape(span.erroneous_text)}</error_span>"
            )
            curr_idx = span.end_pos
        if curr_idx < len(draft_text):
            pieces.append(draft_text[curr_idx:])

        return "".join(pieces), spans


class SpanRepairSplicer:
    """Extracts corrections and deterministically splices them into the draft."""

    _CORRECTION_PATTERN = re.compile(
        r'<correction\s+id=["\']?(\w+)["\']?[^>]*>(.*?)</correction>',
        re.DOTALL | re.IGNORECASE,
    )
    _FINAL_TRANSLATION_PATTERN = re.compile(
        r"<final_translation>(.*?)</final_translation>",
        re.DOTALL | re.IGNORECASE,
    )

    def splice_repairs(
        self,
        original_draft: str,
        spans: list[MQMErrorSpan],
        model_output: str,
    ) -> tuple[str, bool]:
        """Splice targeted repairs into original_draft.

        Returns:
            (repaired_text, is_in_place_spliced)
        """
        final_match = self._FINAL_TRANSLATION_PATTERN.search(model_output)

        if not spans:
            # Fall back to extracting final translation tag or cleaning model output
            if final_match:
                return html.unescape(final_match.group(1).strip()), False
            return model_output.strip(), False

        # If any span was an insertion anchored at end-of-text (e.g. missing number),
        # an in-place splice at len(original_draft) would blindly append the number to the end.
        # When <final_translation> is provided, prioritize it over blind end-concatenation.
        has_end_insertion = any(
            span.start_pos == span.end_pos == len(original_draft) for span in spans
        )
        if has_end_insertion and final_match:
            cleaned = final_match.group(1).strip()
            cleaned = re.sub(r"</?error_span[^>]*>", "", cleaned)
            # The annotated draft HTML-escapes span content (see annotate_draft),
            # so a model that faithfully echoes AT&amp;T must be unescaped on
            # every <final_translation> return path or the entity ships as text.
            return html.unescape(cleaned), False

        corrections: dict[str, str] = {}
        for match in self._CORRECTION_PATTERN.finditer(model_output):
            c_id = match.group(1).strip()
            val = match.group(2).strip()
            # Clean possible nested error_span tags inside correction
            val = re.sub(r"</?error_span[^>]*>", "", val)
            val = html.unescape(val)
            corrections[c_id] = val

        # If at least one span correction was found:
        if corrections:
            # Sort spans by start_pos descending to perform safe in-place replacement by offset
            sorted_spans = sorted(spans, key=lambda s: s.start_pos, reverse=True)
            result = original_draft
            spliced_any = False
            for span in sorted_spans:
                if span.id in corrections:
                    replacement = corrections[span.id]
                    # Never blindly append a bare missing number/token to the very end of the paragraph
                    if span.start_pos == span.end_pos == len(original_draft):
                        if len(replacement) > len(original_draft) * 0.5:
                            return replacement, False
                        continue
                    result = result[: span.start_pos] + replacement + result[span.end_pos :]
                    spliced_any = True

            if spliced_any:
                return result, True
            if final_match:
                cleaned = final_match.group(1).strip()
                cleaned = re.sub(r"</?error_span[^>]*>", "", cleaned)
                return html.unescape(cleaned), False

        # If no <correction id="..."> was found, check for <final_translation>
        if final_match:
            cleaned = final_match.group(1).strip()
            # Strip any residual <error_span> tags
            cleaned = re.sub(r"</?error_span[^>]*>", "", cleaned)
            return html.unescape(cleaned), False

        # Corrections were parsed but none could be spliced (e.g. an end-of-text
        # insertion with no <final_translation> to anchor it). Returning
        # model_output here would ship the raw <correction id="N">…</correction>
        # repair protocol as the translation — and it can pass the numeric gate
        # when the correction carries the missing number. The unmodified draft
        # is real translated prose; prefer it over protocol markup.
        if corrections:
            return original_draft, False

        # Fall back to cleaned output with tags stripped
        cleaned = re.sub(r"</?error_span[^>]*>", "", model_output.strip())
        cleaned = re.sub(r"</?correction[^>]*>", "", cleaned)
        return html.unescape(cleaned), False
