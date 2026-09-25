"""Single source of truth for defect-flag fragments shared across stages.

FastPass rejections, repair-loop structural failures, and triage severity all
classify the same ``error_flags`` strings. Sharing one marker table — read by
both the quality gate and triage — keeps a defect that is fatal in one stage
from being auto-passable in another (e.g. a numeric fidelity failure must not
clear the QE gate and never reach triage).
"""

from collections.abc import Iterable

# --- The "not translated" (echo) class --------------------------------------
# FastPass rejects an untranslated passage with one of two phrasings: the exact
# echo, and the near-verbatim variant (``is_near_verbatim_echo``) for a target
# that kept nearly all source words while changing a few. They are the same
# defect, so every table below — and the QE score classifier — has to match
# both; matching only one would let a never-translated paragraph fall through to
# "other structural rejection" (0.70), read as a generic structural problem, and
# get auto-passed by any threshold at or above it. The markers live here so the
# reason strings and the tables cannot drift.
ECHO_MARKER = "Target identical to source"
NEAR_ECHO_MARKER = "Target keeps nearly all source words"
ECHO_MARKERS: tuple[str, ...] = (ECHO_MARKER, NEAR_ECHO_MARKER)

# Fragments that mark structural / factual corruption: the draft cannot be
# trusted even when the score looks acceptable, and must never be auto-passed.
STRUCTURAL_DEFECT_MARKERS: tuple[str, ...] = (
    "Empty target text",
    "html_tag_mismatch",
    "HTML delta failure",
    "Prompt template XML artifacts",
    "Repetitive loop hallucination",
    "repetition",
    "loop hallucination",
    "repair_structural_failure",
    "Repair error",
    "Drafting error",
    "Math span mismatch",
    "Undelimited math",
    "Hallucinated LaTeX",
    # Masked-token loss: the model dropped/mutated a protects-math or
    # delimiter-free (soup) span during drafting. The restoration reports a
    # mismatch, so the target is corrupt even when FastPass scores it clean.
    "math_token_corrupt",
    "soup_token_corrupt",
    # Same class for the other two masker namespaces. The draft stage already
    # keeps every unclean restore out of the auto-pass path by persisting it as
    # REPAIR_PENDING, so these two markers are defence in depth: they keep the
    # verdict fatal at ANY QE threshold and "major" at triage even if such a
    # block reaches the gate by another route. All four masker namespaces are
    # listed so this table is genuinely the single source of truth.
    "code_token_corrupt",
    "cite_token_corrupt",
    "Omission suspected",
    "Numeric fidelity",
    "Visual witness discrepancy",
    # Lossy / wrong-script classes: a permissive QE threshold must not release
    # a truncated or wrong-language target.
    "Target text suspiciously truncated",
    "Target text suspiciously inflated",
    "script density",
    # Content the source never had. Fabrication is exactly the
    # case this table exists for — an echoed paragraph reads well and classifies
    # as "other structural rejection" (0.70), so any QE threshold at or below
    # 0.70 would auto-release it *and* wipe its error flags. The marker keeps the
    # verdict fatal regardless of the configured threshold.
    "Added reference",
    "Prompt scaffold",
    # A fluent target that drops or alters an ENFORCED glossary
    # term passes every structural gate. Terminology is a correctness defect,
    # not a style note, so it is fatal here like the classes above; the export
    # stage's advisory copy of the same check runs too late to change anything.
    "Glossary term violation",
    # A translated table that lost or gained a column renders as a broken grid;
    # the emitter cannot recover the intended shape downstream.
    "Table grid mismatch",
    # A source table whose structure collapsed to flat text before the model ever
    # saw it. FastPass rejects it, but the reason has no classifier branch, so it
    # scores as "other structural rejection" (0.70) — one dual-witness +0.05
    # boost away from the default 0.75 threshold, where the gate would mark it
    # MTQE_PASSED and wipe the flag. The marker keeps a dropped table fatal at
    # any threshold; content loss is not a style note.
    "Table dropped",
    # A target that repeats the source is not a translation. The density gate
    # cannot catch it for same-script pairs (and not for ja->zh, where kanji
    # clear the Chinese bar), so without these markers an untouched paragraph
    # would be released as MTQE_PASSED by the quality gate.
    *ECHO_MARKERS,
)

# Format-only defects are cheap to fix and may bypass expensive reasoning.
# These are the PRODUCTION reason substrings emitted by fast_pass.py, not
# symbolic names: only the real emitted strings make ``is_format_only`` ever
# return True and keep the cheap path live.
FORMAT_ONLY_MARKERS: tuple[str, ...] = (
    "HTML delta failure",
    "html_tag_mismatch",
    "Target text suspiciously truncated",
    "Target text suspiciously inflated",
)


# Subset of the structural markers that make the draft untrustworthy rather
# than merely awkward: protected-token loss, numeric/omission/fabrication
# defects, math drift, repetition and prompt leakage. Triage maps these to
# MQM Critical so an unrepaired block is quarantined (BLOCKED_HUMAN) instead of
# shipped as a NEEDS_HUMAN draft. Terminology is deliberately absent (Major per
# the triage contract), as are the format-only classes (HTML delta, length)
# and the provenance-only "Visual witness discrepancy" marker.
CRITICAL_DEFECT_MARKERS: tuple[str, ...] = (
    "Empty target text",
    "math_token_corrupt",
    "soup_token_corrupt",
    "code_token_corrupt",
    "cite_token_corrupt",
    "Numeric fidelity",
    "Math span mismatch",
    "Undelimited math",
    "Hallucinated LaTeX",
    "Omission suspected",
    # A dropped table is wholesale content loss, the same class as omission.
    "Table dropped",
    "Added reference",
    "Prompt scaffold",
    "Prompt template XML artifacts",
    "Repetitive loop hallucination",
    "loop hallucination",
    "repair_structural_failure",
    # An echo that survived repair is a completeness failure, not a style
    # problem: the reader gets source text where the target belongs.
    *ECHO_MARKERS,
)


def has_structural_defect(flags: Iterable[str]) -> bool:
    """True when any flag contains a structural/factual defect marker."""
    return any(marker in flag for flag in flags for marker in STRUCTURAL_DEFECT_MARKERS)


def has_critical_defect(flags: Iterable[str]) -> bool:
    """True when any flag marks an untrustworthy (MQM Critical) draft."""
    return any(marker in flag for flag in flags for marker in CRITICAL_DEFECT_MARKERS)


def is_format_only(flags: Iterable[str]) -> bool:
    """True when every flag is a cheap-to-fix format defect."""
    materialized = [f for f in flags if f]
    return bool(materialized) and all(
        any(marker in flag for marker in FORMAT_ONLY_MARKERS) for flag in materialized
    )


# --- Lifecycle failure markers (not quality defects) ------------------------
# The same ``error_flags`` strings double as the resume/queue signal: which
# failures a resume may re-queue, and which are permanent triage verdicts.
# Writers are the draft/repair/export stages; the reader is the ledger's
# ``reset_transient_failures``. Both sides share these single-sourced literals
# (rather than the ledger keeping its own copy) so a writer rename cannot
# silently stop matching the reader and strand blocks across resumes.
DRAFTING_ERROR_PREFIX = "Drafting error:"
# A drafting failure the router already classified as non-retryable (401/402/
# 400/404/422 — bad credential, no credit, malformed request, missing model).
# It is deliberately NOT in ``TRANSIENT_FAILURE_PREFIXES``: the same error will
# recur on every resume, so re-queueing the block only re-bills it. It must also
# not *start with* ``DRAFTING_ERROR_PREFIX``, or ``startswith`` would match it
# as transient anyway — hence the different wording.
NON_RETRYABLE_DRAFT_PREFIX = "Drafting unrecoverable:"
REPAIR_ERROR_PREFIX = "Repair error:"
# Export's stale-block sweep force-fails any block still non-terminal when the
# render stage starts (a stage crashed mid flight); without this prefix a resume
# treats those as permanent and the block can never be retried.
UNTRANSLATED_PREFIX = "untranslated:"

TRANSIENT_FAILURE_PREFIXES: tuple[str, ...] = (
    DRAFTING_ERROR_PREFIX,
    REPAIR_ERROR_PREFIX,
    UNTRANSLATED_PREFIX,
)

FLAG_NEEDS_HUMAN_REVIEW = "needs_human_review"
FLAG_MQM_CRITICAL_BLOCKED = "mqm_critical_blocked"

#: Triage verdicts that permanently quarantine a block (never auto-reset; the
#: PE queue must not silently lose members on resume).
TRIAGE_VERDICT_FLAGS: tuple[str, ...] = (FLAG_NEEDS_HUMAN_REVIEW, FLAG_MQM_CRITICAL_BLOCKED)


def is_transient_failure(flags: Iterable[str]) -> bool:
    """True when any flag marks a failure a resume may safely re-queue."""
    return any(flag.startswith(prefix) for flag in flags for prefix in TRANSIENT_FAILURE_PREFIXES)


def is_repair_only_transient_failure(flags: Iterable[str]) -> bool:
    """True when every transient marker is a *repair* marker.

    A repair-stage failure leaves the already-billed ``target_text`` behind
    (``stages/repair.py`` persists it deliberately). Re-queueing such a block as
    PENDING would NULL that text and make the next run pay for the same draft
    twice, so the resume path must send these back to REPAIR_PENDING instead.
    A block that also carries a drafting/untranslated marker never had a usable
    draft, so it keeps the PENDING reset.
    """
    transient = [
        flag
        for flag in flags
        if any(flag.startswith(prefix) for prefix in TRANSIENT_FAILURE_PREFIXES)
    ]
    if not transient:
        return False
    return all(flag.startswith(REPAIR_ERROR_PREFIX) for flag in transient)


def is_transient_lifecycle_only(flags: Iterable[str]) -> bool:
    """True when every flag is a lifecycle failure marker, not a quality defect.

    A drafting/repair call that failed transiently (timeout, 5xx, connect error)
    is a provider problem, not a judgement on a draft: triage must leave it
    FAILED/REPAIR_PENDING so ``reset_transient_failures`` can re-queue it. Without
    this, triage classifies the ``Drafting error`` marker as structural (it is in
    ``STRUCTURAL_DEFECT_MARKERS``), writes a permanent ``NEEDS_HUMAN`` verdict,
    and the resume path then refuses to retry the never-drafted block — turning a
    temporary outage into manual work (and, above ``export_min_completion_ratio``,
    into a job that can never complete).

    A permanent ``NON_RETRYABLE_DRAFT_PREFIX`` block is excluded on purpose: it
    will fail identically on every resume, so triage is the correct destination.
    """
    materialized = [flag for flag in flags if flag]
    if not any(
        flag.startswith(prefix) for flag in materialized for prefix in TRANSIENT_FAILURE_PREFIXES
    ):
        return False
    lifecycle_prefixes = (*TRANSIENT_FAILURE_PREFIXES, NON_RETRYABLE_DRAFT_PREFIX)
    return not any(
        not any(flag.startswith(prefix) for prefix in lifecycle_prefixes) for flag in materialized
    )


def has_triage_verdict(flags: Iterable[str]) -> bool:
    """True when any flag is a triage verdict (permanent quarantine)."""
    materialized = list(flags)
    return any(verdict in materialized for verdict in TRIAGE_VERDICT_FLAGS)
