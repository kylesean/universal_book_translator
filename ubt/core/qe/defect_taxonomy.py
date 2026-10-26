"""Single source of truth for defect-flag fragments shared across stages.

FastPass rejections, repair-loop structural failures, and triage severity all
classify the same ``error_flags`` strings. Sharing one marker table — read by
both the quality gate and triage — keeps a defect that is fatal in one stage
from being auto-passable in another (e.g. a numeric fidelity failure must not
clear the QE gate and never reach triage).

**Why the substring tables are still here, and what would let them go.**
The structured ``<category>:<severity>:<detail>`` form exists so a flag can
carry its own classification instead of being guessed at. Three families
cannot use it yet, and each is load-bearing:

* **Lifecycle notes** (``Drafting error:``, ``Repair error:``,
  ``repair_structural_failure:``). They are not quality defects, but the
  structural table contains them *on purpose*: it is what keeps a block whose
  provider call failed out of the auto-pass path. They have no severity to
  declare — the defect is about the call, not the text — so converting them to
  the structured form would mean inventing a severity for something that is
  not a defect, and the resume path (``reset_transient_failures``) matches them
  by prefix.
* **Echo phrasings** (``Target identical to source`` and its two variants).
  Their reason text doubles as the repair instruction, so changing the
  encoding changes the prompt the model is sent.
* **Flags written by an older build that the upgrade declines.** A stored
  ``Repair error: …`` from a job that will never be re-run still has to
  classify correctly today.

Deleting a table is therefore gated on more than "the migration ran": it needs
every legacy flag family to have a structured equivalent that preserves its
routing verdict, *and* the stored corpus to be provably free of the families
that do not. Until then the tables are the classifier of record for anything
the structured form cannot express, and ``tests/unit/qe/test_flag_provenance``
pins both directions (every marker is emitted, every emitted flag is
classified or exempt).
"""

import re
from collections.abc import Iterable

# --- The "not translated" (echo) class --------------------------------------
# FastPass rejects an untranslated passage with one of three phrasings: the
# exact echo, the near-verbatim variant (``is_near_verbatim_echo``) for a target
# that kept nearly all source words while changing a few, and the same-script
# lexical variant (``same_script_identity_gate``) for a target whose script
# classes cannot separate the pair and whose frequent function words are still
# the source language's. They are the same defect, so every table below — and
# the QE score classifier — has to match all of them; matching only some would
# let a never-translated paragraph fall through to "other structural rejection"
# (0.70), read as a generic structural problem, and get auto-passed by any
# threshold at or above it. The markers live here so the reason strings and the
# tables cannot drift.
ECHO_MARKER = "Target identical to source"
NEAR_ECHO_MARKER = "Target keeps nearly all source words"
UNTRANSLATED_RESIDUE_MARKER = "Target reads as source-language residue"
ECHO_MARKERS: tuple[str, ...] = (ECHO_MARKER, NEAR_ECHO_MARKER, UNTRANSLATED_RESIDUE_MARKER)

#: The consistency stage's repair instruction prefix ("render X as Y").
#: Single-sourced here rather than in :mod:`ubt.core.qe.consistency_enforce`
#: because this module's marker tables must classify it, and that module
#: imports the term scanner — a back-edge would cycle.
CONSTRAINT_PREFIX = "terminology_consistency:"

# Fragments that mark structural / factual corruption: the draft cannot be
# trusted even when the score looks acceptable, and must never be auto-passed.
STRUCTURAL_DEFECT_MARKERS: tuple[str, ...] = (
    "Empty target text",
    # The export stage's HTML-delta check emits ``html_structure_mismatch: …``.
    # This table listed ``html_tag_mismatch``, which no module has ever emitted:
    # the entry matched nothing, so a block whose target lost its markup
    # structure fell through to "other structural rejection" (0.70) and could be
    # released by any threshold at or below it.
    "html_structure_mismatch",
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
    # Masked-token loss: the model dropped/mutated a protected span during
    # drafting. The restoration reports a mismatch, so the target is corrupt
    # even when FastPass scores it clean. ``PlaceholderEngine`` emits five
    # namespaces (email, soup, math, citation, code) and every one is listed
    # here so this table is genuinely the single source of truth: an unlisted
    # namespace falls through to the "other structural rejection" band and is
    # auto-passed at any threshold at or above it.
    "email_token_corrupt",
    "math_token_corrupt",
    "soup_token_corrupt",
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
    # A target with no letters at all while the source had prose: the model
    # emitted punctuation/symbols. Same class as an empty target — nothing the
    # reader can use came back — and it slipped through every gate while the
    # density check read the empty residue as ratio 1.0.
    "carries no letters at all",
    # A target containing a character from a script the target language never
    # uses ("遵循查询기반范式" — the model emitted Hangul for a Chinese target).
    # FastPass rejects it outright, but no marker matched the reason, so it fell
    # through to ``SEMANTIC_DOUBT``: repair skipped it as unverifiable and the
    # block landed in the human queue even though the same check can verify a
    # rewrite. Same family as ``script density``, which was already here.
    "Foreign-script text",
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
    # The consistency stage's repair *instruction* ("render X as Y") is the
    # same defect class as the gate's violation marker, emitted by a different
    # producer. It is listed here because ``RepairLoop._flags_are_recheckable``
    # reads this table to decide whether a no-span defect is still verifiable:
    # without the entry every terminology repair the consistency stage planned
    # was refused as "unverifiable" and the block was written FAILED without a
    # model call — the one defect class the stage exists to fix.
    CONSTRAINT_PREFIX,
    # A heading that dropped its section number or appendix letter breaks every
    # cross-reference and is invisible to every other gate. The FastPass check
    # emitted this reason with no classifier branch, so it scored as "other
    # structural rejection" (0.70) and could be released by a permissive
    # threshold, or silently dropped from the report's defect classes.
    "Heading label dropped",
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
    "html_structure_mismatch",
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
    "email_token_corrupt",
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
    # No letters at all is wholesale loss too.
    "carries no letters at all",
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


#: Numeric flags that are **advisory**: reported on the block, never blocking.
#: ``consistency.py`` emits these for an order change (legal in translation) and
#: for an elided restatement (legal too). They share the ``numeric`` stem with
#: the real defect class, so any consumer that asks "was a number problem
#: flagged?" by substring must exclude them — the span annotator did not, and
#: built MQM error spans for notes the pipeline deliberately does not act on.
NUMERIC_ADVISORY_PREFIXES: tuple[str, ...] = (
    "numeric_advisory:",
    "numeric_order_differs:",
    "numeric_repeat_shortfall:",
)


def is_numeric_defect(flags: Iterable[str]) -> bool:
    """Whether any flag reports a *blocking* numeric problem.

    Case-insensitive on the ``numeric`` stem so a flag written in caps by an
    older build still matches, but advisory classes are excluded by prefix.
    """
    return any(
        "numeric" in flag.lower() and not flag.startswith(NUMERIC_ADVISORY_PREFIXES)
        for flag in flags
    )


#: Structured categories that are structural **regardless of severity**.
#:
#: Severity alone cannot express "structural but minor": a dropped markup tag
#: is cheap to fix (``FORMAT_ONLY_MARKERS``) *and* must never be auto-passed
#: (``STRUCTURAL_DEFECT_MARKERS``) — the legacy tables say both. A structured
#: flag carries no marker, so the category has to carry that half of the
#: meaning. Without this, upgrading ``HTML delta failure`` to
#: ``html_formatting_drift:minor:…`` flipped its routing from
#: ``DETERMINISTIC_DEFECT`` to ``PASS``; ``flag_upgrade``'s safety check caught
#: it and refused the rewrite, which is why the set exists rather than a bug.
STRUCTURAL_CATEGORIES: tuple[str, ...] = (
    "empty_target",
    "numeric_fidelity",
    "omission",
    "table_dropped",
    "table_grid",
    "heading_label",
    "added_reference",
    "prompt_scaffold",
    "prompt_leak",
    "repetition",
    "math_span",
    "undelimited_math",
    "hallucinated_latex",
    "terminology_consistency",
    "html_structure_mismatch",
    "html_formatting_drift",
    "email_token_corrupt",
    "soup_token_corrupt",
    "math_token_corrupt",
    "cite_token_corrupt",
    "code_token_corrupt",
    "visual_witness",
    "length_truncated",
    "length_inflated",
    "script_density",
    "foreign_script",
)


def has_structural_defect(flags: Iterable[str]) -> bool:
    """True when any flag marks a structural/factual defect.

    Three ways a flag can say so, and all three must agree with the legacy
    tables for a migration to be safe:

    * it contains a marker from ``STRUCTURAL_DEFECT_MARKERS`` (legacy prose);
    * it declares ``major`` or ``critical`` (structured, severity carries it);
    * its category is in ``STRUCTURAL_CATEGORIES`` (structured, severity does
      not carry it — the format-only classes are structural but minor).

    Making this predicate — rather than each caller — honour the declaration is
    what keeps every consumer (the quality gate's pass check, triage,
    ``RepairLoop._flags_are_recheckable``) in step: a writer that adopts the
    format is classified correctly everywhere without touching any of them.
    """
    materialized = list(flags)
    if any(marker in flag for flag in materialized for marker in STRUCTURAL_DEFECT_MARKERS):
        return True
    if structured_severity(materialized) in ("major", "critical"):
        return True
    return any(
        parsed is not None and parsed[0] in STRUCTURAL_CATEGORIES
        for parsed in (parse_defect_flag(flag) for flag in materialized)
    )


def has_critical_defect(flags: Iterable[str]) -> bool:
    """True when any flag marks an untrustworthy (MQM Critical) draft."""
    materialized = list(flags)
    if any(marker in flag for flag in materialized for marker in CRITICAL_DEFECT_MARKERS):
        return True
    return structured_severity(materialized) == "critical"


#: Structured categories whose fix is mechanical (a re-emit), for the "all
#: flags are format-only" question. Derived from ``FORMAT_ONLY_MARKERS``' own
#: subject matter: HTML/markup drift and length over/under-shoot. Every entry
#: here must also be in ``STRUCTURAL_CATEGORIES`` — the legacy tables say a
#: format-only defect is *both* mechanical and structural, and a category that
#: is one but not the other changes a verdict on migration.
FORMAT_ONLY_CATEGORIES: tuple[str, ...] = (
    "html_formatting_drift",
    "html_structure_mismatch",
    "length_truncated",
    "length_inflated",
)


def is_format_only(flags: Iterable[str]) -> bool:
    """True when every flag is a cheap-to-fix format defect.

    The quantifier is "all", which has no structured equivalent: a structured
    flag's severity says how bad it is, not whether the fix is mechanical. So
    the structured half is expressed through the *category* instead, and only
    the categories whose fix is a re-emit belong here. A flag that is neither
    a known format marker nor a format category makes the answer False — the
    conservative direction, since the caller uses this to skip expensive
    reasoning.
    """
    materialized = [f for f in flags if f]
    if not materialized:
        return False
    return all(_is_format_flag(flag) for flag in materialized)


def _is_format_flag(flag: str) -> bool:
    if any(marker in flag for marker in FORMAT_ONLY_MARKERS):
        return True
    parsed = parse_defect_flag(flag)
    return parsed is not None and parsed[0] in FORMAT_ONLY_CATEGORIES


# --- Structured flags: a machine-readable alternative to prose matching -----
# Every table above classifies a *free-text* flag by substring, which is why a
# rename in one stage silently changes a verdict in another. The tables stay
# (thousands of ledger rows carry the old spellings), but a flag written through
# :func:`format_defect_flag` carries its own category and severity, so a new
# writer never has to be added to four marker tuples to be classified.
#
# Format: ``<category>:<severity>:<detail>`` — e.g.
# ``consistency:major:terminology_drift: render 'ALU' as '算术逻辑单元'``.
# ``detail`` may itself contain colons; only the first two fields are fixed.
DEFECT_SEVERITIES: tuple[str, ...] = ("minor", "major", "critical")
STRUCTURED_FLAG_RE = re.compile(
    r"^(?P<category>[a-z][a-z0-9_]*):(?P<severity>minor|major|critical):"
)


def format_defect_flag(category: str, severity: str, detail: str) -> str:
    """Render a structured flag; ``severity`` must be one of the three tiers."""
    if severity not in DEFECT_SEVERITIES:
        raise ValueError(
            f"unknown defect severity {severity!r}; expected one of {DEFECT_SEVERITIES}"
        )
    return f"{category}:{severity}:{detail}"


def parse_defect_flag(flag: str) -> tuple[str, str] | None:
    """``(category, severity)`` for a structured flag, ``None`` for a legacy one.

    Deliberately narrow: a flag that merely *contains* a colon (``Drafting
    error: 500``) is not structured, because its first field is not a lowercase
    category token followed by a known severity. Callers fall back to the
    substring tables for those.
    """
    match = STRUCTURED_FLAG_RE.match(flag)
    if match is None:
        return None
    return match.group("category"), match.group("severity")


def structured_severity(flags: Iterable[str]) -> str | None:
    """The highest severity any structured flag declares, if any do."""
    best: str | None = None
    for flag in flags:
        parsed = parse_defect_flag(flag)
        if parsed is None:
            continue
        severity = parsed[1]
        if best is None or DEFECT_SEVERITIES.index(severity) > DEFECT_SEVERITIES.index(best):
            best = severity
    return best


def declared_severity(flags: Iterable[str]) -> str | None:
    """The highest severity across *both* encodings, or ``None``.

    The whole point of a structured flag is that its author states the verdict
    instead of hoping a substring table guesses it. Two rules follow, and the
    incremental landing got both wrong:

    * A structured flag must never *weaken* a verdict. Returning the structured
      severity whenever one is present let a single ``advisory:minor:`` note
      downgrade a block that also carried a legacy ``Numeric fidelity`` flag —
      triage then released it through the minor auto-pass branch. The answer is
      the worse of the two encodings.
    * A structured flag must never be *invisible* to the routing rule.
      ``classify_defect`` read only the tables, so a writer that declared
      ``critical`` produced a flag that classified as ``PASS``.

    ``None`` means neither encoding expressed an opinion, which is different
    from a declared ``minor``: the latter is a deliberate "this is advisory".
    """
    declared = structured_severity(flags)
    if has_critical_defect(flags):
        return "critical"
    if has_structural_defect(flags):
        # A structural marker floors at major; keep a declared critical.
        if declared == "critical":
            return "critical"
        return "major"
    return declared


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
# A provider configuration failure (missing route/model, rejected credential)
# fails identically until the operator fixes that configuration. It still trips
# the fail-fast breaker, but unlike a malformed source it becomes translatable
# after the fix, so an explicit resume may re-queue it.
RETRYABLE_CONFIG_DRAFT_PREFIX = "Drafting config error:"
REPAIR_ERROR_PREFIX = "Repair error:"
# Export's stale-block sweep force-fails any block still non-terminal when the
# render stage starts (a stage crashed mid flight); without this prefix a resume
# treats those as permanent and the block can never be retried.
UNTRANSLATED_PREFIX = "untranslated:"

TRANSIENT_FAILURE_PREFIXES: tuple[str, ...] = (
    DRAFTING_ERROR_PREFIX,
    RETRYABLE_CONFIG_DRAFT_PREFIX,
    REPAIR_ERROR_PREFIX,
    UNTRANSLATED_PREFIX,
)

FLAG_NEEDS_HUMAN_REVIEW = "needs_human_review"
FLAG_MQM_CRITICAL_BLOCKED = "mqm_critical_blocked"

#: Triage verdicts that permanently quarantine a block (never auto-reset; the
#: PE queue must not silently lose members on resume).
TRIAGE_VERDICT_FLAGS: tuple[str, ...] = (FLAG_NEEDS_HUMAN_REVIEW, FLAG_MQM_CRITICAL_BLOCKED)

#: Render/inplace skip flags that mean "source element deliberately kept"
#: (chrome, non-prose, policy, footer), as opposed to a fail-closed skip
#: (spill/no_zone/math_unrenderable) that lost a translation. Shared by the
#: reporter's coverage split and the export ledger pass so the two cannot drift.


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


# --------------------------------------------------------------------------- #
# The one decision surface
#
# Before this, "should this block be repaired automatically?" was answered in
# two places that could not see each other: the quality gate routed every
# FastPass failure to REPAIR_PENDING, and the repair loop re-decided with
# ``error_flags or score < threshold * 0.8``. Because every block reaching the
# second check already carried a flag, the score clause was effectively dead —
# and the ``* 0.8`` constant was a hidden second pass line that no documented
# knob controlled. Both are gone; this function is the answer.
# --------------------------------------------------------------------------- #

#: Flags that only a *probabilistic* judge can produce. A block whose flags are
#: all in this set is in SEMANTIC_DOUBT: nothing deterministic failed, so a
#: repair cannot be verified against the check that triggered it.
SEMANTIC_ONLY_MARKERS: tuple[str, ...] = (
    "judge",
    "Semantic",
    "semantic",
)


def classify_defect(flags: Iterable[str], *, score: float | None, threshold: float) -> str:
    """The pipeline's only routing decision for a scored block.

    Returns one of :data:`~ubt.core.qe.heuristic.DEFECT_OUTCOMES`:

    ``PASS``
        Nothing deterministic failed and the score clears ``threshold``.
    ``DETERMINISTIC_DEFECT``
        A named invariant failed (or the score is below ``threshold`` with no
        semantic-only explanation). Automatic repair is legitimate here because
        the same check that rejected the draft will judge the repair.
    ``SEMANTIC_DOUBT``
        The only reason to doubt this block is a probabilistic judge. Automatic
        repair is *not* legitimate: the judge has its own noise, so a rewrite
        cannot be shown to have helped. Route to a human queue.

    The distinction is the whole point of the simplification. Repairing a
    dropped number is verifiable; repairing "the judge thought this read
    awkwardly" is not, and doing it anyway spends tokens to move text around
    with no evidence of improvement.
    """
    materialized = [flag for flag in flags if flag]
    if not materialized and (score is None or score >= threshold):
        return "PASS"
    # Any deterministic marker outranks a semantic one: a block that dropped a
    # number is a deterministic defect even if the judge also disliked it.
    if has_critical_defect(materialized) or has_structural_defect(materialized):
        return "DETERMINISTIC_DEFECT"
    if materialized and all(
        any(marker in flag for marker in SEMANTIC_ONLY_MARKERS) for flag in materialized
    ):
        return "SEMANTIC_DOUBT"
    if score is not None and score < threshold:
        # Below the line with no deterministic explanation: the score is the
        # only evidence, and a score is a judgement. Treat it as doubt.
        return "SEMANTIC_DOUBT"
    return "PASS"
