"""Every flag a module emits must be classified, or declared exempt on purpose.

``defect_taxonomy``'s tables answer "what does this flag mean?" by substring
matching, and the only existing guard checks one direction: every marker in a
table must be emitted somewhere (so a dead entry cannot hide). The other
direction had no guard at all — a module could start emitting a flag that no
table recognises and nothing failed. That is not hypothetical:

* ``html_tag_mismatch`` was in the table while the exporter emitted
  ``html_structure_mismatch``; the dead entry matched nothing, so a target that
  lost its markup classified as "other structural rejection" (0.70) and any
  threshold at or below it released the block.
* Nine flags the pipeline really emits (``numeric_advisory``,
  ``render_skip:non_prose``, …) classify as ``PASS`` today. Most of that is
  *correct* — they are advisories, render skips or lifecycle notes, not quality
  defects — but it is correct by accident: nothing records the intent, so the
  next reader cannot tell a deliberate exemption from an oversight.

This module turns the accident into a contract. ``EXEMPT`` names every emitted
flag family that is deliberately not a quality defect, with the reason. The
guard fails when a module emits a flag that is neither classified nor exempt.

Adding a flag is therefore a two-line decision, and the reviewer sees it:

* a quality defect -> add its marker to ``STRUCTURAL_DEFECT_MARKERS`` (and
  ``CRITICAL_DEFECT_MARKERS`` if it is untrustworthy), or write it through
  ``format_defect_flag``;
* anything else -> add a row to ``EXEMPT`` saying what it is.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from ubt.core.qe.defect_taxonomy import (
    STRUCTURAL_DEFECT_MARKERS,
    has_critical_defect,
    has_structural_defect,
    parse_defect_flag,
)

pytestmark = pytest.mark.fast

_ROOT = pathlib.Path(__file__).resolve().parents[3] / "ubt"

#: Emitted flag families that are deliberately **not** quality defects, keyed by
#: the prefix the emitter writes, valued by why. A row here is a promise: the
#: flag never has to reach a gate, a repair loop or the human queue.
EXEMPT: dict[str, str] = {
    # --- lifecycle: about the *call*, not the text ---------------------------
    "Drafting error": "provider call failed transiently; the resume path re-queues it",
    "Drafting config error": "provider route/auth was broken; retry after configuration changes",
    "Drafting unrecoverable": "non-retryable provider failure; re-running would re-bill it",
    "Repair error": "repair call failed transiently; the resume path re-queues it",
    "untranslated:": "export's stale-block sweep; a resume must retry it",
    "repair_overreach": "the repair was rejected for rewriting out-of-span content",
    "repair_structural_failure": "the repair produced structurally invalid text",
    "repair_unverifiable": "no span and no re-checkable defect; no rewrite was attempted",
    # --- triage verdicts: the block already left the machine path ------------
    "needs_human_review": "triage verdict; permanent, never auto-reset",
    "mqm_critical_blocked": "triage verdict; permanent, never auto-reset",
    # --- render / layout: the deliverable was adjusted, the text was not ------
    "render_skip:": "the renderer kept the source element; not a text defect",
    "non_prose_degraded:": (
        "an advisory on a block that is already NEEDS_HUMAN; the renderer still "
        "typesets its draft, so nothing was skipped and nothing was lost"
    ),
    "column_count_mismatch": "table reconstruction refused; reported, never blocked",
    "row_count_mismatch": "table reconstruction refused; reported, never blocked",
    "target_not_a_pipe_table": "table reconstruction refused; reported, never blocked",
    "source_not_a_pipe_table": "table reconstruction refused; reported, never blocked",
    "html_formatting_drift:": "the HTML delta validator's advisory tier",  # --- advisory: reported, never blocked -----------------------------------
    "numeric_advisory:": "numeric order/repeat notes are advisory by design",
    "numeric_order_differs:": "advisory: order may legitimately change in translation",
    "numeric_repeat_shortfall:": "advisory: an elided restatement is legal",
    "font_encoding_damage:": "a PDF extraction warning about the source, not the target",
    "escalation error:": "triage's escalated repair call failed; the verdict follows below",
    # --- accepted / informational --------------------------------------------
    "c_text_span_translation:": "a legal \\text{} span translation, recorded as accepted",
    "math_invariant_repair:": "the formula was reset to source; recorded, not a defect",
    "math_debris_source_fallback:": "the source was rendered instead of a hallucinated draft",
    "glossary_quarantine_compound": "the enforcer declined a risky compound substitution",
}

#: Emitted flag families that ARE quality defects, listed so the guard can tell
#: a classification from an exemption. Most are matched by a marker in
#: ``STRUCTURAL_DEFECT_MARKERS``; the structured-flag producers are covered by
#: ``parse_defect_flag`` and need no row here.
DEFECT_PREFIXES: tuple[str, ...] = (
    "Empty target text",
    "Numeric fidelity",
    "Omission suspected",
    "Glossary term violation",
    "terminology_consistency:",
    "html_structure_mismatch",
    "Table dropped",
    "Table grid mismatch",
    "Heading label dropped",
    "Added reference",
    "Prompt scaffold",
    "Prompt template XML artifacts",
    "Repetitive loop hallucination",
    "Math span mismatch",
    "Undelimited math",
    "Hallucinated LaTeX",
    "email_token_corrupt",
    "soup_token_corrupt",
    "math_token_corrupt",
    "cite_token_corrupt",
    "code_token_corrupt",
    "Visual witness discrepancy",
    "Target text suspiciously truncated",
    "Target text suspiciously inflated",
    "script density",
    "Foreign-script text",
)

#: Emitted strings that are not flags at all (FastPass's own pass messages), and
#: test-only constants. Kept explicit rather than filtered by heuristics.
_NOT_FLAGS: frozenset[str] = frozenset(
    {
        "Flawless: passed all 0-token deterministic invariants",
        "Structural invariants passed",
        "rehearsal run: language heuristics skipped, structural invariants verified",
        "No added content detected",
        "No omission signals",
    }
)

#: Flag families produced **indirectly**, where no literal reaches an
#: ``append`` the scanner can see. Each row names where it comes from, so the
#: registration is reviewable rather than a hole. A new indirect producer must
#: add a row here; that is the point.
INDIRECT: dict[str, str] = {
    "terminology_consistency:": (
        "ConsistencyTask.flag builds it via format_defect_flag(CONSTRAINT_PREFIX…), "
        "not from a literal (ubt/core/qe/consistency_enforce.py)"
    ),
    "email_token_corrupt": "RestoreOutcome.reports yields the label; draft.py appends it",
    "soup_token_corrupt": "RestoreOutcome.reports yields the label; draft.py appends it",
    "math_token_corrupt": "RestoreOutcome.reports yields the label; draft.py appends it",
    "cite_token_corrupt": "RestoreOutcome.reports yields the label; draft.py appends it",
    "code_token_corrupt": "RestoreOutcome.reports yields the label; draft.py appends it",
    "Undelimited math": (
        "FastPass passes _UNDELIMITED_MATH_*_HINT through reason=; the scanner "
        "resolves f-string holes but not a bare conditional expression"
    ),
    "html_structure_mismatch": (
        'export.py binds f"html_structure_mismatch: …" to `flag` and appends the '
        "name; the package-wide constant table is shadowed by later bindings of "
        "the same local name"
    ),
}

#: Flag text assembled from a loop variable rather than a literal. The scanner
#: sees the *template* (``f"{label} missing=…"``) and cannot know the labels;
#: they are ``RestoreOutcome.reports``, which yields the five
#: ``*_token_corrupt`` names listed in ``INDIRECT``.
_TEMPLATE_ONLY: frozenset[str] = frozenset(
    {" missing= mismatched= mutated= reordered= duplicated="}
)


def _flag_literals() -> dict[str, set[str]]:
    """``{module: {literal flags it appends}}`` from the real AST.

    Reads three shapes, each confirmed to reach ``IRBlock.error_flags``:

    * ``<name containing 'flag'>.append(<literal>)`` — the direct writers;
    * ``FastPassDecision(reason=<literal>)`` — the quality gate copies
      ``decision.reason`` into ``error_flags`` (``quality_gate.py``:
      ``b.error_flags = [decision.reason, *prior]``);
    * ``AddedContentDecision(reason=…)`` / ``OmissionDecision(reason=…)`` —
      forwarded into a ``FastPassDecision`` by ``fast_pass.evaluate``.

    Deliberately *not* every ``reason=`` keyword: ``TableReconstruction``,
    ``RouteDecision``, ``JobQueue.fail`` and ``MQMErrorSpan`` all carry a
    ``reason`` that never becomes a block flag, and treating those as flags
    would make this guard demand exemptions for strings that do not exist.

    A literal built by an f-string is reduced to its literal prefix
    (``f"render_skip:{reason}"`` -> ``render_skip:``), which is the family a
    table matches on anyway.
    """
    flag_ctors = {"FastPassDecision", "AddedContentDecision", "OmissionDecision"}
    found: dict[str, set[str]] = {}
    # Constants are collected across the *whole* tree, not per module: the
    # consistency stage builds its flag from ``GLOSSARY_VIOLATION_MARKER``,
    # which it imports from the QE package, so a per-module table would see
    # only the ``": "`` separator.
    global_constants = _global_constants()
    modules: list[tuple[str, ast.Module, set[str]]] = []
    for path in sorted(_ROOT.rglob("*.py")):
        rel = str(path.relative_to(_ROOT.parent))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules.append((rel, tree, _shadowed_bindings(tree)))
    for rel, tree, shadowed in modules:
        # A name bound to a call *in this module* is this module's truth: the
        # package table must not hand its scan another module's same-named
        # literal (``detail`` is bound both to a template in draft.py and to
        # ``format_defect_flag(...)`` in agent_translation.py).
        constants = {n: v for n, v in global_constants.items() if n not in shadowed}
        for node in ast.walk(tree):
            for value in _appended_literals(node, flag_ctors, constants):
                if value and value not in _NOT_FLAGS:
                    found.setdefault(rel, set()).add(value)
    return found


def _global_constants() -> dict[str, str]:
    """Every module-level ``NAME = "literal"`` in the package, by name.

    Names are assumed unique across the package. That is true for the flag
    vocabulary (``ECHO_MARKER``, ``GLOSSARY_VIOLATION_MARKER``, …) and a
    collision would only make this guard *more* permissive, never less: the
    scanner would resolve a hole to the wrong string and report it unknown,
    which fails loudly.
    """
    out: dict[str, str] = {}
    for path in sorted(_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        out.update(_module_constants(tree))
    return out


def _appended_literals(node: ast.AST, flag_ctors: set[str], constants: dict[str, str]) -> list[str]:
    """String literals this node contributes as a flag, if any."""
    out: list[str] = []
    if not isinstance(node, ast.Call):
        return out
    ctor = node.func.id if isinstance(node.func, ast.Name) else ""
    if ctor in flag_ctors:
        for kw in node.keywords:
            if kw.arg == "reason":
                out.extend(_resolve(kw.value, constants))
    if isinstance(node.func, ast.Attribute) and node.func.attr in ("append", "extend"):
        receiver = _receiver_name(node.func.value)
        if ("flag" in receiver or receiver == "error_flags") and node.args:
            out.extend(_resolve(node.args[0], constants))
    return out


def _resolve(node: ast.AST, constants: dict[str, str]) -> list[str]:
    """The literal(s) an appended expression stands for.

    Handles a literal, an f-string, and a **local name bound to a literal in
    the same module**. That last shape is not incidental: the export stage
    writes ``flag = f"html_structure_mismatch: …"`` and appends ``flag``, so
    matching only inline literals made every such producer invisible — the
    guard would have reported its markers dead and, worse, would never have
    flagged a genuinely unclassified one.
    """
    literal = _literal_prefix(node, constants)
    if literal:
        return literal
    if isinstance(node, ast.Name) and node.id in constants:
        return [constants[node.id]]
    return []


def _receiver_name(node: ast.AST) -> str:
    """The trailing name of an attribute chain: ``b.error_flags`` -> ``error_flags``.

    Both spellings are used in the tree (``error_flags.append(...)`` on a local
    list and ``b.error_flags.append(...)`` on the block), and matching only the
    bare name silently skipped every block-level writer.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """``NAME = "literal"`` bindings anywhere in the module, for flag prefixes.

    Not restricted to module level: the export stage binds its flag inside a
    function (``flag = f"html_structure_mismatch: …"``) and appends the name a
    few lines later. Names are collected package-wide, so a same-named local in
    another module could shadow one — a collision can only make this guard
    report a flag *unknown*, never silently accept one.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    for literal in _literal_prefix(node.value, {}):
                        out[target.id] = literal
    return out


def _shadowed_bindings(tree: ast.Module) -> set[str]:
    """Names this module binds to a call, not to a literal.

    ``detail = format_defect_flag(...)`` declares its category and severity at
    the call site, so the name does not stand for a free-text literal. Keeping
    it out of the scan is what makes a structured writer invisible to the
    legacy-writer ratchet — and keeps it from *inheriting* a same-named
    literal that some other module binds.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.add(target.id)
    return out


def _literal_prefix(node: ast.AST, constants: dict[str, str]) -> list[str]:
    """The literal text an expression contributes, with f-string holes joined.

    Joining every constant part rather than stopping at the first one matters:
    ``f"Insufficient {code} script density (ratio=…)"`` only matches the
    ``script density`` marker once the tail is kept.

    A hole that is a module-level constant is resolved too. The consistency
    stage writes ``f"{GLOSSARY_VIOLATION_MARKER}: {check.message}"``, so without
    this the scanner would see only ``": "`` and the guard would demand an
    exemption for a flag that is already classified.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                inner = value.value
                if isinstance(inner, ast.Name) and inner.id in constants:
                    parts.append(constants[inner.id])
        joined = "".join(parts)
        return [joined] if joined.strip() else []
    return []


def _classify(value: str) -> str:
    """``"defect"``, ``"exempt"`` or ``"unknown"`` for one emitted literal."""
    if any(value.startswith(prefix) or prefix in value for prefix in DEFECT_PREFIXES):
        return "defect"
    if has_structural_defect([value]) or has_critical_defect([value]):
        return "defect"
    if parse_defect_flag(value) is not None:
        return "defect"
    if any(value.startswith(prefix) or prefix in value for prefix in EXEMPT):
        return "exempt"
    return "unknown"


def test_every_emitted_flag_is_classified_or_exempt() -> None:
    """The reverse guard: no module may emit a flag nobody has accounted for."""
    unknown: dict[str, list[str]] = {}
    for module, values in _flag_literals().items():
        for value in sorted(values):
            if value in _TEMPLATE_ONLY:
                continue
            if _classify(value) == "unknown":
                unknown.setdefault(module, []).append(value)

    assert unknown == {}, (
        "these modules emit a flag that is neither a classified defect nor a "
        "declared exemption. Decide which it is:\n"
        "  * a quality defect -> add a marker to STRUCTURAL_DEFECT_MARKERS "
        "(and CRITICAL_DEFECT_MARKERS if untrustworthy), or write it through "
        "format_defect_flag();\n"
        "  * not a defect -> add a row to EXEMPT in this file saying what it is.\n"
        + "\n".join(f"  {m}: {v}" for m, v in sorted(unknown.items()))
    )


def test_every_exemption_is_actually_emitted() -> None:
    """A stale exemption hides the next real gap, exactly like a dead marker.

    Checked against the **source text**, not the AST: a flag reaches
    ``error_flags`` through too many shapes to enumerate statically (a tuple
    yielded from a property, a constant selected by a conditional, a name bound
    in a loop), and demanding a literal ``append`` would make this guard
    demand that the code be written for the guard's convenience. The AST scan
    above answers "is this accounted for?"; this answers "does this string
    still exist anywhere in the package?".
    """
    sources = _source_text()
    dead = [
        prefix for prefix in EXEMPT if prefix not in INDIRECT and not _mentioned(prefix, sources)
    ]
    assert dead == [], f"exemptions no module mentions any more: {dead}"


def _mentioned(prefix: str, sources: dict[pathlib.Path, str]) -> bool:
    """Whether the package mentions this flag family, by text or by constant.

    A lifecycle prefix lives in ``defect_taxonomy`` as a shared constant
    (``NON_RETRYABLE_DRAFT_PREFIX``) and is imported by name, so its *literal*
    text appears nowhere else. Search for the constant too rather than
    declaring an exemption for a string the package plainly does emit.
    """
    candidates = [prefix]
    from ubt.core.qe import defect_taxonomy

    for name, value in vars(defect_taxonomy).items():
        if isinstance(value, str) and value == prefix:
            candidates.append(name)
    return any(candidate in text for text in sources.values() for candidate in candidates)


def test_every_defect_prefix_is_actually_emitted() -> None:
    """The same one-directional check the marker tables already have."""
    sources = _source_text()
    dead = [
        prefix
        for prefix in DEFECT_PREFIXES
        if prefix not in INDIRECT and not _mentioned(prefix, sources)
    ]
    assert dead == [], f"defect prefixes no module mentions any more: {dead}"


def _source_text() -> dict[pathlib.Path, str]:
    """Every module's text, **including** the taxonomy.

    Unlike the marker-table guard, this scan must see ``defect_taxonomy.py``:
    it is where the shared lifecycle constants are *defined*, and their
    literal text appears nowhere else (every consumer imports the name).
    """
    return {path: path.read_text(encoding="utf-8") for path in _ROOT.rglob("*.py")}


def test_a_new_unclassified_flag_fails_the_guard() -> None:
    """Prove the guard bites, rather than trusting that it would.

    Written as a direct call on the classifier: a module that appends a brand
    new string must be reported ``unknown``. Without this, a guard whose
    scanner silently stopped matching would pass forever.
    """
    assert _classify("some_brand_new_defect: 42") == "unknown"
    assert _classify("Numeric fidelity failure: dropped 42") == "defect"
    assert _classify("render_skip:non_prose") == "exempt"
    assert _classify("terminology_consistency:major:x") == "defect"


def test_the_guard_sees_the_literals_it_claims_to() -> None:
    """Sanity on the scanner itself: it must find the known producers."""
    literals = _flag_literals()
    assert "ubt/core/qe/fast_pass.py" in literals
    assert "ubt/core/engine/stages/export.py" in literals
    fast_pass = literals["ubt/core/qe/fast_pass.py"]
    assert "Empty target text" in fast_pass


# --------------------------------------------------------------------------- #
# The predicate layers must agree with the shared vocabulary
# --------------------------------------------------------------------------- #


def test_the_workbench_classifies_every_structural_marker() -> None:
    """``api/review`` is a view over this vocabulary, not a parallel copy."""
    from ubt.api.review import _ISSUE_MARKERS

    covered = {m for markers in _ISSUE_MARKERS.values() for m in markers}
    missing = [
        marker
        for marker in STRUCTURAL_DEFECT_MARKERS
        if marker not in covered and not any(marker in c for c in covered)
    ]
    # The two lifecycle prefixes are about a failed call, not a defect in the
    # text; the workbench reports those through the block status instead.
    from ubt.core.qe.defect_taxonomy import DRAFTING_ERROR_PREFIX, REPAIR_ERROR_PREFIX

    lifecycle = {DRAFTING_ERROR_PREFIX.rstrip(":"), REPAIR_ERROR_PREFIX.rstrip(":")}
    assert [m for m in missing if m not in lifecycle] == []


def test_the_fatal_leak_table_is_a_subset_of_the_critical_table() -> None:
    """``layout_policy`` decides whether to strip machine output.

    Its own two-entry table is a *policy* view, but every entry must also be a
    critical defect — a leak that strips the target but that triage does not
    treat as critical would ship a quarantined-looking block as a draft.
    """
    from ubt.core.policy.layout_policy import FATAL_LEAK_MARKERS

    not_critical = [m for m in FATAL_LEAK_MARKERS if not has_critical_defect([m])]
    assert not_critical == [], (
        f"fatal-leak markers the taxonomy does not call critical: {not_critical}"
    )


def test_the_span_annotator_ignores_advisory_numeric_flags() -> None:
    """A ``numeric_advisory`` note must not become an MQM error span.

    The annotator used a local ``"numeric" in flag`` test, which also matched
    the two advisory classes the pipeline emits precisely so they are *not*
    blocking. The result was review-queue spans no gate had asked for.
    """
    from ubt.core.validators.span_repair import MQMSpanAnnotator

    annotator = MQMSpanAnnotator()
    src = "The table lists 42 rows in 3 groups."
    draft = "该表列出了 41 行，共 3 组。"

    _, advisory_spans = annotator.annotate_draft(src, draft, ["numeric_advisory: order differs"])
    assert advisory_spans == [], "an advisory flag produced an error span"

    _, defect_spans = annotator.annotate_draft(src, draft, ["Numeric fidelity failure: dropped 42"])
    assert defect_spans, "a real numeric defect must still produce a span"


def test_the_numeric_predicate_excludes_every_advisory_class() -> None:
    """One definition of "was a number problem flagged?", used everywhere.

    The advisory classes share the ``numeric`` stem with the real defect class.
    A consumer that asks by substring alone acts on notes the pipeline emits
    precisely because they are *not* actionable.
    """
    from ubt.core.qe.defect_taxonomy import NUMERIC_ADVISORY_PREFIXES, is_numeric_defect

    for prefix in NUMERIC_ADVISORY_PREFIXES:
        assert not is_numeric_defect([f"{prefix} 3 -> 5"]), prefix
    assert is_numeric_defect(["Numeric fidelity failure: dropped 42"])
    assert is_numeric_defect(["NUMERIC"])  # older builds wrote it in caps
    assert not is_numeric_defect(["Glossary term violation: ALU"])


# --------------------------------------------------------------------------- #
# Phase 1 ratchet: the legacy-writer set may only shrink
# --------------------------------------------------------------------------- #

#: Modules that still write free-text flags. This list is a **ratchet**: a new
#: writer must use ``format_defect_flag`` (so its severity is declared rather
#: than guessed), and the only permitted edit to this set is removing an entry.
#: Freezing it now is what makes the migration monotone — without it, "new
#: writers use the structured form" would be a convention nobody enforces.
LEGACY_WRITERS: frozenset[str] = frozenset(
    {
        "ubt/core/qe/fast_pass.py",
        "ubt/core/qe/added_content.py",
        "ubt/core/qe/omission.py",
        "ubt/core/engine/stages/export.py",
        "ubt/core/engine/stages/quality_gate.py",
        "ubt/core/engine/stages/consistency.py",
        "ubt/core/engine/stages/triage.py",
        "ubt/core/engine/repair_loop.py",
        "ubt/core/engine/visual_verify_loop.py",
        "ubt/core/validators/math_guard.py",
        "ubt/adapters/pdf/extraction_witness.py",
        "ubt/core/engine/stages/draft.py",
    }
)


def test_no_new_module_starts_writing_free_text_flags() -> None:
    """The ratchet. A new producer must declare its severity."""
    writers = set(_flag_literals())
    new = sorted(writers - LEGACY_WRITERS)
    assert new == [], (
        "these modules started emitting free-text flags. Write them through "
        "format_defect_flag(<category>, <severity>, <detail>) so the severity "
        "is declared instead of guessed by a substring table:\n" + "\n".join(f"  {m}" for m in new)
    )


def test_every_structured_flag_declares_a_known_severity() -> None:
    """The writer-side contract: a typo fails at the call, not in triage."""

    from ubt.core.qe.defect_taxonomy import DEFECT_SEVERITIES

    pattern = re.compile(r'format_defect_flag\(\s*[^,]+,\s*"([a-z]+)"')
    bad: list[str] = []
    for path in _ROOT.rglob("*.py"):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            if match.group(1) not in DEFECT_SEVERITIES:
                bad.append(f"{path}: {match.group(1)!r}")
    assert bad == [], f"unknown severity literals: {bad}"


# --------------------------------------------------------------------------- #
# Phase 2: consumers read both encodings
# --------------------------------------------------------------------------- #


def test_is_format_only_accepts_a_structured_format_flag() -> None:
    """The "all flags are mechanical" quantifier has no structured severity
    equivalent, so the structured half is expressed by category."""
    from ubt.core.qe.defect_taxonomy import format_defect_flag, is_format_only

    assert is_format_only(["HTML delta failure: <b> -> <strong>"])
    assert is_format_only([format_defect_flag("html_formatting_drift", "minor", "<b> -> <strong>")])
    # Mixed: one legacy format flag and one structured format flag.
    assert is_format_only(
        [
            "HTML delta failure: x",
            format_defect_flag("html_formatting_drift", "minor", "y"),
        ]
    )


def test_a_structured_non_format_flag_breaks_the_format_only_quantifier() -> None:
    from ubt.core.qe.defect_taxonomy import format_defect_flag, is_format_only

    assert not is_format_only(
        [
            format_defect_flag("html_formatting_drift", "minor", "x"),
            format_defect_flag("terminology_consistency", "major", "y"),
        ]
    )


def test_repair_loop_reads_the_declared_severity_for_recheckability() -> None:
    """A structured structural flag needs no span to be worth a repair.

    ``_flags_are_recheckable`` reads ``STRUCTURAL_DEFECT_MARKERS`` by
    substring. A structured flag has no marker in that table, so without the
    declaration being consulted the repair is refused as unverifiable — the
    same break the consistency stage hit.
    """
    from ubt.core.engine.repair_loop import RepairLoop
    from ubt.core.qe.defect_taxonomy import format_defect_flag

    declared = format_defect_flag("terminology_consistency", "major", "render 'ALU' as 'X'")
    assert RepairLoop._flags_are_recheckable([declared])
    assert not RepairLoop._flags_are_recheckable(["repair failed: provider 500"])


def test_classify_defect_reads_a_structured_flag_end_to_end() -> None:
    """The routing rule must see the declaration, in both directions.

    ``html_formatting_drift`` is *structural but minor*: cheap to fix, and
    still never auto-passed. Severity alone cannot say that, which is why the
    category participates (``STRUCTURAL_CATEGORIES``). A genuinely advisory
    category stays passable on a good score.
    """
    from ubt.core.qe.defect_taxonomy import classify_defect, format_defect_flag

    critical = format_defect_flag("terminology_consistency", "critical", "x")
    assert classify_defect([critical], score=0.99, threshold=0.75) == "DETERMINISTIC_DEFECT"

    structural_minor = format_defect_flag("html_formatting_drift", "minor", "x")
    assert classify_defect([structural_minor], score=0.99, threshold=0.75) == (
        "DETERMINISTIC_DEFECT"
    )

    advisory = format_defect_flag("word_order", "minor", "x")
    assert classify_defect([advisory], score=0.99, threshold=0.75) == "PASS"


def test_the_intentional_preserve_whitelist_is_gone() -> None:
    """Eight prefixes were declared "intentional keeps" and none had a producer.

    ``render_skip:{policy,non_prose,chrome,footer}`` and their four
    ``inplace_skip:`` twins made ``preserved_blocks`` a permanently-zero KPI in
    real runs, and the one flag that looked like it belonged — triage's
    ``render_skip:non_prose`` — was actively wrong: the renderer typesets the
    rejected *draft* for a NEEDS_HUMAN block, so the contract recorded a
    translation the reader received as "text missing". A block ships its source
    for exactly one reason (``skip_translate``), which produces
    ``TextDisposition.VERBATIM`` on its own.
    """
    import ubt.core.qe.defect_taxonomy as taxonomy

    assert not hasattr(taxonomy, "INTENTIONAL_PRESERVED_SKIP_PREFIXES")


def test_no_emitter_uses_the_inplace_skip_spelling() -> None:
    """``inplace_skip:`` was read-compatibility for a spelling nothing wrote.

    Matched against *string literals and identifiers*, not raw file text: the
    removal notes in the docstrings name the spelling they retired, and a plain
    substring scan flags its own explanation.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "ubt"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            literal = ""
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = node.value
            elif isinstance(node, ast.Name):
                literal = node.id
            if "inplace_skip" in literal:
                line = getattr(node, "lineno", 0)
                offenders.append(f"{path.relative_to(root)}:{line}")
    assert offenders == []


def test_a_needs_human_non_prose_block_is_not_marked_as_a_render_skip() -> None:
    """Triage must not write a skip flag for a block it did not skip."""
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[3]
        / "ubt"
        / "core"
        / "engine"
        / "stages"
        / "triage.py"
    ).read_text(encoding="utf-8")
    # The comment explains the retired flag; the *string literal* must be gone.
    assert '"render_skip:non_prose"' not in source
    assert '"non_prose_degraded: structural block kept in the draft"' in source
