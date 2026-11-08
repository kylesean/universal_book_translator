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

EXEMPT: dict[str, str] = {
    "Drafting error": "provider call failed transiently; the resume path re-queues it",
    "Drafting config error": "provider route/auth was broken; retry after configuration changes",
    "Drafting unrecoverable": "non-retryable provider failure; re-running would re-bill it",
    "Repair error": "repair call failed transiently; the resume path re-queues it",
    "untranslated:": "export's stale-block sweep; a resume must retry it",
    "repair_overreach": "the repair was rejected for rewriting out-of-span content",
    "repair_structural_failure": "the repair produced structurally invalid text",
    "repair_unverifiable": "no span and no re-checkable defect; no rewrite was attempted",
    "needs_human_review": "triage verdict; permanent, never auto-reset",
    "mqm_critical_blocked": "triage verdict; permanent, never auto-reset",
    "render_skip:": "the renderer kept the source element; not a text defect",
    "non_prose_degraded:": (
        "an advisory on a block that is already NEEDS_HUMAN; the renderer still "
        "typesets its draft, so nothing was skipped and nothing was lost"
    ),
    "column_count_mismatch": "table reconstruction refused; reported, never blocked",
    "row_count_mismatch": "table reconstruction refused; reported, never blocked",
    "target_not_a_pipe_table": "table reconstruction refused; reported, never blocked",
    "source_not_a_pipe_table": "table reconstruction refused; reported, never blocked",
    "html_formatting_drift:": "the HTML delta validator's advisory tier",
    "numeric_advisory:": "numeric order/repeat notes are advisory by design",
    "numeric_order_differs:": "advisory: order may legitimately change in translation",
    "numeric_repeat_shortfall:": "advisory: an elided restatement is legal",
    "font_encoding_damage:": "a PDF extraction warning about the source, not the target",
    "escalation error:": "triage's escalated repair call failed; the verdict follows below",
    "c_text_span_translation:": "a legal \\text{} span translation, recorded as accepted",
    "math_invariant_repair:": "the formula was reset to source; recorded, not a defect",
    "math_debris_source_fallback:": "the source was rendered instead of a hallucinated draft",
    "glossary_quarantine_compound": "the enforcer declined a risky compound substitution",
}

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

_NOT_FLAGS: frozenset[str] = frozenset(
    {
        "Flawless: passed all 0-token deterministic invariants",
        "Structural invariants passed",
        "rehearsal run: language heuristics skipped, structural invariants verified",
        "No added content detected",
        "No omission signals",
    }
)

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

_TEMPLATE_ONLY: frozenset[str] = frozenset(
    {" missing= mismatched= mutated= reordered= duplicated="}
)


def _flag_literals() -> dict[str, set[str]]:
    flag_ctors = {"FastPassDecision", "AddedContentDecision", "OmissionDecision"}
    found: dict[str, set[str]] = {}
    global_constants = _global_constants()
    modules: list[tuple[str, ast.Module, set[str]]] = []
    for path in sorted(_ROOT.rglob("*.py")):
        rel = str(path.relative_to(_ROOT.parent))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules.append((rel, tree, _shadowed_bindings(tree)))
    for rel, tree, shadowed in modules:
        constants = {n: v for n, v in global_constants.items() if n not in shadowed}
        for node in ast.walk(tree):
            for value in _appended_literals(node, flag_ctors, constants):
                if value and value not in _NOT_FLAGS:
                    found.setdefault(rel, set()).add(value)
    return found


def _global_constants() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        out.update(_module_constants(tree))
    return out


def _appended_literals(node: ast.AST, flag_ctors: set[str], constants: dict[str, str]) -> list[str]:
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
    literal = _literal_prefix(node, constants)
    if literal:
        return literal
    if isinstance(node, ast.Name) and node.id in constants:
        return [constants[node.id]]
    return []


def _receiver_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _module_constants(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    for literal in _literal_prefix(node.value, {}):
                        out[target.id] = literal
    return out


def _shadowed_bindings(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.add(target.id)
    return out


def _literal_prefix(node: ast.AST, constants: dict[str, str]) -> list[str]:
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
    sources = _source_text()
    dead = [
        prefix for prefix in EXEMPT if prefix not in INDIRECT and not _mentioned(prefix, sources)
    ]
    assert dead == [], f"exemptions no module mentions any more: {dead}"


def _mentioned(prefix: str, sources: dict[pathlib.Path, str]) -> bool:
    candidates = [prefix]
    from ubt.core.qe import defect_taxonomy

    for name, value in vars(defect_taxonomy).items():
        if isinstance(value, str) and value == prefix:
            candidates.append(name)
    return any(candidate in text for text in sources.values() for candidate in candidates)


def test_every_defect_prefix_is_actually_emitted() -> None:
    sources = _source_text()
    dead = [
        prefix
        for prefix in DEFECT_PREFIXES
        if prefix not in INDIRECT and not _mentioned(prefix, sources)
    ]
    assert dead == [], f"defect prefixes no module mentions any more: {dead}"


def _source_text() -> dict[pathlib.Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in _ROOT.rglob("*.py")}


def test_a_new_unclassified_flag_fails_the_guard() -> None:
    assert _classify("some_brand_new_defect: 42") == "unknown"
    assert _classify("Numeric fidelity failure: dropped 42") == "defect"
    assert _classify("render_skip:non_prose") == "exempt"
    assert _classify("terminology_consistency:major:x") == "defect"


def test_the_guard_sees_the_literals_it_claims_to() -> None:
    literals = _flag_literals()
    assert "ubt/core/qe/fast_pass.py" in literals
    assert "ubt/core/engine/stages/export.py" in literals
    fast_pass = literals["ubt/core/qe/fast_pass.py"]
    assert "Empty target text" in fast_pass


def test_the_workbench_classifies_every_structural_marker() -> None:
    from ubt.api.review import _ISSUE_MARKERS

    covered = {m for markers in _ISSUE_MARKERS.values() for m in markers}
    missing = [
        marker
        for marker in STRUCTURAL_DEFECT_MARKERS
        if marker not in covered and not any(marker in c for c in covered)
    ]
    from ubt.core.qe.defect_taxonomy import DRAFTING_ERROR_PREFIX, REPAIR_ERROR_PREFIX

    lifecycle = {DRAFTING_ERROR_PREFIX.rstrip(":"), REPAIR_ERROR_PREFIX.rstrip(":")}
    assert [m for m in missing if m not in lifecycle] == []


def test_the_fatal_leak_table_is_a_subset_of_the_critical_table() -> None:
    from ubt.core.policy.layout_policy import FATAL_LEAK_MARKERS

    not_critical = [m for m in FATAL_LEAK_MARKERS if not has_critical_defect([m])]
    assert not_critical == [], (
        f"fatal-leak markers the taxonomy does not call critical: {not_critical}"
    )


def test_the_span_annotator_ignores_advisory_numeric_flags() -> None:
    from ubt.core.validators.span_repair import MQMSpanAnnotator

    annotator = MQMSpanAnnotator()
    src = "The table lists 42 rows in 3 groups."
    draft = "该表列出了 41 行，共 3 组。"

    _, advisory_spans = annotator.annotate_draft(src, draft, ["numeric_advisory: order differs"])
    assert advisory_spans == [], "an advisory flag produced an error span"

    _, defect_spans = annotator.annotate_draft(src, draft, ["Numeric fidelity failure: dropped 42"])
    assert defect_spans, "a real numeric defect must still produce a span"


def test_the_numeric_predicate_excludes_every_advisory_class() -> None:
    from ubt.core.qe.defect_taxonomy import NUMERIC_ADVISORY_PREFIXES, is_numeric_defect

    for prefix in NUMERIC_ADVISORY_PREFIXES:
        assert not is_numeric_defect([f"{prefix} 3 -> 5"]), prefix
    assert is_numeric_defect(["Numeric fidelity failure: dropped 42"])
    assert is_numeric_defect(["NUMERIC"])
    assert not is_numeric_defect(["Glossary term violation: ALU"])


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
    writers = set(_flag_literals())
    new = sorted(writers - LEGACY_WRITERS)
    assert new == [], (
        "these modules started emitting free-text flags. Write them through "
        "format_defect_flag(<category>, <severity>, <detail>) so the severity "
        "is declared instead of guessed by a substring table:\n" + "\n".join(f"  {m}" for m in new)
    )


def test_every_structured_flag_declares_a_known_severity() -> None:

    from ubt.core.qe.defect_taxonomy import DEFECT_SEVERITIES

    pattern = re.compile(r'format_defect_flag\(\s*[^,]+,\s*"([a-z]+)"')
    bad: list[str] = []
    for path in _ROOT.rglob("*.py"):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            if match.group(1) not in DEFECT_SEVERITIES:
                bad.append(f"{path}: {match.group(1)!r}")
    assert bad == [], f"unknown severity literals: {bad}"


def test_is_format_only_accepts_a_structured_format_flag() -> None:
    from ubt.core.qe.defect_taxonomy import format_defect_flag, is_format_only

    assert is_format_only(["HTML delta failure: <b> -> <strong>"])
    assert is_format_only([format_defect_flag("html_formatting_drift", "minor", "<b> -> <strong>")])
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
    from ubt.core.engine.repair_loop import RepairLoop
    from ubt.core.qe.defect_taxonomy import format_defect_flag

    declared = format_defect_flag("terminology_consistency", "major", "render 'ALU' as 'X'")
    assert RepairLoop._flags_are_recheckable([declared])
    assert not RepairLoop._flags_are_recheckable(["repair failed: provider 500"])


def test_classify_defect_reads_a_structured_flag_end_to_end() -> None:
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
    import ubt.core.qe.defect_taxonomy as taxonomy

    assert not hasattr(taxonomy, "INTENTIONAL_PRESERVED_SKIP_PREFIXES")


def test_no_emitter_uses_the_inplace_skip_spelling() -> None:
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
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[3]
        / "ubt"
        / "core"
        / "engine"
        / "stages"
        / "triage.py"
    ).read_text(encoding="utf-8")
    assert '"render_skip:non_prose"' not in source
    assert '"non_prose_degraded: structural block kept in the draft"' in source


def test_classify_error_flag_names_the_routing_verdict() -> None:
    """Every lifecycle family maps to the verdict the resume path routes on.

    The prefix tables stay the classifier of record (see the taxonomy module
    docstring); ``classify_error_flag`` names the verdict each family encodes so
    a reader routes on the kind instead of repeating ``startswith`` archaeology.
    """
    from ubt.core.qe.defect_taxonomy import ErrorFlagKind, classify_error_flag

    assert (
        classify_error_flag("Drafting error: provider timeout") is ErrorFlagKind.DRAFTING_TRANSIENT
    )
    assert (
        classify_error_flag("Drafting unrecoverable: malformed source")
        is ErrorFlagKind.NON_RETRYABLE
    )
    assert (
        classify_error_flag("Drafting config error: missing route/model")
        is ErrorFlagKind.RETRYABLE_CONFIG
    )
    assert classify_error_flag("Repair error: repair stage died") is ErrorFlagKind.REPAIR_TRANSIENT
    assert classify_error_flag("untranslated: stage crash") is ErrorFlagKind.UNTRANSLATED
    assert classify_error_flag("needs_human_review") is ErrorFlagKind.TRIAGE_VERDICT
    assert classify_error_flag("mqm_critical_blocked") is ErrorFlagKind.TRIAGE_VERDICT
    assert classify_error_flag("numeric_mismatch:2:high") is ErrorFlagKind.QUALITY
    assert classify_error_flag("terminology_consistency:1:low") is ErrorFlagKind.QUALITY
