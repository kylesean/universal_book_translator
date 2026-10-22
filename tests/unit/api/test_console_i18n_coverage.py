"""The console must not leave English chrome inside a Chinese screen.

The dictionaries are typed (``TranslationDictionary``), so `tsc -b` already
fails when a key exists in only one locale. That check cannot see the other
half of the problem: a view that renders a *literal*, or a raw engine
identifier, never consults the dictionary at all — so both locales show the
same English word and the type system is satisfied.

These guards cover the literal class, which is invisible to the compiler. They
run in the Python suite because the frontend has no test runner of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VIEWS = _REPO_ROOT / "web" / "src" / "views"


def _source(path: Path) -> str:
    if not path.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    return path.read_text(encoding="utf-8")


def test_no_view_upper_cases_a_status_into_the_ui() -> None:
    """``status.toUpperCase()`` bypasses the dictionary entirely.

    That is how the Chinese console ended up with "COMPLETED" in the header
    badge, the queue rows and the execution-state panel — and "NEEDS_HUMAN" on
    every review card — while the surrounding chrome was Chinese.
    """
    offenders = []
    for view in sorted(_VIEWS.rglob("*.tsx")):
        for number, line in enumerate(_source(view).splitlines(), 1):
            if line.lstrip().startswith(("//", "*", "/*")):
                continue
            # A fallback inside a label helper is fine — it only fires for a
            # status the dictionary does not know. What must not happen is a
            # status rendered directly, i.e. `{status.toUpperCase()}` in JSX.
            if re.search(r"\{\s*\w*[Ss]tatus\s*\.toUpperCase\(\)\s*\}", line):
                offenders.append(f"{view.name}:{number}: {line.strip()}")
    assert not offenders, "a status is upper-cased instead of localized: " + "; ".join(offenders)


def test_the_dictionaries_name_every_engine_state_a_view_shows() -> None:
    """A status the dictionary does not name falls back to a raw identifier.

    The fallback exists so a new engine state is still visible, but the states
    the console already renders must be translated in both locales.
    """
    zh = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "zh.ts")
    en = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts")

    # Job statuses (ubt.core.engine.job_queue.JobStatus).
    for status in ("submitted", "queued", "running", "completed", "failed", "cancelled"):
        assert status in en and status in zh, f"job status {status!r} is not localized"
    # Block statuses (ubt.core.ir.models.BlockStatus) shown on review cards.
    for status in ("mtqe_passed", "needs_human", "blocked_human", "repaired"):
        assert status in en and status in zh, f"block status {status!r} is not localized"
    # Doctor verdicts.
    for status in ("OK", "WARN", "FAIL", "SKIP"):
        assert status in en and status in zh, f"doctor status {status!r} is not localized"


def test_every_deliverable_key_has_a_label() -> None:
    """The label switch fell through to the backend's English ``label``.

    ``DELIVERABLE_LABELS`` names the files the API will serve; the console
    translates them by key. ``html`` and ``xliff`` were missing from the switch,
    so those two buttons were the only deliverables still in English on a
    Chinese screen — and adding a key to the backend would silently repeat it.
    """
    from ubt.api.routes.shared import DELIVERABLE_LABELS

    view = _source(_VIEWS / "quality-gate" / "QualityGateView.tsx")
    switch = view[view.index("function deliverableLabel") :]
    switch = switch[: switch.index("\n}")]

    missing = [key for key in DELIVERABLE_LABELS if f"case '{key}':" not in switch]
    assert not missing, (
        f"these deliverable keys fall back to the backend's English label: {missing}"
    )


def test_the_reparse_confirm_requires_the_job_id_to_be_typed() -> None:
    """A destructive action must cost more than one click.

    Re-parsing discards a paid ledger and re-translates the whole book. A plain
    "are you sure" button is one stray Enter away from that, so the dialog takes
    a phrase only it names. Both halves are pinned here: the dialog must disable
    its confirm until the text matches, and the call site must actually pass one.
    """
    repo = _REPO_ROOT
    dialog = _source(repo / "web" / "src" / "components" / "ui" / "ConfirmDialog.tsx")
    view = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    assert "typeToConfirm" in dialog, "ConfirmDialog has no phrase gate"
    assert "disabled={!phraseOk}" in dialog, "the confirm button is never disabled"
    assert "typed.trim() === typeToConfirm" in dialog, "the typed text is not compared"

    marker = "onClick={() => setPendingReparse(currentJobId)}"
    assert marker in view, "the re-parse entry point disappeared"
    assert "typeToConfirm={pendingReparse ?? ''}" in view, (
        "the re-parse dialog does not require the job id"
    )
    # Remounted per job, so a phrase typed for one cannot confirm another.
    assert "key={pendingReparse ?? 'none'}" in view


def test_the_console_says_re_parsing_is_expensive_and_destructive() -> None:
    """The cost must be stated where the operator decides, not in a changelog."""
    zh = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "zh.ts")
    en = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts")

    # Both locales must name the two things the operator is about to lose.
    for text, words in (
        (zh, ("花费", "人工修改")),
        (en, ("cost", "human revision")),
    ):
        body = text[text.index("reparseConfirm:") :][:900]
        for word in words:
            assert word in body, f"reparseConfirm does not mention {word!r}"


def test_the_client_error_type_carries_the_http_status() -> None:
    """A bare `Error` forced every call site to guess at prose.

    The review surface distinguishes a 409 (another writer holds the job — retry
    once it clears) from a 422 (the request is wrong — retrying is pointless),
    and could not, because `apiError` flattened both into `new Error(detail)`.
    """
    client = _source(_REPO_ROOT / "web" / "src" / "api" / "client.ts")

    assert "export class ApiError extends Error" in client
    assert "readonly status: number" in client
    assert "new ApiError(message, res.status)" in client
    # The malformed label a template-literal refactor left behind.
    assert 'apiError(res, "string")' not in client


def test_the_review_pane_reads_freshness_from_the_ledger() -> None:
    """`reExportPending` in `useState` forgot across sessions what the ledger knows.

    A reviewer who edited, closed the tab, and came back saw a clean toolbar
    button while the delivered files were still stale — the marker lived only in
    the tab that made the edit.
    """
    view = _source(_VIEWS / "review" / "ReviewWorkbench.tsx")

    assert "listDeliverables(jobId)" in view
    assert "setReExportPending(listing.stale)" in view


def test_uncommitted_review_drafts_survive_leaving_the_page() -> None:
    """`drafts` was component state: any navigation discarded unsaved edits.

    On a long book that is real work lost with no warning and no trace.
    """
    view = _source(_VIEWS / "review" / "ReviewWorkbench.tsx")

    assert "_DRAFT_PREFIX" in view
    assert "sessionStorage.setItem(_DRAFT_PREFIX" in view
    assert "useState<Record<string, string>>(() => readDrafts(jobId))" in view
    # And the reviewer can see which block is carrying a restored draft.
    assert "t.review.unsavedDraft" in view


def test_every_destructive_console_action_requires_a_typed_phrase() -> None:
    """Deleting a job destroys more than re-parsing it, and was guarded more weakly.

    Re-parse (discard the ledger, re-translate) demanded the job id typed out;
    delete (ledger AND every delivered file, permanently) was one click.
    """
    view = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    for action in ("pendingReparse", "pendingDelete"):
        marker = f"typeToConfirm={{{action} ?? ''}}"
        assert marker in view, f"{action} does not require the job id"
        assert f"key={{{action} ?? 'none'}}" in view, f"{action} is not remounted per job"


def test_the_gate_never_attests_a_report_it_did_not_load() -> None:
    """A missing report rendered as PRODUCTION READY / PASS / five green pillars.

    `summary` defaults to `{}`, so `failed_blocks ?? 0` is 0 when the fetch
    failed, and the old `verdict ? … : failedBlocks === 0` fallback made the
    absent report read as a clean pass. The report legitimately 409s while a job
    is running and 404s before it is written, so this was reachable by opening
    the gate too early — an attestation of nothing, on the screen whose only job
    is attestation.
    """
    view = _source(_VIEWS / "quality-gate" / "QualityGateView.tsx")

    assert "const hasReport = report !== null" in view
    assert "const isPassed = hasReport &&" in view
    # The verdict block and the pillar list must both be gated on it.
    assert view.count("{hasReport && (") >= 2
    assert "t.quality.noVerdictTitle" in view


def test_pillar_three_reads_the_live_visual_verdict_not_a_deleted_field() -> None:
    """`report.fidelity` was removed with the overlay engine, but stayed read.

    The result was a permanent "0 pages · residual —" scoring 100 and PASS: full
    marks for a measurement nobody took. The real verdict (`passed`,
    `skipped_reason`) lives on the visual report payload.
    """
    view = _source(_VIEWS / "quality-gate" / "QualityGateView.tsx")
    reporter = _source(_REPO_ROOT / "ubt" / "core" / "engine" / "reporter.py")

    assert "fidelity" not in reporter, "the report schema grew fidelity back — recheck the view"
    assert "report?.fidelity" not in view
    assert "visual?.passed" in view and "visual?.skipped_reason" in view
    # A skipped gate must not read as a pass.
    assert "visualSkipped\n        ? 'warn'" in view or "visualSkipped ? 'warn'" in view


def test_an_unconfigured_glossary_is_not_reported_as_clean() -> None:
    """A 409 ("no glossary file configured") was swallowed into `[]`.

    The panel then showed "0 active terms" beside a green "No glossary conflicts
    detected" — two positive claims about a file that does not exist.
    """
    client = _source(_REPO_ROOT / "web" / "src" / "api" / "client.ts")
    view = _source(_VIEWS / "language-assets" / "LanguageAssetsView.tsx")

    assert "configured: false" in client
    assert "glossaryConfigured" in view
    assert "t.assets.glossaryNotConfigured" in view


def test_every_irreversible_console_action_confirms_first() -> None:
    """The cheapest clicks were on the most expensive mistakes."""
    assets = _source(_VIEWS / "language-assets" / "LanguageAssetsView.tsx")
    mission = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    # Glossary term and TM row: an icon click used to rewrite the glossary file
    # and delete TM rows (including human post-editing) with no prompt.
    assert "setPendingDeleteTerm(" in assets and "open={pendingDeleteTerm !== null}" in assets
    assert "setPendingEvictId(" in assets and "open={pendingEvictId !== null}" in assets
    # Cancel: aborts a billable in-flight run. It demanded a single click while
    # delete and re-parse demanded a typed phrase.
    assert "setPendingCancel(true)" in mission and "open={pendingCancel}" in mission


def test_the_mission_control_view_recovers_from_a_missing_job() -> None:
    """`statusStr` defaulted to 'running' and the poll swallowed every error.

    An unknown or deleted id therefore rendered a RUNNING badge, a live Cancel
    button and "waiting for the first frame" indefinitely.
    """
    view = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    assert "statusMissing" in view
    assert "err.status === 404" in view
    assert "t.mission.jobMissingTitle" in view


def test_a_restarted_run_resubscribes_its_stream() -> None:
    """The SSE client closes permanently on a terminal frame.

    Resume and re-parse restart the SAME job id, so without a new subscription
    the panel kept the previous run's "completed · 250/250" lines above a run
    that was actively re-translating.
    """
    view = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    assert "streamGeneration" in view
    assert "}, [currentJobId, appendLog, streamGeneration])" in view
    assert view.count("setStreamGeneration((n) => n + 1)") == 2


def test_the_visual_gate_tiers_are_named_in_both_locales() -> None:
    """Every tier/code the visual gate can report must have a translation.

    The quality-gate view used to print the backend's English
    ``skipped_reason`` sentence verbatim — "vlm judge disabled; tiers run:
    geometry, type, pixel (not run: vlm)" — inside an otherwise Chinese panel.
    It now renders the structured ``tiers_ran``/``tiers_skipped``/``skip_codes``
    through these maps, so a tier or code with no entry falls back to the raw
    English token, which is the same leak with extra steps.
    """
    tiers = {"geometry", "type", "pixel", "vlm"}
    codes = {"vlm_judge_disabled", "page_render_failed", "unknown"}

    for locale in ("en", "zh"):
        text = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / f"{locale}.ts")
        tier_block = re.search(r"visualTierNames:\s*\{(.*?)\}", text, re.DOTALL)
        code_block = re.search(r"visualSkipCodes:\s*\{(.*?)\}", text, re.DOTALL)
        assert tier_block, f"{locale}.ts has no visualTierNames block"
        assert code_block, f"{locale}.ts has no visualSkipCodes block"

        missing_tiers = sorted(
            t for t in tiers if re.search(rf"\b{t}:", tier_block.group(1)) is None
        )
        missing_codes = sorted(
            c for c in codes if re.search(rf"\b{c}:", code_block.group(1)) is None
        )
        assert not missing_tiers, f"{locale}.ts is missing visualTierNames for {missing_tiers}"
        assert not missing_codes, f"{locale}.ts is missing visualSkipCodes for {missing_codes}"


def test_the_quality_gate_does_not_render_the_raw_skipped_reason() -> None:
    """The English sentence must not reach the row header.

    ``skipped_reason`` stays on the payload for logs and the CLI. Rendering it
    in the console both leaked English and overflowed the row, pushing the
    status badge into a two-character column.
    """
    source = _source(_VIEWS / "quality-gate" / "QualityGateView.tsx")
    assert "${t.quality.visualGateSkipped}: ${visualSkipped}" not in source, (
        "the quality gate is rendering the raw English skipped_reason again"
    )
    assert "skipCodes.map(skipLabel)" in source, (
        "the metric must render translated skip codes, not the raw sentence"
    )
