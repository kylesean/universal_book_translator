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
    offenders = []
    for view in sorted(_VIEWS.rglob("*.tsx")):
        for number, line in enumerate(_source(view).splitlines(), 1):
            if line.lstrip().startswith(("//", "*", "/*")):
                continue
            if re.search(r"\{\s*\w*[Ss]tatus\s*\.toUpperCase\(\)\s*\}", line):
                offenders.append(f"{view.name}:{number}: {line.strip()}")
    assert not offenders, "a status is upper-cased instead of localized: " + "; ".join(offenders)


def test_the_dictionaries_name_every_engine_state_a_view_shows() -> None:
    zh = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "zh.ts")
    en = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts")

    for status in ("submitted", "queued", "running", "completed", "failed", "cancelled"):
        assert status in en and status in zh, f"job status {status!r} is not localized"
    for status in ("mtqe_passed", "needs_human", "blocked_human", "repaired"):
        assert status in en and status in zh, f"block status {status!r} is not localized"
    for status in ("OK", "WARN", "FAIL", "SKIP"):
        assert status in en and status in zh, f"doctor status {status!r} is not localized"


def test_every_deliverable_key_has_a_label() -> None:
    from ubt.api.routes.shared import DELIVERABLE_LABELS

    meta = _source(_VIEWS / "quality-gate" / "artifactMeta.ts")
    switch = meta[meta.index("function deliverableName") :]
    switch = switch[: switch.index("\n}")]

    missing = [key for key in DELIVERABLE_LABELS if f"case '{key}':" not in switch]
    assert not missing, (
        f"these deliverable keys fall back to the backend's English label: {missing}"
    )


def test_the_reparse_confirm_names_the_job_it_will_destroy() -> None:
    repo = _REPO_ROOT
    dialog = _source(repo / "web" / "src" / "components" / "ui" / "ConfirmDialog.tsx")
    view = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    assert "typeToConfirm" in dialog, "ConfirmDialog lost its optional phrase gate"
    assert "disabled={!phraseOk}" in dialog, "the confirm button is never disabled"
    assert "typed.trim() === typeToConfirm" in dialog, "the typed text is not compared"

    marker = "onClick={() => setPendingReparse(currentJobId)}"
    assert marker in view, "the re-parse entry point disappeared"
    assert "pendingReparse &&" in view, "the dialog does not render the target job id"
    assert "reparseConfirm" in view


def test_the_console_says_re_parsing_is_expensive_and_destructive() -> None:
    zh = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "zh.ts")
    en = _source(_REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts")

    for text, words in (
        (zh, ("花费", "人工修改")),
        (en, ("cost", "human revision")),
    ):
        body = text[text.index("reparseConfirm:") :][:900]
        for word in words:
            assert word in body, f"reparseConfirm does not mention {word!r}"


def test_the_client_error_type_carries_the_http_status() -> None:
    client = _source(_REPO_ROOT / "web" / "src" / "api" / "client.ts")

    assert "export class ApiError extends Error" in client
    assert "readonly status: number" in client
    assert "new ApiError(message, res.status)" in client
    assert 'apiError(res, "string")' not in client


def test_the_review_pane_reads_freshness_from_the_ledger() -> None:
    view = _source(_VIEWS / "review" / "ReviewWorkbench.tsx")

    assert "listDeliverables(jobId)" in view
    assert "setReExportPending(listing.stale)" in view


def test_uncommitted_review_drafts_survive_leaving_the_page() -> None:
    view = _source(_VIEWS / "review" / "ReviewWorkbench.tsx")

    assert "_DRAFT_PREFIX" in view
    assert "sessionStorage.setItem(_DRAFT_PREFIX" in view
    assert "useState<Record<string, string>>(() => readDrafts(jobId))" in view
    assert "t.review.unsavedDraft" in view


def test_every_destructive_console_action_shows_the_job_it_will_destroy() -> None:
    view = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    for action in ("pendingReparse", "pendingDelete"):
        assert action in view, f"{action} is no longer wired"
    assert view.count("font-mono text-xs text-muted-foreground") >= 1
    assert view.count("font-mono text-xs text-foreground select-all") >= 1


def test_the_gate_never_attests_a_report_it_did_not_load() -> None:
    view = _source(_VIEWS / "quality-gate" / "QualityGateView.tsx")

    assert "const hasReport = report !== null" in view
    assert "const isPassed = hasReport &&" in view
    assert view.count("{hasReport && (") >= 1
    assert "t.quality.noVerdictTitle" in view


def test_pillar_three_reads_the_live_visual_verdict_not_a_deleted_field() -> None:
    view = _source(_VIEWS / "quality-gate" / "QualityGateView.tsx")
    reporter = _source(_REPO_ROOT / "ubt" / "core" / "engine" / "reporter.py")

    assert "fidelity" not in reporter, "the report schema grew fidelity back — recheck the view"
    assert "report?.fidelity" not in view
    assert "visual?.passed" in view and "visual?.skipped_reason" in view
    assert "visualMissing || visualSkipped" in view
    assert "visualMissing || visualSkipped ? 0" in view


def test_an_unconfigured_glossary_is_not_reported_as_clean() -> None:
    client = _source(_REPO_ROOT / "web" / "src" / "api" / "client.ts")
    view = _source(_VIEWS / "language-assets" / "LanguageAssetsView.tsx")

    assert "configured: false" in client
    assert "glossaryConfigured" in view
    assert "t.assets.glossaryNotConfigured" in view


def test_every_irreversible_console_action_confirms_first() -> None:
    assets = _source(_VIEWS / "language-assets" / "LanguageAssetsView.tsx")
    mission = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    assert "setPendingDeleteTerm(" in assets and "open={pendingDeleteTerm !== null}" in assets
    assert "setPendingEvict(" in assets and "open={pendingEvict !== null}" in assets
    assert "setPendingCancel(true)" in mission and "open={pendingCancel}" in mission


def test_the_mission_control_view_recovers_from_a_missing_job() -> None:
    view = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    assert "statusMissing" in view
    assert "err.status === 404" in view
    assert "t.mission.jobMissingTitle" in view
    assert "let missing = false" in view
    assert "if (missing) return" in view
    assert "clearInterval(timer)" in view
    assert "unsubscribe()" in view
    assert "if (isMounted && !missing) setIsLiveStreaming(connected)" in view


def test_a_restarted_run_resubscribes_its_stream() -> None:
    view = _source(_VIEWS / "mission-control" / "MissionControl.tsx")

    assert "streamGeneration" in view
    assert "streamGeneration" in view
    assert "}, [currentJobId, appendLog, activityText, streamGeneration])" in view
    assert view.count("setStreamGeneration((n) => n + 1)") == 2


def test_the_visual_gate_tiers_are_named_in_both_locales() -> None:
    tiers = {"geometry", "type", "pixel", "canvas", "vlm"}
    codes = {
        "vlm_judge_disabled",
        "page_render_failed",
        "canvas_background_absent",
        "canvas_background_render_failed",
        "unknown",
    }

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
    source = _source(_VIEWS / "quality-gate" / "QualityGateView.tsx")
    assert "${t.quality.visualGateSkipped}: ${visualSkipped}" not in source, (
        "the quality gate is rendering the raw English skipped_reason again"
    )
    assert "skipCodes.map(skipLabel)" in source, (
        "the metric must render translated skip codes, not the raw sentence"
    )


def test_the_provider_editor_does_not_offer_output_extraction_as_a_choice() -> None:
    source = _source(_VIEWS / "doctor" / "ProviderEditor.tsx")
    assert "EXTRACTION_STRATEGIES" not in source, (
        "the extraction dropdown is back; it offers values that are worse than "
        "the default and cannot be reasoned about from the form"
    )
    assert "extractionAutoNote" in source, (
        "the panel must still state which parsing behaviour applies"
    )


def test_the_system_prompt_switch_is_a_collapsed_remedy_not_a_peer_choice() -> None:
    source = _source(_VIEWS / "doctor" / "ProviderEditor.tsx")
    assert "capabilityTroubleshootToggle" in source, (
        "the system-prompt switch must sit behind a describing disclosure"
    )
    assert "<Accordion" in source, "the remedy must be collapsed by default"
