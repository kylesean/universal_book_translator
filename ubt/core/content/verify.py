"""Verification entry points for the delivery contract (CLI: ``ubt verify``).

Two ways to obtain the contract:

- :func:`load_contract` reads the ``*_contract.json`` written beside a delivered
  artifact -- the artifact of record.
- :func:`contract_from_ledger` re-derives it from a finished job's ledger blocks
  -- it does not trust the sidecar -- through the same attestation projection the
  export used, so both paths read one account.

:func:`evaluate_expectations` turns a corpus case's thresholds into failures, so
a regression on any render path shows up as an unbalanced book.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.content.adapt import graph_from_blocks
from ubt.core.content.contract import ReconciliationReport
from ubt.core.job_options import sidecar_path

if TYPE_CHECKING:
    from ubt.core.engine.ledger import SQLiteJobLedger


def contract_path_for_artifact(artifact: Path | str) -> Path:
    """The standalone ``*_contract.json`` that sits beside an artifact."""
    return sidecar_path(Path(artifact), "contract.json")


def load_contract_file(path: Path | str) -> ReconciliationReport:
    """Read a ``*_contract.json`` written by the export stage (raises if absent)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReconciliationReport.model_validate(data)


def load_contract(artifact: Path | str) -> ReconciliationReport:
    """Read the contract written beside ``artifact`` (raises if absent/corrupt)."""
    return load_contract_file(contract_path_for_artifact(artifact))


def contract_from_ledger(
    ledger: SQLiteJobLedger, job_id: str, *, engine: str | None = None
) -> ReconciliationReport:
    """Re-derive the contract from a finished job's ledger blocks.

    It does not trust the ``*_contract.json`` sidecar: the blocks come from the
    ledger, and the run's language policy is read back from it (the target_lang
    column and the source_lang the ingest recorded), so the translation
    predicate scores with the same profile the run used.

    The account is the *attestation projection* the export uses (pre-render
    decision plan), not a second, independent balance: one account, one mechanism.
    ``engine`` selects the asset-preservation policy (see
    :func:`ubt.core.content.adapt.graph_from_blocks`); if not explicitly provided,
    it falls back to the job's persisted ``render_engine_effective`` metadata,
    defaulting to ``publication`` (reflow) when unset.
    """
    from ubt.core.qe.fast_pass import FastPassFilter
    from ubt.layout.theme import resolve_theme

    # Lazy: ubt.pipeline.attest imports ubt.core.content, so a module-level
    # import here would be a circular import.
    from ubt.pipeline.attest import attest_document, project_contract
    from ubt.pipeline.delivery import delivery_document, delivery_translations
    from ubt.render.typst_backend import TypstBackend
    from ubt.verify.verifier import build_verifiers

    if engine is None:
        persisted = ledger.get_job_metadata_value(job_id, "render_engine_effective")
        engine = str(persisted) if persisted else "publication"

    blocks = ledger.get_all_blocks(job_id)
    graph = graph_from_blocks(blocks, engine=engine, doc_id=job_id)
    source_lang = str(ledger.get_job_metadata_value(job_id, "source_lang") or "en")
    target_lang = str(ledger.get_job_target_lang(job_id) or "zh")
    report = attest_document(
        delivery_document(blocks, doc_id=job_id),
        TypstBackend(
            delivery_translations(blocks, engine=engine),
            theme=resolve_theme(source_lang, target_lang),
        ),
        build_verifiers(FastPassFilter(source_lang=source_lang, target_lang=target_lang)),
    )
    return project_contract(report, graph)


class CorpusCase(BaseModel):
    """One representative document in the verification corpus."""

    model_config = ConfigDict(extra="ignore")

    id: str
    description: str = ""
    #: Source document to translate (with ``--run``) before verifying. Relative
    #: paths resolve against the corpus directory.
    document: str | None = None
    #: Path to a delivered artifact (its ``*_contract.json`` is the contract).
    artifact: str | None = None
    #: Or a finished job id whose ledger is re-reconciled.
    job: str | None = None
    #: Assertion thresholds; all optional, all independent.
    expect: dict[str, float] = Field(default_factory=dict)


def load_corpus(corpus_dir: Path | str) -> list[CorpusCase]:
    """Read ``cases.json`` from a corpus directory."""
    manifest = Path(corpus_dir) / "cases.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no corpus manifest at {manifest}")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    return [CorpusCase.model_validate(item) for item in raw.get("cases", [])]


def evaluate_expectations(report: ReconciliationReport, expect: dict[str, Any]) -> list[str]:
    """Return the expectation failures for one verified contract (empty = pass).

    Recognised keys:

    - ``max_errors``: ERROR-severity violations allowed (default 0);
    - ``max_missing_assets``: assets that may be lost (default 0);
    - ``max_source_kept``: text nodes that may ship source (translation not placed);
    - ``min_accounted_ratio``: minimum share of text nodes that are accounted for
      as *delivered* (translated) or *verbatim* (intentionally kept). Nodes that
      shipped source (``source_kept``) or were dropped do not count.
    - ``min_delivered_ratio``: minimum share of text nodes actually *translated*.
      ``min_accounted_ratio`` also counts ``verbatim``, so a run that keeps more
      of the book than it translates can clear it; this is the floor that catches
      that trade (a finer block split that turns prose into preserved fragments,
      for instance).
    """
    failures: list[str] = []
    max_errors = int(expect.get("max_errors", 0))
    if len(report.errors) > max_errors:
        failures.append(f"{len(report.errors)} error(s) > allowed {max_errors}")

    max_missing = int(expect.get("max_missing_assets", 0))
    if report.missing_assets > max_missing:
        failures.append(f"{report.missing_assets} missing asset(s) > allowed {max_missing}")

    if "max_source_kept" in expect:
        allowed = int(expect["max_source_kept"])
        if report.source_kept_text > allowed:
            failures.append(
                f"{report.source_kept_text} source-kept text node(s) > allowed {allowed}"
            )

    if "min_accounted_ratio" in expect:
        accounted = report.delivered_text + report.verbatim_text
        ratio = accounted / report.total_text if report.total_text else 1.0
        floor = float(expect["min_accounted_ratio"])
        if ratio < floor:
            failures.append(f"accounted ratio {ratio:.3f} < required {floor:.3f}")

    if "min_delivered_ratio" in expect:
        ratio = report.delivered_text / report.total_text if report.total_text else 1.0
        floor = float(expect["min_delivered_ratio"])
        if ratio < floor:
            failures.append(f"delivered ratio {ratio:.3f} < required {floor:.3f}")

    return failures


__all__ = [
    "CorpusCase",
    "contract_from_ledger",
    "contract_path_for_artifact",
    "evaluate_expectations",
    "load_contract",
    "load_contract_file",
    "load_corpus",
]
