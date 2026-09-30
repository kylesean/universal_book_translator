"""Verification entry points for the delivery contract (CLI: ``ubt verify``).

Two ways to obtain the contract:

- :func:`load_contract` reads the ``*_contract.json`` written beside a delivered
  artifact -- the artifact of record.
- :func:`contract_from_ledger` re-derives it from a finished job's ledger blocks,
  an independent cross-check that does not trust the sidecar.

:func:`evaluate_expectations` turns a corpus case's thresholds into failures, so
a regression on any render path shows up as an unbalanced book.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.content.adapt import graph_from_blocks
from ubt.core.content.contract import ReconciliationReport, reconcile
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
    ledger: SQLiteJobLedger, job_id: str, *, engine: str = "publication"
) -> ReconciliationReport:
    """Re-derive the contract from a finished job's ledger blocks.

    ``engine`` selects the asset-preservation policy (see
    :func:`ubt.core.content.adapt.graph_from_blocks`); it cannot be read back
    reliably from the ledger, so the caller states it (default: reflow).
    """
    blocks = ledger.get_all_blocks(job_id)
    graph = graph_from_blocks(blocks, engine=engine, doc_id=job_id)
    return reconcile(graph)


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
