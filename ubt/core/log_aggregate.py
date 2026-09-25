"""Aggregate repeated third-party warnings into a tiered summary.

Some recoveries are logged once per object: docling's TableFormer snaps a
table cell to the nearest column centroid when no column band matched, and
warns for every one. On a real 92-page paper that was 1492 WARNING lines — and
they are not noise: 55% of those cells were snapped more than 200pt (7cm) from
the column they landed in, i.e. probably the wrong column.

Muting them would hide a correctness signal, so this module does the other
thing: count them, bucket them by how bad the guess was, and emit a handful of
lines instead of thousands. Everything stays queryable through
:func:`noise_report` so the pipeline can write the distribution into
``visual_report.json``.

The aggregator is a :class:`logging.Filter`, not a Handler, so it only ever
*counts* what it matches — every other record from the same logger still
propagates untouched.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

#: Snapshot points for the running summary. A tiered cascade (1, 10, 100, 1000)
#: keeps the log self-limiting without needing teardown coordination, and the
#: final report carries the exact total.
_CASCADE = (1, 10, 100, 1_000)


@dataclass
class NoiseAggregator(logging.Filter):
    """Count matching records instead of letting each one through."""

    name: str
    label: str
    pattern: re.Pattern[str]
    buckets: tuple[tuple[str, float], ...] = ()
    extract: Any = None
    seen: int = field(default=0, init=False)
    counts: dict[str, int] = field(default_factory=dict, init=False)
    worst: float = field(default=0.0, init=False)

    def filter(self, record: logging.LogRecord) -> bool:
        """Count a matching record (return False to drop it) and cascade-log."""
        message = record.getMessage()
        if not self.pattern.search(message):
            return True
        self.seen += 1
        value = self._value_of(message)
        if value is not None:
            self.worst = max(self.worst, value)
            for label, ceiling in self.buckets:
                if value <= ceiling:
                    self.counts[label] = self.counts.get(label, 0) + 1
                    break
        if self.seen in _CASCADE:
            logging.getLogger(__name__).warning("%s", self.summary())
        return False

    def _value_of(self, message: str) -> float | None:
        if self.extract is None:
            return None
        found = self.extract.search(message)
        return float(found.group(1)) if found else None

    def stats(self) -> dict[str, Any]:
        return {
            "total": self.seen,
            "buckets": dict(self.counts),
            "worst": round(self.worst, 1) if self.worst else None,
        }

    def summary(self) -> str:
        spread = " ".join(f"{label}={self.counts.get(label, 0)}" for label, _ in self.buckets)
        worst = f", worst {self.worst:.0f}pt" if self.worst else ""
        return f"{self.label}: {self.seen} event(s) [{spread}{worst}]"


#: Process-wide registry, so a report written later in the run can report what
#: the parse stage saw. Empty until :func:`install_noise_aggregators` runs.
_REGISTRY: dict[str, NoiseAggregator] = {}


def noise_aggregators() -> dict[str, NoiseAggregator]:
    return _REGISTRY


def noise_report() -> dict[str, dict[str, Any]]:
    """The aggregated counts, shaped for ``visual_report.json``."""
    return {name: agg.stats() for name, agg in _REGISTRY.items() if agg.seen}


#: The nearest-column distance thresholds, in PDF points. A typical table cell
#: is 30-80pt wide, so anything past 200pt is a guess spanning a third of a page.
_DIST_BUCKETS = (
    ("<=30pt", 30.0),
    ("30-100pt", 100.0),
    ("100-200pt", 200.0),
    (">200pt", float("inf")),
)

_ORPHAN = re.compile(r"Orphan pdf_cell \d+ recovered to (?:col|row)=\d+")
_DIST = re.compile(r"dist=([0-9.]+)")


def install_noise_aggregators() -> dict[str, NoiseAggregator]:
    """Attach the aggregators to their third-party loggers (idempotent)."""
    if "table_structure_guess" not in _REGISTRY:
        _REGISTRY["table_structure_guess"] = NoiseAggregator(
            name="table_structure_guess",
            label="table structure placed by geometry guess, not by the model",
            pattern=_ORPHAN,
            buckets=_DIST_BUCKETS,
            extract=_DIST,
        )
    agg = _REGISTRY["table_structure_guess"]
    for logger_name in (
        "docling_ibm_models.tableformer.data_management",
        "docling_ibm_models.tableformer.data_management.matching_post_processor",
    ):
        target = logging.getLogger(logger_name)
        for stale in [f for f in target.filters if isinstance(f, NoiseAggregator)]:
            target.removeFilter(stale)
        target.addFilter(agg)
    for handler in logging.getLogger().handlers:
        for stale in [f for f in handler.filters if isinstance(f, NoiseAggregator)]:
            handler.removeFilter(stale)
        handler.addFilter(agg)
    return _REGISTRY
