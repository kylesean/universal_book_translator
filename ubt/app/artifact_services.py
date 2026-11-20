"""Application implementation of the core artifact QA/gate protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class AdapterArtifactServices:
    """Adapter-backed post-render gate, parity, evidence and attestation."""

    def run_gate(
        self,
        pdf_path: Path,
        *,
        expected_page_count: int | None = None,
        faithful_pages: bool = False,
        target_lang: str | None = None,
    ) -> Any:
        from ubt.adapters.pdf.artifact_gate import run_visual_gate

        return run_visual_gate(
            pdf_path,
            expected_page_count=expected_page_count,
            faithful_pages=faithful_pages,
            target_lang=target_lang,
        )

    def crashed_gate(self, message: str) -> Any:
        from ubt.adapters.pdf.artifact_gate import VisualFinding, VisualGateResult

        return VisualGateResult(
            passed=False,
            findings=(
                VisualFinding(
                    severity="critical",
                    code="visual_gate_crashed",
                    message=message,
                ),
            ),
            stats={"total_pages": 0},
        )

    def blocking_gate(
        self,
        findings: Any,
        total_pages: int,
        enabled: bool,
    ) -> list[Any]:
        from ubt.adapters.pdf.artifact_gate import blocking_gate_tripped

        return list(blocking_gate_tripped(findings, total_pages, enabled))

    def read_evidence(self, rendered_path: Path, **kwargs: Any) -> Any:
        from ubt.adapters.pdf.delivery_evidence import read_artifact_evidence

        return read_artifact_evidence(rendered_path, **kwargs)

    def check_artifact(self, blocks: Any, translations: Any, rendered_path: Path) -> Any:
        from ubt.adapters.pdf.artifact_text import check_artifact

        return check_artifact(blocks, translations, rendered_path)

    def parity_findings(
        self,
        *,
        artifact_pdf: Path,
        target_lang: str,
        selected_pages: Any = None,
        source_lang: str | None = None,
    ) -> list[Any]:
        from ubt.adapters.pdf.artifact_parity import check_artifact_parity

        return list(
            check_artifact_parity(
                artifact_pdf=artifact_pdf,
                target_lang=target_lang,
                selected_pages=selected_pages,
                source_lang=source_lang,
            )
        )

    def probe_unavailable(self, code: str, message: str) -> Any:
        from ubt.adapters.pdf.artifact_parity import ParityFinding

        return ParityFinding("info", code, message)


__all__ = ["AdapterArtifactServices"]
