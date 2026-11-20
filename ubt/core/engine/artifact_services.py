"""Core-side artifact QA/gate queries, resolved through the application root.

The export stage needs the post-render gate, artifact parity, delivery
evidence, and the attestation check. All of them live in the PDF adapter stack;
core names only this protocol, and the application composition root installs
the implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ubt.core.engine import _artifact_services_registry as _registry


class ArtifactServices(Protocol):
    """PDF artifact checks the core export stage is allowed to call."""

    def run_gate(
        self,
        pdf_path: Path,
        *,
        expected_page_count: int | None = None,
        faithful_pages: bool = False,
        target_lang: str | None = None,
    ) -> Any: ...

    def crashed_gate(self, message: str) -> Any: ...

    def blocking_gate(
        self,
        findings: Any,
        total_pages: int,
        enabled: bool,
    ) -> list[Any]: ...

    def read_evidence(self, rendered_path: Path, **kwargs: Any) -> Any: ...

    def check_artifact(self, blocks: Any, translations: Any, rendered_path: Path) -> Any: ...

    def parity_findings(
        self,
        *,
        artifact_pdf: Path,
        target_lang: str,
        selected_pages: Any = None,
        source_lang: str | None = None,
    ) -> list[Any]: ...

    def probe_unavailable(self, code: str, message: str) -> Any: ...


def configure_artifact_services(provider: ArtifactServices) -> None:
    """Install the app's implementation (the composition root calls this once)."""
    _registry.provider = provider


def _provider() -> ArtifactServices:
    provider = _registry.provider
    if provider is None:
        raise RuntimeError(
            "Artifact services are not configured; import the application composition root "
            "(ubt.app.bootstrap) before running a job"
        )
    return provider


def run_visual_gate(
    pdf_path: Path,
    *,
    expected_page_count: int | None = None,
    faithful_pages: bool = False,
    target_lang: str | None = None,
) -> Any:
    return _provider().run_gate(
        pdf_path,
        expected_page_count=expected_page_count,
        faithful_pages=faithful_pages,
        target_lang=target_lang,
    )


def crashed_visual_gate_result(message: str) -> Any:
    return _provider().crashed_gate(message)


def blocking_gate_tripped(
    findings: Any,
    total_pages: int,
    enabled: bool,
) -> list[Any]:
    return _provider().blocking_gate(findings, total_pages, enabled)


def read_delivery_evidence(rendered_path: Path, **kwargs: Any) -> Any:
    return _provider().read_evidence(rendered_path, **kwargs)


def check_artifact(blocks: Any, translations: Any, rendered_path: Path) -> Any:
    return _provider().check_artifact(blocks, translations, rendered_path)


def artifact_parity_findings(
    *,
    artifact_pdf: Path,
    target_lang: str,
    selected_pages: Any = None,
    source_lang: str | None = None,
) -> list[Any]:
    return _provider().parity_findings(
        artifact_pdf=artifact_pdf,
        target_lang=target_lang,
        selected_pages=selected_pages,
        source_lang=source_lang,
    )


def probe_unavailable_finding(code: str, message: str) -> Any:
    return _provider().probe_unavailable(code, message)


__all__ = [
    "ArtifactServices",
    "artifact_parity_findings",
    "blocking_gate_tripped",
    "check_artifact",
    "configure_artifact_services",
    "crashed_visual_gate_result",
    "probe_unavailable_finding",
    "read_delivery_evidence",
    "run_visual_gate",
]
