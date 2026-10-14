"""Delivered-artifact identity: the two paths the export stage keys on.

Only path bookkeeping lives here. The artifact *presence check* (does the
delivered PDF actually carry each text block's realization?) is PDF-adapter
machinery and lives in :mod:`ubt.adapters.pdf.artifact_text` — the pipeline
layer must not name ``ubt.adapters``, and the layering acceptance tool
(``scripts/dump_compiler_edges.py``) forbids that edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ubt.core.job_options import SidecarKind, companion_path, sidecar_path


@dataclass(frozen=True, slots=True)
class DeliveredArtifact:
    """One delivered artifact and the files that hang off it.

    Two artifact identities coexist in the export stage: ``target_output`` (the
    path the primary render was *asked* for, which the PE queue keys on) and
    ``rendered_path`` (the file the adapter actually *returned*, which the visual
    gate may rewrite and which every sidecar and companion keys on). Keeping both
    in one value means the stage never has to remember which helper takes which,
    and the sibling/companion naming has a single owner instead of ad-hoc
    ``with_name`` calls spread through the stage.
    """

    rendered_path: Path
    target_output: Path

    def sidecar(self, kind: SidecarKind) -> Path:
        """The derived report belonging to the returned artifact."""
        return sidecar_path(self.rendered_path, kind)

    def companion(self, suffix: str) -> Path:
        """A companion document beside the returned artifact (e.g. ``.xliff``)."""
        return companion_path(self.rendered_path, suffix)

    def sibling(self, suffix: str) -> Path:
        """A second artifact beside the *requested* target (e.g. ``_mono.pdf``)."""
        out = self.target_output
        return out.with_name(f"{out.stem}{suffix}{out.suffix}")


def delivered_artifact(rendered_path: str | Path, target_output: str | Path) -> DeliveredArtifact:
    """Name the two artifact identities once, where the render returns."""
    return DeliveredArtifact(Path(rendered_path), Path(target_output))


__all__ = ["DeliveredArtifact", "delivered_artifact"]
