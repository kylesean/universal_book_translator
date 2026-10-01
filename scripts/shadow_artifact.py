#!/usr/bin/env python
"""Artifact-I/O acceptance: one value names the delivered artifact's files.

The export stage carries two artifact identities -- the path the primary render
was *asked* for (``target_output``) and the file the adapter *returned*
(``rendered_path``, which the visual gate may rewrite). Sidecars and companions
key on the returned file; the PE queue and a companion render key on the
requested one. ``DeliveredArtifact`` holds both so a caller never picks wrong.

This checks the value's derivations against the pure ``job_options`` helpers and
against the naming each stage previously spelled out inline.
"""

from __future__ import annotations

from ubt.core.job_options import SidecarKind, companion_path, sidecar_path
from ubt.pipeline.artifact import delivered_artifact

_KINDS: tuple[SidecarKind, ...] = (
    "quality_report.json",
    "quality_report.md",
    "metrics.json",
    "visual_report.json",
    "contract.json",
)


def main() -> int:
    problems: list[str] = []

    # The two identities differ for an auto-named monolingual primary.
    target = "/out/book_bilingual.pdf"
    rendered = "/out/book_bilingual_mono.pdf"
    artifact = delivered_artifact(rendered, target)

    for kind in _KINDS:
        got = artifact.sidecar(kind)
        want = sidecar_path(rendered, kind)
        if got != want:
            problems.append(f"sidecar({kind!r}) = {got} != {want}")
    if artifact.companion(".xliff") != companion_path(rendered, ".xliff"):
        problems.append("companion is not keyed on the rendered file")

    # A companion render keys on the requested target, not the returned file.
    if str(artifact.sibling("_rigid")) != "/out/book_bilingual_rigid.pdf":
        problems.append(f"sibling('_rigid') = {artifact.sibling('_rigid')}")
    if str(artifact.sibling("_secondary")) != "/out/book_bilingual_secondary.pdf":
        problems.append(f"sibling('_secondary') = {artifact.sibling('_secondary')}")

    # The rendered identity is what the sidecar readers look for.
    if str(artifact.sidecar("contract.json")) != "/out/book_bilingual_mono_pdf_contract.json":
        problems.append("sidecar did not key on the rendered file name")

    print("\nArtifact-I/O acceptance (DeliveredArtifact)")
    print(f"  target={artifact.target_output}")
    print(f"  rendered={artifact.rendered_path}")
    print(f"  problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"    {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
