#!/usr/bin/env python
"""Phase-1 acceptance: the typed AST round-trips the corpus without loss.

Parses each corpus document into ``IRBlock``s, bridges them into a typed
:class:`~ubt.model.ast.Document`, bridges that back to ``IRBlock``s, and asserts
the document-defining fields are identical. This is the ADR's Phase-1 gate: the
new model captures the real documents, not a toy.

Compared (document structure): id, spine_index, block_type, flow_id,
region, source_text, skip_translate, bbox.
Not compared (execution state the AST deliberately does not model): status,
target_text, scores, flags, provenance.

Exit code 0 when every block round-trips, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from ubt.analyze.bridge import blocks_from_document, document_from_blocks
from ubt.core.ir.models import FlowID, IRBlock
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.model.ast import Document, RegionKind
from ubt.verify import build_verifiers, verify_element


def _parse_blocks(document: Path, engine: str) -> list[IRBlock]:
    from ubt.adapters.factory import get_adapter_for_path

    adapter = get_adapter_for_path(document, pdf_engine=engine)

    async def collect() -> list[IRBlock]:
        blocks: list[IRBlock] = []
        async for chapter in adapter.parse_stream(document):
            blocks.extend(chapter.blocks)
        return blocks

    return asyncio.run(collect())


def _norm_region(block: IRBlock) -> str:
    if block.region is not None:
        return block.region.value
    if block.flow_id is FlowID.CAPTION:
        return RegionKind.CAPTION.value
    if block.flow_id is FlowID.FOOTNOTE:
        return RegionKind.FOOTNOTE.value
    return RegionKind.BODY.value


def _projection(block: IRBlock) -> dict[str, object]:
    return {
        "spine_index": block.spine_index,
        "block_type": block.block_type.value,
        "flow_id": block.flow_id.value,
        "region": _norm_region(block),
        "source_text": block.source_text or "",
        "skip_translate": block.skip_translate,
        "bbox": None
        if block.bbox is None
        else (block.bbox.page, block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1),
    }


def _compare(
    blocks: list[IRBlock], *, doc_id: str, path: str
) -> tuple[Counter[str], list[str], Document]:
    document = document_from_blocks(blocks, doc_id=doc_id, path=path)
    rebuilt = blocks_from_document(document)
    original = {block.id: block for block in blocks}
    rebuilt_by_id = {block.id: block for block in rebuilt}

    counts: Counter[str] = Counter()
    details: list[str] = []

    missing = set(original) - set(rebuilt_by_id)
    extra = set(rebuilt_by_id) - set(original)
    for block_id in sorted(missing):
        counts["missing"] += 1
        details.append(f"missing block {block_id}")
    for block_id in sorted(extra):
        counts["extra"] += 1
        details.append(f"extra block {block_id}")

    for block_id, before in original.items():
        after = rebuilt_by_id.get(block_id)
        if after is None:
            continue
        before_proj = _projection(before)
        after_proj = _projection(after)
        diffs = [key for key, value in before_proj.items() if after_proj[key] != value]
        if diffs:
            counts["mismatch"] += 1
            for key in diffs:
                counts[f"field:{key}"] += 1
            if len(details) < 40:
                details.append(f"{block_id}: {diffs}")
        else:
            counts["ok"] += 1
    return counts, details, document


def _judge(document: Document) -> tuple[Counter[str], Counter[str], int]:
    """Run the verifier over every element; count kinds, outcomes, and fall-through."""
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    by_kind: Counter[str] = Counter()
    by_outcome: Counter[str] = Counter()
    unjudged = 0
    for element in document.elements:
        by_kind[element.kind.value] += 1
        try:
            proof = verify_element(element, verifiers)
        except Exception:  # a fall-through is the failure this gate exists to catch
            unjudged += 1
            continue
        by_outcome[proof.outcome.value] += 1
    return by_kind, by_outcome, unjudged


def _load_cases(corpus_dir: Path) -> list[tuple[str, Path]]:
    from ubt.core.content.verify import load_corpus

    cases: list[tuple[str, Path]] = []
    for case in load_corpus(corpus_dir):
        if not case.document:
            continue
        document = Path(case.document)
        if not document.is_absolute():
            document = corpus_dir / document
        cases.append((case.id, document))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    per_case: list[dict[str, object]] = []
    total = 0
    total_ok = 0
    total_elements = 0
    total_unjudged = 0
    all_details: list[str] = []
    for case_id, document in _load_cases(corpus_dir):
        if not document.exists():
            per_case.append({"id": case_id, "status": "missing"})
            continue
        try:
            blocks = _parse_blocks(document, args.engine)
        except Exception as exc:  # one bad document must not sink the acceptance run
            per_case.append({"id": case_id, "status": "error", "reason": str(exc)})
            continue
        counts, details, doc = _compare(blocks, doc_id=case_id, path=str(document))
        by_kind, by_outcome, unjudged = _judge(doc)
        total += len(blocks)
        total_ok += counts["ok"]
        total_elements += sum(by_kind.values())
        total_unjudged += unjudged
        all_details.extend(f"[{case_id}] {line}" for line in details)
        per_case.append(
            {
                "id": case_id,
                "status": "pass"
                if counts["mismatch"] == 0 and counts["missing"] == 0 and unjudged == 0
                else "fail",
                "blocks": len(blocks),
                "ok": counts["ok"],
                "mismatch": counts["mismatch"],
                "missing": counts["missing"],
                "extra": counts["extra"],
                "elements": sum(by_kind.values()),
                "unjudged": unjudged,
                "outcomes": dict(by_outcome),
                "fields": {k: v for k, v in counts.items() if k.startswith("field:")},
            }
        )

    passed = total > 0 and total_ok == total and total_unjudged == 0
    payload = {
        "status": "pass" if passed else "fail",
        "total_blocks": total,
        "round_tripped": total_ok,
        "total_elements": total_elements,
        "unjudged": total_unjudged,
        "cases": per_case,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"\nPhase-1 AST acceptance — {corpus_dir}")
        for entry in per_case:
            fields = entry.get("fields") or ""
            print(
                f"  {entry['id']:<20} {entry['status']:<8} "
                f"blocks={entry.get('blocks', 0)} ok={entry.get('ok', 0)} "
                f"mismatch={entry.get('mismatch', 0)} missing={entry.get('missing', 0)} "
                f"elements={entry.get('elements', 0)} unjudged={entry.get('unjudged', 0)}"
                + (f"  {fields}" if fields else "")
                + (f"  reason={entry['reason']}" if entry.get("reason") else "")
            )
        print(f"\n  round-trip : {total_ok}/{total} blocks lossless")
        print(
            f"  judged     : {total_elements - total_unjudged}/{total_elements} elements, {total_unjudged} unjudged"
        )
        for line in all_details[:30]:
            print(f"    {line}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
