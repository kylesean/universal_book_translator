"""Agent translation workflow: zero model calls, leases, idempotency, rendering."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from pydantic import SecretStr

from ubt.core.config import UBTConfig
from ubt.core.engine.agent_translation import (
    AgentBlockSubmission,
    AgentClaimRequest,
    AgentFinishRequest,
    AgentPrepareRequest,
    AgentSubmitRequest,
    AgentTranslationService,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_mixins import AGENT_SESSION_KEY
from ubt.core.ir.models import BlockStatus

pytestmark = pytest.mark.fast


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        visual_judge_enabled=False,
        ocr_mode="off",
        allow_page_upload=False,
        service_api_key=SecretStr("test-key"),
    )


def _write_book(tmp_path: Path) -> Path:
    source = tmp_path / "book.md"
    source.write_text(
        "# A Tiny Book\n\n"
        "Hello world. This is a test paragraph with $x_i$ math and `code`.\n\n"
        "## Second Section\n\nThe value is 42.\n",
        encoding="utf-8",
    )
    return source


def _translate(masked: str) -> str:
    return (
        masked.replace("A Tiny Book", "一本小书")
        .replace("Hello world.", "你好世界。")
        .replace("This is a test paragraph", "这是一个测试段落")
        .replace("Second Section", "第二部分")
        .replace("The value is 42.", "数值是 42。")
    )


@pytest.mark.asyncio
async def test_prepare_claim_submit_finish_without_any_model_call(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            output_path=str(tmp_path / "translated.md"),
            job_id="agent_e2e",
        )
    )
    assert prepared.model_calls == 0
    assert prepared.total_blocks == 4
    assert prepared.pending_blocks == 4

    while True:
        claim = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
        if claim.package is None:
            break
        submissions = [
            AgentBlockSubmission(
                block_id=block.block_id, translation=_translate(block.masked_source)
            )
            for block in claim.package.blocks
        ]
        result = await service.submit(
            AgentSubmitRequest(
                job_id=prepared.job_id,
                package_id=claim.package.package_id,
                lease_token=claim.package.lease_token,
                blocks=submissions,
            )
        )
        assert result.accepted_blocks == len(submissions)

    finished = await service.finish(AgentFinishRequest(job_id=prepared.job_id))
    assert finished.status == "completed"
    assert finished.output_path is not None
    output = Path(finished.output_path)
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "一本小书" in text
    assert "$x_i$" in text
    assert "`code`" in text
    assert finished.report_path is not None
    assert Path(finished.report_path).exists()


@pytest.mark.asyncio
async def test_prepare_same_id_with_different_inputs_is_rejected(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    request = AgentPrepareRequest(input_path=str(source), job_id="agent_same")
    await service.prepare(request)
    with pytest.raises(Exception, match="different Agent prepare inputs"):
        await service.prepare(
            AgentPrepareRequest(
                input_path=str(source),
                target_lang="ja",
                job_id="agent_same",
            )
        )


@pytest.mark.asyncio
async def test_prepare_refuses_an_unrelated_existing_output(tmp_path: Path) -> None:
    """A pre-existing file at the deliverable path is not the job's to clobber.

    The rejection must land before any block is parsed, and a completed run's
    own artifact stays re-renderable: the exemption reads the job's own
    recorded ``output_file``, not the ``fresh`` flag.
    """
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    victim = tmp_path / "translated.md"
    victim.write_text("someone else's manuscript", encoding="utf-8")
    with pytest.raises(Exception, match="refusing to overwrite"):
        await service.prepare(
            AgentPrepareRequest(
                input_path=str(source),
                output_path=str(victim),
                job_id="agent_victim",
            )
        )
    with SQLiteJobLedger(tmp_path / "db" / "agent_victim.sqlite") as ledger:
        assert ledger.get_job_stats("agent_victim") is None or (
            int(ledger.get_job_stats("agent_victim").get("total_all", 0) or 0) == 0
        )
    assert victim.read_text(encoding="utf-8") == "someone else's manuscript"


@pytest.mark.asyncio
async def test_prepare_replay_may_re_render_its_own_deliverable(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            output_path=str(tmp_path / "translated.md"),
            job_id="agent_own_output",
        )
    )
    # A crashed render may have left a partial artifact; the job that recorded
    # the path owns it.
    Path(prepared.output_path).write_text("partial", encoding="utf-8")
    replayed = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            output_path=str(tmp_path / "translated.md"),
            job_id="agent_own_output",
        )
    )
    assert replayed.job_id == prepared.job_id


@pytest.mark.asyncio
async def test_claim_is_exclusive_and_expired_lease_is_recoverable(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            job_id="agent_lease",
            package_size=2,
            lease_seconds=30,
        )
    )
    first = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert first.package is not None
    second = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert second.package is not None
    assert {b.block_id for b in first.package.blocks}.isdisjoint(
        b.block_id for b in second.package.blocks
    )
    with pytest.raises(Exception, match="not a leased draft|lease"):
        await service.submit(
            AgentSubmitRequest(
                job_id=prepared.job_id,
                package_id=first.package.package_id,
                lease_token="x" * 32,
                blocks=[
                    AgentBlockSubmission(
                        block_id=b.block_id, translation=_translate(b.masked_source)
                    )
                    for b in first.package.blocks
                ],
            )
        )
    # Simulate elapsed leases and prove their blocks become claimable again.
    db_path = tmp_path / "db" / f"{prepared.job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        session = ledger.get_job_metadata_value(prepared.job_id, "agent_session")
        assert isinstance(session, dict)
        for package in session["packages"].values():
            package["expires_at"] = time.time() - 1
        ledger.set_job_metadata_value(prepared.job_id, "agent_session", session)
    recovered = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert recovered.package is not None


@pytest.mark.asyncio
async def test_concurrent_claims_never_lease_the_same_block(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            job_id="agent_concurrent",
            package_size=2,
        )
    )
    first, second = await asyncio.gather(
        service.claim(AgentClaimRequest(job_id=prepared.job_id)),
        service.claim(AgentClaimRequest(job_id=prepared.job_id)),
    )
    assert first.package is not None
    assert second.package is not None
    assert {b.block_id for b in first.package.blocks}.isdisjoint(
        {b.block_id for b in second.package.blocks}
    )


@pytest.mark.asyncio
async def test_submit_is_idempotent_for_same_package(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(input_path=str(source), job_id="agent_idem", package_size=1)
    )
    claim = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert claim.package is not None
    block = claim.package.blocks[0]
    request = AgentSubmitRequest(
        job_id=prepared.job_id,
        package_id=claim.package.package_id,
        lease_token=claim.package.lease_token,
        blocks=[
            AgentBlockSubmission(
                block_id=block.block_id,
                translation=_translate(block.masked_source),
            )
        ],
    )
    first = await service.submit(request)
    replay = await service.submit(request)
    assert first.accepted_blocks == 1
    assert replay.idempotent_replay is True
    assert replay.accepted_blocks == 1


@pytest.mark.asyncio
async def test_broken_placeholder_is_routed_to_repair(tmp_path: Path) -> None:
    source = tmp_path / "placeholder.md"
    source.write_text("The value is $x_i$ and `code`.\n", encoding="utf-8")
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(input_path=str(source), job_id="agent_placeholder")
    )
    claim = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert claim.package is not None
    block = claim.package.blocks[0]
    await service.submit(
        AgentSubmitRequest(
            job_id=prepared.job_id,
            package_id=claim.package.package_id,
            lease_token=claim.package.lease_token,
            blocks=[
                AgentBlockSubmission(
                    block_id=block.block_id,
                    translation="数值被损坏。",
                )
            ],
        )
    )
    db_path = tmp_path / "db" / f"{prepared.job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        saved = ledger.get_block(block.block_id, job_id=prepared.job_id)
    assert saved is not None
    assert saved.status is BlockStatus.REPAIR_PENDING
    assert saved.error_flags
    repair_claim = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert repair_claim.package is not None
    assert repair_claim.package.kind == "repair"
    assert repair_claim.package.blocks[0].draft_text


@pytest.mark.asyncio
async def test_repair_rounds_are_consumed_and_stop_at_human_gate(tmp_path: Path) -> None:
    source = tmp_path / "repair_loop.md"
    source.write_text("The value is $x_i$.\n", encoding="utf-8")
    config = _config(tmp_path).model_copy(update={"max_repair_rounds": 2})
    service = AgentTranslationService(config=config)
    prepared = await service.prepare(
        AgentPrepareRequest(input_path=str(source), job_id="agent_repair_budget")
    )
    claim = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert claim.package is not None
    block = claim.package.blocks[0]
    await service.submit(
        AgentSubmitRequest(
            job_id=prepared.job_id,
            package_id=claim.package.package_id,
            lease_token=claim.package.lease_token,
            blocks=[AgentBlockSubmission(block_id=block.block_id, translation="数值被损坏。")],
        )
    )

    for expected_round in (1, 2, 3):
        repair = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
        assert repair.package is not None
        assert repair.package.kind == "repair"
        repair_block = repair.package.blocks[0]
        await service.submit(
            AgentSubmitRequest(
                job_id=prepared.job_id,
                package_id=repair.package.package_id,
                lease_token=repair.package.lease_token,
                blocks=[
                    AgentBlockSubmission(
                        block_id=repair_block.block_id,
                        translation="数值仍然损坏。",
                    )
                ],
            )
        )
        with SQLiteJobLedger(tmp_path / "db" / f"{prepared.job_id}.sqlite") as ledger:
            saved = ledger.get_block(block.block_id, job_id=prepared.job_id)
        assert saved is not None
        assert saved.repair_rounds == expected_round

    with pytest.raises(Exception, match="below the 50% floor"):
        await service.finish(AgentFinishRequest(job_id=prepared.job_id))
    with SQLiteJobLedger(tmp_path / "db" / f"{prepared.job_id}.sqlite") as ledger:
        saved = ledger.get_block(block.block_id, job_id=prepared.job_id)
        session = ledger.get_job_metadata_value(prepared.job_id, "agent_session")
    assert saved is not None
    assert saved.status is BlockStatus.BLOCKED_HUMAN
    assert saved.repair_rounds == 3
    assert isinstance(session, dict)
    assert session["state"] == "awaiting_agent"


@pytest.mark.asyncio
async def test_finish_refuses_while_another_package_is_actively_leased(
    tmp_path: Path,
) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            job_id="agent_active_finish",
            package_size=2,
        )
    )
    first = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    second = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert first.package is not None
    assert second.package is not None
    await service.submit(
        AgentSubmitRequest(
            job_id=prepared.job_id,
            package_id=first.package.package_id,
            lease_token=first.package.lease_token,
            blocks=[
                AgentBlockSubmission(
                    block_id=block.block_id, translation=_translate(block.masked_source)
                )
                for block in first.package.blocks
            ],
        )
    )

    finished = await service.finish(AgentFinishRequest(job_id=prepared.job_id))
    assert finished.status == "awaiting_agent"
    assert "still actively leased" in finished.message
    with SQLiteJobLedger(tmp_path / "db" / f"{prepared.job_id}.sqlite") as ledger:
        session = ledger.get_job_metadata_value(prepared.job_id, "agent_session")
        blocks = ledger.get_all_blocks(prepared.job_id)
    assert isinstance(session, dict)
    assert session["state"] == "awaiting_agent"
    assert any(block.status is BlockStatus.DRAFTED for block in blocks)


@pytest.mark.asyncio
async def test_repair_packages_do_not_mix_with_pending_blocks(tmp_path: Path) -> None:
    source = tmp_path / "mixed.md"
    source.write_text("First $x$ value.\n\nSecond ordinary paragraph.\n", encoding="utf-8")
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            job_id="agent_mixed_claim",
            package_size=8,
        )
    )
    first = await service.claim(AgentClaimRequest(job_id=prepared.job_id, max_blocks=1))
    assert first.package is not None
    block = first.package.blocks[0]
    await service.submit(
        AgentSubmitRequest(
            job_id=prepared.job_id,
            package_id=first.package.package_id,
            lease_token=first.package.lease_token,
            blocks=[AgentBlockSubmission(block_id=block.block_id, translation="损坏。")],
        )
    )

    repair = await service.claim(AgentClaimRequest(job_id=prepared.job_id, max_blocks=8))
    assert repair.package is not None
    assert repair.package.kind == "repair"
    assert all(block.draft_text for block in repair.package.blocks)


@pytest.mark.asyncio
async def test_submit_replay_with_different_text_is_a_conflict(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            job_id="agent_replay_conflict",
            package_size=1,
        )
    )
    claim = await service.claim(AgentClaimRequest(job_id=prepared.job_id))
    assert claim.package is not None
    block = claim.package.blocks[0]
    request = AgentSubmitRequest(
        job_id=prepared.job_id,
        package_id=claim.package.package_id,
        lease_token=claim.package.lease_token,
        blocks=[
            AgentBlockSubmission(
                block_id=block.block_id,
                translation=_translate(block.masked_source),
            )
        ],
    )
    await service.submit(request)
    changed = request.model_copy(
        update={
            "blocks": [AgentBlockSubmission(block_id=block.block_id, translation="不同的译文。")]
        }
    )
    with pytest.raises(Exception, match="different text"):
        await service.submit(changed)


@pytest.mark.asyncio
async def test_prepare_rejects_provider_job_and_completed_agent_job(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(input_path=str(source), job_id="agent_completed_guard")
    )
    db_path = tmp_path / "db" / f"{prepared.job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.save_checkpoints_batch(
            [
                {
                    "block_id": block.id,
                    "status": BlockStatus.MTQE_PASSED,
                    "target_text": "完成",
                }
                for block in ledger.get_all_blocks(prepared.job_id)
            ],
            job_id=prepared.job_id,
        )
        ledger.finalize_job(prepared.job_id, status="completed")
    with pytest.raises(Exception, match="already completed"):
        await service.prepare(AgentPrepareRequest(input_path=str(source), job_id=prepared.job_id))

    provider = _write_book(tmp_path)
    with SQLiteJobLedger(tmp_path / "db" / "provider_job.sqlite") as ledger:
        from ubt.core.ir.models import BookManifest

        ledger.init_job_from_manifest(
            "provider_job",
            BookManifest(doc_id="provider_job", title="t", source_path=str(provider)),
        )
        ledger.set_job_metadata_value("provider_job", "translation_mode", "provider")
    with pytest.raises(Exception, match="provider translation job"):
        await service.prepare(AgentPrepareRequest(input_path=str(provider), job_id="provider_job"))


@pytest.mark.asyncio
async def test_layout_options_are_persisted_and_used(tmp_path: Path) -> None:
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(
            input_path=str(source),
            job_id="agent_layout",
            dual_mode="dual",
            layout_columns=2,
        )
    )
    with SQLiteJobLedger(tmp_path / "db" / f"{prepared.job_id}.sqlite") as ledger:
        agent_config = ledger.get_job_metadata_value(prepared.job_id, "agent_config")
        render_plan = ledger.get_job_metadata_value(prepared.job_id, "render_plan")
    assert isinstance(agent_config, dict)
    assert agent_config["dual_mode"] == "dual"
    assert agent_config["layout_columns"] == 2
    assert isinstance(render_plan, dict)
    assert render_plan["bilingual_mode"] == "bilingual"


def test_agent_session_survives_as_awaiting_agent_in_catalog(tmp_path: Path) -> None:
    from ubt.api.job_catalog import summarize_job

    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = asyncio.run(
        service.prepare(AgentPrepareRequest(input_path=str(source), job_id="agent_catalog"))
    )
    summary = summarize_job(tmp_path / "db" / f"{prepared.job_id}.sqlite")
    assert summary is not None
    assert summary["status"] == "awaiting_agent"


@pytest.mark.asyncio
async def test_submitted_packages_compact_and_replay_reports_their_count(
    tmp_path: Path,
) -> None:
    """A submitted package keeps its identity, not its per-block payload.

    Every claim and submit rewrites the whole ``agent_session`` metadata, so a
    package that kept ``block_ids`` and ``expected_statuses`` made each write
    re-serialize every prior package's blocks — O(n²) in blocks across a run.
    Compaction drops exactly those two fields at submit time; everything the
    replay surface reads (``state``, ``lease_token_hash``, ``submission_hash``)
    stays, and ``block_count`` preserves the count the replay report shows.
    A two-block package pins that count instead of stumbling on a 1-block
    coincidence.
    """
    source = _write_book(tmp_path)
    service = AgentTranslationService(config=_config(tmp_path))
    prepared = await service.prepare(
        AgentPrepareRequest(input_path=str(source), job_id="agent_compact", package_size=2)
    )
    job_id = prepared.job_id
    accepted: list[int] = []
    for _ in range(2):
        claim = await service.claim(AgentClaimRequest(job_id=job_id))
        assert claim.package is not None
        request = AgentSubmitRequest(
            job_id=job_id,
            package_id=claim.package.package_id,
            lease_token=claim.package.lease_token,
            blocks=[
                AgentBlockSubmission(
                    block_id=block.block_id,
                    translation=_translate(block.masked_source),
                )
                for block in claim.package.blocks
            ],
        )
        result = await service.submit(request)
        accepted.append(result.accepted_blocks)
        # The replay must survive compaction: same request, same verdict.
        replay = await service.submit(request)
        assert replay.idempotent_replay is True
        assert replay.accepted_blocks == result.accepted_blocks

    assert accepted == [2, 2]
    with SQLiteJobLedger(tmp_path / "db" / f"{job_id}.sqlite") as ledger:
        session = ledger.get_job_metadata_value(job_id, AGENT_SESSION_KEY)
        assert isinstance(session, dict), "the agent session must survive compaction"
        packages = session["packages"]
        assert len(packages) == 2
        for package in packages.values():
            assert package["state"] == "submitted"
            assert "block_ids" not in package, "the spent per-block payload must be gone"
            assert "expected_statuses" not in package
            assert package["block_count"] == 2
            assert package["lease_token_hash"], "replay needs the stored token digest"
            assert package["submission_hash"], "replay needs the stored submission digest"
        # And the metadata the next claim rewrites is payload-free: the raw
        # JSON carries no per-block list for any submitted package.
        import json as _json

        raw = _json.dumps(session, ensure_ascii=False)
        assert "block_ids" not in raw
