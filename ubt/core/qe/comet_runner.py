"""Local zero-token MTQE scoring runner with subprocess air-gapping.

Honesty note: the heuristic runner does NOT produce a calibrated
quality score. It maps a deterministic fast-pass rejection reason — plus the
terminology signal below — to a **discrete defect class** (see
``QE_DEFECT_CLASS_LEGEND``): a routing signal, not a measurement. The ``0.92``
"pass" value carries no quality information beyond "no deterministic invariant
failed". Downstream surfaces (report, progress events) must present it as such;
a calibrated neural score requires the CometKiwi L2 runner
(``SubprocessQERunner``) or a separate calibration track.

A fluent translation can use the WRONG enforced glossary term and
pass every structural gate. The heuristic path therefore also checks the
translation against the run's glossary and classifies a violation as
``QE_SCORE_GLOSSARY_VIOLATION`` (:data:`GLOSSARY_VIOLATION_MARKER`), which the
quality gate refuses to auto-pass via ``STRUCTURAL_DEFECT_MARKERS``.
"""

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from ubt.core.env import subprocess_env
from ubt.core.exceptions import MTQEEvaluationError
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.defect_taxonomy import ECHO_MARKERS
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.validators.consistency import GlossaryConsistencyValidator

logger = logging.getLogger(__name__)

#: Non-JSON stdout lines to skip on the resident path before declaring desync.
#: Neural loaders print progress there; the one-shot parser already tolerates it.
_MAX_RESIDENT_CHATTER_LINES = 100

# The exact score values the heuristic emits, and what each band
# actually means. Twelve discrete values — nothing between them is reachable.
# Single source of truth: the named constants below feed BOTH
# this legend and ``HeuristicQERunner.score_from_decision_reason`` so the two
# can never drift apart.
QE_SCORE_PASS = 0.92
QE_SCORE_STRUCTURAL_OTHER = 0.70
QE_SCORE_LENGTH = 0.60
QE_SCORE_NUMERIC = 0.55
QE_SCORE_SCRIPT_DENSITY = 0.40
QE_SCORE_OMISSION = 0.35
QE_SCORE_HTML_DELTA = 0.30
QE_SCORE_GLOSSARY_VIOLATION = 0.25
QE_SCORE_REPETITION = 0.20
QE_SCORE_FABRICATED = 0.15
QE_SCORE_LEAK = 0.10
QE_SCORE_EMPTY = 0.0

#: Error-flag fragment for a terminology violation. Registered in
#: ``ubt.core.qe.defect_taxonomy.STRUCTURAL_DEFECT_MARKERS`` so the quality
#: gate's existing ``has_structural_defect`` check keeps such a block out of
#: auto-pass even when a neural runner returns a high score.
GLOSSARY_VIOLATION_MARKER = "Glossary term violation"

QE_DEFECT_CLASS_LEGEND: tuple[tuple[float, str], ...] = (
    (QE_SCORE_PASS, "pass — no deterministic invariant failed (NOT a quality estimate)"),
    (QE_SCORE_STRUCTURAL_OTHER, "other structural rejection (no specific classifier matched)"),
    (QE_SCORE_LENGTH, "suspicious length ratio (truncated or inflated target)"),
    (QE_SCORE_NUMERIC, "numeric fidelity failure (missing/altered figures)"),
    (QE_SCORE_SCRIPT_DENSITY, "script density anomaly (target not in expected script)"),
    (
        QE_SCORE_OMISSION,
        "omission suspected (dropped sentences / missing identifier terms / truncated residue)",
    ),
    (QE_SCORE_HTML_DELTA, "HTML tag delta failure (markup corrupted or dropped)"),
    (
        QE_SCORE_GLOSSARY_VIOLATION,
        "enforced glossary term dropped or altered in the target (terminology violation)",
    ),
    (QE_SCORE_REPETITION, "repetition-loop hallucination"),
    (QE_SCORE_FABRICATED, "untranslated residue / fabricated content vs empty source"),
    (QE_SCORE_LEAK, "prompt-template leak / empty-target via rejection path"),
    (QE_SCORE_EMPTY, "empty target (short-circuit in score_pairs)"),
)


def glossary_violation_flag(
    source_text: str,
    target_text: str,
    validator: GlossaryConsistencyValidator | None,
) -> str | None:
    """Terminology defect flag for a target that drops/alters an enforced term.

    Returns ``None`` when the check cannot run (no glossary) or the target
    carries every expected rendering. Detection is delegated to the
    export-side validator so QE and export can never disagree about what a
    violation is; building the validator once and passing it in keeps the
    per-block cost a single pass over the sorted glossary.
    """
    if validator is None:
        return None
    result = validator.validate(source_text, target_text)
    if result.is_valid:
        return None
    return f"{GLOSSARY_VIOLATION_MARKER}: {result.message}"


async def _reap_subprocess(proc: asyncio.subprocess.Process) -> None:
    """Kill the scorer subprocess if still alive, then reap it."""
    with suppress(ProcessLookupError):
        proc.kill()  # already exited → nothing to collect
    with suppress(Exception):
        await proc.wait()  # cleanup must never mask the triggering error


class _ResidentRestart(Exception):
    """Retryable resident-session failure (kill the scorer and re-attempt)."""


class _ResidentDown(Exception):
    """Resident session unusable; the caller falls back to per-call spawn."""


class SubprocessQERunner(BaseQERunner):
    """Executes CometKiwi or xTOWER in an isolated subprocess via JSON IPC.

    The IPC reply is ``{"scores": [...], "engine": "neural"|"heuristic_fallback"}``:
    the subprocess falls back to the deterministic heuristic when
    ``unbabel-comet``/``torch`` are unimportable, and a fallback batch must
    never masquerade as a calibrated neural score. ``is_calibrated`` flips
    accordingly so the repair loop stops ranking candidates by fake numbers.
    """

    def __init__(
        self,
        python_bin: Path,
        script_path: Path,
        model_name: str = "Unbabel/wmt22-cometkiwi-da",
        timeout_seconds: int = 300,
        idle_reap_seconds: float = 300.0,
    ) -> None:
        self.python_bin = python_bin
        self.script_path = script_path
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        # Idle time after which the resident scorer is torn down on the next
        # use (weights cost GBs of RAM; a queue that pauses should not hold
        # them hostage). Reaped lazily -- the process also dies with its
        # stdin pipe, so a parent exit never orphans it.
        self.idle_reap_seconds = idle_reap_seconds
        # Engine reported by the most recent IPC reply; None until the first
        # batch is scored.
        self._last_engine: str | None = None
        self._resident: asyncio.subprocess.Process | None = None
        self._resident_lock = asyncio.Lock()
        self._req_seq = 0
        self._last_used = 0.0
        self._reaper: asyncio.Task[None] | None = None
        # Set after the resident protocol fails twice in one request: the
        # rest of the run falls back to per-call invocation (also manual
        # rollback: UBT_COMET_RESIDENT=0 disables residency from the start).
        self._resident_broken = False
        if model_name.startswith("Unbabel/wmt22-cometkiwi"):
            # The default QE weights are CC-BY-NC-SA-4.0 (non-commercial),
            # which clashes with the commercial (KDP) delivery this gate is
            # used in. Warn at selection time; the qe_engine default
            # remains the zero-dependency heuristic runner.
            logger.warning(
                "QE model '%s' is licensed CC-BY-NC-SA-4.0 (non-commercial). "
                "Commercial deliveries gated on its scores should switch "
                "UBT_COMET_MODEL to a commercially usable scorer first.",
                model_name,
            )

    @property
    def last_engine(self) -> str | None:
        """Engine reported by the most recent IPC reply (None = not scored yet)."""
        return self._last_engine

    def is_calibrated(self) -> bool:
        """False until a batch actually reports the neural engine.

        The subprocess can silently downgrade when its COMET/torch import
        fails; ``repair_loop`` must not best-of-n-rerank candidates ranked by
        twelve discrete heuristic bands pretending to be CometKiwi. "Not scored
        yet" is not evidence of calibration — treating it as such let the first
        rerank of a run rank on scores the gate had not measured.
        """
        return self._last_engine == "neural"

    def reset_residency(self) -> None:
        """Re-enable the resident scorer for a new run (see BaseQERunner)."""
        self._resident_broken = False

    def _coerce_reply(self, decoded: Any, stdout_str: str) -> tuple[list[float], str]:
        """Extract (scores, engine) from a decoded IPC reply.

        Shared by the per-call object payload and the resident JSON-lines
        envelope. Unlabelled bare-array replies cannot prove neural provenance,
        so they are reported as ``heuristic_fallback`` rather than calibrated.
        """
        if isinstance(decoded, dict):
            raw_scores = decoded.get("scores")
            engine = str(decoded.get("engine", "unknown"))
            if not isinstance(raw_scores, list):
                raise MTQEEvaluationError(
                    f"QE subprocess 'scores' field is not a JSON list: {stdout_str[:200]}",
                    details={"stdout": stdout_str},
                )
        elif isinstance(decoded, list):
            # Unlabelled array protocol: scorer output lacks engine label; calibration unknown.
            raw_scores = decoded
            engine = "heuristic_fallback"
            logger.warning(
                "QE subprocess returned unlabelled array protocol; "
                "treating scores as uncalibrated (best-of-n rerank disabled). "
                "Upgrade the scorer script to emit {'scores', 'engine'}."
            )
        else:
            raise MTQEEvaluationError(
                f"QE subprocess output is not a JSON object or list: {stdout_str[:200]}",
                details={"stdout": stdout_str},
            )
        return [round(float(s), 4) for s in raw_scores], engine

    def _parse_ipc_output(self, stdout_str: str) -> tuple[list[float], str]:
        """Parse a one-shot IPC reply into (scores, engine)."""
        decoded: Any = None
        decode_error: json.JSONDecodeError | None = None
        try:
            decoded = json.loads(stdout_str)
        except json.JSONDecodeError as err:
            decode_error = err
            # Tolerate stray chatter on stdout (torch writes progress there):
            # retry on the outermost JSON object.
            start_idx = stdout_str.find("{")
            end_idx = stdout_str.rfind("}")
            if start_idx != -1 and end_idx > start_idx:
                try:
                    decoded = json.loads(stdout_str[start_idx : end_idx + 1])
                except json.JSONDecodeError:
                    decoded = None
        if decode_error is not None and decoded is None:
            raise MTQEEvaluationError(
                f"Failed to parse QE subprocess output as JSON: {stdout_str[:200]}",
                details={"stdout": stdout_str},
            )
        return self._coerce_reply(decoded, stdout_str)

    def _note_engine(self, engine: str, pair_count: int, stderr_tail: str = "") -> None:
        """Record the reported engine and warn on any non-neural batch."""
        self._last_engine = engine
        if engine != "neural":
            logger.warning(
                "QE subprocess scored %d pair(s) with engine=%s — these are "
                "discrete heuristic bands, not calibrated CometKiwi scores. "
                "Install unbabel-comet + torch in the scorer environment: %s",
                pair_count,
                engine,
                stderr_tail,
            )

    def _schedule_idle_reap(self) -> None:
        """(Re)arm the background reaper that frees idle resident GPU/RAM."""
        if self._reaper is not None and not self._reaper.done():
            self._reaper.cancel()
        self._reaper = asyncio.get_running_loop().create_task(self._reap_when_idle())

    async def _reap_when_idle(self) -> None:
        await asyncio.sleep(self.idle_reap_seconds)
        async with self._resident_lock:
            idle = asyncio.get_running_loop().time() - self._last_used
            if self._resident is not None and idle >= self.idle_reap_seconds - 1.0:
                await self._teardown_resident()

    def _resident_enabled(self) -> bool:
        return not self._resident_broken and os.getenv("UBT_COMET_RESIDENT", "1").lower() not in (
            "0",
            "false",
            "no",
        )

    async def _teardown_resident(self, proc: asyncio.subprocess.Process | None = None) -> None:
        """Best-effort quit+reap of the resident scorer (never raises)."""
        proc = proc if proc is not None else self._resident
        self._resident = None
        if proc is None:
            return
        with suppress(Exception):
            if proc.stdin is not None and proc.returncode is None:
                proc.stdin.write(json.dumps({"cmd": "quit"}).encode() + b"\n")
                await proc.stdin.drain()
        with suppress(Exception):
            await asyncio.wait_for(_reap_subprocess(proc), timeout=5.0)

    async def aclose(self) -> None:
        """Stop the resident scorer and cancel its idle-reap task."""
        reaper = self._reaper
        self._reaper = None
        if reaper is not None and not reaper.done():
            reaper.cancel()
            with suppress(asyncio.CancelledError):
                await reaper
        async with self._resident_lock:
            await self._teardown_resident()

    async def _ensure_resident(self) -> asyncio.subprocess.Process:
        """Return a live resident scorer, spawning one when needed."""
        proc = self._resident
        now = asyncio.get_running_loop().time()
        if (
            proc is not None
            and proc.returncode is None
            and now - self._last_used <= self.idle_reap_seconds
        ):
            return proc
        await self._teardown_resident()
        proc = await asyncio.create_subprocess_exec(
            str(self.python_bin),
            str(self.script_path),
            "--serve",
            "--model",
            self.model_name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Discarded, not piped: nothing on the resident path ever reads
            # stderr, so a piped child that fills the ~64 KiB buffer (model
            # download progress, torch warnings) blocks in write() and stops
            # answering -- the parent then stalls until the 300 s request
            # timeout. The per-call path still captures stderr for diagnostics.
            stderr=asyncio.subprocess.DEVNULL,
            env=subprocess_env(),
        )
        self._resident = proc
        return proc

    async def _resident_score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        """Score via the long-lived scorer; one transparent restart per request.

        Protocol desync (unparseable line, id mismatch) tears the process
        down and retries once with a fresh one — a second failure marks
        residency broken for the rest of the run so ``score_pairs`` falls
        back to the historical per-call path.
        """
        for attempt in (1, 2):
            proc: asyncio.subprocess.Process | None = None
            try:
                try:
                    proc = await self._ensure_resident()
                except Exception as err:
                    # Spawn failed (missing script/python): retrying a
                    # per-call spawn surfaces the real error there.
                    raise _ResidentDown(f"scorer spawn failed: {err}") from err
                self._req_seq += 1
                req_id = self._req_seq
                if proc.stdin is None or proc.stdout is None:
                    raise _ResidentRestart("scorer pipes unavailable")
                proc.stdin.write(
                    (json.dumps({"id": req_id, "pairs": pairs}, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                )
                await proc.stdin.drain()
                reply: Any = None
                # ONE deadline for the whole reply. Re-arming wait_for per line
                # would multiply the bound by the chatter cap, so a scorer that
                # trickled junk could delay desync detection far past
                # timeout_seconds; every line now shares the single budget.
                deadline = asyncio.get_running_loop().time() + self.timeout_seconds
                for _ in range(_MAX_RESIDENT_CHATTER_LINES):
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise _ResidentRestart("resident scorer reply deadline exceeded")
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                    if not line:
                        raise _ResidentRestart("scorer closed stdout (crashed?)")
                    try:
                        reply = json.loads(line.decode("utf-8", errors="replace").strip())
                    except json.JSONDecodeError:
                        # torch/neural loaders print progress to stdout; skip the
                        # chatter instead of tearing the session down (the one-shot
                        # parser was already hardened against exactly this).
                        logger.debug("resident scorer chatter: %r", line[:120])
                        continue
                    break
                if not isinstance(reply, dict) or reply.get("id") != req_id:
                    raise _ResidentRestart(f"reply id mismatch: {reply}")
                if "error" in reply:
                    # The session is healthy (the envelope decoded and matched);
                    # only this request failed -- surface it, do not restart.
                    raise MTQEEvaluationError(
                        f"Resident QE scorer rejected request: {reply['error']}",
                        details={"reply": reply},
                    )
                scores, engine = self._coerce_reply(reply, str(reply)[:200])
                if len(scores) != len(pairs):
                    raise MTQEEvaluationError(
                        f"Resident QE scorer returned {len(scores)} scores for "
                        f"{len(pairs)} pairs",
                        details={"expected": len(pairs), "got": len(scores)},
                    )
                self._last_used = asyncio.get_running_loop().time()
                self._note_engine(engine, len(pairs))
                self._schedule_idle_reap()
                return scores
            except _ResidentRestart as err:
                await self._teardown_resident(proc)
                if attempt == 2:
                    raise _ResidentDown(str(err)) from err
            except TimeoutError as err:
                await self._teardown_resident(proc)
                raise _ResidentDown(
                    f"resident scorer timed out after {self.timeout_seconds}s"
                ) from err
            except MTQEEvaluationError:
                raise
            except Exception as err:
                # Dead child surfacing as BrokenPipeError/pipe EOF mid-request:
                # the session is unusable; restart once, then give up.
                await self._teardown_resident(proc)
                if attempt == 2:
                    raise _ResidentDown(f"resident session error: {err}") from err
        raise AssertionError("unreachable")

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        """Score one batch, preferring the resident scorer.

        Model weights dominate per-call cost: the one-shot path reloads the
        checkpoint for every QE batch. The resident ``--serve`` session loads
        once and answers JSON-lines requests; the per-call path remains as the
        failure fallback and the documented rollback (UBT_COMET_RESIDENT=0).
        """
        if not pairs:
            return []
        if self._resident_enabled():
            try:
                async with self._resident_lock:
                    return await self._resident_score_pairs(pairs)
            except _ResidentDown as err:
                self._resident_broken = True
                logger.warning(
                    "Resident QE scorer unavailable (%s); this run falls back "
                    "to per-call subprocess scoring (UBT_COMET_RESIDENT=0 to "
                    "select it from the start).",
                    err,
                )
            except MTQEEvaluationError:
                # A per-request error envelope (``{"id": N, "error": ...}``) is
                # raised only after the reply decoded and its id matched, so the
                # session is in sync and only this request failed. Propagate it
                # without discarding the multi-GB resident checkpoint.
                raise
            except BaseException:
                # A cancellation mid-request leaves the reply in the pipe;
                # drop the session so a later request does not read the stale
                # line first. Synchronous kill only -- awaits can be
                # re-cancelled on this path, and the child exits at stdin EOF.
                proc = self._resident
                self._resident = None
                if proc is not None:
                    with suppress(ProcessLookupError):
                        proc.kill()
                raise
        return await self._score_via_spawn(pairs)

    async def _score_via_spawn(self, pairs: list[dict[str, str]]) -> list[float]:
        if not pairs:
            return []

        cmd = [
            str(self.python_bin),
            str(self.script_path),
            "--score-stdin",
            "--model",
            self.model_name,
        ]

        payload = json.dumps(pairs, ensure_ascii=False).encode("utf-8")

        # Bound before the try: if ``create_subprocess_exec`` itself raises (the
        # scorer script or its venv python is missing), every handler below
        # would otherwise reap an unbound name and replace the real error with
        # an UnboundLocalError.
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # H-2: the scorer needs HF_HOME/TORCH_*/PATH, not this host's keys.
                env=subprocess_env(),
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=payload),
                timeout=self.timeout_seconds,
            )

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace")[-1000:]
                raise MTQEEvaluationError(
                    f"Subprocess QE failed with exit code {proc.returncode}: {err_msg}",
                    details={"returncode": proc.returncode, "stderr": err_msg},
                )

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            scores, engine = self._parse_ipc_output(stdout_str)
            if len(scores) != len(pairs):
                # A third-party scorer need not honour the count; a short reply
                # silently misaligned candidates downstream (IndexError in the
                # repair rerank). Fail loudly here instead.
                raise MTQEEvaluationError(
                    f"QE subprocess returned {len(scores)} scores for {len(pairs)} pairs",
                    details={"expected": len(pairs), "got": len(scores)},
                )
            self._note_engine(
                engine,
                len(pairs),
                stderr.decode("utf-8", errors="replace")[-400:],
            )
            return scores

        except TimeoutError as err:
            if proc is not None:
                await _reap_subprocess(proc)
            raise MTQEEvaluationError(
                f"Subprocess QE timed out after {self.timeout_seconds}s",
                details={"timeout": self.timeout_seconds},
            ) from err
        except Exception as err:
            if proc is not None:
                await _reap_subprocess(proc)
            if isinstance(err, MTQEEvaluationError):
                raise
            raise MTQEEvaluationError(
                f"Failed to execute QE runner: {err}",
                details={"error": str(err)},
            ) from err
        except BaseException:
            # ``CancelledError`` derives from BaseException, so neither handler
            # above sees it: without this the scorer subprocess outlived the
            # cancelled task and kept holding its model weights in GPU/RAM.
            if proc is not None:
                await _reap_subprocess(proc)
            raise


class MockQERunner(BaseQERunner):
    """Deterministic fast mock runner for unit and integration testing without PyTorch."""

    def __init__(self, default_score: float = 0.85) -> None:
        self.default_score = default_score

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        scores: list[float] = []
        for p in pairs:
            src = p.get("src", "")
            mt = p.get("mt", "")
            # Deterministic variation based on length matching
            if not mt:
                scores.append(0.0)
            elif "<issues>" in mt:
                scores.append(0.35)
            else:
                ratio = min(len(mt), len(src)) / max(1, max(len(mt), len(src)))
                score = min(1.0, max(0.5, self.default_score * (0.8 + 0.2 * ratio)))
                scores.append(round(score, 4))
        return scores


class HeuristicQERunner(BaseQERunner):
    """Deterministic zero-token heuristic QE runner based on structural, lexical, and script invariants.

    Provides a production-grade default score (0.0 ~ 1.0) without requiring PyTorch or HuggingFace models.

    the score is a **discrete defect-class proxy** (see
    ``QE_DEFECT_CLASS_LEGEND``), not a calibrated quality estimate — there
    are only twelve reachable values and the gaps between them are
    meaningless. Use it for routing/repair triggering; use the L2
    ``SubprocessQERunner`` (CometKiwi) when an actual quality number is
    needed.

    When constructed with the run's ``glossary``, a target that drops or
    alters an enforced term scores ``QE_SCORE_GLOSSARY_VIOLATION`` instead of
    the pass band.
    """

    def __init__(
        self,
        target_lang: str = "zh",
        source_lang: str = "en",
        glossary: list[dict[str, Any]] | None = None,
    ) -> None:
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.fast_pass = FastPassFilter(source_lang=source_lang, target_lang=target_lang)
        self._glossary = glossary or None
        self._glossary_validator = (
            GlossaryConsistencyValidator(glossary=glossary) if glossary else None
        )

    def with_languages(self, source_lang: str, target_lang: str) -> "HeuristicQERunner":
        if (self.source_lang, self.target_lang) == (source_lang, target_lang):
            return self
        return HeuristicQERunner(
            target_lang=target_lang,
            source_lang=source_lang,
            glossary=self._glossary,
        )

    def with_glossary(self, glossary: list[dict[str, Any]] | None) -> "HeuristicQERunner":
        """Bind the run's enforced terminology once the bible stage has it.

        The orchestrator constructs the runner before the glossary exists;
        this rebinds it so the repair loop's re-score sees a term violation
        instead of passing the repaired text at 0.92.
        """
        if not glossary or glossary is self._glossary:
            return self
        return HeuristicQERunner(
            target_lang=self.target_lang,
            source_lang=self.source_lang,
            glossary=glossary,
        )

    def is_glossary_aware(self) -> bool:
        """True once a glossary is bound, so a term violation caps the score."""
        return self._glossary_validator is not None

    def is_calibrated(self) -> bool:
        """Always False: the twelve legend bands are a routing class, not a
        graded quality estimate, so they must never rank repair candidates."""
        return False

    def glossary_violation(self, source_text: str, target_text: str) -> str | None:
        """Terminology defect flag for this pair, or ``None`` (granularity-free)."""
        return glossary_violation_flag(source_text, target_text, self._glossary_validator)

    @staticmethod
    def score_from_decision_reason(reason: str) -> float:
        """Map a fast-pass structural rejection reason to its defect class.

        despite the "score" name this is a classification, not a
        graded measurement — each rejection reason maps to one of the fixed
        bands in ``QE_DEFECT_CLASS_LEGEND``.
        """
        r = reason.lower()
        if "empty" in r or "leak" in r:
            return QE_SCORE_LEAK
        if any(marker.lower() in r for marker in ECHO_MARKERS):
            # Same class as the fabricated-content band. Scoring a
            # near-verbatim untranslated target as QE_SCORE_STRUCTURAL_OTHER
            # (0.70) would leave it close enough to the default threshold to be
            # auto-passed, so it maps to the fabricated band instead.
            return QE_SCORE_FABRICATED
        if "hallucination" in r:
            return QE_SCORE_REPETITION
        if "added reference" in r or "fabricated" in r or "prompt scaffold" in r:
            # Fabricated content the source never had, including fabricated
            # citations and equation numbers. Without this branch the reason falls through
            # to 0.70 ("no specific classifier matched"), which is only 0.05
            # below the default threshold and reads as a generic structural
            # problem in the report's score distribution.
            # NOTE: scaffold *headings* ("Prompt scaffold leaked into the
            # target...") intentionally stay in the leak band above — see
            # test_d2_echo_regression. Only non-"leak" scaffold phrasings
            # reach this branch.
            return QE_SCORE_FABRICATED
        if "html" in r:
            return QE_SCORE_HTML_DELTA
        if "omission" in r:
            return QE_SCORE_OMISSION
        if "script density" in r:
            return QE_SCORE_SCRIPT_DENSITY
        if "numeric" in r:
            return QE_SCORE_NUMERIC
        if "truncated" in r or "inflated" in r:
            return QE_SCORE_LENGTH
        return QE_SCORE_STRUCTURAL_OTHER

    @classmethod
    def score_from_flags(cls, flags: Sequence[str]) -> float:
        """Defect class for a block's error flags, terminology included.

        A glossary violation caps the class at ``QE_SCORE_GLOSSARY_VIOLATION``
        so it can never reach the auto-pass band; an already-lower structural
        class (leak 0.10, fabrication 0.15, repetition 0.20) keeps its existing
        value, so every non-terminology condition is unchanged.
        """
        score = cls.score_from_decision_reason(flags[0] if flags else "")
        if any(GLOSSARY_VIOLATION_MARKER in flag for flag in flags):
            return min(score, QE_SCORE_GLOSSARY_VIOLATION)
        return score

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        scores: list[float] = []
        for p in pairs:
            src = p.get("src", "").strip()
            mt = p.get("mt", "").strip()
            if not mt:
                scores.append(QE_SCORE_EMPTY)
                continue
            if not src:
                # Non-empty MT against an empty source is fabricated content,
                # not a flawless translation — score it like untranslated residue.
                scores.append(QE_SCORE_FABRICATED)
                continue

            # Untranslated identical prose is decided in one place:
            # ``FastPassFilter.evaluate`` below rejects the echo and
            # ``score_from_decision_reason`` maps it to the fabricated band.
            decision = self.fast_pass.evaluate(src, mt)
            if not decision.passed:
                scores.append(self.score_from_decision_reason(decision.reason))
                continue
            # Passing every structural gate is not enough — the target must
            # also carry the enforced glossary renderings.
            if self.glossary_violation(src, mt) is not None:
                scores.append(QE_SCORE_GLOSSARY_VIOLATION)
            else:
                scores.append(QE_SCORE_PASS)
        return scores
