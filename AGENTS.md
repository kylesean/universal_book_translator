# AGENTS.md — Testing & Verification Standard

## 1. Universal Verification Invariants

- **No Post-Hoc Tests**: Never write tests after production code. Tests written after-the-fact only mirror the implementation.
- **Failure-First Specification**: Before implementing complex or branching logic, enumerate edge cases and failure modes directly as FAILING test assertions (RED). Production code exists only to satisfy these checks (GREEN).
- **E2E as Ground Truth**: Do not rely solely on unit tests with heavy mocking. Every feature must culminate in an end-to-end verification step that produces a verifiable, repeatable artifact (file, persistent DB state, or serialized response).
- **Unit Tests for Dense Logic Only**: Use isolated tests exclusively for pure computation, data transformation, and edge-case error branches that are prohibitively slow or flaky to orchestrate in E2E.
- **One Behavior, One Home**: Name a test file for the module or behaviour it pins (``test_<module>.py``), never for the review activity that produced it. Do **not** add ``test_review_<date>_*``, ``test_roundN_*`` or ``*_fixes.py`` files: a dated file scatters one module's coverage across every review that touched it, so the next reviewer cannot see what is already covered and re-tests it. A defect's regression case goes into that module's canonical file; only a behaviour with no canonical file gets a new, module-named one. Rationale and the frozen legacy exceptions: `tests/unit/regressions/README.md`.
- **Guard Optional Dependencies**: A test that needs a package outside the `dev` extra (numpy / rapidocr / torch / transformers / docling …) must `pytest.importorskip` it, or stub the import boundary — so the default `uv sync --extra dev` matrix stays green and a test never depends on the maintainer's machine. A dependency-guarded test must not be marked `fast` unless it still runs unguarded.
- **No Silent Bargaining**: Never relax an assertion, widen tolerances, or raise timeouts to make a failure disappear. If a test looks flawed or contradictory: (1) freeze and record the failing assertion plus a minimal repro; (2) fix the test through the same RED-to-GREEN path with evidence (a second failing case or an E2E artifact), never by editing the expectation to match the output; (3) log the rationale in the test comment or PR. Emit `[HALT_TEST_CONTRADICTION: <path> - <rationale>]` only when the spec itself is self-contradictory and no explicit user override exists; an explicit user instruction to continue overrides the halt.

## 2. Project Harness & Execution

- **Fast Inner-Loop (< 5s)**: `uv run pytest -m fast -q`
- **Local Gate (active)**: run `uv run pre-commit install` once; lint/format then run on every commit, and `mypy --strict ubt tests` + `pytest -m fast -q` run on every push. Remote CI is intentionally parked in `.github/workflows.disabled/` while the repo is private and iterating fast — the phased rollout is documented in `docs/guides/CI_AND_QUALITY_GATES.md`. Do not re-enable CI that is not already green locally.
- **Final E2E Verification**: `uv run pytest tests/baselines tests/e2e -q`
- **Expected Artifact**: a finalized `job_meta` row plus terminal `blocks` rows in `<job_id>.sqlite`, plus the bilingual artifact on disk and `*_quality_report.json`

## 3. Evidence-First Verification Protocol

- **Red-Phase Proof**: When executing Failure-First steps, you must execute the test against current code and output the terminal trace confirming failure (Exit Code != 0) BEFORE touching production code.
- **Green-Phase Proof**: Before declaring completion, you must execute <FAST_TEST_CMD> and <E2E_TEST_CMD>. Your final response MUST display the raw test runner console stdout showing 100% pass rate (Exit Code == 0). Textual claims without raw CLI execution traces are strictly invalid.
