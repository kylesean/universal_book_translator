# AGENTS.md — Testing & Verification Standard

## 1. Universal Verification Invariants

- **No Post-Hoc Tests**: Never write tests after production code. Tests written after-the-fact only mirror the implementation.
- **Failure-First Specification**: Before implementing complex or branching logic, enumerate edge cases and failure modes directly as FAILING test assertions (RED). Production code exists only to satisfy these checks (GREEN).
- **E2E as Ground Truth**: Do not rely solely on unit tests with heavy mocking. Every feature must culminate in an end-to-end verification step that produces a verifiable, repeatable artifact (file, persistent DB state, or serialized response).
- **Unit Tests for Dense Logic Only**: Use isolated tests exclusively for pure computation, data transformation, and edge-case error branches that are prohibitively slow or flaky to orchestrate in E2E.
- **No Silent Bargaining**: Never relax an assertion, widen tolerances, or raise timeouts to make a failure disappear. If a test looks flawed or contradictory: (1) freeze and record the failing assertion plus a minimal repro; (2) fix the test through the same RED-to-GREEN path with evidence (a second failing case or an E2E artifact), never by editing the expectation to match the output; (3) log the rationale in the test comment or PR. Emit `[HALT_TEST_CONTRADICTION: <path> - <rationale>]` only when the spec itself is self-contradictory and no explicit user override exists; an explicit user instruction to continue overrides the halt.

## 2. Project Harness & Execution

- **Fast Inner-Loop (< 5s)**: `uv run pytest -m fast -q`
- **Final E2E Verification**: `uv run pytest tests/baselines tests/e2e -q`
- **Expected Artifact**: a finalized `job_meta` row plus terminal `blocks` rows in `<job_id>.sqlite`, plus the bilingual artifact on disk and `*_quality_report.json`
