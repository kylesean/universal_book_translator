# Contributing to Universal Book Translator (UBT)

Thank you for your interest in contributing to UBT! We welcome contributions from developers, researchers, and book translation enthusiasts.

This guide outlines our development workflow, coding standards, and how to submit contributions.

---

## Getting Started

### Prerequisites
- **Python**: `>= 3.12` (Python 3.12+ is required)
- **uv**: Recommended fast package and project manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Typst**: Pinned binary `0.15.1` on `PATH` for PDF rendering tests
- **Node.js & pnpm**: Only required if developing or rebuilding the Web Console frontend (`web/`)

### Setting Up the Environment

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/universal_book_translator.git
   cd universal_book_translator
   ```

2. Install dependencies with `uv`:
   ```bash
   # Development environment (linting, type checking, unit tests, fast tier)
   uv sync --extra dev

   # Optional extras for PDF and OCR development
   uv sync --all-extras
   ```

3. (Optional) Set up pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

---

## Development Workflow

### Code Style and Quality Checks
UBT enforces strict code quality gates in CI:

1. **Linting and Formatting (Ruff)**:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   # To auto-format:
   uv run ruff format .
   ```

2. **Strict Type Checking (Mypy)**:
   ```bash
   uv run mypy --strict ubt tests
   ```

3. **Compiler Layering Invariant**:
   UBT enforces strict architectural boundaries (no reverse imports from `core` into `adapters` or `render`):
   ```bash
   uv run python scripts/dump_compiler_edges.py
   ```

### Running Tests

We split tests into fast deterministic unit tests and slower end-to-end integration tests:

1. **Fast Unit Tests** (runs in seconds with coverage check):
   ```bash
   uv run pytest -m fast -q
   ```

2. **Running Specific Test Files**:
   ```bash
   uv run pytest tests/unit/api/test_review_endpoints.py
   ```

3. **Running the Full Suite**:
   ```bash
   uv run pytest
   ```

---

## Web Console Development

The Web Console frontend is located in `web/` (React 19 + TypeScript + Vite + Tailwind CSS):

```bash
cd web
pnpm install
pnpm dev         # Starts Vite dev server with HMR on http://localhost:5173
```

When building for production:
```bash
# Builds the SPA and syncs bundled assets to ubt/api/static
pnpm build
python scripts/build_console_assets.py
```

---

## Submitting a Pull Request

1. Create a descriptive branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes with clear, focused commits.
3. Ensure all tests and linting pass locally before pushing:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy --strict ubt tests
   uv run python scripts/dump_compiler_edges.py
   uv run pytest -m fast
   ```
4. Push your branch and open a Pull Request against the `main` branch.
5. Provide a clear PR description using the PR template.
