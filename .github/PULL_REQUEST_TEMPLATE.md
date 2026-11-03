## Description

Please include a summary of the changes and the related issue/feature motivation.

- Fixes #(issue)
- Closes #(issue)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring / Performance improvement

## Checklist

- [ ] I have read the [CONTRIBUTING.md](CONTRIBUTING.md) guide.
- [ ] I have run `uv run ruff check .` and `uv run ruff format --check .` with no issues.
- [ ] I have run `uv run mypy --strict ubt tests` and it passes.
- [ ] I have run `uv run python scripts/dump_compiler_edges.py` and the layering invariant passes.
- [ ] I have run `uv run pytest -m fast` and all tests pass.
- [ ] I have added tests that prove my fix is effective or that my feature works.
