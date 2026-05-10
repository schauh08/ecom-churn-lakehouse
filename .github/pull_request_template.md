## Summary

Describe the change in 2–5 sentences.

## Why this change

Explain the problem being solved and the repo layer it touches.

## Validation

- [ ] `uv run ruff check .`
- [ ] `uv run black --check .`
- [ ] `uv run mypy src services`
- [ ] `uv run pytest tests/unit tests/contract services/api/tests -q`
- [ ] `uv run pytest tests/integration/test_slice_e2e.py -q -m e2e` (if applicable)

## Checklist

- [ ] Scope is small and focused
- [ ] Contracts or schemas updated if needed
- [ ] Tests added or updated
- [ ] Docs/runbook updated if behavior changed
- [ ] No secrets or local paths committed