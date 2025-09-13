# CODING_STANDARDS (quantumleap-imu-engine)

## Python
- Black formatting, 120 cols:
  - `black --line-length 120 .`
- Ruff linting (add later to CI):
  - `ruff check .`
- Use type hints for public APIs. Prefer dataclasses/yaml configs.

## Training
- Keep quick-start configs under `configs/`.
- Write trainers to be deterministic where possible (seeds, eval mode).
- Save checkpoints to `out*/checkpoints/`.

## Tests
- Pytest smoke tests should validate model I/O shapes and basic forward pass.
- Avoid GPU assumptions in tests; run on CPU.

## Commits & PRs
- Conventional commits (feat, fix, chore, docs, refactor, test, ci).
- Keep PRs small; include a brief description and run instructions when relevant.

## Docs
- Keep `README.md` updated with data generation, training, and Core ML export steps.
- Document input/output tensor shapes and label mapping.
