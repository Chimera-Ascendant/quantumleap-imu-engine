# TASK_BACKLOG (quantumleap-imu-engine)

version: 1
updated: 2025-09-12

```yaml
repo: quantumleap-imu-engine
sprint:
  name: Perception Baseline v0.1
  dates: [2025-09-13, 2025-09-20]
owners:
  - jonathan
labels:
  - data_engine
  - training
  - export
  - coreml
  - tests

tasks:
  - id: synth_gen_v2
    title: Synthetic generator v2 (per-exercise motion params + richer form errors)
    status: pending
    priority: high
    labels: [data_engine]
    acceptance_criteria:
      - YAML config includes per-exercise amplitude/tempo ranges
      - New error types mapped to labels

  - id: train_metrics
    title: Add training metrics & checkpoints (AUC, F1, loss curves)
    status: pending
    priority: medium
    labels: [training]
    acceptance_criteria:
      - Log CSV in out/ with loss per head and global
      - Save best checkpoint by val loss

  - id: coreml_validate
    title: Core ML export validation + sample runner
    status: pending
    priority: medium
    labels: [export, coreml]
    acceptance_criteria:
      - Minimal Swift runner verifies I/O shapes
      - Readme section on .mlpackage integration

  - id: ci_lint
    title: Add ruff/black checks to CI
    status: pending
    priority: low
    labels: [ci]
    acceptance_criteria:
      - CI fails on style errors
```
