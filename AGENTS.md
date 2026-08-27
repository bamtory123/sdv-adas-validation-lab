# Repository working rules

Keep changes small, testable, and within the active phase in `PLAN.md`.

## Documentation synchronization

Update documentation in the same commit when a change affects it:

- Architecture, interfaces, artifact layout, or scope boundary: update `ARCHITECTURE.md`.
- Milestone, acceptance criteria, experiment matrix, or ordering: update `PLAN.md`.
- Completed work, test evidence, current limitation, or immediate next task: update `STATUS.md`.
- Dependency/GPU compatibility expectation: update `configs/compatibility.yaml` and, once implemented, the run manifest schema.

Do not claim experiments, runtime behavior, or vehicle applicability that has not been verified and recorded.
