# Plan

This is the executable v0.1 plan. The detailed rationale remains in the original project plan; this file owns implementation order and acceptance criteria.

## Release target: `v0.1.0-portfolio`

Demonstrate repeatable offline camera replay under controlled transport delay, comparing ONNX reference, TensorRT FP32, and TensorRT FP16. Formal experiments use the same fixed model, dataset subset, preprocessing, and warm-up policy. Conditions are repeated three times; warm-up runs are retained but excluded from formal KPI.

## Milestones

| Phase | Deliverable | Verification | Status |
| --- | --- | --- | --- |
| 0 | Repository, pinned environment, preflight | imports GPU runtimes; manifest-ready environment facts | Completed |
| 1 | Versioned `SensorFrame`, replay source, stream preflight | contract and replay fixture tests | Completed |
| 2 | ONNX reference runtime and fixed preprocessing | deterministic output on sample frames | Completed |
| 3 | TensorRT FP32/FP16 build and runtime | output comparison and local GPU smoke | Completed |
| 4 | Non-blocking delay/drop faults | fake-clock FIFO/deadline/overflow tests | Completed |
| 5 | Manifest, KPI, verdict, batch runner, report | hand-calculated KPI and report fixture tests | Completed |
| 6 | Formal experiments and evidence | 3 retained runs per runtime/fault condition | Completed: bounded fixture evidence |
| 7 | Portfolio release | CI, documentation, sample results, tagged public release | Completed: [`v0.1.0-portfolio`](https://github.com/bamtory123/sdv-adas-validation-lab/releases/tag/v0.1.0-portfolio) |
| 8 | Full VOC reference evaluation | one fixed full validation fixture, ONNX/FP32/FP16 ground-truth summaries and provenance manifests | Completed: external-cache evidence only |

## Immediate execution order

1. Preserve the v0.1 evidence boundary: the public release is bounded replay evidence, not a vehicle, HIL, DRIVE AGX, or closed-loop claim.
2. Treat the completed full-VOC accuracy check as external-cache v0.2 reference evidence; it does not change the tagged v0.1 release or establish driving performance.
3. Scope additional fault campaigns and any closed-loop simulator work as separate post-v0.1 milestones.

## Formal v0.1 experiment matrix

- Runtime baseline: reference ONNX, TensorRT FP32, TensorRT FP16; three measured runs each.
- Delay study: fixed runtime TensorRT FP16; 0/50/100/150 ms; three measured runs each.
- Frame drop and image-quality faults are implementation/test scope in v0.1; they are not required formal experiments.

## Evidence required for release

- Per-run manifests and logs, including invalid attempts/retries.
- Latency, throughput, deadline-miss, output-age, frame-coverage, queue/drop, and segmentation-accuracy KPI.
- Explicit `valid/pass`, `valid/fail`, or `invalid/not_evaluated` verdict.
- Unit tests and CPU-only report generation in CI.
- Documentation of WSL2 limits and the synthetic/offline validation boundary.
