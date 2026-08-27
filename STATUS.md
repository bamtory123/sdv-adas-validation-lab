# Status

Last updated: 2026-08-28 (Asia/Seoul)

## Current state

Phase 0 and Phase 1 are underway. The repository and independent WSL environment exist; the sensor contract and basic stream validator are tested. No model, replay dataset, TensorRT engine, fault queue, formal benchmark, or release result exists yet.

## Completed

- Public repository created: `bamtory123/sdv-adas-validation-lab`.
- WSL2 runtime verified on NVIDIA GeForce RTX 4080, driver 610.47.
- Installed and import-verified: ONNX 1.22.0, ONNX Runtime GPU 1.29.0, TensorRT 11.2.1.2.
- Added `SensorFrame` contract (`sensor-frame/v1`) with immutable fields and basic validation.
- Added stream monotonicity validation.
- Added runtime preflight collector and `configs/compatibility.yaml`.
- Unit tests: 8 passed on 2026-08-28.

## Not yet implemented

- Replay source and dataset/model decision gate.
- Reference inference, TensorRT engine build, FP32/FP16 comparison.
- Fault queue, run state machine, manifest, KPI, verdict, report, batch runner, CI, and formal experiments.

## Next task

Implement a deterministic replay fixture and the Phase 1 `ReplaySource`, then expand stream-preflight tests for sequence, timestamp, dimensions, and coverage.

## Historical baseline

The earlier openpilot/MetaDrive work remains in its separate repository/branch as a future v0.2 closed-loop adapter baseline. It is not evidence for this v0.1 runtime benchmark.
