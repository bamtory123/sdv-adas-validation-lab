# Status

Last updated: 2026-08-28 (Asia/Seoul)

## Current state

Phase 0 and Phase 1 are underway; Phase 4 has a tested minimal queue implementation. The repository and independent WSL environment exist. No model, real replay dataset, TensorRT engine, formal benchmark, or release result exists yet.

## Completed

- Public repository created: `bamtory123/sdv-adas-validation-lab`.
- WSL2 runtime verified on NVIDIA GeForce RTX 4080, driver 610.47.
- Installed and import-verified: ONNX 1.22.0, ONNX Runtime GPU 1.29.0, TensorRT 11.2.1.2.
- Added `SensorFrame` contract (`sensor-frame/v1`) with immutable fields and basic validation.
- Added stream monotonicity validation.
- Added JSONL file-backed and deterministic synthetic replay sources.
- Added a non-blocking delay/drop queue. Producer submission never sleeps; the publisher releases frames only at a monotonic deadline. Zero delay uses the same queue path.
- Added a repeatable replay/fault CLI which writes isolated `manifest.json`, `frames.csv`, `events.jsonl`, and `summary.json` artifacts per run.
- Smoke-tested two repeated runs: 8 synthetic frames, 10 ms delay, drop every third frame; each produced 6 published frames and 2 configured drops with `valid/pass`.
- Added runtime preflight collector and `configs/compatibility.yaml`.
- Unit tests: 15 passed on 2026-08-28.

## Not yet implemented

- Replay source content hashing, stream coverage checks, and the dataset/model decision gate.
- Reference inference, TensorRT engine build, FP32/FP16 comparison.
- Fault queue, run state machine, manifest, KPI, verdict, report, batch runner, CI, and formal experiments.

## Next task

Implement a fixed ONNX reference model and preprocessing path over the replay interface; then compare its deterministic output before adding TensorRT engine builds.

## Historical baseline

The earlier openpilot/MetaDrive work remains in its separate repository/branch as a future v0.2 closed-loop adapter baseline. It is not evidence for this v0.1 runtime benchmark.
