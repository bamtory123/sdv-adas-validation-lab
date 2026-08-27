# Status

Last updated: 2026-08-28 (Asia/Seoul)

## Current state

Phase 0 and Phase 1 are underway; Phase 4 has a tested minimal queue implementation. The repository and independent WSL environment exist. No model, real replay dataset, TensorRT engine, formal benchmark, or release result exists yet.

## Completed

- Public repository created: `bamtory123/sdv-adas-validation-lab`.
- WSL2 runtime verified on NVIDIA GeForce RTX 4080, driver 610.47.
- Installed and import-verified: ONNX 1.22.0, ONNX Runtime GPU 1.29.0, TensorRT 10.16.1.11.
- Added `SensorFrame` contract (`sensor-frame/v1`) with immutable fields and basic validation.
- Added stream monotonicity validation.
- Added JSONL file-backed and deterministic synthetic replay sources.
- Added a non-blocking delay/drop queue. Producer submission never sleeps; the publisher releases frames only at a monotonic deadline. Zero delay uses the same queue path.
- Added a repeatable replay/fault CLI which writes isolated `manifest.json`, `frames.csv`, `events.jsonl`, and `summary.json` artifacts per run.
- Added an ONNX Runtime reference adapter with fixed RGB8-to-NCHW float32 `[0,1]` preprocessing. Published frames can now produce per-frame `predictions.csv` latency and output-shape records.
- Added local TensorRT FP32 engine build and static-shape direct GPU execution via CUDA Python bindings. TensorRT is selectable in the same replay/fault harness as the ONNX reference runtime.
- GPU smoke-tested two repeated TensorRT runs: 5 synthetic frames, 5 ms delay, no configured drops; each produced 5 published and inferred frames with `valid/pass`. This uses a generated Identity ONNX model and is runtime-integration evidence only, not segmentation accuracy evidence.
- Pinned TensorRT 10.16.1.11 because it exposes explicit FP16 precision control. FP32 and FP16 generated-model engine build/execution smoke both pass locally (2 passed).
- Selected the MIT-licensed ONNX Model Zoo FCN-ResNet50 opset-11 semantic-segmentation candidate and recorded source checksums. Its FP32 TensorRT engine parses and builds with a fixed `1x3x520x520` profile. The full-model FP16 build has not yet produced a completed engine artifact and remains unresolved.
- Smoke-tested two repeated runs: 8 synthetic frames, 10 ms delay, drop every third frame; each produced 6 published frames and 2 configured drops with `valid/pass`.
- Added runtime preflight collector and `configs/compatibility.yaml`.
- Unit tests: 18 passed plus 2 opt-in local GPU smoke tests on 2026-08-28.

## Not yet implemented

- Replay source content hashing, stream coverage checks, and the dataset/model decision gate.
- A redistribution-compatible segmentation model, model-specific output decoding, TensorRT FP16, and FP32/FP16 accuracy comparison.
- Fault queue, run state machine, manifest, KPI, verdict, report, batch runner, CI, and formal experiments.

## Next task

Implement FCN-specific resize/normalization and output decoding, then execute the selected model through the ONNX reference and FP32 TensorRT paths before diagnosing the full-model FP16 build.

## Historical baseline

The earlier openpilot/MetaDrive work remains in its separate repository/branch as a future v0.2 closed-loop adapter baseline. It is not evidence for this v0.1 runtime benchmark.
