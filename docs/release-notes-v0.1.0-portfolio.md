# v0.1.0-portfolio

## Included

- Immutable camera-frame contract, deterministic replay, stream validation, and non-blocking delay/drop fault queue.
- ONNX reference plus TensorRT FP32/FP16 adapters with fixed FCN-ResNet50 VOC preprocessing.
- Per-run manifests, validity/outcome verdicts, transport/inference KPI, repeat orchestration, Markdown reports, and CPU-only CI evidence generation.
- Bounded VOC repeatability and hold-out evidence documented in [the FCN/VOC record](model-fcn-resnet50.md).

## Reproduce

1. Follow the pinned environment details in `configs/compatibility.yaml`.
2. Supply externally obtained model/data artifacts outside the repository.
3. Use the fixture, evaluation, experiment, report, and evidence CLIs documented in the source tree.
4. Run `uv run pytest -q`; CI also validates the redacted evidence sample.

## Scope boundary

This release demonstrates offline camera-replay/runtime validation under controlled transport delay. It does not validate a vehicle, openpilot closed-loop driving, CARLA/MetaDrive, HIL, DRIVE AGX/DriveOS, Hyundai production software, real-time operation, or safety performance. Raw VOC data, models, TensorRT engines, and local environment manifests are not included.
