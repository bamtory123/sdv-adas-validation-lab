# SDV ADAS Sensor Runtime Validation Lab

Reproducible, offline validation of a camera frame contract, replay pipeline, transport faults, and inference-runtime timing.

`v0.1` targets WSL2 GPU development with ONNX Runtime and TensorRT. It does not validate a real vehicle, HIL system, DRIVE AGX/DriveOS, or Hyundai production software.

## Current scope

- Versioned camera `SensorFrame` contract
- Replay-ready immutable frame representation
- Stream-integrity validation and unit tests
- Environment preflight recorded in run manifests
- JSONL/synthetic replay and repeatable non-blocking delay/drop fault smoke harness
- CPU-only Markdown evidence generation from ground-truth and replay artifacts
- Full PASCAL VOC 2012 validation accuracy reference for ONNX, TensorRT FP32, and TensorRT FP16, retained only in the local external-data cache

Openpilot/MetaDrive work remains a separate `v0.2` closed-loop adapter baseline; it is not copied into this repository.

## Working documents

- [Architecture](ARCHITECTURE.md): component boundaries, data path, and invariants.
- [Plan](PLAN.md): v0.1 milestones and release evidence.
- [Status](STATUS.md): completed work, verified evidence, and the next task.
- [FCN/VOC evidence](docs/model-fcn-resnet50.md): model contract, dataset boundaries, and bounded results.
- [Release readiness](docs/release-readiness.md): evidence boundary and publication checklist.
- [Redacted evidence sample](samples/evidence/README.md): CI-verified release-safe report input.
