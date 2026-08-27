# SDV ADAS Sensor Runtime Validation Lab

Reproducible, offline validation of a camera frame contract, replay pipeline, transport faults, and inference-runtime timing.

`v0.1` targets WSL2 GPU development with ONNX Runtime and TensorRT. It does not validate a real vehicle, HIL system, DRIVE AGX/DriveOS, or Hyundai production software.

## Current scope

- Versioned camera `SensorFrame` contract
- Replay-ready immutable frame representation
- Stream-integrity validation and unit tests
- Environment preflight recorded in run manifests

Openpilot/MetaDrive work remains a separate `v0.2` closed-loop adapter baseline; it is not copied into this repository.

## Working documents

- [Architecture](ARCHITECTURE.md): component boundaries, data path, and invariants.
- [Plan](PLAN.md): v0.1 milestones and release evidence.
- [Status](STATUS.md): completed work, verified evidence, and the next task.
