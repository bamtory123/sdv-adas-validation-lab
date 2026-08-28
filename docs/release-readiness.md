# Release readiness

## Current evidence

- Pinned WSL2 environment, replay contract, non-blocking delay/drop queue, manifests, verdicts, CI, and CPU-only report generation are implemented and tested.
- The deterministic 50-image VOC repeatability fixture has three retained zero-delay runs for ONNX CPU reference, TensorRT FP32, and TensorRT FP16; the FP16 delay study has three retained runs at 0/50/100/150ms.
- A non-overlapping 200-image VOC hold-out was evaluated once without tuning. Source manifests, runtime manifests, and the consolidated Markdown evidence report remain in the local cache because they reference external data and untracked engines.

## Release boundary

This evidence supports a bounded offline camera-replay portfolio demonstration. It does not support claims of:

- full VOC benchmark performance;
- driving, openpilot, CARLA, MetaDrive, vehicle, HIL, or production-ADAS validation;
- real-time guarantees; or
- Hyundai/NVIDIA production-system equivalence.

## Published release

`v0.1.0-portfolio` is published as a bounded offline replay portfolio release. Its asset is the CI-verified redacted bundle; raw VOC data, ONNX models, TensorRT engines, and engine-bound manifests remain excluded.

CI also runs a release-content audit that rejects tracked `.onnx`, `.engine`, raw RGB, JPEG, and PNG artifacts.

The repository contains the published [release notes](release-notes-v0.1.0-portfolio.md).

`python -m sdv_adas_validation_lab.release_bundle --output <directory> --archive <zip-path>` builds a redacted release bundle, audits its content, writes a manifest, and optionally creates a ZIP release asset.

The full declared VOC split remains a post-v0.1 scope decision.
