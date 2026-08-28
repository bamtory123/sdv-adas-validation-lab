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

## Before a public tag

1. Decide whether the bounded fixture evidence is the intended `v0.1.0-portfolio` scope, or run a declared larger split first.
2. If publishing, attach only the CI-verified [redacted sample input](../samples/evidence/README.md), a generated Markdown report, and reproduction commands—never raw VOC data, ONNX models, TensorRT engines, or engine-bound manifests.
3. Confirm README and release notes retain the scope boundary above.

The repository contains a [draft release note](release-notes-v0.1.0-portfolio.md) for this review; it is not a published GitHub release.

No Git tag or public GitHub release has been created by this document.
