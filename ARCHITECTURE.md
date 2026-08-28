# Architecture

## Purpose

`sdv-adas-validation-lab` is an offline, reproducible validation lab for an ADAS camera stream and NVIDIA GPU inference runtimes in WSL2. v0.1 compares a reference ONNX runtime with TensorRT FP32 and FP16, then measures controlled transport-fault effects.

It is not a vehicle, HIL, DRIVE AGX/DriveOS, Hyundai-production, or real-time-guarantee validation.

## v0.1 data path

```text
ReplaySource -> SensorFrame -> StreamValidator -> FaultQueue -> RuntimeAdapter
     |               |               |                 |              |
     +----------- frames.csv --------+------ events.jsonl ------- predictions
                                                                  |
                                                          KPI / verdict / report
```

- `ReplaySource`: deterministic recorded-frame input. The only v0.1 source.
- `SensorFrame`: immutable, versioned contract. IDs and monotonic timestamps are the shared truth between stages.
- `StreamValidator`: rejects malformed, non-monotonic, missing, or incompatible input before a benchmark.
- `FaultQueue`: a non-blocking publisher path. Every delay condition, including 0 ms, uses the same queue. Deterministic configured drops are recorded; queue overflow is invalid.
- `RuntimeAdapter`: reference ONNX Runtime, TensorRT FP32, or TensorRT FP16 with identical preprocessing and output decoding.
- `KPI / verdict / report`: separates execution validity from performance outcome and generates reproducible artifacts.

The parity comparator uses the same `ReplaySource` frames for both runtimes and aggregates primary segmentation label agreement. It is a runtime-consistency KPI; ground-truth accuracy requires labeled data and is tracked separately. A labeled replay manifest adds one `label_path` per frame; labels must be indexed/grayscale and are resized with nearest-neighbor sampling to the runtime's output shape before the metric is calculated.

Delay experiments execute one warm-up per condition, then shuffle all delay conditions independently within each measured repeat block using a recorded deterministic seed. The aggregate manifest records the exact block order, so elapsed GPU/WSL effects are not systematically assigned to one delay condition.

The replay loop records all ready publications before invoking the runtime adapter. Inference latency is still captured per frame, but runtime work is downstream of camera transport and cannot block the simulated camera producer.

`configs/evaluation-policy.yaml` separates explicit development smoke IDs, the seeded repeatability fixture, and a non-overlapping seeded hold-out. Each fixture's local `source.json` records selected IDs and selection metadata; hold-out metrics are recorded once and must not guide tuning.

The CPU-only evidence generator combines named ground-truth summaries with one or more replay-run directories. It reads only artifacts already produced by the harness and writes Markdown; it never accesses models, raw images, labels, or TensorRT engines.

The replay harness records actual publish delay rather than only the requested fault delay, plus inference latency and published/captured coverage. This isolates transport scheduling from runtime execution time.

Configured frame drops are expected fault events and are excluded from the coverage denominator; overflow and unaccounted loss remain invalid conditions.

The report renderer is CPU-only and reads saved run summaries; GPU inference is never required in CI to regenerate a report.

## Boundary and versions

- Host: Windows + WSL2; GPU: NVIDIA GeForce RTX 4080.
- Runtime versions and expected hardware belong in `configs/compatibility.yaml`.
- Each run records actual software, GPU/driver, source/model/config hashes, command, and Git state in its manifest. A TensorRT engine is a local build artifact and is never committed.
- `openpilot`/MetaDrive is a v0.2 source and closed-loop adapter. CARLA is only a connectivity/interface smoke-test topic until a later release.

## Run artifacts

The implemented parity run writes an isolated output directory containing:

- `manifest.json`: actual environment, Git state, command, hashes, and configuration.
- `frames.csv`: source/fault timestamps, queue depth, publish/drop outcome.
- `predictions.csv`: runtime output timing and accuracy inputs.
- `events.jsonl`: state transitions and errors.
- `summary.json` and `report.md`: verdict and KPI summary.

## Design invariants

1. Use monotonic nanoseconds for performance measurements; wall-clock is only for human-readable run metadata.
2. Do not block the source thread for delay injection.
3. Compare runtimes with identical frames, preprocessing, output decoding, warm-up policy, and measurement window.
4. Preserve invalid attempts and retries; do not silently discard them.
5. `validity` answers whether a run can be evaluated; `outcome` answers whether evaluated KPI thresholds passed.

## Current reference-runtime contract

The implemented reference adapter accepts `rgb8` frames whose byte count is exactly `width * height * 3`, converts them to contiguous NCHW `float32` in `[0,1]`, and writes per-frame inference latency and output shapes. The FCN profile additionally resizes to 520×520, applies ImageNet normalization, and derives the primary argmax label map for runtime-parity comparison.

The implemented TensorRT adapter builds local static-shape FP32 or FP16 engines, allocates CUDA buffers/stream per inference, and uses the same preprocessing. TensorRT 10.16.1.11 is pinned because it exposes explicit FP16 build control. Engine files stay local and are excluded from Git.
