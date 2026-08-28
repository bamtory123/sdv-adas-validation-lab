# Status

Last updated: 2026-08-28 (Asia/Seoul)

## Current state

Phases 0–6 are complete for bounded offline evidence in the independent WSL environment. The FCN ONNX model, external replay fixtures, and FP32/FP16 TensorRT engines exist only in the local cache. A public release is prepared but not published; no full-dataset benchmark or real-world validation result exists.

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
- Selected the MIT-licensed ONNX Model Zoo FCN-ResNet50 opset-11 semantic-segmentation candidate and recorded source checksums. Its FP32 and FP16 TensorRT engines parse and build with a fixed `1x3x520x520` profile.
- Implemented FCN RGB resize/normalization and primary label-map hashing. On the downloaded sample image, ONNX reference and TensorRT FP32 output shapes were equal and primary pixel-label agreement was 99.9815%; raw output checksums differed, so exact float equality is not used as the parity criterion.
- Added a local fixture preparer and multi-image parity CLI. On both provided FCN samples (540,800 resized pixels), TensorRT FP32 achieved 99.9906% primary label agreement against the ONNX reference. This is runtime parity, not ground-truth segmentation accuracy.
- Promoted parity output to an isolated run artifact: `manifest.json` records preflight and source/model/engine SHA-256; `summary.json` separates `validity` and `outcome`; `events.jsonl` records state transitions. The two-image FCN FP32 run is `valid/pass` with a 0.999 agreement threshold.
- The same two-image parity run for FCN TensorRT FP16 is `valid/pass`: 540,627 of 540,800 primary labels agree with ONNX reference (99.9680%). The local serialized FP16 engine is about 68MB and remains untracked.
- Added actual transport-delay and inference-latency KPI to replay run summaries. FCN FP16 smoke runs for target 0/50/100/150 ms had coverage 1.0 and median actual transport delay of approximately 0.01/55.56/106.30/153.93 ms. These are one-repeat, two-frame queue/KPI integration checks, not formal benchmark evidence.
- Replay harness manifests now include deterministic source-content SHA-256 and the executing Git commit/dirty state.
- Added CPU-only Markdown batch report generation over run `summary.json` files, with validity/outcome, actual median delay, inference latency, and coverage columns.
- Configured frame drops are now counted separately from unexpected loss. A six-frame delay/drop smoke (50 ms, every second frame dropped) produced 3 published and 3 expected drops with coverage 1.0 and `valid/pass`.
- Added queue-capacity control and overflow coverage. A three-frame, 100 ms, depth-one smoke produced two overflow events and correctly returned `invalid/not_evaluated`.
- Added a GitHub Actions CPU-only workflow for unit tests and report generation. TensorRT engine smoke remains an explicit local GPU test.
- GitHub Actions CI passed after GPU-free preflight handling was added.
- Retained a pre-formal FCN FP16 delay matrix: 0/50/100/150 ms, three runs per condition, all `valid/pass` with coverage 1.0. The first-run warm-up effect is not yet excluded, so these artifacts demonstrate repeatability plumbing rather than formal release KPI.
- Added `--warmups`: warm-up runs are retained in distinct artifact directories but excluded from returned measured results and reports.
- Added a batch delay-experiment runner with `experiment_manifest.json`. A warm-up-excluded FCN FP16 matrix completed: four delay conditions × three measured runs, all `valid/pass`, coverage 1.0. Its report excludes warm-up directories. This is a two-image micro-fixture validation of execution mechanics, not the final release benchmark.
- Inspected the Model Zoo FCN archive: it provides ONNX golden outputs, not semantic ground-truth labels. Added tested pixel-accuracy/mIoU calculation with ignored-label support and a `label_path` replay-manifest adapter.
- Selected external PASCAL VOC 2012 class segmentation for the FCN/VOC 21-class mapping. Its replay fixture builder and published train/validation archive MD5 verification are tested; raw images and labels stay in the local cache and are not committed.
- Added a label-evaluation adapter: indexed/grayscale label PNGs are resized only with nearest-neighbor sampling and aggregated with runtime label maps for pixel accuracy/mIoU.
- Downloaded and MD5-verified the external VOC 2012 train/validation archive, then built a fixed eight-image validation replay fixture. ONNX reference measured mIoU 0.595712 and pixel accuracy 0.970537; TensorRT FP16 measured mIoU 0.595641 and pixel accuracy 0.970542 across 2,057,767 non-void pixels. This is a ground-truth integration smoke only, not a full VOC benchmark.
- Ground-truth evaluation output can write provenance manifests with replay and runtime-artifact SHA-256 plus runtime preflight data.
- Corrected the delay experiment runner so delay conditions are shuffled within each measured repeat block and the exact seed/order is recorded. The prior grouped-order artifact is retained but excluded from interpretation because it confounded elapsed runtime effects with the final condition.
- Re-ran the warm-up-excluded VOC eight-image TensorRT FP16 matrix: 0/50/100/150 ms, three measured runs per condition, all `valid/pass` with coverage 1.0. Actual median delay ranges were 0.0055–0.0058, 50.43–50.81, 100.47–101.22, and 150.47–150.94 ms. This is repeatability evidence for the micro-fixture only.
- Discovered that the original harness synchronously invoked runtime inference while publishing ready frames, which inflated later transport timing on longer fixtures. Runtime inference is now downstream of transport; a slow-runtime unit test verifies it cannot stall producer timing. Earlier 50-image delay artifact is retained but excluded from interpretation and will be re-run.
- Built a deterministic, non-overlapping 50-image VOC validation fixture (seed `20260828`, 12,671,063 non-void pixels). ONNX measured mIoU 0.643037 / pixel accuracy 0.921312; TensorRT FP16 measured 0.643125 / 0.921344. The corrected warm-up-excluded 12-run FP16 delay matrix is all `valid/pass`, coverage 1.0, with actual median delay ranges 0.0033–0.0039, 50.36–50.94, 100.06–100.81, and 150.47–150.83 ms for the four target conditions.
- Added `configs/evaluation-policy.yaml` and fixture selection provenance. The non-overlapping 200-image hold-out (seed `20260829`) was evaluated once without tuning: 51,230,864 non-void pixels; ONNX mIoU 0.713040 / pixel accuracy 0.938304 and TensorRT FP16 mIoU 0.713115 / pixel accuracy 0.938329. It is a held-out reference, not full-dataset or real-world validation.
- Added and executed a CPU-only evidence generator that combines named ground-truth JSON summaries and replay-run artifacts into one Markdown report. The generated cache artifact includes both 50-image/hold-out metrics and all 12 corrected delay runs.
- Completed the bounded 50-image runtime baseline: TensorRT FP32 ground-truth mIoU 0.642985 / pixel accuracy 0.921310. Three warm-up-excluded zero-delay runs per runtime were all `valid/pass`, coverage 1.0; local median inference ranges were ONNX CPU reference 481.64–490.31ms, TensorRT FP32 43.24–46.71ms, and TensorRT FP16 21.19–22.06ms. These are local replay measurements, not real-time guarantees.
- Added a release-safe redacted evidence sample and CI evidence-generation check. It includes only compact result summaries; no raw data, models, engines, local paths, or environment manifests are tracked.
- Added a `v0.1.0-portfolio` draft release note with reproduction boundaries. It is not a published GitHub release or tag.
- Added and locally generated an unpublished release-bundle artifact containing only redacted documents, a redacted evidence report, and SHA-256 manifest. Bundle generation and its content audit run in CI.
- Smoke-tested two repeated runs: 8 synthetic frames, 10 ms delay, drop every third frame; each produced 6 published frames and 2 configured drops with `valid/pass`.
- Added runtime preflight collector and `configs/compatibility.yaml`.
- Unit tests: 43 passed plus 2 opt-in local GPU smoke tests on 2026-08-28.

## Not yet implemented

- A declared larger/full VOC split, if the bounded evidence is not the intended public-release scope.
- Public Git tag and GitHub release creation, after release-scope approval.

## Next task

Review [release readiness](docs/release-readiness.md) and decide whether to expand to a full declared VOC split or publish the bounded evidence; no public tag/release has been created.

## Historical baseline

The earlier openpilot/MetaDrive work remains in its separate repository/branch as a future v0.2 closed-loop adapter baseline. It is not evidence for this v0.1 runtime benchmark.
