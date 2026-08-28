# FCN-ResNet50 model decision

`v0.1` integration candidate is ONNX Model Zoo's FCN-ResNet50 semantic-segmentation model, opset 11.

- Source: `onnx/models`, `validated/vision/object_detection_segmentation/fcn`
- License: MIT (repository and model README SPDX header)
- Model URL: `https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.onnx`
- Model SHA-256: `8abd3ae6c258e6fd210d8fb296261ea5c0603c6cc2f568db6097f6fbaaeff0d5`
- Sample-image SHA-256: `80c12ee468dccf34ca0bd5261ee484c59d8e63d33e87e524a7e3437044e5d082`
- Input: dynamic `NCHW`, RGB `[0,1]`, then ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- Outputs: `out` and `aux`, each with 21 VOC classes. `out` is the primary segmentation output.

Raw models, images, replay data, and TensorRT engines are local artifacts and must not be committed. This choice enables runtime and output-parity work; it does not establish ADAS or vehicle-performance accuracy.

## Accuracy-data boundary

The Model Zoo sample-data archive includes ONNX test inputs and golden outputs, not semantic ground-truth labels. Those tensors are valid for ONNX regression checks only; they must not be reported as segmentation accuracy. The repository has a tested mIoU/pixel-accuracy metric implementation, but a final accuracy result requires a separately licensed labeled replay source with an explicit class mapping.

## Ground-truth source

The external evaluation source is PASCAL VOC 2012 class segmentation. Its `SegmentationClass` PNGs use the 21-class VOC index space expected by this model; index `255` is void and is ignored by the metric. The dataset is downloaded to a local cache and never committed or redistributed by this repository. The train/validation archive is accepted only after it matches the public MD5 `6cd6e144f989b92b3379bac3b3de84fd`. `sdv_adas_validation_lab.voc.build_labeled_fixture` converts explicitly selected image IDs from an external `VOCdevkit/VOC2012` directory into this project's raw-RGB replay manifest with a paired `label_path`.

## Ground-truth smoke result

On 2026-08-28, the fixed VOC validation image IDs `2007_000033`, `2007_000042`, `2007_000061`, `2007_000123`, `2007_000129`, `2007_000175`, `2007_000187`, and `2007_000323` produced 2,057,767 non-void pixels after nearest-neighbor label resizing to 520×520. The ONNX reference measured pixel accuracy 0.970537 and mIoU 0.595712; TensorRT FP16 measured 0.970542 and 0.595641. Per-runtime summary and provenance manifests stay in the local cache with the source data. This is a fixed eight-image integration smoke, not a full VOC benchmark or release KPI.

The same eight-image fixture completed a warm-up-excluded TensorRT FP16 delay matrix with three measured runs per 0/50/100/150ms condition. All twelve runs were `valid/pass` with coverage 1.0. Their actual median transport-delay ranges were 0.0055–0.0058ms, 50.43–50.81ms, 100.47–101.22ms, and 150.47–150.94ms respectively. The matrix uses seed `20260828` and records the per-block order; it validates transport-delay repeatability, not full-dataset model accuracy or a release KPI.
