"""Ground-truth label evaluation adapter for a replay/runtime pair."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import time_ns
from typing import Protocol

import numpy as np
from PIL import Image

from .contract import SensorFrame
from .metrics import SegmentationMetrics, segmentation_metrics
from .preflight import collect
from .replay import ReplaySource
from .runtime import Prediction


class RuntimeAdapter(Protocol):
  def infer(self, frame: SensorFrame) -> Prediction: ...


def load_label(path, *, shape: tuple[int, int]) -> np.ndarray:
  """Load an indexed label PNG and resize only with nearest-neighbor sampling."""
  label = Image.open(path)
  if label.mode not in {"L", "P"}:
    raise ValueError(f"label image must be indexed/grayscale, got {label.mode}")
  return np.asarray(label.resize((shape[1], shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)


def load_labeled_replay(manifest_path: Path) -> tuple[ReplaySource, tuple[np.ndarray, ...]]:
  """Load a replay manifest whose every frame names an indexed label image."""
  source = ReplaySource.from_jsonl(manifest_path)
  labels: list[np.ndarray] = []
  records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
  for frame, record in zip(source, records, strict=True):
    if "label_path" not in record:
      raise ValueError(f"missing label_path for frame {frame.frame_id}")
    labels.append(load_label(manifest_path.parent / record["label_path"], shape=(frame.height, frame.width)))
  return source, tuple(labels)


def evaluate_manifest(
  runtime: RuntimeAdapter,
  manifest_path: Path,
  *,
  class_count: int,
  ignore_label: int = 255,
) -> SegmentationMetrics:
  """Evaluate a labeled replay after resizing each label to the runtime output."""
  source = ReplaySource.from_jsonl(manifest_path)
  records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
  predictions: list[np.ndarray] = []
  targets: list[np.ndarray] = []
  for frame, record in zip(source, records, strict=True):
    if "label_path" not in record:
      raise ValueError(f"missing label_path for frame {frame.frame_id}")
    prediction = runtime.infer(frame).primary_labels
    if prediction is None:
      raise ValueError("runtime did not produce a primary label map")
    predictions.append(prediction)
    targets.append(load_label(manifest_path.parent / record["label_path"], shape=prediction.shape))
  if not predictions:
    raise ValueError("no labeled frames")
  return segmentation_metrics(
    np.concatenate(predictions, axis=0),
    np.concatenate(targets, axis=0),
    class_count=class_count,
    ignore_label=ignore_label,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Evaluate a labeled replay against one segmentation runtime.")
  parser.add_argument("--replay", required=True, type=Path)
  parser.add_argument("--onnx-model", type=Path)
  parser.add_argument("--tensorrt-engine", type=Path)
  parser.add_argument("--profile", default="fcn_resnet50_voc")
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--manifest", type=Path, help="optional provenance manifest path")
  parser.add_argument("--class-count", type=int, default=21)
  parser.add_argument("--ignore-label", type=int, default=255)
  args = parser.parse_args()
  if bool(args.onnx_model) == bool(args.tensorrt_engine):
    parser.error("provide exactly one of --onnx-model or --tensorrt-engine")
  if args.onnx_model:
    from .runtime import OnnxReferenceRuntime

    runtime: RuntimeAdapter = OnnxReferenceRuntime(args.onnx_model, profile=args.profile)
  else:
    from .tensorrt_runtime import TensorRtRuntime

    runtime = TensorRtRuntime(args.tensorrt_engine, profile=args.profile)
  result = asdict(evaluate_manifest(runtime, args.replay, class_count=args.class_count, ignore_label=args.ignore_label))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if args.manifest:
    artifact = args.onnx_model or args.tensorrt_engine
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
      json.dumps(
        {
          "kind": "ground_truth_evaluation/v1",
          "created_wall_ns": time_ns(),
          "preflight": collect(),
          "replay_manifest_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest(),
          "runtime": "onnx_reference" if args.onnx_model else "tensorrt",
          "runtime_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        indent=2,
        sort_keys=True,
      )
      + "\n",
      encoding="utf-8",
    )
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()


def evaluate(
  runtime: RuntimeAdapter,
  frames: Iterable[SensorFrame],
  labels: Iterable[np.ndarray],
  *,
  class_count: int,
  ignore_label: int = 255,
) -> SegmentationMetrics:
  predictions: list[np.ndarray] = []
  targets: list[np.ndarray] = []
  for frame, target in zip(frames, labels, strict=True):
    prediction = runtime.infer(frame).primary_labels
    if prediction is None:
      raise ValueError("runtime did not produce a primary label map")
    if prediction.shape != target.shape:
      raise ValueError(f"label shape mismatch for frame {frame.frame_id}")
    predictions.append(prediction)
    targets.append(target)
  if not predictions:
    raise ValueError("no labeled frames")
  return segmentation_metrics(np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0), class_count=class_count, ignore_label=ignore_label)
