"""Ground-truth label evaluation adapter for a replay/runtime pair."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .contract import SensorFrame
from .metrics import SegmentationMetrics, segmentation_metrics
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
