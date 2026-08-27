"""Ground-truth semantic-segmentation metrics with explicit ignored labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SegmentationMetrics:
  evaluated_pixels: int
  pixel_accuracy: float
  mean_iou: float
  class_iou: tuple[float | None, ...]


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray, *, class_count: int, ignore_label: int = 255) -> SegmentationMetrics:
  if prediction.shape != target.shape or prediction.ndim != 2:
    raise ValueError("prediction and target must be equal-shape 2D label maps")
  if not np.issubdtype(prediction.dtype, np.integer) or not np.issubdtype(target.dtype, np.integer):
    raise ValueError("label maps must use integer dtypes")
  valid = target != ignore_label
  if np.any(prediction[valid] < 0) or np.any(prediction[valid] >= class_count) or np.any(target[valid] >= class_count):
    raise ValueError("labels exceed class_count")
  matrix = np.bincount(
    class_count * target[valid].astype(np.int64) + prediction[valid].astype(np.int64), minlength=class_count**2
  ).reshape(class_count, class_count)
  ious: list[float | None] = []
  for label in range(class_count):
    intersection = matrix[label, label]
    union = matrix[label, :].sum() + matrix[:, label].sum() - intersection
    ious.append(float(intersection / union) if union else None)
  present = [value for value in ious if value is not None]
  evaluated = int(valid.sum())
  return SegmentationMetrics(
    evaluated,
    float(np.trace(matrix) / evaluated) if evaluated else 0.0,
    float(np.mean(present)) if present else 0.0,
    tuple(ious),
  )
