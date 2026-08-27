import numpy as np
import pytest

from sdv_adas_validation_lab.metrics import segmentation_metrics


def test_hand_calculated_segmentation_metrics() -> None:
  result = segmentation_metrics(np.array([[0, 1], [1, 2]]), np.array([[0, 1], [2, 2]]), class_count=3)
  assert result.evaluated_pixels == 4
  assert result.pixel_accuracy == 0.75
  assert result.class_iou == (1.0, 0.5, 0.5)
  assert result.mean_iou == pytest.approx(2 / 3)


def test_ignore_label_is_excluded() -> None:
  result = segmentation_metrics(np.array([[0, 1]]), np.array([[0, 255]]), class_count=2)
  assert result.evaluated_pixels == 1
  assert result.pixel_accuracy == 1.0


def test_invalid_label_map_is_rejected() -> None:
  with pytest.raises(ValueError):
    segmentation_metrics(np.zeros((1, 2)), np.zeros((1, 2)), class_count=2)
