import numpy as np

from sdv_adas_validation_lab.parity import compare
from sdv_adas_validation_lab.replay import ReplaySource
from sdv_adas_validation_lab.runtime import Prediction


class Runtime:
  def __init__(self, labels: np.ndarray) -> None:
    self.labels = labels

  def infer(self, frame):
    return Prediction(frame.frame_id, 1, 2, ((1, 2, 2),), primary_labels=self.labels)


def test_parity_aggregates_pixel_agreement() -> None:
  source = ReplaySource.synthetic(2)
  summary = compare(source, Runtime(np.array([[0, 1], [1, 0]])), Runtime(np.array([[0, 1], [0, 0]])))
  assert (summary.frames, summary.pixels, summary.matching_pixels) == (2, 8, 6)
  assert summary.label_agreement == 0.75
