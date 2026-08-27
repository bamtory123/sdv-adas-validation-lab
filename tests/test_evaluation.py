import numpy as np
from PIL import Image

from sdv_adas_validation_lab.contract import SensorFrame
from sdv_adas_validation_lab.evaluation import evaluate, load_label
from sdv_adas_validation_lab.runtime import Prediction


class Runtime:
  def infer(self, frame):
    return Prediction(frame.frame_id, 1, 2, ((1, 2, 2),), primary_labels=np.array([[0, 1], [1, 0]], dtype=np.uint8))


def test_evaluation_uses_all_labeled_frames() -> None:
  frame = SensorFrame(0, 1, b"rgb", 1, 1)
  result = evaluate(Runtime(), [frame, frame], [np.array([[0, 1], [1, 0]], dtype=np.uint8)] * 2, class_count=2)
  assert (result.evaluated_pixels, result.pixel_accuracy, result.mean_iou) == (8, 1.0, 1.0)


def test_label_loader_uses_nearest_neighbor(tmp_path) -> None:
  path = tmp_path / "label.png"
  Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(path)
  loaded = load_label(path, shape=(2, 2))
  assert loaded.tolist() == [[0, 1], [0, 1]]
