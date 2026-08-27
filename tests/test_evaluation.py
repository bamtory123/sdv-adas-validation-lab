import json

import numpy as np
from PIL import Image

from sdv_adas_validation_lab.contract import SensorFrame
from sdv_adas_validation_lab.evaluation import evaluate, evaluate_manifest, load_label, load_labeled_replay
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


def test_labeled_replay_pairs_each_frame_with_its_indexed_label(tmp_path) -> None:
  (tmp_path / "image.rgb").write_bytes(b"rgb")
  Image.fromarray(np.array([[1]], dtype=np.uint8)).save(tmp_path / "label.png")
  (tmp_path / "frames.jsonl").write_text(
    json.dumps({"frame_id": 4, "capture_monotonic_ns": 10, "image_path": "image.rgb", "label_path": "label.png", "width": 1, "height": 1}) + "\n",
    encoding="utf-8",
  )
  source, labels = load_labeled_replay(tmp_path / "frames.jsonl")
  assert [frame.frame_id for frame in source] == [4]
  assert labels[0].tolist() == [[1]]


def test_manifest_evaluation_resizes_labels_to_runtime_output(tmp_path) -> None:
  (tmp_path / "image.rgb").write_bytes(b"rgb")
  Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(tmp_path / "label.png")
  (tmp_path / "frames.jsonl").write_text(
    json.dumps({"frame_id": 0, "capture_monotonic_ns": 1, "image_path": "image.rgb", "label_path": "label.png", "width": 1, "height": 1}) + "\n",
    encoding="utf-8",
  )
  result = evaluate_manifest(Runtime(), tmp_path / "frames.jsonl", class_count=2)
  assert result.evaluated_pixels == 4
