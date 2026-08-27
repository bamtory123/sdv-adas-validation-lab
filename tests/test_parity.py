import numpy as np

import json

from sdv_adas_validation_lab.parity import ParitySummary, compare, write_run
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


def test_run_artifacts_separate_validity_and_outcome(tmp_path) -> None:
  replay = tmp_path / "frames.jsonl"
  model = tmp_path / "model.onnx"
  engine = tmp_path / "model.engine"
  replay.write_text("{}\n")
  model.write_bytes(b"model")
  engine.write_bytes(b"engine")
  result = write_run(tmp_path / "run", ParitySummary(1, 10, 9, 0.9), replay_manifest=replay, onnx_model=model, tensorrt_engine=engine, minimum_agreement=0.95)
  assert (result["validity"], result["outcome"]) == ("valid", "fail")
  assert json.loads((tmp_path / "run" / "summary.json").read_text())["matching_pixels"] == 9
  assert len((tmp_path / "run" / "events.jsonl").read_text().splitlines()) == 3
