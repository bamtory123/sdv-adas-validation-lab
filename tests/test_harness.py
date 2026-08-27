import json

from sdv_adas_validation_lab.faults import FaultConfig
from sdv_adas_validation_lab.harness import run_repeated
from sdv_adas_validation_lab.replay import ReplaySource
from test_runtime import identity_model
from sdv_adas_validation_lab.runtime import OnnxReferenceRuntime


def test_repeated_harness_writes_isolated_artifacts(tmp_path) -> None:
  summaries = run_repeated(ReplaySource.synthetic(3, period_ns=1), FaultConfig(), tmp_path, repeats=2)
  runs = sorted(tmp_path.iterdir())
  assert [summary["published_frames"] for summary in summaries] == [3, 3]
  assert len(runs) == 2
  assert all((run / "manifest.json").exists() and (run / "frames.csv").exists() for run in runs)
  assert json.loads((runs[0] / "summary.json").read_text())["validity"] == "valid"
  assert summaries[0]["coverage"] == 1.0
  assert summaries[0]["transport_delay"]["max_ms"] is not None


def test_harness_writes_predictions_when_runtime_is_supplied(tmp_path) -> None:
  runtime = OnnxReferenceRuntime(identity_model(tmp_path / "identity.onnx"))
  output = tmp_path / "runs"
  summary = run_repeated(ReplaySource.synthetic(2), FaultConfig(), output, repeats=1, runtime=runtime)[0]
  assert summary["inference_frames"] == 2
  assert (next(output.iterdir()) / "predictions.csv").exists()
