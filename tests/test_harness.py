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
  assert json.loads((runs[0] / "manifest.json").read_text())["source_hash"]
  assert summaries[0]["coverage"] == 1.0
  assert summaries[0]["transport_delay"]["max_ms"] is not None


def test_harness_writes_predictions_when_runtime_is_supplied(tmp_path) -> None:
  runtime = OnnxReferenceRuntime(identity_model(tmp_path / "identity.onnx"))
  output = tmp_path / "runs"
  summary = run_repeated(ReplaySource.synthetic(2), FaultConfig(), output, repeats=1, runtime=runtime)[0]
  assert summary["inference_frames"] == 2
  assert (next(output.iterdir()) / "predictions.csv").exists()


def test_configured_drop_preserves_expected_coverage(tmp_path) -> None:
  summary = run_repeated(ReplaySource.synthetic(3), FaultConfig(drop_every=2), tmp_path, repeats=1)[0]
  assert (summary["dropped_frames"], summary["expected_dropped_frames"], summary["coverage"]) == (1, 1, 1.0)
  assert summary["outcome"] == "pass"


def test_queue_overflow_is_invalid_not_evaluated(tmp_path) -> None:
  summary = run_repeated(ReplaySource.synthetic(3, period_ns=1), FaultConfig(delay_ms=100, max_pending=1), tmp_path, repeats=1)[0]
  assert (summary["validity"], summary["outcome"]) == ("invalid", "not_evaluated")
  assert summary["invalid_events"] > 0


def test_warmup_artifact_is_excluded_from_returned_measurements(tmp_path) -> None:
  summaries = run_repeated(ReplaySource.synthetic(2), FaultConfig(), tmp_path, repeats=1, warmups=1)
  assert len(summaries) == 1
  assert len(list(tmp_path.glob("warmup-*"))) == 1
  assert len(list(tmp_path.glob("run-*"))) == 1
