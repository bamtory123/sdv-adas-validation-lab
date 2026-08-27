import json

from sdv_adas_validation_lab.faults import FaultConfig
from sdv_adas_validation_lab.harness import run_repeated
from sdv_adas_validation_lab.replay import ReplaySource


def test_repeated_harness_writes_isolated_artifacts(tmp_path) -> None:
  summaries = run_repeated(ReplaySource.synthetic(3, period_ns=1), FaultConfig(), tmp_path, repeats=2)
  runs = sorted(tmp_path.iterdir())
  assert [summary["published_frames"] for summary in summaries] == [3, 3]
  assert len(runs) == 2
  assert all((run / "manifest.json").exists() and (run / "frames.csv").exists() for run in runs)
  assert json.loads((runs[0] / "summary.json").read_text())["validity"] == "valid"
