import json

from sdv_adas_validation_lab.report import render


def test_report_renders_run_summary(tmp_path) -> None:
  run = tmp_path / "delay-50" / "run-001"
  run.mkdir(parents=True)
  (run / "summary.json").write_text(
    json.dumps({"validity": "valid", "outcome": "pass", "coverage": 1.0, "transport_delay": {"median_ms": 50}, "inference_latency": {"median_ms": 4}})
  )
  text = render(tmp_path)
  assert "| run-001 | valid | pass | 50 | 4 | 1.0 |" in text


def test_report_excludes_warmup_runs(tmp_path) -> None:
  for name in ("warmup-001", "run-001"):
    path = tmp_path / name
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({"validity": "valid", "outcome": "pass"}))
  text = render(tmp_path)
  assert "warmup-001" not in text
  assert "run-001" in text
