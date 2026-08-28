import json

from sdv_adas_validation_lab.evidence import render_evidence


def test_evidence_renders_metric_and_replay_sections(tmp_path) -> None:
  evaluation = tmp_path / "evaluation.json"
  evaluation.write_text(json.dumps({"evaluated_pixels": 12, "mean_iou": 0.5, "pixel_accuracy": 0.75}), encoding="utf-8")
  run = tmp_path / "runs" / "run-001"
  run.mkdir(parents=True)
  (run / "summary.json").write_text(json.dumps({"validity": "valid", "outcome": "pass"}), encoding="utf-8")

  text = render_evidence([("holdout", evaluation)], [tmp_path / "runs"])

  assert "| holdout | 12 | 0.500000 | 0.750000 |" in text
  assert "## Replay runs: runs" in text
  assert "run-001" in text
