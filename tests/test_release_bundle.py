import json

from sdv_adas_validation_lab.release_bundle import FILES, archive_bundle, build


def test_release_bundle_contains_only_redacted_documents(tmp_path) -> None:
  for relative in FILES:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(relative, encoding="utf-8")
  sample = tmp_path / "samples" / "evidence"
  (sample / "replay" / "run-001").mkdir(parents=True)
  (sample / "ground-truth.json").write_text(json.dumps({"evaluated_pixels": 1, "mean_iou": 1.0, "pixel_accuracy": 1.0}), encoding="utf-8")
  (sample / "replay" / "run-001" / "summary.json").write_text(json.dumps({"validity": "valid", "outcome": "pass"}), encoding="utf-8")

  manifest = build(tmp_path, tmp_path / "bundle")

  assert manifest["kind"] == "portfolio_release_bundle/v1"
  assert (tmp_path / "bundle" / "evidence-sample.md").is_file()
  assert (tmp_path / "bundle" / "manifest.json").is_file()
  assert archive_bundle(tmp_path / "bundle", tmp_path / "portfolio.zip").is_file()
