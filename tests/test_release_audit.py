from pathlib import Path

from sdv_adas_validation_lab.release_audit import forbidden_paths


def test_release_audit_blocks_external_runtime_artifacts() -> None:
  blocked = forbidden_paths([Path("model.onnx"), Path("runtime.engine"), Path("frame.rgb"), Path("photo.jpg"), Path("label.png")])
  assert len(blocked) == 5


def test_release_audit_allows_redacted_summaries() -> None:
  assert forbidden_paths([Path("samples/evidence/ground-truth.json"), Path("docs/release-notes.md")]) == []
