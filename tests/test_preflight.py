from sdv_adas_validation_lab import preflight


def test_collect_includes_reproducibility_fields(monkeypatch) -> None:
  monkeypatch.setattr(preflight, "_gpu", lambda: ["Test GPU, 1.0, 1 MiB"])
  data = preflight.collect()

  assert data["python"]
  assert data["gpu"] == ["Test GPU, 1.0, 1 MiB"]
  assert set(data["packages"]) == {"onnx", "onnxruntime-gpu", "tensorrt"}
