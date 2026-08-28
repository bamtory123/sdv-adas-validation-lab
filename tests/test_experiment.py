import json

import pytest

from sdv_adas_validation_lab.experiment import run_delay_experiment
from sdv_adas_validation_lab.replay import ReplaySource


def test_experiment_requires_a_new_output_directory(tmp_path) -> None:
  with pytest.raises(FileExistsError):
    run_delay_experiment(ReplaySource.synthetic(1), tmp_path / "missing.engine", tmp_path, delays_ms=[0], repeats=1, warmups=0, profile="unit")


def test_experiment_manifest_shape_without_runtime(monkeypatch, tmp_path) -> None:
  engine = tmp_path / "engine"
  engine.write_bytes(b"engine")
  monkeypatch.setattr("sdv_adas_validation_lab.experiment.TensorRtRuntime", lambda *args, **kwargs: object())
  monkeypatch.setattr("sdv_adas_validation_lab.experiment.run_once", lambda *args, **kwargs: {})
  manifest = run_delay_experiment(
    ReplaySource.synthetic(1), engine, tmp_path / "output", delays_ms=[0, 50], repeats=1, warmups=0, profile="unit", order_seed=7
  )
  assert [item["delay_ms"] for item in manifest["conditions"]] == [0, 50]
  stored = json.loads((tmp_path / "output" / "experiment_manifest.json").read_text())
  assert stored["repeats"] == 1
  assert sorted(stored["block_orders_ms"][0]) == [0, 50]
