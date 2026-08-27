import os

import pytest

from sdv_adas_validation_lab.contract import SensorFrame
from sdv_adas_validation_lab.tensorrt_runtime import TensorRtRuntime, build_engine
from test_runtime import identity_model


@pytest.mark.skipif(os.environ.get("RUN_TENSORRT_GPU") != "1", reason="local GPU smoke only")
def test_fp32_engine_build_and_execution(tmp_path) -> None:
  engine = build_engine(identity_model(tmp_path / "identity.onnx"), tmp_path / "identity.fp32.engine")
  prediction = TensorRtRuntime(engine.engine_path).infer(SensorFrame(0, 1, bytes(range(12)), 2, 2))
  assert engine.engine_path.exists()
  assert prediction.output_shapes == ((1, 3, 2, 2),)
