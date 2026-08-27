from pathlib import Path

import onnx
from onnx import TensorProto, helper
import pytest

from sdv_adas_validation_lab.contract import SensorFrame
from sdv_adas_validation_lab.runtime import OnnxReferenceRuntime


def identity_model(path: Path) -> Path:
  graph = helper.make_graph(
    [helper.make_node("Identity", ["input"], ["output"])],
    "identity",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 2])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 2])],
  )
  onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
  return path


def test_reference_runtime_infers_one_rgb_frame(tmp_path) -> None:
  runtime = OnnxReferenceRuntime(identity_model(tmp_path / "identity.onnx"))
  prediction = runtime.infer(SensorFrame(4, 1, bytes(range(12)), 2, 2))
  assert prediction.frame_id == 4
  assert prediction.output_shapes == ((1, 3, 2, 2),)
  assert prediction.latency_ns >= 0


def test_preprocess_rejects_wrong_rgb_size(tmp_path) -> None:
  runtime = OnnxReferenceRuntime(identity_model(tmp_path / "identity.onnx"))
  with pytest.raises(ValueError, match="image length"):
    runtime.infer(SensorFrame(0, 1, b"too short", 2, 2))
