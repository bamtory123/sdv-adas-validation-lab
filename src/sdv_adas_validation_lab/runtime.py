"""Reference ONNX Runtime adapter with explicit, reproducible preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .contract import SensorFrame


@dataclass(frozen=True, slots=True)
class Prediction:
  frame_id: int
  started_ns: int
  finished_ns: int
  output_shapes: tuple[tuple[int, ...], ...]

  @property
  def latency_ns(self) -> int:
    return self.finished_ns - self.started_ns


class OnnxReferenceRuntime:
  """ONNX reference path: RGB8 bytes -> NCHW float32 in the [0, 1] range."""

  def __init__(self, model_path: Path, *, provider: str = "CPUExecutionProvider") -> None:
    self._session = ort.InferenceSession(str(model_path), providers=[provider])
    self.provider = self._session.get_providers()[0]
    self._input = self._session.get_inputs()[0]

  @staticmethod
  def preprocess(frame: SensorFrame) -> np.ndarray:
    if frame.pixel_format != "rgb8":
      raise ValueError(f"unsupported pixel format: {frame.pixel_format}")
    expected = frame.width * frame.height * 3
    if len(frame.image) != expected:
      raise ValueError(f"rgb8 image length {len(frame.image)} does not match {expected}")
    image = np.frombuffer(frame.image, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0

  def infer(self, frame: SensorFrame) -> Prediction:
    started_ns = monotonic_ns()
    outputs = self._session.run(None, {self._input.name: self.preprocess(frame)})
    finished_ns = monotonic_ns()
    return Prediction(frame.frame_id, started_ns, finished_ns, tuple(tuple(output.shape) for output in outputs))
