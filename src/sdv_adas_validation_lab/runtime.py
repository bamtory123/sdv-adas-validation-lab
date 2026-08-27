"""Reference ONNX Runtime adapter with explicit, reproducible preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from time import monotonic_ns
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .contract import SensorFrame


@dataclass(frozen=True, slots=True)
class Prediction:
  frame_id: int
  started_ns: int
  finished_ns: int
  output_shapes: tuple[tuple[int, ...], ...]
  output_hashes: tuple[str, ...] = ()
  primary_label_hash: str | None = None
  primary_labels: np.ndarray | None = None

  @property
  def latency_ns(self) -> int:
    return self.finished_ns - self.started_ns


def preprocess_rgb8(frame: SensorFrame, *, profile: str = "unit") -> np.ndarray:
  """Shared preprocessing contract for reference and TensorRT runtimes."""
  if frame.pixel_format != "rgb8":
    raise ValueError(f"unsupported pixel format: {frame.pixel_format}")
  expected = frame.width * frame.height * 3
  if len(frame.image) != expected:
    raise ValueError(f"rgb8 image length {len(frame.image)} does not match {expected}")
  image = np.frombuffer(frame.image, dtype=np.uint8).reshape(frame.height, frame.width, 3)
  if profile == "fcn_resnet50_voc":
    image = np.asarray(Image.fromarray(image).resize((520, 520), Image.Resampling.BILINEAR))
  elif profile != "unit":
    raise ValueError(f"unsupported preprocessing profile: {profile}")
  value = np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0
  if profile == "fcn_resnet50_voc":
    value = (value - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]) / np.asarray(
      [0.229, 0.224, 0.225], dtype=np.float32
    )[None, :, None, None]
  return value


def _prediction(frame_id: int, started_ns: int, finished_ns: int, outputs: list[np.ndarray]) -> Prediction:
  hashes = tuple(hashlib.sha256(output.tobytes()).hexdigest() for output in outputs)
  primary_labels = np.argmax(outputs[0][0], axis=0).astype(np.uint8)
  return Prediction(
    frame_id,
    started_ns,
    finished_ns,
    tuple(tuple(output.shape) for output in outputs),
    hashes,
    hashlib.sha256(primary_labels.tobytes()).hexdigest(),
    primary_labels,
  )


class OnnxReferenceRuntime:
  """ONNX reference path: RGB8 bytes -> NCHW float32 in the [0, 1] range."""

  def __init__(self, model_path: Path, *, provider: str = "CPUExecutionProvider", profile: str = "unit") -> None:
    self._session = ort.InferenceSession(str(model_path), providers=[provider])
    self.provider = self._session.get_providers()[0]
    self._input = self._session.get_inputs()[0]
    self.profile = profile

  def infer(self, frame: SensorFrame) -> Prediction:
    started_ns = monotonic_ns()
    outputs = self._session.run(None, {self._input.name: preprocess_rgb8(frame, profile=self.profile)})
    finished_ns = monotonic_ns()
    return _prediction(frame.frame_id, started_ns, finished_ns, outputs)
