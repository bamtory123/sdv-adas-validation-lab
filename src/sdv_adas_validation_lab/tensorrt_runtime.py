"""Local TensorRT engine build and static-shape FP32 execution support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from .contract import SensorFrame
from .runtime import Prediction, _prediction, preprocess_rgb8


def _check(result: tuple[object, ...], operation: str) -> tuple[object, ...]:
  if result[0] != cudart.cudaError_t.cudaSuccess:
    raise RuntimeError(f"CUDA {operation} failed: {result[0]}")
  return result[1:]


@dataclass(frozen=True, slots=True)
class EngineBuild:
  precision: str
  engine_path: Path


def build_engine(
  model_path: Path, engine_path: Path, *, precision: str = "fp32", input_shape: tuple[int, ...] | None = None
) -> EngineBuild:
  """Build a local TensorRT engine. Engines are intentionally not portable artifacts."""
  if precision not in {"fp32", "fp16"}:
    raise ValueError("precision must be fp32 or fp16")
  logger = trt.Logger(trt.Logger.ERROR)
  builder = trt.Builder(logger)
  # TensorRT 11 uses explicit batch dimensions by default; the old flag was removed.
  network = builder.create_network(0)
  parser = trt.OnnxParser(network, logger)
  if not parser.parse_from_file(str(model_path)):
    errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
    raise ValueError(f"TensorRT could not parse {model_path}: {errors}")
  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024)
  input_tensor = network.get_input(0)
  if any(dimension == -1 for dimension in input_tensor.shape):
    if input_shape is None:
      raise ValueError("dynamic ONNX input requires an explicit input_shape")
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
  if precision == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)
  serialized = builder.build_serialized_network(network, config)
  if serialized is None:
    raise RuntimeError("TensorRT engine build failed")
  engine_path.parent.mkdir(parents=True, exist_ok=True)
  engine_path.write_bytes(bytes(serialized))
  return EngineBuild(precision, engine_path)


class TensorRtRuntime:
  """Static-shape TensorRT adapter using the same preprocessing as the ONNX reference."""

  def __init__(self, engine_path: Path, *, profile: str = "unit") -> None:
    logger = trt.Logger(trt.Logger.ERROR)
    self._runtime = trt.Runtime(logger)
    self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if self._engine is None:
      raise ValueError(f"could not deserialize TensorRT engine: {engine_path}")
    self._context = self._engine.create_execution_context()
    self.profile = profile
    self._input_name = next(
      self._engine.get_tensor_name(index)
      for index in range(self._engine.num_io_tensors)
      if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
    )
    self._output_names = tuple(
      self._engine.get_tensor_name(index)
      for index in range(self._engine.num_io_tensors)
      if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
    )

  def infer(self, frame: SensorFrame) -> Prediction:
    value = preprocess_rgb8(frame, profile=self.profile)
    started_ns = monotonic_ns()
    stream, = _check(cudart.cudaStreamCreate(), "stream create")
    allocations: list[int] = []
    try:
      input_ptr, = _check(cudart.cudaMalloc(value.nbytes), "input allocation")
      allocations.append(input_ptr)
      _check(
        cudart.cudaMemcpyAsync(input_ptr, value.ctypes.data, value.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream),
        "input copy",
      )
      if not self._context.set_input_shape(self._input_name, value.shape):
        raise ValueError(f"TensorRT engine does not accept input shape {value.shape}")
      self._context.set_tensor_address(self._input_name, input_ptr)
      outputs: list[np.ndarray] = []
      for name in self._output_names:
        shape = tuple(self._context.get_tensor_shape(name))
        output = np.empty(shape, dtype=trt.nptype(self._engine.get_tensor_dtype(name)))
        output_ptr, = _check(cudart.cudaMalloc(output.nbytes), "output allocation")
        allocations.append(output_ptr)
        self._context.set_tensor_address(name, output_ptr)
        outputs.append(output)
      if not self._context.execute_async_v3(stream):
        raise RuntimeError("TensorRT inference execution failed")
      for name, output, output_ptr in zip(self._output_names, outputs, allocations[1:], strict=True):
        _check(
          cudart.cudaMemcpyAsync(output.ctypes.data, output_ptr, output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream),
          f"output copy {name}",
        )
      _check(cudart.cudaStreamSynchronize(stream), "stream synchronize")
      return _prediction(frame.frame_id, started_ns, monotonic_ns(), outputs)
    finally:
      for pointer in allocations:
        cudart.cudaFree(pointer)
      cudart.cudaStreamDestroy(stream)
