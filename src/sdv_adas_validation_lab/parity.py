"""Semantic-output parity KPI for two runtimes on identical replay frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .contract import SensorFrame
from .replay import ReplaySource
from .runtime import OnnxReferenceRuntime, Prediction
from .tensorrt_runtime import TensorRtRuntime


class RuntimeAdapter(Protocol):
  def infer(self, frame: SensorFrame) -> Prediction: ...


@dataclass(frozen=True, slots=True)
class ParitySummary:
  frames: int
  pixels: int
  matching_pixels: int
  label_agreement: float


def compare(source: ReplaySource, reference: RuntimeAdapter, candidate: RuntimeAdapter) -> ParitySummary:
  frames = pixels = matching = 0
  for frame in source:
    expected = reference.infer(frame).primary_labels
    observed = candidate.infer(frame).primary_labels
    if expected is None or observed is None or expected.shape != observed.shape:
      raise ValueError(f"runtime output labels are incompatible for frame {frame.frame_id}")
    frames += 1
    pixels += expected.size
    matching += int(np.count_nonzero(expected == observed))
  return ParitySummary(frames, pixels, matching, matching / pixels if pixels else 0.0)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", required=True, type=Path)
  parser.add_argument("--onnx-model", required=True, type=Path)
  parser.add_argument("--tensorrt-engine", required=True, type=Path)
  parser.add_argument("--profile", default="fcn_resnet50_voc")
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  summary = compare(
    ReplaySource.from_jsonl(args.replay),
    OnnxReferenceRuntime(args.onnx_model, profile=args.profile),
    TensorRtRuntime(args.tensorrt_engine, profile=args.profile),
  )
  args.output.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
  print(json.dumps(asdict(summary)))


if __name__ == "__main__":
  main()
